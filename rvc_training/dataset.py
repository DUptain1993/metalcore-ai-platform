"""Vocal dataset preparation for RVC training.

Takes isolated vocal stems (see :mod:`rvc_training.isolate`) organised per
vocalist, splits them on silence, cuts fixed-length clips, drops bad/near-silent
fragments, resamples to the RVC training rate, and writes a training-ready
dataset. Supports two blend strategies for combining multiple vocalists:

* ``merged``      -> pool all vocalists into ONE dataset (single blended voice);
* ``per_speaker`` -> keep one dataset per vocalist.

Self-contained (numpy + librosa + soundfile); no GPU required, fully testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from metalcore.audio_io import AUDIO_EXTS, load_audio, save_audio
from rvc_training.config import RVCConfig


@dataclass
class PrepStats:
    """Summary of a dataset-preparation run."""

    speakers: int
    files: int
    clips: int
    seconds: float
    dropped: int


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return -120.0 if rms <= 1e-9 else 20.0 * float(np.log10(rms))


def split_on_silence(y: np.ndarray, sr: int, top_db: float) -> List[Tuple[int, int]]:
    """Return non-silent ``(start, end)`` sample intervals via librosa."""
    import librosa

    if y.ndim == 2:
        y = y[0]
    intervals = librosa.effects.split(y, top_db=top_db)
    return [(int(s), int(e)) for s, e in intervals]


def segment_audio(y: np.ndarray, sr: int, cfg: RVCConfig) -> Tuple[List[np.ndarray], int]:
    """Split ``y`` on silence then cut fixed-length clips.

    Returns ``(clips, dropped)`` where each clip is mono ``(1, N)`` and
    ``dropped`` counts fragments rejected for being too short or too quiet.
    """
    if y.ndim == 2:
        y = y[0]
    clip_len = int(round(cfg.segment_seconds * sr))
    min_len = int(round(cfg.min_segment_seconds * sr))

    clips: List[np.ndarray] = []
    dropped = 0
    for start, end in split_on_silence(y, sr, cfg.silence_top_db):
        region = y[start:end]
        pos = 0
        while pos < region.shape[0]:
            clip = region[pos : pos + clip_len]
            pos += clip_len
            if clip.shape[0] < min_len:
                dropped += 1
                continue
            if _rms_db(clip) < cfg.min_rms_db:
                dropped += 1
                continue
            clips.append(clip[np.newaxis, :].astype(np.float32))
    return clips, dropped


def _list_audio(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def prepare_speaker(
    speaker_dir: Path,
    out_dir: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
    prefix: str = "",
) -> Tuple[int, int, float, int]:
    """Prepare all vocal files for one speaker into ``out_dir``.

    Returns ``(files, clips, seconds, dropped)``. Respects
    ``cfg.max_minutes_per_speaker`` to keep a merged blend balanced.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_audio(speaker_dir)
    max_seconds = cfg.max_minutes_per_speaker * 60.0

    n_clips = 0
    total_seconds = 0.0
    dropped_total = 0
    for path in files:
        try:
            y, _ = load_audio(path, sr=cfg.prep_sample_rate, mono=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path.name, exc)
            continue
        clips, dropped = segment_audio(y, cfg.prep_sample_rate, cfg)
        dropped_total += dropped
        for clip in clips:
            if total_seconds >= max_seconds:
                logger.info("Reached per-speaker cap (%.1f min) for %s", cfg.max_minutes_per_speaker, speaker_dir.name)
                break
            name = f"{prefix}{path.stem}_{n_clips:05d}.wav"
            save_audio(out_dir / name, clip, cfg.prep_sample_rate)
            n_clips += 1
            total_seconds += clip.shape[-1] / cfg.prep_sample_rate
        if total_seconds >= max_seconds:
            break

    logger.info(
        "Speaker '%s': %d file(s) -> %d clip(s) (%.1fs, %d dropped)",
        speaker_dir.name,
        len(files),
        n_clips,
        total_seconds,
        dropped_total,
    )
    return len(files), n_clips, total_seconds, dropped_total


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> PrepStats:
    """Prepare an RVC dataset from a folder of per-vocalist subdirectories.

    Expected input layout::

        input_dir/
        ├── vocalist_a/  *.wav (isolated vocals)
        ├── vocalist_b/  *.wav
        └── vocalist_c/  *.wav

    Output layout depends on ``cfg.blend_mode``:

    * ``merged``      -> ``output_dir/merged/`` (all clips, speaker-prefixed);
    * ``per_speaker`` -> ``output_dir/<speaker>/`` per vocalist.

    Returns aggregate :class:`PrepStats`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    speaker_dirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    if not speaker_dirs:
        # Allow a flat folder of wavs as a single unnamed speaker.
        if _list_audio(input_dir):
            speaker_dirs = [input_dir]
        else:
            raise RuntimeError(f"No vocalist subfolders or audio found under {input_dir}")

    logger.info("Preparing %d vocalist(s) in '%s' mode", len(speaker_dirs), cfg.blend_mode)

    total_files = total_clips = total_dropped = 0
    total_seconds = 0.0

    for speaker in speaker_dirs:
        if cfg.blend_mode == "merged":
            target = output_dir / "merged"
            prefix = f"{speaker.name}__"
        else:
            target = output_dir / speaker.name
            prefix = ""
        files, clips, seconds, dropped = prepare_speaker(speaker, target, cfg, logger, prefix)
        total_files += files
        total_clips += clips
        total_seconds += seconds
        total_dropped += dropped

    stats = PrepStats(
        speakers=len(speaker_dirs),
        files=total_files,
        clips=total_clips,
        seconds=total_seconds,
        dropped=total_dropped,
    )
    logger.info(
        "Dataset ready: %d speaker(s), %d clip(s), %.1f min, %d dropped -> %s",
        stats.speakers,
        stats.clips,
        stats.seconds / 60.0,
        stats.dropped,
        output_dir,
    )
    return stats
