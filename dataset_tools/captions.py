"""Auto-generate MusicGen captions from audio features + style tags.

No manual labelling is required: each chunk's caption combines the configured
style tags with a librosa-estimated tempo and musical key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dataset_tools.config import DatasetConfig
from dataset_tools.metadata import ChunkRecord, write_jsonl
from metalcore.audio_io import load_audio

METADATA_JSONL = "metadata.jsonl"

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (major / minor), used to estimate key.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def estimate_tempo(y: np.ndarray, sr: int) -> float:
    """Estimate tempo (BPM) with librosa, returning 0.0 on failure."""
    try:
        import librosa

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        value = float(np.atleast_1d(tempo)[0])
        return value if np.isfinite(value) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def estimate_key(y: np.ndarray, sr: int) -> Optional[str]:
    """Estimate the musical key (e.g. ``"E minor"``) via chroma correlation.

    Returns ``None`` if estimation fails.
    """
    try:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        if not np.any(chroma_mean):
            return None

        best_score = -np.inf
        best_key: Optional[str] = None
        for shift in range(12):
            major_corr = _corr(np.roll(_MAJOR_PROFILE, shift), chroma_mean)
            minor_corr = _corr(np.roll(_MINOR_PROFILE, shift), chroma_mean)
            if major_corr > best_score:
                best_score, best_key = major_corr, f"{_PITCH_CLASSES[shift]} major"
            if minor_corr > best_score:
                best_score, best_key = minor_corr, f"{_PITCH_CLASSES[shift]} minor"
        return best_key
    except Exception:  # noqa: BLE001
        return None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_caption(
    style_tags: List[str], tempo: float, key: Optional[str]
) -> str:
    """Compose a caption string from style tags and estimated features."""
    parts: List[str] = list(style_tags)
    if tempo and tempo > 0:
        parts.append(f"around {int(round(tempo))} BPM")
    if key:
        parts.append(f"in {key}")
    return ", ".join(parts)


def caption_chunk(path: Path, cfg: DatasetConfig) -> Tuple[str, float, Optional[str]]:
    """Return ``(caption, tempo, key)`` for a single chunk."""
    y, sr = load_audio(path, sr=cfg.sample_rate, mono=True)
    y = y[0]
    tempo = estimate_tempo(y, sr)
    key = estimate_key(y, sr)
    return build_caption(cfg.style_tags, tempo, key), tempo, key


def run_captions(
    records: List[ChunkRecord],
    dataset_root: Path,
    cfg: DatasetConfig,
    logger: logging.Logger,
) -> List[ChunkRecord]:
    """Fill the ``caption`` field for each record and write ``metadata.jsonl``.

    Args:
        records: Chunk records (from the preprocess step).
        dataset_root: Folder that contains the ``chunks/`` directory; chunk
            ``audio`` paths are resolved relative to it.
        cfg: Dataset configuration (supplies style tags).
        logger: Logger for progress.

    Returns:
        The records with captions populated.
    """
    dataset_root = Path(dataset_root)
    total = len(records)
    for i, record in enumerate(records, start=1):
        chunk_path = dataset_root / record.audio
        try:
            caption, tempo, key = caption_chunk(chunk_path, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Caption failed for %s: %s", chunk_path.name, exc)
            caption = ", ".join(cfg.style_tags)
            tempo, key = 0.0, None
        record.caption = caption
        if i % 25 == 0 or i == total:
            logger.info("Captioned %d/%d chunk(s)", i, total)
        else:
            logger.debug("Captioned %s: %s", chunk_path.name, caption)

    jsonl_path = dataset_root / METADATA_JSONL
    write_jsonl(jsonl_path, records)
    logger.info("Wrote metadata with captions -> %s", jsonl_path)
    return records
