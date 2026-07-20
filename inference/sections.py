"""Generate and stitch MusicGen instrumental sections into a full arrangement.

MusicGen is coherent for ~30 s, so a full song is built from per-section clips
that are equal-power cross-faded together. Generation reuses Stage 2 and is
imported lazily; the stitching DSP is self-contained and testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from inference.config import AssemblyConfig
from metalcore.audio_io import load_audio, save_audio


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    """Equal-power cross-fade mono ``a`` into mono ``b`` over ``overlap`` samples."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    overlap = int(max(0, min(overlap, a.shape[0], b.shape[0])))
    if overlap == 0:
        return np.concatenate([a, b]).astype(np.float32)

    t = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)

    head = a[:-overlap]
    mixed = a[-overlap:] * fade_out + b[:overlap] * fade_in
    tail = b[overlap:]
    return np.concatenate([head, mixed, tail]).astype(np.float32)


def stitch_sections(section_audio: List[np.ndarray], sr: int, crossfade_seconds: float) -> np.ndarray:
    """Cross-fade a list of mono section clips into one continuous track."""
    if not section_audio:
        raise ValueError("No sections to stitch.")
    overlap = int(round(crossfade_seconds * sr))
    result = np.asarray(section_audio[0], dtype=np.float32).reshape(-1)
    for clip in section_audio[1:]:
        result = equal_power_crossfade(result, np.asarray(clip, dtype=np.float32).reshape(-1), overlap)
    return result


def generate_sections(
    music_config: str,
    adapter_dir: str,
    cfg: AssemblyConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> List[Tuple[str, Path]]:
    """Generate one WAV per configured section using the trained MusicGen LoRA.

    Returns a list of ``(section_name, wav_path)`` in arrangement order.
    """
    from music_training.generate import generate  # lazy (torch)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Tuple[str, Path]] = []

    for i, section in enumerate(cfg.sections):
        name = section["name"]
        logger.info("Generating section %d/%d: %s", i + 1, len(cfg.sections), name)
        section_out = out_dir / name
        written = generate(
            config_path=music_config,
            adapter_dir=adapter_dir,
            prompt=section["prompt"],
            out_dir=str(section_out),
            seconds=float(section["seconds"]),
            num_samples=1,
            guidance_scale=cfg.guidance_scale,
            logger=logger,
            seed=cfg.seed + i,
        )
        produced.append((name, written[0]))
    return produced


def load_and_stitch(
    section_paths: List[Path],
    cfg: AssemblyConfig,
    logger: logging.Logger,
    out_path: Path,
) -> Path:
    """Load section WAVs (resampling to the work rate) and stitch them."""
    clips: List[np.ndarray] = []
    for path in section_paths:
        audio, _ = load_audio(path, sr=cfg.work_sample_rate, mono=True)
        clips.append(audio[0])
    full = stitch_sections(clips, cfg.work_sample_rate, cfg.crossfade_seconds)
    save_audio(out_path, full[np.newaxis, :], cfg.work_sample_rate)
    logger.info("Stitched %d section(s) -> %s (%.1fs)", len(clips), out_path, full.shape[0] / cfg.work_sample_rate)
    return Path(out_path)
