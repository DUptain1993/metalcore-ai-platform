"""Orchestrate a full song: instrumental (+ optional vocal) -> mixed master."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from inference import mix as mixmod
from inference import master as mastermod
from inference import sections as sectionmod
from inference.config import AssemblyConfig
from metalcore.audio_io import load_audio


def assemble_song(
    out_base: Path,
    cfg: AssemblyConfig,
    logger: logging.Logger,
    music_config: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    instrumental_path: Optional[str] = None,
    vocal_path: Optional[str] = None,
    work_dir: Optional[Path] = None,
) -> List[Path]:
    """Build and export a full song.

    Instrumental source resolution (first match wins):
        1. ``instrumental_path`` -- an existing full instrumental WAV;
        2. otherwise generate per-section clips with MusicGen (needs
           ``music_config`` + ``adapter_dir``) and stitch them.

    If ``vocal_path`` is given it is mixed over the instrumental; otherwise the
    instrumental is mastered on its own.

    Args:
        out_base: Output path without extension.
        cfg: Assembly configuration.
        logger: Logger.
        music_config: Path to ``music_lora.yaml`` (for generation).
        adapter_dir: Trained MusicGen adapter dir (for generation).
        instrumental_path: Optional existing instrumental WAV.
        vocal_path: Optional vocal stem WAV.
        work_dir: Scratch directory for section clips/intermediates.

    Returns:
        List of exported file paths (WAV, and MP3 if enabled).
    """
    out_base = Path(out_base)
    work_dir = Path(work_dir) if work_dir else out_base.parent / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instrumental
    if instrumental_path:
        logger.info("Using existing instrumental: %s", instrumental_path)
        instrumental, _ = load_audio(instrumental_path, sr=cfg.work_sample_rate, mono=True)
        instrumental = instrumental[0]
    else:
        if not (music_config and adapter_dir):
            raise ValueError(
                "Provide either --instrumental, or both --music-config and --adapter to generate one."
            )
        produced = sectionmod.generate_sections(music_config, adapter_dir, cfg, work_dir / "sections", logger)
        clips = []
        for _, path in produced:
            audio, _ = load_audio(path, sr=cfg.work_sample_rate, mono=True)
            clips.append(audio[0])
        instrumental = sectionmod.stitch_sections(clips, cfg.work_sample_rate, cfg.crossfade_seconds)

    # 2. Mix (or instrumental-only)
    if vocal_path:
        logger.info("Mixing vocal stem: %s", vocal_path)
        vocal, _ = load_audio(vocal_path, sr=cfg.work_sample_rate, mono=True)
        mixed = mixmod.mix(instrumental, vocal[0], cfg, logger)
    else:
        logger.info("No vocal stem; mastering instrumental only.")
        mixed = mixmod.to_stereo(
            instrumental * mixmod.db_to_gain(cfg.instrumental_gain_db), cfg.stereo_width
        )

    # 3. Master + export
    return mastermod.export(mixed, cfg.work_sample_rate, out_base, cfg, logger)
