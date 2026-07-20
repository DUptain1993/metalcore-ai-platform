"""Vocal isolation via Demucs (for building vocal datasets from full mixes).

Wraps the ``demucs`` CLI. Demucs is pip-installable and runs on a Kaggle T4.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List

from metalcore.audio_io import AUDIO_EXTS
from rvc_training.config import RVCConfig


def build_demucs_cmd(input_path: Path, out_dir: Path, model: str) -> List[str]:
    """Construct the Demucs command for two-stem (vocals) separation."""
    return [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        model,
        "-o",
        str(out_dir),
        str(input_path),
    ]


def _run(cmd: List[str], logger: logging.Logger) -> None:
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Command failed (%d): %s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"Demucs failed with code {result.returncode}")


def isolate_file(input_path: Path, out_dir: Path, cfg: RVCConfig, logger: logging.Logger) -> Path:
    """Isolate vocals from one file; returns the path to the vocals stem.

    Demucs writes ``out_dir/<model>/<track>/vocals.wav``.
    """
    _run(build_demucs_cmd(input_path, out_dir, cfg.demucs_model), logger)
    vocals = out_dir / cfg.demucs_model / input_path.stem / "vocals.wav"
    if not vocals.is_file():
        raise FileNotFoundError(f"Expected Demucs output not found: {vocals}")
    return vocals


def isolate_dir(
    input_dir: Path,
    out_dir: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> List[Path]:
    """Isolate vocals from every audio file under ``input_dir`` (recursively).

    Returns the list of produced ``vocals.wav`` paths.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    files = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    logger.info("Isolating vocals from %d file(s) with Demucs '%s'", len(files), cfg.demucs_model)

    produced: List[Path] = []
    for i, path in enumerate(files, start=1):
        try:
            produced.append(isolate_file(path, out_dir, cfg, logger))
            logger.info("[%d/%d] isolated %s", i, len(files), path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%d/%d] failed %s: %s", i, len(files), path.name, exc)
    return produced
