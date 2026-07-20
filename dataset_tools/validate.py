"""Validate raw audio files: detect corrupt, too-short, or silent tracks.

Bad files are moved into a ``quarantine/`` directory and recorded in a JSON
report so nothing is silently discarded.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from dataset_tools.config import DatasetConfig
from metalcore.audio_io import AUDIO_EXTS, load_audio, probe_audio


@dataclass
class ValidationResult:
    """Outcome of validating a single file."""

    path: str
    ok: bool
    reason: str
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0


def find_audio_files(root: Path, exts: List[str]) -> List[Path]:
    """Recursively collect audio files under ``root`` matching ``exts``."""
    allowed = {e.lower() for e in exts} & set(AUDIO_EXTS) or {e.lower() for e in exts}
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in allowed
    ]
    return files


def _rms_db(data: np.ndarray) -> float:
    """Return overall RMS level in dBFS (``-inf`` guarded to a low finite value)."""
    if data.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * float(np.log10(rms))


def validate_file(path: Path, cfg: DatasetConfig) -> ValidationResult:
    """Validate one audio file.

    Checks, in order: readability, minimum duration, and non-silence.

    Args:
        path: Audio file to validate.
        cfg: Dataset configuration with thresholds.

    Returns:
        A :class:`ValidationResult`.
    """
    try:
        info = probe_audio(path)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(path=str(path), ok=False, reason=f"unreadable: {exc}")

    if info.duration < cfg.min_duration:
        return ValidationResult(
            path=str(path),
            ok=False,
            reason=f"too_short: {info.duration:.2f}s < {cfg.min_duration}s",
            duration=info.duration,
            sample_rate=info.sample_rate,
            channels=info.channels,
        )

    # Load (mono) to check for silence. This also confirms the samples decode.
    try:
        data, _ = load_audio(path, sr=None, mono=True)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            path=str(path), ok=False, reason=f"decode_failed: {exc}"
        )

    level = _rms_db(data)
    if level < cfg.silence_rms_db:
        return ValidationResult(
            path=str(path),
            ok=False,
            reason=f"silent: {level:.1f} dB < {cfg.silence_rms_db} dB",
            duration=info.duration,
            sample_rate=info.sample_rate,
            channels=info.channels,
        )

    return ValidationResult(
        path=str(path),
        ok=True,
        reason="ok",
        duration=info.duration,
        sample_rate=info.sample_rate,
        channels=info.channels,
    )


def run_validate(
    input_dir: Path,
    output_dir: Path,
    cfg: DatasetConfig,
    logger: logging.Logger,
    quarantine: bool = True,
) -> List[ValidationResult]:
    """Validate every audio file under ``input_dir``.

    Args:
        input_dir: Directory of raw audio (searched recursively).
        output_dir: Where the report (and quarantine folder) are written.
        cfg: Dataset configuration.
        logger: Logger for progress/errors.
        quarantine: If ``True``, move invalid files into ``output_dir/quarantine``.

    Returns:
        The list of validation results (both valid and invalid).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(input_dir, cfg.audio_exts)
    logger.info("Found %d candidate audio file(s) under %s", len(files), input_dir)
    if not files:
        logger.warning("No audio files matched extensions %s", cfg.audio_exts)

    results: List[ValidationResult] = []
    quarantine_dir = output_dir / "quarantine"

    for i, path in enumerate(files, start=1):
        result = validate_file(path, cfg)
        results.append(result)
        if result.ok:
            logger.info("[%d/%d] OK      %s (%.1fs)", i, len(files), path.name, result.duration)
        else:
            logger.warning("[%d/%d] REJECT  %s -> %s", i, len(files), path.name, result.reason)
            if quarantine:
                _quarantine_file(path, input_dir, quarantine_dir, logger)

    n_ok = sum(r.ok for r in results)
    logger.info("Validation complete: %d valid, %d rejected", n_ok, len(results) - n_ok)

    report_path = output_dir / "report_validate.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in results], handle, indent=2)
    logger.info("Wrote validation report -> %s", report_path)

    return results


def _quarantine_file(
    path: Path, input_root: Path, quarantine_dir: Path, logger: logging.Logger
) -> None:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    dest = quarantine_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, dest)
        logger.debug("Quarantined copy -> %s", dest)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to quarantine %s: %s", path, exc)


def valid_paths(results: List[ValidationResult]) -> List[Path]:
    """Extract the paths of files that passed validation."""
    return [Path(r.path) for r in results if r.ok]
