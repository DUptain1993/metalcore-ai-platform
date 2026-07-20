"""Mastering: loudness normalisation + WAV/MP3 export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from inference.config import AssemblyConfig
from metalcore.audio_io import resample, save_audio


def loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalise integrated loudness to ``target_lufs`` and guard against clipping.

    Args:
        audio: ``(channels, frames)`` float audio.
        sr: Sample rate.
        target_lufs: Target integrated loudness (LUFS).

    Returns:
        Loudness-normalised audio, peak-limited to <= 0.99.
    """
    import pyloudnorm as pyln

    # pyloudnorm expects (frames,) or (frames, channels).
    data = audio.T if audio.ndim == 2 else audio
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
    except Exception:  # noqa: BLE001
        loudness = float("-inf")

    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    else:
        normalized = data

    normalized = np.asarray(normalized, dtype=np.float32)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    out = normalized.T if audio.ndim == 2 else normalized
    return out.astype(np.float32)


def export(
    audio: np.ndarray,
    sr: int,
    out_base: Path,
    cfg: AssemblyConfig,
    logger: logging.Logger,
) -> List[Path]:
    """Master and export ``audio`` as WAV (and MP3 if enabled).

    Args:
        audio: ``(channels, frames)`` mixed audio at ``sr``.
        sr: Sample rate of ``audio``.
        out_base: Output path without extension (e.g. ``outputs/songs/track``).
        cfg: Assembly configuration.
        logger: Logger.

    Returns:
        List of written file paths.
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if sr != cfg.export_sample_rate:
        audio = resample(audio, sr, cfg.export_sample_rate)
        sr = cfg.export_sample_rate

    mastered = loudness_normalize(audio, sr, cfg.target_lufs)

    written: List[Path] = []
    wav_path = out_base.with_suffix(".wav")
    save_audio(wav_path, mastered, sr)
    written.append(wav_path)
    logger.info("Exported WAV -> %s", wav_path)

    if cfg.export_mp3:
        try:
            mp3_path = _export_mp3(wav_path, out_base.with_suffix(".mp3"), cfg.mp3_bitrate)
            written.append(mp3_path)
            logger.info("Exported MP3 -> %s", mp3_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("MP3 export failed (is ffmpeg installed?): %s", exc)

    return written


def _export_mp3(wav_path: Path, mp3_path: Path, bitrate: str) -> Path:
    from pydub import AudioSegment

    seg = AudioSegment.from_wav(str(wav_path))
    seg.export(str(mp3_path), format="mp3", bitrate=bitrate)
    return mp3_path
