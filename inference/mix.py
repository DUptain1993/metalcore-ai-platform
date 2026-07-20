"""Mix a vocal stem over an instrumental into a stereo track."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from inference.config import AssemblyConfig


def db_to_gain(db: float) -> float:
    """Convert decibels to a linear amplitude factor."""
    return float(10.0 ** (db / 20.0))


def _fit(mono: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad a mono signal to ``length`` samples."""
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.shape[0] == length:
        return mono
    if mono.shape[0] > length:
        return mono[:length]
    return np.pad(mono, (0, length - mono.shape[0])).astype(np.float32)


def to_stereo(mono: np.ndarray, width: float) -> np.ndarray:
    """Turn mono into a ``(2, N)`` stereo pair with a Haas-style width.

    ``width`` in [0, 1]; 0 is dual-mono, larger values add a short inter-channel
    delay + slight de-correlation for a wider image.
    """
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if width <= 0.0:
        return np.stack([mono, mono])
    delay = int(round(width * 20))  # up to ~20 samples of Haas delay
    right = np.concatenate([np.zeros(delay, dtype=np.float32), mono])[: mono.shape[0]] if delay else mono.copy()
    left = mono
    return np.stack([left, right]).astype(np.float32)


def mix(
    instrumental: np.ndarray,
    vocal: np.ndarray,
    cfg: AssemblyConfig,
    logger: logging.Logger,
) -> np.ndarray:
    """Mix mono ``instrumental`` and mono ``vocal`` into a stereo ``(2, N)`` track.

    The vocal is fit to the instrumental length. Both stems get their configured
    gain, the instrumental is widened to stereo, and the (centred) vocal is summed
    in. A soft safety limit prevents clipping before mastering.
    """
    instrumental = np.asarray(instrumental, dtype=np.float32).reshape(-1)
    length = instrumental.shape[0]
    vocal = _fit(vocal, length)

    inst_g = db_to_gain(cfg.instrumental_gain_db)
    voc_g = db_to_gain(cfg.vocal_gain_db)

    inst_stereo = to_stereo(instrumental * inst_g, cfg.stereo_width)
    voc_stereo = np.stack([vocal * voc_g, vocal * voc_g])  # vocal centred

    mixed = inst_stereo + voc_stereo

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = np.tanh(mixed * 0.9).astype(np.float32)  # gentle bus limiter
        logger.info("Applied soft bus limiter (pre-mix peak %.2f)", peak)
    return mixed.astype(np.float32)
