"""Self-contained vocal FX chain: clean / harsh / scream presets.

Pure numpy + scipy (+ optional librosa) DSP so it runs and is testable anywhere,
with no GPU or external model. Applied to a converted (RVC) or raw vocal to push
it toward a metalcore texture.

Honest note: waveshaping + roughness + spectral darkening *approximate* a scream;
they do not synthesise the non-harmonic vocal-fold chaos of a real fry scream.
See docs/VOCALS_GUIDE.md.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _as_mono_1d(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    return arr.astype(np.float32, copy=False)


def _sos(kind: str, cutoff, sr: int, order: int = 4):
    return butter(order, cutoff, btype=kind, fs=sr, output="sos")


def highpass(x: np.ndarray, sr: int, freq: float, order: int = 4) -> np.ndarray:
    if freq <= 0:
        return x
    return sosfiltfilt(_sos("highpass", freq, sr, order), x).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, freq: float, order: int = 4) -> np.ndarray:
    if freq >= sr / 2:
        return x
    return sosfiltfilt(_sos("lowpass", freq, sr, order), x).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    high = min(high, sr / 2 - 1)
    return sosfiltfilt(_sos("bandpass", [low, high], sr, order), x).astype(np.float32)


def band_boost(x: np.ndarray, sr: int, low: float, high: float, gain: float) -> np.ndarray:
    """Additive band boost: y = x + (gain-1) * bandpass(x)."""
    if gain == 1.0:
        return x
    return (x + (gain - 1.0) * bandpass(x, sr, low, high)).astype(np.float32)


def soft_saturate(x: np.ndarray, drive: float) -> np.ndarray:
    """Symmetric tanh saturation, gain-normalised."""
    if drive <= 0:
        return x
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


def asymmetric_saturate(x: np.ndarray, drive: float, bias: float = 0.12) -> np.ndarray:
    """Asymmetric waveshaping -> even + odd harmonics (a growl-ier distortion)."""
    if drive <= 0:
        return x
    y = np.tanh(drive * (x + bias)) - np.tanh(drive * bias)
    peak = float(np.max(np.abs(y))) + 1e-9
    return (y / peak).astype(np.float32)


def bitcrush(x: np.ndarray, bits: int) -> np.ndarray:
    """Quantise amplitude to ``bits`` bits for a gritty digital edge."""
    if bits <= 0 or bits >= 16:
        return x
    levels = float(2 ** bits)
    return (np.round(x * levels) / levels).astype(np.float32)


def add_grit(x: np.ndarray, sr: int, amount: float, seed: int = 0) -> np.ndarray:
    """Add fry-like roughness via low-band noisy amplitude modulation."""
    if amount <= 0:
        return x
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.shape[0]).astype(np.float32)
    mod_noise = sosfiltfilt(_sos("bandpass", [40, 220], sr, 2), noise)
    mod_noise = mod_noise / (float(np.max(np.abs(mod_noise))) + 1e-9)
    mod = 1.0 + amount * mod_noise
    return (x * mod).astype(np.float32)


def low_shelf(x: np.ndarray, sr: int, freq: float, gain: float) -> np.ndarray:
    """Boost/cut everything below ``freq`` by ``gain`` (phase-safe, additive).

    ``gain > 1`` darkens/enlarges the voice (more low-mid body); ``gain < 1``
    thins it. Implemented as ``x + (gain-1) * lowpass(x)`` so it is a real filter
    with no magnitude/phase-surgery artefacts.
    """
    if gain == 1.0:
        return x
    return (x + (gain - 1.0) * lowpass(x, sr, freq)).astype(np.float32)


def high_shelf(x: np.ndarray, sr: int, freq: float, gain: float) -> np.ndarray:
    """Boost/cut everything above ``freq`` by ``gain`` (phase-safe, additive)."""
    if gain == 1.0:
        return x
    return (x + (gain - 1.0) * highpass(x, sr, freq)).astype(np.float32)


def _match_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[0] == n:
        return y
    if y.shape[0] > n:
        return y[:n]
    return np.pad(y, (0, n - y.shape[0]))


def peak_normalize(x: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 0:
        return x
    return (x * (peak / m)).astype(np.float32)


# --- Presets -----------------------------------------------------------------

def _clean(x: np.ndarray, sr: int, seed: int) -> np.ndarray:
    y = highpass(x, sr, 80.0)
    y = soft_saturate(y, drive=1.3)
    y = band_boost(y, sr, 3000.0, 6000.0, gain=1.15)
    return peak_normalize(y)


def _harsh(x: np.ndarray, sr: int, seed: int) -> np.ndarray:
    y = highpass(x, sr, 110.0)
    y = asymmetric_saturate(y, drive=4.5, bias=0.1)
    y = add_grit(y, sr, amount=0.15, seed=seed)
    y = band_boost(y, sr, 2500.0, 5500.0, gain=1.4)
    y = lowpass(y, sr, 9500.0)
    return peak_normalize(y)


def _scream(x: np.ndarray, sr: int, seed: int) -> np.ndarray:
    y = highpass(x, sr, 150.0)
    y = low_shelf(y, sr, 450.0, gain=1.4)      # darken/enlarge (pseudo formant drop)
    y = asymmetric_saturate(y, drive=9.0, bias=0.18)
    y = add_grit(y, sr, amount=0.4, seed=seed)
    y = bitcrush(y, bits=10)
    y = bandpass(y, sr, 250.0, 6800.0)
    y = band_boost(y, sr, 2000.0, 5000.0, gain=1.5)  # presence/rasp
    return peak_normalize(y)


STYLES: Dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "clean": _clean,
    "harsh": _harsh,
    "scream": _scream,
}


def process(
    audio: np.ndarray,
    sr: int,
    style: str = "harsh",
    dry_wet: float = 1.0,
    output_peak: float = 0.97,
    seed: int = 0,
) -> np.ndarray:
    """Apply a vocal FX preset.

    Args:
        audio: Mono ``(N,)`` or ``(1, N)`` / ``(C, N)`` float audio.
        sr: Sample rate.
        style: ``"clean"``, ``"harsh"`` or ``"scream"``.
        dry_wet: 0 = dry, 1 = fully processed.
        output_peak: Final peak-normalisation target.
        seed: RNG seed for the grit generator (reproducible).

    Returns:
        Mono ``(1, N)`` float32 audio.
    """
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; choose from {sorted(STYLES)}.")
    x = _as_mono_1d(audio)
    wet = STYLES[style](x, sr, seed)
    wet = _match_length(wet, x.shape[0])
    out = (1.0 - dry_wet) * x + dry_wet * wet
    out = peak_normalize(out, output_peak)
    return out[np.newaxis, :].astype(np.float32)
