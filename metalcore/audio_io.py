"""Robust audio loading, conversion and saving.

These helpers wrap :mod:`soundfile` and :mod:`librosa` with defensive error
handling so a single corrupt file never crashes a batch job. Audio is always
represented as a float32 numpy array of shape ``(channels, frames)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

#: Audio file extensions the pipeline recognises.
AUDIO_EXTS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aiff", ".aif", ".aac"}
)


@dataclass(frozen=True)
class AudioInfo:
    """Lightweight metadata describing an audio file."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    duration: float


def probe_audio(path: Union[str, Path]) -> AudioInfo:
    """Return metadata for an audio file without loading all samples.

    Falls back to a full decode (via :func:`load_audio`) for formats that
    :mod:`soundfile` cannot stat directly (e.g. some MP3/M4A files).

    Args:
        path: Path to the audio file.

    Returns:
        An :class:`AudioInfo` record.

    Raises:
        RuntimeError: If the file cannot be read by any backend.
    """
    import soundfile as sf

    path = str(path)
    try:
        info = sf.info(path)
        return AudioInfo(
            path=path,
            sample_rate=info.samplerate,
            channels=info.channels,
            frames=info.frames,
            duration=float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
        )
    except Exception:  # noqa: BLE001 - fall back to a full decode below.
        pass

    try:
        data, sr = load_audio(path, sr=None, mono=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read audio file: {path} ({exc})") from exc

    channels, frames = data.shape
    return AudioInfo(
        path=path,
        sample_rate=sr,
        channels=channels,
        frames=frames,
        duration=float(frames) / float(sr) if sr else 0.0,
    )


def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load an audio file as ``float32`` samples.

    Args:
        path: Path to the audio file.
        sr: Target sample rate. ``None`` keeps the file's native rate.
        mono: If ``True``, downmix to a single channel.

    Returns:
        Tuple ``(data, sample_rate)`` where ``data`` has shape
        ``(channels, frames)`` and dtype ``float32``.

    Raises:
        RuntimeError: If neither backend can decode the file.
    """
    path = str(path)
    data: Optional[np.ndarray] = None
    file_sr: Optional[int] = None

    # Primary backend: soundfile (fast, handles WAV/FLAC/OGG and modern MP3).
    try:
        import soundfile as sf

        raw, file_sr = sf.read(path, dtype="float32", always_2d=True)
        data = raw.T  # (frames, channels) -> (channels, frames)
    except Exception:  # noqa: BLE001 - try librosa/audioread next.
        data = None

    # Fallback backend: librosa (uses audioread/ffmpeg for exotic codecs).
    if data is None:
        try:
            import librosa

            raw, file_sr = librosa.load(path, sr=None, mono=False)
            data = raw[np.newaxis, :] if raw.ndim == 1 else raw
            data = data.astype(np.float32, copy=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to decode audio: {path} ({exc})") from exc

    assert data is not None and file_sr is not None

    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)

    if sr is not None and sr != file_sr:
        data = resample(data, file_sr, sr)
        file_sr = sr

    return np.ascontiguousarray(data, dtype=np.float32), int(file_sr)


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample multi-channel audio to ``target_sr``.

    Args:
        data: Array of shape ``(channels, frames)``.
        orig_sr: Original sample rate.
        target_sr: Desired sample rate.

    Returns:
        Resampled array of shape ``(channels, new_frames)``.
    """
    if orig_sr == target_sr:
        return data
    import librosa

    channels = [
        librosa.resample(np.ascontiguousarray(ch), orig_sr=orig_sr, target_sr=target_sr)
        for ch in data
    ]
    return np.stack(channels).astype(np.float32, copy=False)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Downmix ``(channels, frames)`` audio to shape ``(1, frames)``."""
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.shape[0] == 1:
        return data
    return data.mean(axis=0, keepdims=True).astype(np.float32, copy=False)


def save_audio(
    path: Union[str, Path],
    data: np.ndarray,
    sr: int,
    subtype: str = "PCM_16",
) -> None:
    """Write audio to disk, creating parent directories as needed.

    Args:
        path: Destination path (extension selects the container).
        data: Array of shape ``(channels, frames)`` or ``(frames,)``.
        sr: Sample rate.
        subtype: soundfile subtype (e.g. ``"PCM_16"`` or ``"FLOAT"``).
    """
    import soundfile as sf

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        array = array.T  # (channels, frames) -> (frames, channels)
    sf.write(str(out), array, sr, subtype=subtype)


def peak_normalize(data: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale audio so its absolute peak equals ``peak`` (no-op for silence)."""
    max_abs = float(np.max(np.abs(data))) if data.size else 0.0
    if max_abs <= 0.0:
        return data
    return (data * (peak / max_abs)).astype(np.float32, copy=False)
