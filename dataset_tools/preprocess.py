"""Normalise loudness, resample, and chunk validated audio into training clips."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import numpy as np

from dataset_tools.config import DatasetConfig
from dataset_tools.metadata import ChunkRecord, write_jsonl
from metalcore.audio_io import load_audio, peak_normalize, save_audio

CHUNKS_JSONL = "chunks.jsonl"


def loudness_normalize(data: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Loudness-normalise mono audio to ``target_lufs`` and clip peaks.

    Uses ITU-R BS.1770 integrated loudness (via :mod:`pyloudnorm`). If loudness
    cannot be measured (e.g. clip too short/quiet) the input is peak-normalised
    instead so downstream stages still receive a sane level.

    Args:
        data: Audio of shape ``(1, frames)``.
        sr: Sample rate.
        target_lufs: Target integrated loudness.

    Returns:
        Normalised audio of shape ``(1, frames)`` in ``float32`` clipped to
        ``[-1, 1]``.
    """
    import pyloudnorm as pyln

    mono = data[0] if data.ndim == 2 else data
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(mono)
    except Exception:  # noqa: BLE001 - clip likely shorter than a measurement block.
        loudness = float("-inf")

    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(mono, loudness, target_lufs)
    else:
        normalized = peak_normalize(mono[np.newaxis, :])[0]

    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)
    return normalized[np.newaxis, :]


def chunk_audio(
    data: np.ndarray,
    sr: int,
    chunk_seconds: float,
    overlap: float,
    keep_tail_ratio: float,
) -> List[np.ndarray]:
    """Split audio into fixed-length, zero-padded chunks.

    Args:
        data: Audio of shape ``(1, frames)``.
        sr: Sample rate.
        chunk_seconds: Chunk length in seconds.
        overlap: Overlap between consecutive chunks in seconds.
        keep_tail_ratio: Keep the final partial chunk only if it is at least this
            fraction of a full chunk; kept tails are zero-padded to full length.

    Returns:
        A list of chunks, each of shape ``(1, chunk_len)``.
    """
    n = data.shape[-1]
    chunk_len = int(round(chunk_seconds * sr))
    hop = max(1, int(round((chunk_seconds - overlap) * sr)))
    min_tail = int(round(keep_tail_ratio * chunk_len))

    chunks: List[np.ndarray] = []
    start = 0
    while start < n:
        seg = data[..., start : start + chunk_len]
        seg_len = seg.shape[-1]
        if seg_len < chunk_len:
            if seg_len >= min_tail and seg_len > 0:
                pad = chunk_len - seg_len
                seg = np.pad(seg, ((0, 0), (0, pad)))
                chunks.append(seg.astype(np.float32, copy=False))
            break
        chunks.append(seg.astype(np.float32, copy=False))
        start += hop
    return chunks


def _safe_stem(path: Path, input_root: Path) -> str:
    """Build a filesystem-safe, collision-resistant stem from a file path."""
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    raw = rel.with_suffix("").as_posix().replace("/", "__")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return stem or "track"


def run_preprocess(
    valid_files: List[Path],
    input_dir: Path,
    output_dir: Path,
    cfg: DatasetConfig,
    logger: logging.Logger,
) -> List[ChunkRecord]:
    """Normalise and chunk every valid file, writing chunks + ``chunks.jsonl``.

    Args:
        valid_files: Paths that passed validation.
        input_dir: Root of the raw audio (for stable relative naming).
        output_dir: Dataset output root. Chunks go under ``output_dir/chunks``.
        cfg: Dataset configuration.
        logger: Logger for progress/errors.

    Returns:
        The list of produced :class:`ChunkRecord` objects.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    seen_stems: dict[str, int] = {}
    records: List[ChunkRecord] = []

    for i, path in enumerate(valid_files, start=1):
        try:
            data, sr = load_audio(path, sr=cfg.sample_rate, mono=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%d/%d] Skipping %s (load failed: %s)", i, len(valid_files), path.name, exc)
            continue

        data = loudness_normalize(data, sr, cfg.target_lufs)
        segments = chunk_audio(
            data, sr, cfg.chunk_seconds, cfg.chunk_overlap, cfg.keep_tail_ratio
        )
        if not segments:
            logger.warning("[%d/%d] No usable chunks from %s", i, len(valid_files), path.name)
            continue

        stem = _safe_stem(path, input_dir)
        if stem in seen_stems:
            seen_stems[stem] += 1
            stem = f"{stem}-{seen_stems[stem]}"
        else:
            seen_stems[stem] = 0

        for idx, seg in enumerate(segments):
            name = f"{stem}_{idx:04d}.wav"
            out_path = chunks_dir / name
            save_audio(out_path, seg, sr)
            records.append(
                ChunkRecord(
                    audio=(Path("chunks") / name).as_posix(),
                    source=stem,
                    duration=seg.shape[-1] / sr,
                    sample_rate=sr,
                )
            )
        logger.info(
            "[%d/%d] %s -> %d chunk(s)", i, len(valid_files), path.name, len(segments)
        )

    jsonl_path = output_dir / CHUNKS_JSONL
    n = write_jsonl(jsonl_path, records)
    logger.info("Wrote %d chunk record(s) -> %s", n, jsonl_path)
    return records
