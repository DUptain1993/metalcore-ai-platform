"""Audio -> EnCodec code caching and the training Dataset for MusicGen.

To avoid re-encoding audio on every Kaggle session restart, chunk audio is
encoded to discrete EnCodec codes once and stored as a single int16 memmap on
disk (``codes.i16``) with a sidecar ``captions.json`` and ``meta.json``. The
:class:`MusicCodesDataset` then memmaps that file for fast, low-RAM access.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dataset_tools.metadata import ChunkRecord
from metalcore.audio_io import load_audio
from music_training.config import MusicLoRAConfig
from music_training.model import unwrap

CODES_FILE = "codes.i16"
CAPTIONS_FILE = "captions.json"
META_FILE = "meta.json"


def _fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad mono audio ``(1, frames)`` to exactly ``target_len``."""
    frames = audio.shape[-1]
    if frames == target_len:
        return audio
    if frames > target_len:
        return audio[..., :target_len]
    pad = target_len - frames
    return np.pad(audio, ((0, 0), (0, pad)))


def _cache_is_valid(cache_dir: Path, n_records: int, cfg: MusicLoRAConfig) -> bool:
    meta_path = cache_dir / META_FILE
    if not (cache_dir / CODES_FILE).is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return (
        meta.get("n") == n_records
        and meta.get("model_id") == cfg.model_id
        and abs(float(meta.get("train_seconds", -1)) - cfg.train_seconds) < 1e-6
    )


def build_code_cache(
    records: List[ChunkRecord],
    dataset_root: Path,
    model: Any,
    device: str,
    cfg: MusicLoRAConfig,
    cache_dir: Path,
    logger: logging.Logger,
    encode_batch: int = 8,
    force: bool = False,
) -> Path:
    """Encode all chunks to EnCodec codes and persist them as a memmap.

    Args:
        records: Chunk records for one split.
        dataset_root: Folder containing the ``chunks/`` directory.
        model: Loaded MusicGen model (PEFT-wrapped or not).
        device: Torch device for encoding.
        cfg: Music LoRA configuration.
        cache_dir: Destination directory for the cache.
        logger: Logger.
        encode_batch: Number of clips encoded per forward pass.
        force: Rebuild even if a valid cache exists.

    Returns:
        The ``cache_dir`` path.
    """
    import torch

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force and _cache_is_valid(cache_dir, len(records), cfg):
        logger.info("Reusing valid code cache at %s (%d clips)", cache_dir, len(records))
        return cache_dir

    base = unwrap(model)
    sampling_rate = int(base.config.audio_encoder.sampling_rate)
    target_len = int(round(cfg.train_seconds * sampling_rate))

    logger.info(
        "Encoding %d clip(s) to EnCodec codes @ %d Hz (%.1fs each)...",
        len(records),
        sampling_rate,
        cfg.train_seconds,
    )

    def _encode(batch_audio: np.ndarray) -> np.ndarray:
        # batch_audio: (B, 1, samples)
        input_values = torch.from_numpy(batch_audio).to(device)
        with torch.no_grad():
            encoded = base.audio_encoder.encode(input_values, padding_mask=None)
        codes = encoded.audio_codes  # (num_chunks=1, B, K, T)
        if codes.dim() != 4 or codes.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected EnCodec code shape {tuple(codes.shape)}; expected (1, B, K, T)."
            )
        return codes[0].to("cpu").numpy().astype(np.int16)  # (B, K, T)

    # Encode a first batch to learn (K, T), then allocate the memmap.
    codes_memmap: np.memmap | None = None
    num_codebooks = seq_len = 0
    write_idx = 0
    captions: List[str] = []
    buffer: List[np.ndarray] = []
    buffer_caps: List[str] = []

    def _flush() -> None:
        nonlocal codes_memmap, num_codebooks, seq_len, write_idx
        if not buffer:
            return
        batch_audio = np.stack(buffer).astype(np.float32)  # (B, 1, samples)
        batch_codes = _encode(batch_audio)  # (B, K, T)
        if codes_memmap is None:
            num_codebooks, seq_len = batch_codes.shape[1], batch_codes.shape[2]
            codes_memmap = np.memmap(
                cache_dir / CODES_FILE,
                dtype=np.int16,
                mode="w+",
                shape=(len(records), num_codebooks, seq_len),
            )
        codes_memmap[write_idx : write_idx + batch_codes.shape[0]] = batch_codes
        write_idx += batch_codes.shape[0]
        captions.extend(buffer_caps)
        buffer.clear()
        buffer_caps.clear()

    for i, record in enumerate(records, start=1):
        chunk_path = dataset_root / record.audio
        audio, _ = load_audio(chunk_path, sr=sampling_rate, mono=True)  # (1, frames)
        audio = _fit_length(audio, target_len)
        buffer.append(audio)
        buffer_caps.append(record.caption or "metalcore")
        if len(buffer) >= encode_batch:
            _flush()
        if i % 50 == 0 or i == len(records):
            logger.info("Encoded %d/%d clip(s)", i, len(records))
    _flush()

    if codes_memmap is None:
        raise RuntimeError("No clips were encoded; cannot build cache.")

    codes_memmap.flush()
    del codes_memmap

    (cache_dir / CAPTIONS_FILE).write_text(
        json.dumps(captions, ensure_ascii=False), encoding="utf-8"
    )
    (cache_dir / META_FILE).write_text(
        json.dumps(
            {
                "n": len(records),
                "num_codebooks": num_codebooks,
                "seq_len": seq_len,
                "sampling_rate": sampling_rate,
                "model_id": cfg.model_id,
                "train_seconds": cfg.train_seconds,
            }
        ),
        encoding="utf-8",
    )
    logger.info(
        "Wrote code cache: %d clips, %d codebooks, %d frames -> %s",
        len(records),
        num_codebooks,
        seq_len,
        cache_dir / CODES_FILE,
    )
    return cache_dir


class MusicCodesDataset:
    """Memmap-backed dataset yielding ``{"codes": (K, T) int64, "caption": str}``.

    Implemented against the ``torch.utils.data.Dataset`` protocol (``__len__`` /
    ``__getitem__``) without importing torch at module import time.
    """

    def __init__(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / META_FILE).read_text(encoding="utf-8"))
        self.n: int = int(meta["n"])
        self.num_codebooks: int = int(meta["num_codebooks"])
        self.seq_len: int = int(meta["seq_len"])
        self.sampling_rate: int = int(meta["sampling_rate"])
        self._codes = np.memmap(
            cache_dir / CODES_FILE,
            dtype=np.int16,
            mode="r",
            shape=(self.n, self.num_codebooks, self.seq_len),
        )
        self._captions: List[str] = json.loads(
            (cache_dir / CAPTIONS_FILE).read_text(encoding="utf-8")
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Dict[str, Any]:
        codes = np.asarray(self._codes[index], dtype=np.int64)
        return {"codes": codes, "caption": self._captions[index]}


class MusicCollator:
    """Collate function: tokenise captions and stack code labels into a batch.

    Implements classifier-free-guidance conditioning dropout: with probability
    ``guidance_dropout`` an individual caption is replaced by the empty string so
    the model also learns the unconditional distribution used at generation time.
    """

    def __init__(self, processor: Any, guidance_dropout: float = 0.0, seed: int = 0) -> None:
        import random

        self.processor = processor
        self.guidance_dropout = float(guidance_dropout)
        self._rng = random.Random(seed)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        captions = [
            "" if self.guidance_dropout and self._rng.random() < self.guidance_dropout
            else item["caption"]
            for item in batch
        ]
        codes = np.stack([item["codes"] for item in batch]).astype(np.int64)  # (B, K, T)

        text_inputs = self.processor(
            text=captions, padding=True, return_tensors="pt"
        )
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": torch.from_numpy(codes),
        }
