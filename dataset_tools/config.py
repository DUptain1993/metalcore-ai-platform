"""Configuration dataclass for Stage 1 dataset processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    """Knobs for the dataset-processing pipeline.

    See ``configs/dataset.yaml`` for documented defaults.
    """

    sample_rate: int = 32000
    chunk_seconds: float = 15.0
    chunk_overlap: float = 0.0
    target_lufs: float = -14.0
    min_duration: float = 2.0
    silence_rms_db: float = -50.0
    keep_tail_ratio: float = 0.5
    val_ratio: float = 0.1
    seed: int = 42
    style_tags: List[str] = field(
        default_factory=lambda: [
            "aggressive modern metalcore",
            "downtuned rhythm guitar",
            "double bass drums",
            "ambient lead textures",
        ]
    )
    audio_exts: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    )

    def __post_init__(self) -> None:
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        if not 0.0 <= self.chunk_overlap < self.chunk_seconds:
            raise ValueError("chunk_overlap must be in [0, chunk_seconds).")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in (0, 1).")
        if not 0.0 <= self.keep_tail_ratio <= 1.0:
            raise ValueError("keep_tail_ratio must be in [0, 1].")
        # Normalise extensions to lower-case with a leading dot.
        self.audio_exts = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.audio_exts
        ]
