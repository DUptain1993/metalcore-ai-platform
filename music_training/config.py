"""Configuration dataclass for Stage 2 MusicGen LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MusicLoRAConfig:
    """Hyper-parameters for MusicGen LoRA training and validation generation.

    See ``configs/music_lora.yaml`` for documented defaults.
    """

    model_id: str = "facebook/musicgen-medium"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    # Optimisation
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 50
    max_steps: int = 2000
    batch_size: int = 1
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0
    guidance_dropout: float = 0.1

    # Precision / memory
    mixed_precision: str = "fp16"  # fp16 | bf16 | no
    gradient_checkpointing: bool = True

    # Data
    train_seconds: float = 15.0

    # Checkpointing / validation cadence
    save_every: int = 200
    val_every: int = 500
    keep_last_checkpoints: int = 2

    # Validation generation
    val_prompt: str = "aggressive modern metalcore breakdown, downtuned guitars"
    val_seconds: float = 8.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mixed_precision not in {"fp16", "bf16", "no"}:
            raise ValueError("mixed_precision must be one of: fp16, bf16, no.")
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
