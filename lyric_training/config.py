"""Configuration dataclass for Stage 3 lyrics QLoRA fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LyricsLoRAConfig:
    """Hyper-parameters for lyrics QLoRA training and generation.

    See ``configs/lyrics_lora.yaml`` for documented defaults.
    """

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Quantisation (QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # auto-falls back to fp16 on T4

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Optimisation
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 3.0
    batch_size: int = 2
    grad_accum_steps: int = 4
    max_grad_norm: float = 0.3
    max_seq_length: int = 1024

    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 10
    keep_last_checkpoints: int = 2

    # Data
    val_ratio: float = 0.1

    # Structure / themes
    sections: List[str] = field(
        default_factory=lambda: [
            "Verse",
            "Chorus",
            "Verse",
            "Bridge",
            "Breakdown",
            "Final Chorus",
        ]
    )
    themes: List[str] = field(
        default_factory=lambda: [
            "depression",
            "recovery",
            "addiction",
            "self destruction",
            "hope",
            "betrayal",
            "resilience",
            "emotional struggle",
        ]
    )

    seed: int = 42

    def __post_init__(self) -> None:
        if self.bnb_4bit_compute_dtype not in {"bfloat16", "float16"}:
            raise ValueError("bnb_4bit_compute_dtype must be 'bfloat16' or 'float16'.")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1).")
