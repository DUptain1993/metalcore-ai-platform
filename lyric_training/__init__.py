"""Stage 3 - metalcore lyrics QLoRA fine-tuning and generation.

Heavy dependencies (torch/transformers/peft/trl/bitsandbytes) are imported
lazily so ``python -m lyric_training.cli --help`` works without a GPU.
"""

from __future__ import annotations

from lyric_training.config import LyricsLoRAConfig

__all__ = ["LyricsLoRAConfig"]
