"""Stage 2 - MusicGen LoRA fine-tuning and generation.

Heavy dependencies (torch/transformers/peft) are imported lazily inside the
functions that need them, so ``python -m music_training.cli --help`` and static
tooling work on machines without a GPU.
"""

from __future__ import annotations

from music_training.config import MusicLoRAConfig

__all__ = ["MusicLoRAConfig"]
