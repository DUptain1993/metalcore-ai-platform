"""MusicGen model loading, LoRA attachment and memory-saving helpers.

All heavy imports happen inside functions so this module can be imported without
torch/transformers installed.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from music_training.config import MusicLoRAConfig


def resolve_device() -> str:
    """Return ``"cuda"`` when a GPU is available, else ``"cpu"``."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_from_precision(mixed_precision: str) -> Any:
    """Map a precision string to a torch dtype for autocast."""
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }[mixed_precision]


def load_musicgen(model_id: str, device: str) -> Tuple[Any, Any]:
    """Load a MusicGen model and its processor in float32 (weights stay full
    precision; training uses autocast for the forward/backward math).

    Args:
        model_id: HF model id, e.g. ``facebook/musicgen-medium``.
        device: Target device.

    Returns:
        Tuple ``(model, processor)``.
    """
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    return model, processor


def unwrap(model: Any) -> Any:
    """Return the underlying MusicGen model whether or not it is PEFT-wrapped."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def attach_lora(model: Any, cfg: MusicLoRAConfig, logger: logging.Logger) -> Any:
    """Wrap ``model`` with LoRA adapters on the decoder attention projections.

    The configured ``target_modules`` (``q_proj``/``k_proj``/``v_proj``/
    ``out_proj``) match only the MusicGen decoder, so the T5 text encoder and the
    EnCodec audio codec remain frozen.

    Args:
        model: A loaded MusicGen model.
        cfg: LoRA configuration.
        logger: Logger for the trainable-parameter summary.

    Returns:
        The PEFT-wrapped model.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    trainable, total = _count_parameters(peft_model)
    logger.info(
        "LoRA attached: %s trainable / %s total params (%.3f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
    return peft_model


def enable_memory_savings(model: Any, cfg: MusicLoRAConfig) -> None:
    """Enable gradient checkpointing and disable the KV cache for training."""
    base = unwrap(model)
    # The decoder's config drives use_cache; disable it so grad-checkpointing works.
    try:
        base.config.use_cache = False
    except Exception:  # noqa: BLE001
        pass
    if hasattr(base, "decoder") and hasattr(base.decoder, "config"):
        base.decoder.config.use_cache = False

    if cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # Required for grad-checkpointing to propagate through frozen embeddings
        # when using PEFT.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()


def _count_parameters(model: Any) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total
