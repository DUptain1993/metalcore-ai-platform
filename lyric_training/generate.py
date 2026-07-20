"""Generate structured metalcore lyrics from a trained (or base) model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from lyric_training.config import LyricsLoRAConfig
from lyric_training.dataset import build_instruction


def generate(
    config_path: str,
    adapter_dir: Optional[str],
    themes: List[str],
    out_path: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logger: logging.Logger,
    seed: Optional[int] = None,
) -> str:
    """Generate one structured lyric set.

    Args:
        config_path: Path to ``lyrics_lora.yaml`` (for base ``model_id`` +
            section list).
        adapter_dir: LoRA adapter directory (e.g. ``.../adapter``). ``None`` uses
            the base instruct model.
        themes: Themes to condition on (e.g. ``["addiction", "hope"]``).
        out_path: Optional file to write the lyrics to. Always returned as a str.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        logger: Logger.
        seed: Optional RNG seed.

    Returns:
        The generated lyric text.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from metalcore.config import load_config

    cfg = load_config(config_path, LyricsLoRAConfig)
    if seed is not None:
        torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16

    tokenizer_src = adapter_dir if adapter_dir and Path(adapter_dir, "tokenizer_config.json").is_file() else cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model '%s'...", cfg.model_id)
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, quantization_config=bnb_config, device_map={"": 0}
        )
    else:
        logger.warning("No GPU detected; loading in float32 on CPU (slow).")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id)

    if adapter_dir:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()

    instruction = build_instruction(themes, cfg.sections)
    messages = [{"role": "user", "content": instruction}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    logger.info("Generating lyrics | themes: %s", ", ".join(themes))
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True).strip()

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("Wrote lyrics -> %s", out)

    return text
