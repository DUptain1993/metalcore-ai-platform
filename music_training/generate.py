"""Generate audio from a text prompt using a trained MusicGen LoRA adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from music_training.config import MusicLoRAConfig
from music_training.model import resolve_device


def generate(
    config_path: str,
    adapter_dir: Optional[str],
    prompt: str,
    out_dir: str,
    seconds: float,
    num_samples: int,
    guidance_scale: float,
    logger: logging.Logger,
    seed: Optional[int] = None,
) -> List[Path]:
    """Generate one or more audio clips.

    Args:
        config_path: Path to the ``music_lora.yaml`` used for training (for the
            base ``model_id``).
        adapter_dir: Directory of a saved LoRA adapter (e.g.
            ``.../checkpoints/step_002000/adapter``). ``None`` uses the base model.
        prompt: Text prompt describing the desired music.
        out_dir: Directory to write generated WAV files.
        seconds: Length of each generated clip.
        num_samples: How many clips to generate.
        guidance_scale: Classifier-free guidance scale.
        logger: Logger.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of written file paths.
    """
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    from metalcore.audio_io import save_audio
    from metalcore.config import load_config

    cfg = load_config(config_path, MusicLoRAConfig)
    device = resolve_device()
    if seed is not None:
        torch.manual_seed(seed)

    logger.info("Loading base model '%s'...", cfg.model_id)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(cfg.model_id)

    if adapter_dir:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    else:
        logger.warning("No adapter provided; generating with the base model.")

    model.to(device)
    model.eval()

    sr = int(model.config.audio_encoder.sampling_rate)
    frame_rate = int(model.config.audio_encoder.frame_rate)
    max_new_tokens = int(seconds * frame_rate)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    inputs = processor(
        text=[prompt] * num_samples, padding=True, return_tensors="pt"
    ).to(device)

    logger.info(
        "Generating %d clip(s) of ~%.1fs (%d tokens) | prompt: %s",
        num_samples,
        seconds,
        max_new_tokens,
        prompt,
    )
    with torch.no_grad():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
        )

    written: List[Path] = []
    for i in range(audio.shape[0]):
        wav = audio[i].to("cpu").float().numpy()  # (channels, frames)
        out_path = out_dir_p / f"generated_{i:02d}.wav"
        save_audio(out_path, wav, sr)
        written.append(out_path)
        logger.info("Wrote %s", out_path)

    return written
