"""MusicGen LoRA training loop.

Features required for Kaggle's ephemeral sessions:

* mixed-precision (fp16/bf16) with gradient accumulation and clipping;
* gradient checkpointing (enabled in :mod:`music_training.model`);
* periodic + on-signal (SIGTERM/SIGINT) checkpointing to ``/kaggle/working``;
* ``--resume`` that restores adapter, optimizer, scheduler, scaler and step;
* periodic validation loss + a generated audio sample.
"""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path
from typing import Any, List, Optional

from dataset_tools.metadata import read_jsonl
from music_training.config import MusicLoRAConfig
from music_training.dataset import (
    MusicCodesDataset,
    MusicCollator,
    build_code_cache,
)
from music_training.model import (
    attach_lora,
    dtype_from_precision,
    enable_memory_savings,
    load_musicgen,
    resolve_device,
    unwrap,
)

LATEST_POINTER = "latest.txt"


class _StopFlag:
    """Captures SIGTERM/SIGINT so the loop can checkpoint before exiting."""

    def __init__(self, logger: logging.Logger) -> None:
        self.stop = False
        self._logger = logger

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):  # not in main thread / unsupported
                pass

    def _handle(self, signum: int, _frame: Any) -> None:
        self._logger.warning("Received signal %s; will checkpoint and stop.", signum)
        self.stop = True


def _infinite_batches(dataloader: Any):
    while True:
        for batch in dataloader:
            yield batch


def _save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    global_step: int,
    ckpt_root: Path,
    cfg: MusicLoRAConfig,
    logger: logging.Logger,
) -> None:
    import torch

    ckpt_dir = ckpt_root / f"step_{global_step:06d}"
    adapter_dir = ckpt_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(adapter_dir)
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        ckpt_dir / "trainer_state.pt",
    )
    (ckpt_root / LATEST_POINTER).write_text(ckpt_dir.name, encoding="utf-8")
    logger.info("Saved checkpoint -> %s", ckpt_dir)
    _prune_checkpoints(ckpt_root, cfg.keep_last_checkpoints, logger)


def _prune_checkpoints(ckpt_root: Path, keep: int, logger: logging.Logger) -> None:
    if keep <= 0:
        return
    ckpts = sorted(
        (p for p in ckpt_root.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in ckpts[:-keep]:
        import shutil

        shutil.rmtree(old, ignore_errors=True)
        logger.debug("Pruned old checkpoint %s", old.name)


def _load_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    ckpt_root: Path,
    device: str,
    logger: logging.Logger,
) -> int:
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict

    pointer = ckpt_root / LATEST_POINTER
    if not pointer.is_file():
        logger.info("No checkpoint to resume from; starting fresh.")
        return 0

    ckpt_dir = ckpt_root / pointer.read_text(encoding="utf-8").strip()
    if not ckpt_dir.is_dir():
        logger.warning("Latest pointer references missing dir %s; starting fresh.", ckpt_dir)
        return 0

    weights = load_peft_weights(str(ckpt_dir / "adapter"))
    set_peft_model_state_dict(model, weights)

    state = torch.load(ckpt_dir / "trainer_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    global_step = int(state["global_step"])
    logger.info("Resumed from %s at step %d", ckpt_dir, global_step)
    return global_step


def _validate(
    model: Any,
    val_loader: Any,
    device: str,
    autocast_dtype: Any,
    use_amp: bool,
    max_batches: int,
    logger: logging.Logger,
) -> Optional[float]:
    import torch

    if val_loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                out = model(**batch)
            total += float(out.loss.item())
            count += 1
    model.train()
    if count == 0:
        return None
    avg = total / count
    logger.info("Validation loss: %.4f (%d batch)", avg, count)
    return avg


def _generate_sample(
    model: Any,
    processor: Any,
    cfg: MusicLoRAConfig,
    device: str,
    out_path: Path,
    logger: logging.Logger,
) -> None:
    import torch

    from metalcore.audio_io import save_audio

    base = unwrap(model)
    sr = int(base.config.audio_encoder.sampling_rate)
    frame_rate = int(base.config.audio_encoder.frame_rate)
    max_new_tokens = int(cfg.val_seconds * frame_rate)

    model.eval()
    prev_cache = base.config.use_cache
    base.config.use_cache = True
    try:
        inputs = processor(text=[cfg.val_prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            audio = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
        wav = audio[0].to("cpu").float().numpy()  # (channels, frames) or (frames,)
        save_audio(out_path, wav, sr)
        logger.info("Wrote validation audio -> %s", out_path)
    except Exception as exc:  # noqa: BLE001 - never let sampling kill training.
        logger.error("Validation generation failed: %s", exc)
    finally:
        base.config.use_cache = prev_cache
        model.train()


def train(
    config_path: str,
    dataset_dir: str,
    output_dir: str,
    resume: bool,
    logger: logging.Logger,
) -> None:
    """Run MusicGen LoRA training.

    Args:
        config_path: Path to a ``music_lora.yaml`` config.
        dataset_dir: Folder containing ``train.jsonl`` / ``val.jsonl`` and ``chunks/``.
        output_dir: Where caches, checkpoints and samples are written.
        resume: If ``True``, resume from the latest checkpoint if present.
        logger: Logger.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    from metalcore.config import load_config

    cfg = load_config(config_path, MusicLoRAConfig)
    torch.manual_seed(cfg.seed)

    device = resolve_device()
    if device != "cuda":
        logger.warning("No CUDA GPU detected; MusicGen training will be extremely slow.")

    dataset_dir_p = Path(dataset_dir)
    output_dir_p = Path(output_dir)
    ckpt_root = output_dir_p / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MusicGen model '%s'...", cfg.model_id)
    model, processor = load_musicgen(cfg.model_id, device)
    model = attach_lora(model, cfg, logger)
    enable_memory_savings(model, cfg)
    model.train()

    # --- Data ---
    train_records = read_jsonl(dataset_dir_p / "train.jsonl")
    val_path = dataset_dir_p / "val.jsonl"
    val_records = read_jsonl(val_path) if val_path.is_file() else []
    if not train_records:
        raise RuntimeError(f"No training records in {dataset_dir_p / 'train.jsonl'}")

    cache_root = output_dir_p / "cache"
    train_cache = build_code_cache(
        train_records, dataset_dir_p, model, device, cfg, cache_root / "train", logger
    )
    train_ds = MusicCodesDataset(train_cache)
    collate = MusicCollator(processor, guidance_dropout=cfg.guidance_dropout, seed=cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    val_loader = None
    if val_records:
        val_cache = build_code_cache(
            val_records, dataset_dir_p, model, device, cfg, cache_root / "val", logger
        )
        val_ds = MusicCodesDataset(val_cache)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=MusicCollator(processor),
            num_workers=2,
        )

    # --- Optimiser / scheduler / scaler ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps
    )
    use_amp = cfg.mixed_precision == "fp16" and device == "cuda"
    autocast_dtype = dtype_from_precision(cfg.mixed_precision)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    if resume:
        global_step = _load_checkpoint(model, optimizer, scheduler, scaler, ckpt_root, device, logger)

    stop = _StopFlag(logger)
    stop.install()

    logger.info(
        "Starting training: %d step(s), batch=%d x accum=%d, lr=%.2e, amp=%s",
        cfg.max_steps,
        cfg.batch_size,
        cfg.grad_accum_steps,
        cfg.learning_rate,
        use_amp or (cfg.mixed_precision == "bf16"),
    )

    batches = _infinite_batches(train_loader)
    running_loss = 0.0
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < cfg.max_steps and not stop.stop:
        batch = next(batches)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device == "cuda" and cfg.mixed_precision != "no")):
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += float(loss.item()) * cfg.grad_accum_steps
        accum_count += 1

        if accum_count == cfg.grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accum_count = 0

            if global_step % 10 == 0:
                avg = running_loss / (10 * cfg.grad_accum_steps)
                lr = scheduler.get_last_lr()[0]
                logger.info("step %d/%d | loss %.4f | lr %.2e", global_step, cfg.max_steps, avg, lr)
                running_loss = 0.0

            if global_step % cfg.save_every == 0:
                _save_checkpoint(model, optimizer, scheduler, scaler, global_step, ckpt_root, cfg, logger)

            if cfg.val_every and global_step % cfg.val_every == 0:
                _validate(model, val_loader, device, autocast_dtype, use_amp, max_batches=8, logger=logger)
                sample_path = output_dir_p / "outputs" / "music_val" / f"step_{global_step:06d}.wav"
                _generate_sample(model, processor, cfg, device, sample_path, logger)

    # Final checkpoint (also covers the signal-triggered exit).
    _save_checkpoint(model, optimizer, scheduler, scaler, global_step, ckpt_root, cfg, logger)
    logger.info("Training finished at step %d.", global_step)
