"""Lyrics QLoRA supervised fine-tuning via TRL's ``SFTTrainer``.

Uses 4-bit (nf4) quantisation + LoRA so a small instruct model fine-tunes
within a T4's memory. Checkpointing/resume are handled by the HF ``Trainer``
machinery (``save_steps`` + ``resume_from_checkpoint``), which survives Kaggle
session restarts when ``output_dir`` lives under ``/kaggle/working``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lyric_training.config import LyricsLoRAConfig
from lyric_training.dataset import TRAIN_JSONL, VAL_JSONL, read_jsonl


def _resolve_precision(cfg: LyricsLoRAConfig) -> Tuple[Any, bool, bool]:
    """Return ``(compute_dtype, use_bf16, use_fp16)`` honouring GPU capability."""
    import torch

    want_bf16 = cfg.bnb_4bit_compute_dtype == "bfloat16"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if want_bf16 and bf16_ok:
        return torch.bfloat16, True, False
    return torch.float16, False, True


def _has_checkpoint(ckpt_dir: Path) -> bool:
    return ckpt_dir.is_dir() and any(ckpt_dir.glob("checkpoint-*"))


def train(
    config_path: str,
    data_dir: str,
    output_dir: str,
    resume: bool,
    logger: logging.Logger,
) -> None:
    """Fine-tune a lyrics model with QLoRA.

    Args:
        config_path: Path to ``lyrics_lora.yaml``.
        data_dir: Folder containing ``lyrics_train.jsonl`` (and optionally
            ``lyrics_val.jsonl``) produced by the ``build`` step.
        output_dir: Output directory for HF checkpoints and the final adapter.
        resume: Resume from the latest checkpoint if one exists.
        logger: Logger.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    from metalcore.config import load_config

    cfg = load_config(config_path, LyricsLoRAConfig)
    data_dir_p = Path(data_dir)
    output_dir_p = Path(output_dir)
    ckpt_dir = output_dir_p / "checkpoints"
    final_dir = output_dir_p / "adapter"

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected; 4-bit QLoRA requires a GPU and will fail on CPU.")

    compute_dtype, use_bf16, use_fp16 = _resolve_precision(cfg)
    logger.info(
        "Loading '%s' in 4-bit (compute dtype: %s)...",
        cfg.model_id,
        "bf16" if use_bf16 else "fp16",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Data ---
    train_rows = read_jsonl(data_dir_p / TRAIN_JSONL)
    train_ds = Dataset.from_list(train_rows)
    val_path = data_dir_p / VAL_JSONL
    val_ds = Dataset.from_list(read_jsonl(val_path)) if val_path.is_file() else None
    logger.info("Loaded %d train / %s val example(s)", len(train_ds), len(val_ds) if val_ds else 0)

    def formatting_func(examples: dict) -> List[str]:
        texts: List[str] = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False)
            )
        return texts

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.keep_last_checkpoints,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=cfg.save_steps if val_ds is not None else None,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
    )

    resume_flag = resume and _has_checkpoint(ckpt_dir)
    if resume and not resume_flag:
        logger.info("Resume requested but no checkpoint found; starting fresh.")

    logger.info("Starting lyrics QLoRA training...")
    trainer.train(resume_from_checkpoint=resume_flag)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final adapter -> %s", final_dir)
