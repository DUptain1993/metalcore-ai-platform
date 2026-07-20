"""Command-line interface for Stage 3 lyrics QLoRA training / generation.

Examples::

    # Build the dataset from a folder of .txt lyrics:
    python -m lyric_training.cli build \
        --lyrics data/lyrics --output data/lyrics_dataset

    # Fine-tune (resume-safe):
    python -m lyric_training.cli train \
        --config configs/lyrics_lora.yaml \
        --data data/lyrics_dataset --output outputs/lyrics --resume

    # Generate structured lyrics:
    python -m lyric_training.cli generate \
        --config configs/lyrics_lora.yaml \
        --adapter outputs/lyrics/adapter \
        --themes "addiction, hope" --output outputs/lyrics/song.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from metalcore.logging_utils import get_logger

DEFAULT_CONFIG = "configs/lyrics_lora.yaml"


def _parse_themes(raw: str) -> List[str]:
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def _cmd_build(args: argparse.Namespace) -> int:
    from lyric_training.config import LyricsLoRAConfig
    from lyric_training.dataset import build_dataset
    from metalcore.config import load_config

    logger = get_logger("lyrics.build", f"{args.output}/logs/lyrics.log")
    cfg = (
        load_config(args.config, LyricsLoRAConfig)
        if Path(args.config).is_file()
        else LyricsLoRAConfig()
    )
    build_dataset(Path(args.lyrics), Path(args.output), cfg, logger)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from lyric_training.train import train

    logger = get_logger("lyrics.train", f"{args.output}/logs/lyrics_train.log")
    train(
        config_path=args.config,
        data_dir=args.data,
        output_dir=args.output,
        resume=args.resume,
        logger=logger,
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from lyric_training.generate import generate

    logger = get_logger("lyrics.generate", "outputs/logs/lyrics_generate.log")
    text = generate(
        config_path=args.config,
        adapter_dir=args.adapter,
        themes=_parse_themes(args.themes),
        out_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        logger=logger,
        seed=args.seed,
    )
    print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyric_training",
        description="Stage 3 - metalcore lyrics QLoRA fine-tuning and generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the instruction dataset from .txt lyrics.")
    p_build.add_argument("--config", default=DEFAULT_CONFIG, help="Path to lyrics_lora.yaml.")
    p_build.add_argument("--lyrics", required=True, help="Folder of .txt lyric files.")
    p_build.add_argument("--output", required=True, help="Dataset output directory.")
    p_build.set_defaults(func=_cmd_build)

    p_train = sub.add_parser("train", help="Fine-tune the lyrics model with QLoRA.")
    p_train.add_argument("--config", default=DEFAULT_CONFIG, help="Path to lyrics_lora.yaml.")
    p_train.add_argument("--data", required=True, help="Dataset dir (lyrics_train.jsonl).")
    p_train.add_argument("--output", required=True, help="Output dir for checkpoints/adapter.")
    p_train.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    p_train.set_defaults(func=_cmd_train)

    p_gen = sub.add_parser("generate", help="Generate structured lyrics.")
    p_gen.add_argument("--config", default=DEFAULT_CONFIG, help="Path to lyrics_lora.yaml.")
    p_gen.add_argument("--adapter", default=None, help="LoRA adapter directory (optional).")
    p_gen.add_argument("--themes", default="emotional struggle", help="Comma-separated themes.")
    p_gen.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens", help="Generation length.")
    p_gen.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    p_gen.add_argument("--top-p", type=float, default=0.95, dest="top_p", help="Nucleus sampling p.")
    p_gen.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    p_gen.add_argument("--output", default=None, help="Optional file to write lyrics to.")
    p_gen.set_defaults(func=_cmd_generate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
