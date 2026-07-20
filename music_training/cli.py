"""Command-line interface for Stage 2 MusicGen LoRA training / generation.

Examples::

    # Train (resume-safe):
    python -m music_training.cli train \
        --config configs/music_lora.yaml \
        --dataset data/dataset --output outputs/music --resume

    # Generate from a trained adapter:
    python -m music_training.cli generate \
        --config configs/music_lora.yaml \
        --adapter outputs/music/checkpoints/step_002000/adapter \
        --prompt "melodic metalcore chorus, soaring lead guitar" \
        --seconds 12 --num 2 --output outputs/music/generated
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from metalcore.logging_utils import get_logger

DEFAULT_CONFIG = "configs/music_lora.yaml"


def _cmd_train(args: argparse.Namespace) -> int:
    from music_training.train import train

    logger = get_logger("music.train", f"{args.output}/logs/music_train.log")
    train(
        config_path=args.config,
        dataset_dir=args.dataset,
        output_dir=args.output,
        resume=args.resume,
        logger=logger,
    )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from music_training.generate import generate

    logger = get_logger("music.generate", f"{args.output}/logs/music_generate.log")
    generate(
        config_path=args.config,
        adapter_dir=args.adapter,
        prompt=args.prompt,
        out_dir=args.output,
        seconds=args.seconds,
        num_samples=args.num,
        guidance_scale=args.guidance_scale,
        logger=logger,
        seed=args.seed,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music_training",
        description="Stage 2 - MusicGen LoRA fine-tuning and generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune MusicGen with LoRA.")
    p_train.add_argument("--config", default=DEFAULT_CONFIG, help="Path to music_lora.yaml.")
    p_train.add_argument("--dataset", required=True, help="Dataset dir (train.jsonl/val.jsonl/chunks).")
    p_train.add_argument("--output", required=True, help="Output dir for checkpoints/cache/samples.")
    p_train.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    p_train.set_defaults(func=_cmd_train)

    p_gen = sub.add_parser("generate", help="Generate audio from a prompt.")
    p_gen.add_argument("--config", default=DEFAULT_CONFIG, help="Path to music_lora.yaml.")
    p_gen.add_argument("--adapter", default=None, help="LoRA adapter directory (optional).")
    p_gen.add_argument("--prompt", required=True, help="Text prompt.")
    p_gen.add_argument("--seconds", type=float, default=12.0, help="Clip length in seconds.")
    p_gen.add_argument("--num", type=int, default=1, help="Number of clips to generate.")
    p_gen.add_argument("--guidance-scale", type=float, default=3.0, dest="guidance_scale", help="CFG scale.")
    p_gen.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    p_gen.add_argument("--output", required=True, help="Output directory for WAV files.")
    p_gen.set_defaults(func=_cmd_generate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
