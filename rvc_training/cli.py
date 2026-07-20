"""Command-line interface for Stage 4 vocal generation.

Examples::

    # One-time: clone RVC-Project + download pretrained weights.
    python -m rvc_training.cli setup --config configs/rvc.yaml

    # Build a vocal dataset from full mixes (isolate) then prepare clips.
    python -m rvc_training.cli isolate --input data/refs --output data/vocals_raw
    python -m rvc_training.cli prepare --input data/vocals_raw --output data/rvc_dataset

    # Train the blended RVC voice, then convert lyrics to a vocal stem.
    python -m rvc_training.cli train --dataset data/rvc_dataset/merged --name blend
    python -m rvc_training.cli vocal --lyrics outputs/lyrics/song.txt \
        --model .../logs/blend/blend.pth --index .../logs/blend/added.index \
        --output outputs/vocals/song_vocal.wav

    # Or just re-skin an existing WAV with the FX chain:
    python -m rvc_training.cli fx --input in.wav --output out.wav --style scream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from metalcore.config import load_config
from metalcore.logging_utils import get_logger
from rvc_training.config import RVCConfig

DEFAULT_CONFIG = "configs/rvc.yaml"


def _load_cfg(path: Optional[str]) -> RVCConfig:
    if path and Path(path).is_file():
        return load_config(path, RVCConfig)
    if Path(DEFAULT_CONFIG).is_file():
        return load_config(DEFAULT_CONFIG, RVCConfig)
    return RVCConfig()


def _cmd_setup(args: argparse.Namespace) -> int:
    from rvc_training.rvc import setup

    logger = get_logger("rvc.setup", "outputs/logs/rvc.log")
    setup(_load_cfg(args.config), logger)
    return 0


def _cmd_isolate(args: argparse.Namespace) -> int:
    from rvc_training.isolate import isolate_dir

    logger = get_logger("rvc.isolate", "outputs/logs/rvc.log")
    produced = isolate_dir(Path(args.input), Path(args.output), _load_cfg(args.config), logger)
    return 0 if produced else 1


def _cmd_prepare(args: argparse.Namespace) -> int:
    from rvc_training.dataset import prepare_dataset

    logger = get_logger("rvc.prepare", "outputs/logs/rvc.log")
    stats = prepare_dataset(Path(args.input), Path(args.output), _load_cfg(args.config), logger)
    return 0 if stats.clips > 0 else 1


def _cmd_train(args: argparse.Namespace) -> int:
    from rvc_training.rvc import train_all

    logger = get_logger("rvc.train", "outputs/logs/rvc.log")
    train_all(Path(args.dataset), args.name, _load_cfg(args.config), logger)
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    from rvc_training.rvc import infer

    logger = get_logger("rvc.infer", "outputs/logs/rvc.log")
    infer(args.model, args.index, args.input, args.output, _load_cfg(args.config), logger)
    return 0


def _cmd_tts(args: argparse.Namespace) -> int:
    from rvc_training.tts import synthesize

    logger = get_logger("rvc.tts", "outputs/logs/rvc.log")
    text = Path(args.text).read_text(encoding="utf-8") if Path(args.text).is_file() else args.text
    synthesize(text, Path(args.output), _load_cfg(args.config), logger)
    return 0


def _cmd_fx(args: argparse.Namespace) -> int:
    from metalcore.audio_io import load_audio, save_audio
    from rvc_training import fx

    logger = get_logger("rvc.fx", "outputs/logs/rvc.log")
    audio, sr = load_audio(args.input, sr=None, mono=True)
    out = fx.process(audio, sr, style=args.style, dry_wet=args.dry_wet)
    save_audio(args.output, out, sr)
    logger.info("Wrote %s FX -> %s", args.style, args.output)
    return 0


def _cmd_vocal(args: argparse.Namespace) -> int:
    from rvc_training.pipeline import generate_vocal

    logger = get_logger("rvc.vocal", "outputs/logs/rvc.log")
    cfg = _load_cfg(args.config)
    lyrics = Path(args.lyrics).read_text(encoding="utf-8") if Path(args.lyrics).is_file() else args.lyrics
    work = Path(args.output).parent / "_work"
    generate_vocal(lyrics, args.model, args.index, Path(args.output), cfg, logger, work)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rvc_training",
        description="Stage 4 - vocal generation (isolation, RVC, TTS, FX).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to rvc.yaml.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="Clone RVC-Project + download pretrained weights.")
    p.set_defaults(func=_cmd_setup)

    p = sub.add_parser("isolate", help="Isolate vocals from full mixes (Demucs).")
    p.add_argument("--input", required=True, help="Folder of reference tracks.")
    p.add_argument("--output", required=True, help="Output folder for vocal stems.")
    p.set_defaults(func=_cmd_isolate)

    p = sub.add_parser("prepare", help="Prepare an RVC dataset from vocal stems.")
    p.add_argument("--input", required=True, help="Folder with per-vocalist subfolders of vocals.")
    p.add_argument("--output", required=True, help="Output dataset directory.")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("train", help="Run the full RVC training pipeline.")
    p.add_argument("--dataset", required=True, help="Prepared dataset dir (e.g. .../merged).")
    p.add_argument("--name", required=True, help="Experiment name.")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("infer", help="RVC voice conversion of a WAV.")
    p.add_argument("--model", required=True, help="Trained .pth model.")
    p.add_argument("--index", required=True, help="Feature .index file.")
    p.add_argument("--input", required=True, help="Input WAV.")
    p.add_argument("--output", required=True, help="Output WAV.")
    p.set_defaults(func=_cmd_infer)

    p = sub.add_parser("tts", help="Piper TTS of text/file to a WAV.")
    p.add_argument("--text", required=True, help="Text string or path to a .txt file.")
    p.add_argument("--output", required=True, help="Output WAV.")
    p.set_defaults(func=_cmd_tts)

    p = sub.add_parser("fx", help="Apply a vocal FX preset to a WAV.")
    p.add_argument("--input", required=True, help="Input WAV.")
    p.add_argument("--output", required=True, help="Output WAV.")
    p.add_argument("--style", default="harsh", choices=["clean", "harsh", "scream"], help="FX preset.")
    p.add_argument("--dry-wet", type=float, default=1.0, dest="dry_wet", help="0=dry, 1=wet.")
    p.set_defaults(func=_cmd_fx)

    p = sub.add_parser("vocal", help="Full pipeline: lyrics -> TTS -> RVC -> FX.")
    p.add_argument("--lyrics", required=True, help="Lyrics text or path to a .txt file.")
    p.add_argument("--model", required=True, help="Trained RVC .pth model.")
    p.add_argument("--index", required=True, help="RVC feature .index file.")
    p.add_argument("--output", required=True, help="Output vocal stem WAV.")
    p.set_defaults(func=_cmd_vocal)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
