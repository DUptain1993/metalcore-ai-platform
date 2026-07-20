"""Command-line interface for Stage 5 song assembly.

Examples::

    # Full song: generate instrumental sections + mix a vocal stem + master.
    python -m inference.cli song \
        --config configs/assembly.yaml \
        --music-config configs/music_lora.yaml \
        --adapter outputs/music/checkpoints/step_002000/adapter \
        --vocal outputs/vocals/song_vocal.wav \
        --output outputs/songs/track01

    # Just stitch existing section WAVs into one instrumental:
    python -m inference.cli stitch --sections a.wav b.wav c.wav --output outputs/songs/inst.wav

    # Master (loudness + WAV/MP3) an existing mix:
    python -m inference.cli master --input mix.wav --output outputs/songs/track01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from inference.config import AssemblyConfig
from metalcore.config import load_config
from metalcore.logging_utils import get_logger

DEFAULT_CONFIG = "configs/assembly.yaml"


def _load_cfg(path: Optional[str]) -> AssemblyConfig:
    if path and Path(path).is_file():
        return load_config(path, AssemblyConfig)
    if Path(DEFAULT_CONFIG).is_file():
        return load_config(DEFAULT_CONFIG, AssemblyConfig)
    return AssemblyConfig()


def _cmd_song(args: argparse.Namespace) -> int:
    from inference.assemble import assemble_song

    logger = get_logger("assembly.song", "outputs/logs/assembly.log")
    assemble_song(
        out_base=Path(args.output),
        cfg=_load_cfg(args.config),
        logger=logger,
        music_config=args.music_config,
        adapter_dir=args.adapter,
        instrumental_path=args.instrumental,
        vocal_path=args.vocal,
    )
    return 0


def _cmd_stitch(args: argparse.Namespace) -> int:
    from inference.sections import load_and_stitch

    logger = get_logger("assembly.stitch", "outputs/logs/assembly.log")
    load_and_stitch([Path(p) for p in args.sections], _load_cfg(args.config), logger, Path(args.output))
    return 0


def _cmd_master(args: argparse.Namespace) -> int:
    from inference.master import export
    from metalcore.audio_io import load_audio

    logger = get_logger("assembly.master", "outputs/logs/assembly.log")
    cfg = _load_cfg(args.config)
    audio, sr = load_audio(args.input, sr=None, mono=False)
    export(audio, sr, Path(args.output), cfg, logger)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference",
        description="Stage 5 - song assembly & rendering.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to assembly.yaml.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("song", help="Assemble a full song (instrumental + optional vocal).")
    p.add_argument("--music-config", default=None, dest="music_config", help="music_lora.yaml (to generate instrumentals).")
    p.add_argument("--adapter", default=None, help="MusicGen adapter dir (to generate instrumentals).")
    p.add_argument("--instrumental", default=None, help="Existing instrumental WAV (skips generation).")
    p.add_argument("--vocal", default=None, help="Vocal stem WAV to mix in (optional).")
    p.add_argument("--output", required=True, help="Output path without extension.")
    p.set_defaults(func=_cmd_song)

    p = sub.add_parser("stitch", help="Cross-fade section WAVs into one instrumental.")
    p.add_argument("--sections", nargs="+", required=True, help="Section WAVs in order.")
    p.add_argument("--output", required=True, help="Output WAV path.")
    p.set_defaults(func=_cmd_stitch)

    p = sub.add_parser("master", help="Loudness-normalise + export WAV/MP3.")
    p.add_argument("--input", required=True, help="Input mix WAV.")
    p.add_argument("--output", required=True, help="Output path without extension.")
    p.set_defaults(func=_cmd_master)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
