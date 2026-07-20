"""Command-line interface for Stage 1 dataset processing.

Examples::

    # Run the whole pipeline (validate -> preprocess -> caption -> split):
    python -m dataset_tools.cli all --input data/raw --output data/dataset

    # Or run individual steps:
    python -m dataset_tools.cli validate   --input data/raw --output data/dataset
    python -m dataset_tools.cli preprocess --input data/raw --output data/dataset
    python -m dataset_tools.cli caption    --output data/dataset
    python -m dataset_tools.cli split      --output data/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from dataset_tools.captions import run_captions
from dataset_tools.config import DatasetConfig
from dataset_tools.metadata import ChunkRecord, read_jsonl
from dataset_tools.preprocess import CHUNKS_JSONL, run_preprocess
from dataset_tools.split import run_split
from dataset_tools.validate import ValidationResult, run_validate, valid_paths
from metalcore.config import load_config
from metalcore.logging_utils import get_logger

DEFAULT_CONFIG = "configs/dataset.yaml"


def _load_cfg(path: Optional[str]) -> DatasetConfig:
    if path and Path(path).is_file():
        return load_config(path, DatasetConfig)
    if path:
        raise FileNotFoundError(f"Config not found: {path}")
    # Fall back to defaults if the standard config file is absent.
    if Path(DEFAULT_CONFIG).is_file():
        return load_config(DEFAULT_CONFIG, DatasetConfig)
    return DatasetConfig()


def _valid_paths_from_report(output_dir: Path) -> List[Path]:
    report = output_dir / "report_validate.json"
    if not report.is_file():
        raise FileNotFoundError(
            f"No validation report at {report}. Run the 'validate' step first "
            f"(or use 'all')."
        )
    with report.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    results = [ValidationResult(**item) for item in data]
    return valid_paths(results)


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    output = Path(args.output)
    logger = get_logger("dataset.validate", output / "logs" / "dataset.log")
    results = run_validate(Path(args.input), output, cfg, logger, quarantine=not args.no_quarantine)
    return 0 if any(r.ok for r in results) else 1


def _cmd_preprocess(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    output = Path(args.output)
    logger = get_logger("dataset.preprocess", output / "logs" / "dataset.log")
    files = _valid_paths_from_report(output)
    if not files:
        logger.error("No valid files to preprocess.")
        return 1
    records = run_preprocess(files, Path(args.input), output, cfg, logger)
    return 0 if records else 1


def _cmd_caption(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    output = Path(args.output)
    logger = get_logger("dataset.caption", output / "logs" / "dataset.log")
    records = read_jsonl(output / CHUNKS_JSONL)
    if not records:
        logger.error("No chunk records found. Run 'preprocess' first.")
        return 1
    run_captions(records, output, cfg, logger)
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    output = Path(args.output)
    logger = get_logger("dataset.split", output / "logs" / "dataset.log")
    records = read_jsonl(output / "metadata.jsonl")
    if not records:
        logger.error("No metadata records found. Run 'caption' first.")
        return 1
    run_split(records, output, cfg, logger)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    output = Path(args.output)
    logger = get_logger("dataset.all", output / "logs" / "dataset.log")

    logger.info("=== Stage 1: dataset processing ===")
    results = run_validate(Path(args.input), output, cfg, logger, quarantine=not args.no_quarantine)
    files = valid_paths(results)
    if not files:
        logger.error("No valid audio files; aborting.")
        return 1

    records = run_preprocess(files, Path(args.input), output, cfg, logger)
    if not records:
        logger.error("No chunks produced; aborting.")
        return 1

    records = run_captions(records, output, cfg, logger)
    run_split(records, output, cfg, logger)
    logger.info("=== Stage 1 complete ===")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_tools",
        description="Stage 1 - validate, preprocess, caption and split an audio dataset.",
    )
    parser.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG} if present).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, need_input: bool) -> None:
        if need_input:
            sp.add_argument("--input", required=True, help="Directory of raw audio (searched recursively).")
        sp.add_argument("--output", required=True, help="Dataset output directory.")

    p_val = sub.add_parser("validate", help="Validate raw audio files.")
    add_common(p_val, need_input=True)
    p_val.add_argument("--no-quarantine", action="store_true", help="Do not copy invalid files to quarantine/.")
    p_val.set_defaults(func=_cmd_validate)

    p_pre = sub.add_parser("preprocess", help="Normalise + chunk validated audio.")
    add_common(p_pre, need_input=True)
    p_pre.set_defaults(func=_cmd_preprocess)

    p_cap = sub.add_parser("caption", help="Generate captions -> metadata.jsonl.")
    add_common(p_cap, need_input=False)
    p_cap.set_defaults(func=_cmd_caption)

    p_spl = sub.add_parser("split", help="Write train.jsonl / val.jsonl.")
    add_common(p_spl, need_input=False)
    p_spl.set_defaults(func=_cmd_split)

    p_all = sub.add_parser("all", help="Run the entire Stage 1 pipeline.")
    add_common(p_all, need_input=True)
    p_all.add_argument("--no-quarantine", action="store_true", help="Do not copy invalid files to quarantine/.")
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
