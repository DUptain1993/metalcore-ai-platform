"""Deterministic train/validation split grouped by source track.

Grouping by ``source`` guarantees that chunks from the same track never appear
in both splits, preventing optimistic validation metrics from data leakage.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from dataset_tools.config import DatasetConfig
from dataset_tools.metadata import ChunkRecord, write_jsonl

TRAIN_JSONL = "train.jsonl"
VAL_JSONL = "val.jsonl"


def split_records(
    records: List[ChunkRecord], val_ratio: float, seed: int
) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
    """Split records into train/val by source track.

    Args:
        records: All chunk records.
        val_ratio: Approximate fraction of *tracks* assigned to validation.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple ``(train_records, val_records)``.
    """
    by_source: Dict[str, List[ChunkRecord]] = {}
    for record in records:
        by_source.setdefault(record.source, []).append(record)

    sources = sorted(by_source)
    rng = random.Random(seed)
    rng.shuffle(sources)

    n_val = int(round(len(sources) * val_ratio))
    # With very few tracks, keep at least one in each split when possible.
    if len(sources) >= 2:
        n_val = min(max(n_val, 1), len(sources) - 1)
    val_sources = set(sources[:n_val])

    train: List[ChunkRecord] = []
    val: List[ChunkRecord] = []
    for source in sources:
        target = val if source in val_sources else train
        for record in by_source[source]:
            record.split = "val" if source in val_sources else "train"
            target.append(record)

    return train, val


def run_split(
    records: List[ChunkRecord],
    dataset_root: Path,
    cfg: DatasetConfig,
    logger: logging.Logger,
) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
    """Split records and write ``train.jsonl`` / ``val.jsonl``.

    Args:
        records: Captioned chunk records.
        dataset_root: Folder to write the split files into.
        cfg: Dataset configuration (val ratio + seed).
        logger: Logger for progress.

    Returns:
        Tuple ``(train_records, val_records)``.
    """
    dataset_root = Path(dataset_root)
    train, val = split_records(records, cfg.val_ratio, cfg.seed)

    n_train = write_jsonl(dataset_root / TRAIN_JSONL, train)
    n_val = write_jsonl(dataset_root / VAL_JSONL, val)

    n_sources = len({r.source for r in records})
    logger.info(
        "Split %d chunk(s) from %d track(s): %d train / %d val -> %s, %s",
        len(records),
        n_sources,
        n_train,
        n_val,
        dataset_root / TRAIN_JSONL,
        dataset_root / VAL_JSONL,
    )
    if n_val == 0:
        logger.warning(
            "Validation split is empty (need >= 2 source tracks for a val split)."
        )
    return train, val
