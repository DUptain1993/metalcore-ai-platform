"""Dataset record model and JSONL read/write helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


@dataclass
class ChunkRecord:
    """One training chunk.

    Attributes:
        audio: Path to the chunk WAV, relative to the dataset root (the folder
            that contains ``metadata.jsonl``).
        source: Identifier of the source track the chunk came from. Used to
            keep all chunks of a track on the same side of the train/val split.
        duration: Chunk duration in seconds.
        sample_rate: Chunk sample rate in Hz.
        caption: Text description used to condition MusicGen (filled by the
            captioning step).
        split: ``"train"`` or ``"val"`` (filled by the split step).
    """

    audio: str
    source: str
    duration: float
    sample_rate: int
    caption: str = ""
    split: str = ""


def _record_from_dict(data: Dict[str, Any]) -> ChunkRecord:
    field_names = {f.name for f in fields(ChunkRecord)}
    return ChunkRecord(**{k: v for k, v in data.items() if k in field_names})


def write_jsonl(path: Union[str, Path], records: Iterable[ChunkRecord]) -> int:
    """Write records as JSON-lines. Returns the number of records written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Union[str, Path]) -> List[ChunkRecord]:
    """Read a JSON-lines file into :class:`ChunkRecord` objects."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Metadata file not found: {src}")
    records: List[ChunkRecord] = []
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_dict(json.loads(line)))
    return records
