"""Turn a folder of ``.txt`` lyrics into an instruction-tuning dataset.

Each ``.txt`` file is treated as one song. An optional front-matter line such as
``#theme: depression, hope`` sets the song's themes; otherwise themes are
inferred from the text via a keyword lexicon. Records are written as JSONL with
``instruction`` / ``output`` / ``themes`` fields consumed by the trainer.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lyric_training.config import LyricsLoRAConfig

TRAIN_JSONL = "lyrics_train.jsonl"
VAL_JSONL = "lyrics_val.jsonl"

#: Keyword lexicon used to infer themes when a file has no explicit tags.
THEME_KEYWORDS: Dict[str, List[str]] = {
    "depression": ["empty", "numb", "hollow", "drown", "alone", "cold", "void", "sink", "darkness"],
    "recovery": ["heal", "rise", "stronger", "recover", "mend", "breathe", "forward", "reborn"],
    "addiction": ["needle", "vein", "crave", "habit", "poison", "chase", "relapse", "substance"],
    "self destruction": ["destroy", "burn", "ruin", "blade", "scars", "collapse", "self-hate"],
    "hope": ["light", "hope", "dawn", "hold on", "believe", "tomorrow", "shine", "faith"],
    "betrayal": ["lie", "lied", "betray", "knife", "trust", "traitor", "deceive", "backstab"],
    "resilience": ["survive", "endure", "fight", "stand", "unbroken", "withstand", "overcome"],
    "emotional struggle": ["pain", "struggle", "tears", "weight", "suffocate", "battle", "torn"],
}

_FRONTMATTER_RE = re.compile(r"^\s*#\s*themes?\s*:\s*(.+)$", re.IGNORECASE)
_META_LINE_RE = re.compile(r"^\s*#\s*(title|artist|album)\s*:", re.IGNORECASE)


def infer_themes(text: str, themes: List[str], max_themes: int = 3) -> List[str]:
    """Infer up to ``max_themes`` themes from lyric text via keyword counts."""
    lowered = text.lower()
    scores: List[Tuple[int, str]] = []
    for theme in themes:
        keywords = THEME_KEYWORDS.get(theme, [theme])
        count = sum(lowered.count(kw) for kw in keywords)
        if count > 0:
            scores.append((count, theme))
    scores.sort(reverse=True)
    inferred = [theme for _, theme in scores[:max_themes]]
    if not inferred:
        # Nothing matched; fall back to two generic metalcore themes.
        inferred = [t for t in ("emotional struggle", "resilience") if t in themes][:2]
    return inferred


def parse_lyric_file(
    path: Path, cfg: LyricsLoRAConfig
) -> Optional[Tuple[str, List[str]]]:
    """Parse a lyric file into ``(text, themes)``.

    Returns ``None`` if the file has no usable lyric content.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    themes: List[str] = []
    body_lines: List[str] = []

    for line in raw.splitlines():
        match = _FRONTMATTER_RE.match(line)
        if match:
            themes = [t.strip().lower() for t in match.group(1).split(",") if t.strip()]
            continue
        if _META_LINE_RE.match(line):
            continue  # drop title/artist/album metadata lines
        body_lines.append(line)

    text = "\n".join(body_lines).strip()
    if not text:
        return None

    # Keep only themes the config knows about; infer if none were provided.
    themes = [t for t in themes if t in cfg.themes]
    if not themes:
        themes = infer_themes(text, cfg.themes)
    return text, themes


def build_instruction(themes: List[str], sections: List[str]) -> str:
    """Compose the user instruction used for a training example."""
    theme_str = ", ".join(themes) if themes else "emotional struggle"
    section_str = ", ".join(sections)
    return (
        "Write original metalcore song lyrics exploring the theme(s) of "
        f"{theme_str}. Structure the song with these sections in order: "
        f"{section_str}. Label each section clearly and make the writing "
        "intense, emotional and vivid."
    )


def build_examples(
    lyrics_dir: Path, cfg: LyricsLoRAConfig, logger: logging.Logger
) -> List[Dict[str, object]]:
    """Build instruction/output examples from every ``.txt`` file in a folder."""
    lyrics_dir = Path(lyrics_dir)
    files = sorted(lyrics_dir.rglob("*.txt"))
    logger.info("Found %d lyric file(s) under %s", len(files), lyrics_dir)

    examples: List[Dict[str, object]] = []
    for path in files:
        parsed = parse_lyric_file(path, cfg)
        if parsed is None:
            logger.warning("Skipping empty lyric file: %s", path.name)
            continue
        text, themes = parsed
        examples.append(
            {
                "instruction": build_instruction(themes, cfg.sections),
                "output": text,
                "themes": themes,
            }
        )
    logger.info("Built %d training example(s)", len(examples))
    return examples


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read a lyrics JSONL file into a list of dicts."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Lyrics dataset not found: {src}")
    rows: List[Dict[str, object]] = []
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(
    lyrics_dir: Path,
    output_dir: Path,
    cfg: LyricsLoRAConfig,
    logger: logging.Logger,
) -> Tuple[Path, Optional[Path]]:
    """Build and split the lyrics dataset, writing train/val JSONL files.

    Returns:
        Tuple ``(train_path, val_path_or_None)``.
    """
    output_dir = Path(output_dir)
    examples = build_examples(lyrics_dir, cfg, logger)
    if not examples:
        raise RuntimeError(f"No usable lyric files found under {lyrics_dir}")

    rng = random.Random(cfg.seed)
    rng.shuffle(examples)

    n_val = int(round(len(examples) * cfg.val_ratio))
    if len(examples) >= 2 and cfg.val_ratio > 0:
        n_val = min(max(n_val, 1), len(examples) - 1)
    else:
        n_val = 0

    val_rows = examples[:n_val]
    train_rows = examples[n_val:]

    train_path = output_dir / TRAIN_JSONL
    _write_jsonl(train_path, train_rows)
    logger.info("Wrote %d train example(s) -> %s", len(train_rows), train_path)

    val_path: Optional[Path] = None
    if val_rows:
        val_path = output_dir / VAL_JSONL
        _write_jsonl(val_path, val_rows)
        logger.info("Wrote %d val example(s) -> %s", len(val_rows), val_path)

    return train_path, val_path
