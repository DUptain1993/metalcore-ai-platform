"""Text-to-speech front-end (Piper) that produces the base voice for RVC.

Piper is MIT-licensed, fast, and Kaggle-friendly. It produces *spoken-cadence*
audio; RVC then converts the timbre and the FX chain adds aggression. Piper does
not sing melodically -- this is the documented limitation of the TTS->RVC path.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

from rvc_training.config import RVCConfig


def build_piper_cmd(voice_model: str, out_path: Path) -> List[str]:
    """Construct the Piper CLI command (text is fed via stdin)."""
    return ["piper", "--model", str(voice_model), "--output_file", str(out_path)]


def synthesize(
    text: str,
    out_path: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> Path:
    """Synthesize ``text`` to a WAV at ``out_path`` using Piper.

    Args:
        text: Text to speak (a lyric line or full stanza).
        out_path: Destination WAV path.
        cfg: RVC configuration (supplies the Piper voice model path).
        logger: Logger.

    Returns:
        The output path.

    Raises:
        FileNotFoundError: If the Piper voice model is missing.
        RuntimeError: If Piper exits non-zero.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(cfg.piper_voice).is_file():
        raise FileNotFoundError(
            f"Piper voice model not found: {cfg.piper_voice}. "
            f"Download one (see docs/VOCALS_GUIDE.md) and set 'piper_voice' in configs/rvc.yaml."
        )

    cmd = build_piper_cmd(cfg.piper_voice, out_path)
    logger.info("$ %s  (stdin: %d chars)", " ".join(cmd), len(text))
    result = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Piper failed (%d): %s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"Piper failed with code {result.returncode}")
    if not out_path.is_file():
        raise RuntimeError(f"Piper reported success but no output at {out_path}")
    return out_path


def synthesize_lines(
    lines: List[str],
    out_dir: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> List[Path]:
    """Synthesize each non-empty line to its own WAV. Returns the paths in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        out = out_dir / f"line_{idx:04d}.wav"
        synthesize(line, out, cfg, logger)
        paths.append(out)
        idx += 1
    logger.info("Synthesized %d line(s) -> %s", len(paths), out_dir)
    return paths
