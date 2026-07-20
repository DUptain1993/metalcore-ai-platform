"""End-to-end vocal generation: lyrics -> TTS -> RVC -> FX -> vocal stem."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from metalcore.audio_io import load_audio, save_audio
from rvc_training import fx, rvc, tts
from rvc_training.config import RVCConfig

_SECTION_HEADER = re.compile(r"^\s*[\[(].*[\])]\s*$")


def extract_lyric_lines(text: str) -> List[str]:
    """Return lyric lines with section headers (``[Verse]`` etc.) removed."""
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _SECTION_HEADER.match(stripped):
            continue
        lines.append(stripped)
    return lines


def apply_fx_file(
    input_wav: Path,
    output_wav: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> Path:
    """Load a WAV, apply the configured vocal FX style, and save the result."""
    audio, sr = load_audio(input_wav, sr=None, mono=True)
    processed = fx.process(audio, sr, style=cfg.vocal_style, dry_wet=cfg.fx_dry_wet)
    save_audio(output_wav, processed, sr)
    logger.info("Applied '%s' FX: %s -> %s", cfg.vocal_style, Path(input_wav).name, output_wav)
    return Path(output_wav)


def generate_vocal(
    lyrics_text: str,
    model_pth: str,
    index_path: str,
    out_path: Path,
    cfg: RVCConfig,
    logger: logging.Logger,
    work_dir: Path,
) -> Path:
    """Turn lyric text into a processed vocal stem.

    Steps:
        1. Piper TTS synthesises the lyrics to a base voice.
        2. RVC converts that voice to the trained (blended) timbre.
        3. The FX chain applies the configured metalcore vocal style.

    Args:
        lyrics_text: Full lyrics (section headers are stripped for TTS).
        model_pth: Trained RVC model ``.pth``.
        index_path: RVC feature ``.index``.
        out_path: Destination for the final vocal stem WAV.
        cfg: RVC configuration.
        logger: Logger.
        work_dir: Scratch directory for intermediate files.

    Returns:
        The output vocal stem path.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lines = extract_lyric_lines(lyrics_text)
    if not lines:
        raise ValueError("No lyric content found after stripping section headers.")
    tts_text = "\n".join(lines)

    logger.info("[1/3] TTS (%d line(s))", len(lines))
    tts_wav = tts.synthesize(tts_text, work_dir / "tts_raw.wav", cfg, logger)

    logger.info("[2/3] RVC voice conversion")
    rvc_wav = rvc.infer(model_pth, index_path, str(tts_wav), str(work_dir / "rvc.wav"), cfg, logger)

    logger.info("[3/3] Vocal FX ('%s')", cfg.vocal_style)
    return apply_fx_file(rvc_wav, out_path, cfg, logger)
