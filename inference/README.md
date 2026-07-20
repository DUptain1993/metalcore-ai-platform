# Stage 5 — Song assembly & rendering

Combines the outputs of the earlier stages into a finished track:

```
MusicGen sections ──▶ stitch (cross-fade) ──▶ full instrumental
                                                     │
vocal stem (Stage 4) ─────────────────────────▶ mix (stereo)
                                                     │
                                                     ▼
                                       master (loudness) ──▶ WAV + MP3
```

## Modules

| Module | Kind | Notes |
|---|---|---|
| `sections.py` | ✅ self-contained stitch (+ lazy MusicGen gen) | equal-power cross-fade |
| `mix.py` | ✅ self-contained | vocal-over-instrumental, stereo widening, bus limit |
| `master.py` | ✅ self-contained | loudness normalise (pyloudnorm) + WAV/MP3 export |
| `assemble.py` | orchestration | end-to-end song builder |

All DSP is numpy/scipy/pydub and unit-tested; only instrumental *generation*
needs a GPU (it reuses Stage 2).

## Quickstart

```bash
pip install -r requirements-assembly.txt   # + ffmpeg for MP3

# Full song: generate instrumental sections, mix the vocal stem, master.
python -m inference.cli song \
    --config configs/assembly.yaml \
    --music-config configs/music_lora.yaml \
    --adapter outputs/music/checkpoints/step_002000/adapter \
    --vocal   outputs/vocals/song_vocal.wav \
    --output  outputs/songs/track01

# Instrumental only (no --vocal), or bring your own instrumental (--instrumental).
# Utility subcommands:
python -m inference.cli stitch --sections a.wav b.wav c.wav --output inst.wav
python -m inference.cli master --input mix.wav --output outputs/songs/track01
```

Edit the `sections:` list in `configs/assembly.yaml` to change the arrangement,
per-section prompts, and lengths. Full guide:
[`docs/ASSEMBLY_GUIDE.md`](../docs/ASSEMBLY_GUIDE.md).

## Notes

- MusicGen is mono/32 kHz; the assembler stitches at 32 kHz, then upsamples to
  `export_sample_rate` (44.1 kHz) on export.
- `target_lufs` defaults to a loud, metalcore-style −9 LUFS. Peaks are limited to
  −0.1 dBFS to avoid clipping.
