# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Git repository initialisation and first push to GitHub.
- `LICENSE` (MIT) with an explicit note on model/data/third-party licensing.
- `docs/ARCHITECTURE.md` — module-by-module internals, data-flow diagram, and a
  "where to make common changes" map.
- `docs/CONFIG_REFERENCE.md` — every config key across all five stages, with
  type, default, and effect.
- `docs/README.md` — documentation index.
- `docs/VOICE_SAMPLES_TUTORIAL.md` — thorough, style-by-style guide to collecting
  vocal samples for Stage 4 (sourcing, recording, isolation, cleaning, layout).
- `CONTRIBUTING.md` — code standards and the CPU-path local checks.
- `.github/workflows/ci.yml` — CI running the Stage 1 smoke test, the
  self-contained DSP smoke test, and byte-compilation on every push/PR.
- `Makefile` — convenience targets (`install`, `smoke`, `dsp`, `compile`, `lint`).

### Changed
- Expanded the top-level `README.md` with a pipeline diagram and a documentation
  index.
- `.gitignore` now excludes editor state, Claude Code local state, personal
  scratch (`c.txt`), and Stage 1 `quarantine/`.

## [0.1.0]

### Added
- **Stage 1 — dataset processing** (`dataset_tools/`): validate, loudness
  normalise, resample, chunk, caption (tempo/key + style tags), leakage-safe
  train/val split. CPU-only; verified via `scripts/smoke_test.sh`.
- **Stage 2 — MusicGen LoRA** (`music_training/`): `facebook/musicgen-medium`
  fine-tune via `transformers` + `peft`, fp16 AMP, gradient checkpointing,
  EnCodec code caching, resume-safe checkpointing, validation generation.
- **Stage 3 — lyrics QLoRA** (`lyric_training/`): 4-bit QLoRA on
  `Qwen2.5-1.5B-Instruct` via `trl`, structured (Verse/Chorus/…/Final Chorus)
  generation, theme conditioning.
- **Stage 4 — vocals** (`rvc_training/`): Demucs isolation, vocal dataset
  prep/blend, RVC-Project integration, Piper TTS, self-contained FX scream chain
  (`clean`/`harsh`/`scream`), full lyrics→TTS→RVC→FX pipeline.
- **Stage 5 — assembly** (`inference/`): equal-power section stitching, mix
  (stereo widen + vocal fit + bus limit), master (loudness normalise + WAV/MP3).
- Kaggle notebooks `01`→`05`, per-stage `requirements-*.txt`, `configs/*.yaml`,
  and the `docs/` guides (hardware, troubleshooting, dataset, vocals, assembly).
- Self-contained DSP unit tests (`scripts/smoke_test_dsp.py`).
