# Contributing

Thanks for your interest in improving the Metalcore AI platform. This is a
Kaggle-first, CPU-authoring / GPU-training project, so a few conventions keep it
healthy.

## Ground rules

- **No committed data or model artefacts.** Audio, checkpoints, adapters, and
  indexes are `.gitignore`d — store them on Kaggle Datasets, not in git.
- **Respect the legal guardrails.** Do not add features whose primary purpose is
  cloning specific copyrighted songs or real voices for distribution. See
  [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) and
  [`docs/VOCALS_GUIDE.md`](docs/VOCALS_GUIDE.md).

## Code standards (enforced by review + CI)

- **Type hints** on public functions.
- **Logging, not `print`** for pipeline steps — go through
  `metalcore.logging_utils.get_logger`.
- **Error handling** with actionable messages (name the file/flag/config key).
- **CPU-first import hygiene.** Import `torch`/`transformers`/`peft`/`demucs`
  *inside functions*, never at module top level, so `--help` and Stage 1 run on a
  no-GPU machine.
- **Config → dataclass.** New config keys go in both `configs/*.yaml` and the
  matching `*/config.py` dataclass, and get a row in
  [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md).
- Every stage keeps a working `argparse` CLI (`python -m <stage>.cli --help`).

## Local checks before you push

Only the CPU path can run on a laptop; everything GPU is verified on Kaggle.

```bash
# 1. Stage 1 end-to-end (synthesises audio, no GPU/data needed)
pip install -r requirements-dataset.txt
bash scripts/smoke_test.sh

# 2. Self-contained DSP for Stages 4 & 5 (FX, segmentation, stitch/mix/master)
pip install -r requirements-vocals.txt -r requirements-assembly.txt
PYTHONPATH=. python scripts/smoke_test_dsp.py

# 3. Byte-compile everything (catches syntax errors without importing GPU deps)
python -m compileall -q metalcore dataset_tools music_training \
    lyric_training rvc_training inference
```

CI (`.github/workflows/ci.yml`) runs checks 1–3 on every push and PR.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; update the relevant doc(s) in the same PR.
3. Ensure the local checks above pass.
4. Describe what you verified (CPU checks locally, and any GPU stage you ran on
   Kaggle).
