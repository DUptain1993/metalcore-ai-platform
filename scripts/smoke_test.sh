#!/usr/bin/env bash
#
# End-to-end smoke test for Stage 1 (dataset processing) plus CLI import checks
# for Stages 2 and 3. Requires only the Stage-1 CPU dependencies:
#
#     pip install -r requirements-dataset.txt
#
# It synthesises a few short WAV files (no real audio or GPU needed), runs the
# full validate -> preprocess -> caption -> split pipeline, and asserts the
# expected outputs exist and are non-empty.
#
# Usage:  bash scripts/smoke_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK="$(mktemp -d)"
RAW="$WORK/raw"
OUT="$WORK/dataset"
mkdir -p "$RAW"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "== Synthesising test audio in $RAW =="
python - "$RAW" <<'PY'
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

raw = Path(sys.argv[1])
sr = 44100  # deliberately not 32k so resampling is exercised
rng = np.random.default_rng(0)

# Three valid "tracks" of ~40s each (tones + noise so they are not silent).
for i, freq in enumerate([110.0, 146.83, 196.0]):
    t = np.linspace(0, 40.0, int(sr * 40.0), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    noise = 0.02 * rng.standard_normal(t.shape)
    sf.write(raw / f"track_{i}.wav", (tone + noise).astype("float32"), sr)

# One near-silent file (should be rejected).
sf.write(raw / "silent.wav", np.zeros(sr * 5, dtype="float32"), sr)

# One too-short file (should be rejected).
sf.write(raw / "tiny.wav", (0.3 * np.sin(np.linspace(0, 6.28, sr // 2))).astype("float32"), sr)
print("wrote 5 files")
PY

echo "== Running Stage 1 pipeline =="
python -m dataset_tools.cli all --input "$RAW" --output "$OUT"

echo "== Verifying outputs =="
fail=0
for f in metadata.jsonl train.jsonl chunks.jsonl report_validate.json; do
    if [[ ! -s "$OUT/$f" ]]; then
        echo "MISSING or empty: $OUT/$f"
        fail=1
    fi
done

n_chunks=$(find "$OUT/chunks" -name '*.wav' | wc -l | tr -d ' ')
echo "Produced $n_chunks chunk file(s)."
if [[ "$n_chunks" -lt 1 ]]; then
    echo "No chunks were produced."
    fail=1
fi

echo "== CLI import checks for all stages =="
python -m dataset_tools.cli --help  >/dev/null && echo "dataset_tools CLI OK"
python -m music_training.cli --help >/dev/null && echo "music_training CLI OK"
python -m lyric_training.cli --help >/dev/null && echo "lyric_training CLI OK"
python -m rvc_training.cli --help   >/dev/null && echo "rvc_training CLI OK"
python -m inference.cli --help      >/dev/null && echo "inference CLI OK"

if [[ "$fail" -ne 0 ]]; then
    echo "SMOKE TEST FAILED"
    exit 1
fi
echo "SMOKE TEST PASSED"
