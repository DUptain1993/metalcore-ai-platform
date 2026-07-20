# Stage 4 — Vocal generation (RVC + FX scream chain)

Turns generated lyrics into a metalcore vocal stem:

```
lyrics ──▶ Piper TTS ──▶ RVC voice conversion ──▶ FX chain ──▶ vocal stem
                              (blended voice)      (clean/harsh/scream)
```

Plus the dataset side: **Demucs** isolates vocals from reference tracks, and the
prep tools cut/clean/organise them for RVC training — with a **merged** mode that
blends multiple vocalists into one voice.

## What's self-contained vs integrated

| Module | Kind | Notes |
|---|---|---|
| `dataset.py` | ✅ self-contained | segment/validate/blend vocal clips (numpy/librosa) |
| `fx.py` | ✅ self-contained | clean/harsh/scream DSP (numpy/scipy), unit-tested |
| `isolate.py` | wrapper | Demucs CLI (pip-installable) |
| `tts.py` | wrapper | Piper CLI (MIT) |
| `rvc.py` | wrapper | integrates **RVC-Project** (clone + pretrained weights) |
| `pipeline.py` | orchestration | lyrics → TTS → RVC → FX |

`rvc.py` **integrates** the MIT-licensed `RVC-Project` rather than reimplementing
it. Its command builders target that repo's canonical script layout; adjust them
if you use a fork. Full setup in [`docs/VOCALS_GUIDE.md`](../docs/VOCALS_GUIDE.md).

## Quickstart (Kaggle GPU)

```bash
pip install -r requirements-vocals.txt

# 1) One-time: clone RVC-Project + download pretrained weights, then install its deps.
python -m rvc_training.cli setup
pip install -r /kaggle/working/Retrieval-based-Voice-Conversion-WebUI/requirements.txt

# 2) Build the vocal dataset (isolate vocals, then prepare clips).
python -m rvc_training.cli isolate --input data/refs        --output data/vocals_raw
python -m rvc_training.cli prepare --input data/vocals_raw  --output data/rvc_dataset

# 3) Train the blended voice.
python -m rvc_training.cli train --dataset data/rvc_dataset/merged --name blend

# 4) Generate a vocal stem from lyrics.
python -m rvc_training.cli vocal \
    --lyrics outputs/lyrics/song.txt \
    --model  .../logs/blend/blend.pth \
    --index  .../logs/blend/added_*.index \
    --output outputs/vocals/song_vocal.wav

# Or just re-skin any WAV with the FX chain (no RVC needed):
python -m rvc_training.cli fx --input in.wav --output out.wav --style scream
```

## Honest limitations

- **Screams have a real quality ceiling.** RVC is f0-based voice conversion; the
  FX chain (waveshaping + roughness + spectral darkening) *approximates* a scream
  but does not synthesise true fry/false-cord chaos.
- **Piper is spoken-cadence**, not singing. The vocal follows speech rhythm; RVC
  changes timbre, not melody.
- **Voice likeness / legal.** Training on real, identifiable vocalists carries
  publicity/likeness considerations — keep for personal/research use and don't
  present output as the real artists. See the vocals guide.
- **Disk/VRAM.** RVC + Demucs + pretrained weights are the heaviest parts on
  Kaggle; budgets are in [`docs/HARDWARE.md`](../docs/HARDWARE.md).
