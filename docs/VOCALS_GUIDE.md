# Vocals guide (Stage 4) — RVC setup, blending & the scream ceiling

This stage converts lyrics into a metalcore vocal stem, and can **blend multiple
vocalists** into one voice. It integrates the external, MIT-licensed
`RVC-Project`. Read the legal note first.

> **New to collecting vocal audio?** See
> [`VOICE_SAMPLES_TUTORIAL.md`](VOICE_SAMPLES_TUTORIAL.md) — a thorough,
> style-by-style guide (Currents, Fit For A King, Oceans Ate Alaska, Wind
> Walkers, Aviana, The Plot In You) for sourcing, recording, isolating, cleaning,
> and organising the samples this stage trains on.

## Legal & ethical note (read first)

Training a voice model on a real, identifiable vocalist raises **likeness and
publicity-rights** concerns that vary by jurisdiction. Keep such models for
**personal/research use**, and do **not** present or distribute the output as the
real artists. Prefer voices you have rights to (your own, licensed, or
permissively-licensed sources). See also `docs/DATASET_GUIDE.md`.

## The scream ceiling (honest expectations)

- **RVC** is f0 (pitch)-based voice *conversion*. It changes timbre; it does not
  invent the non-harmonic, chaotic excitation of a real fry/false-cord scream.
- **Piper TTS** provides a *spoken-cadence* base voice — it does not sing melody.
- The **FX chain** (`rvc_training/fx.py`) pushes the converted voice toward
  aggression with waveshaping, roughness (noisy AM), bitcrush, band presence and
  spectral darkening. Styles: `clean`, `harsh`, `scream`.
- Net: expect convincing *harsh/aggressive spoken* vocals; true screams remain
  approximate. This is a fundamental limitation of the open TTS→RVC path, not a
  bug.

## Data layout

Organise **isolated** vocal stems per vocalist:

```
data/vocals_raw/
├── vocalist_a/  *.wav
├── vocalist_b/  *.wav
└── vocalist_c/  *.wav
```

If you only have full mixes, isolate first:

```bash
python -m rvc_training.cli isolate --input data/refs --output data/vocals_raw
```

(Demucs writes `data/vocals_raw/<model>/<track>/vocals.wav`; reorganise those
into per-vocalist folders before `prepare`.)

## Blending three vocalists

`configs/rvc.yaml` → `blend_mode`:

- **`merged`** (default): all three vocalists are pooled into **one** dataset
  (`data/rvc_dataset/merged/`, clips prefixed by speaker) and trained into a
  single blended voice. `max_minutes_per_speaker` keeps the blend balanced.
- **`per_speaker`**: one dataset per vocalist → train separate models, then pick
  per section.

```bash
python -m rvc_training.cli prepare --input data/vocals_raw --output data/rvc_dataset
```

## RVC setup (one-time, on Kaggle)

```bash
python -m rvc_training.cli setup      # clones RVC-Project + downloads weights
pip install -r /kaggle/working/Retrieval-based-Voice-Conversion-WebUI/requirements.txt
```

Downloaded into the repo's `assets/`: `hubert_base.pt`, `rmvpe.pt`,
`pretrained_v2/f0G40k.pth`, `pretrained_v2/f0D40k.pth`.

> The training/inference wrappers in `rvc_training/rvc.py` target RVC-Project's
> **canonical** script interface (`infer/modules/train/*`, `tools/infer_cli.py`).
> If you clone a fork with a different CLI, adjust the `*_cmd` builders or point
> `rvc_repo` / `rvc_python` at the right paths in `configs/rvc.yaml`.

## Piper voice

Download a Piper voice (`.onnx` + `.onnx.json`) from the Piper voices release and
set `piper_voice` in `configs/rvc.yaml`. A neutral English voice works well as an
RVC base since RVC replaces the timbre anyway.

## Train & generate

```bash
python -m rvc_training.cli train --dataset data/rvc_dataset/merged --name blend
python -m rvc_training.cli vocal \
    --lyrics outputs/lyrics/song.txt \
    --model  .../logs/blend/blend.pth \
    --index  .../logs/blend/added_*.index \
    --output outputs/vocals/song_vocal.wav
```

Set the aggression with `vocal_style` (`clean|harsh|scream`) and `fx_dry_wet` in
the config, or ad-hoc:

```bash
python -m rvc_training.cli fx --input any.wav --output out.wav --style scream --dry-wet 0.8
```
