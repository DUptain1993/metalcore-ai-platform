# Documentation index

Start at the top-level [`../README.md`](../README.md) for the project overview and
quickstart. This folder holds the deep-dive docs.

## By purpose

| Doc | Read it when you want to… |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Understand how the code fits together — module map, data flow, where to make changes. |
| [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) | Look up any config key (all 5 stages) — type, default, effect. |
| [HARDWARE.md](HARDWARE.md) | Check VRAM estimates, disk budgets, and expected training durations per stage. |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Fix a specific error — organised by stage, symptom → fix. |

## By stage (how-to guides)

| Stage | Guide | Also see |
|---|---|---|
| 1 · Dataset | [DATASET_GUIDE.md](DATASET_GUIDE.md) — collecting & laying out data (legally) | `dataset_tools/` |
| 2 · Music | (README quickstart §3) | `music_training/`, [CONFIG_REFERENCE](CONFIG_REFERENCE.md) |
| 3 · Lyrics | [DATASET_GUIDE.md#stage-3--lyrics](DATASET_GUIDE.md) | `lyric_training/` |
| 4 · Vocals | [VOCALS_GUIDE.md](VOCALS_GUIDE.md) — RVC setup, blending, the scream ceiling · [VOICE_SAMPLES_TUTORIAL.md](VOICE_SAMPLES_TUTORIAL.md) — how to collect vocal samples | [`../rvc_training/README.md`](../rvc_training/README.md) |
| 5 · Assembly | [ASSEMBLY_GUIDE.md](ASSEMBLY_GUIDE.md) — arrangement, mixing, mastering | [`../inference/README.md`](../inference/README.md) |

## Kaggle workflow

The [`../notebooks/`](../notebooks) folder has ready-to-run notebooks `01`→`05`,
one per stage. They add the repo to `sys.path`, install each stage's
`requirements-*.txt`, and call the same CLIs documented in the guides above.

## Legal & ethical

Read the guardrails in [DATASET_GUIDE.md](DATASET_GUIDE.md) and
[VOCALS_GUIDE.md](VOCALS_GUIDE.md), and the model/data note in
[`../LICENSE`](../LICENSE), before training on any audio or voice.
