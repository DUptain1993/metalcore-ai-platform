# Hardware requirements, VRAM estimates & training durations

All figures target Kaggle's **free** accelerators. Numbers are indicative and
scale with dataset size, chunk length and sequence length.

## Kaggle environment (free tier)

| Resource | T4 x2 | P100 | Notes |
|---|---|---|---|
| GPU VRAM | 16 GB (x2, but use 1) | 16 GB | This project uses a **single** GPU. |
| System RAM | ~29 GB | ~16 GB | Stage 1 is RAM-light; caching helps. |
| Writable disk (`/kaggle/working`) | ~20 GB | ~20 GB | Keep caches + checkpoints under this. |
| `/kaggle/temp` + input | varies | varies | Datasets are read-only under `/kaggle/input`. |
| Session limit | 9–12 h | 9–12 h | Checkpoint often; resume across sessions. |
| Weekly GPU quota | ~30 h | ~30 h | Budget training runs accordingly. |

> Prefer **T4** for these workloads: it supports the bitsandbytes 4-bit kernels
> used by Stage 3 and has enough VRAM for MusicGen-medium + LoRA. The P100 works
> for Stage 2 but lacks efficient int8/4-bit support for Stage 3.

## Stage 1 — dataset processing (CPU)

- **GPU:** none. **RAM:** < 2 GB (one track loaded at a time).
- **Disk:** output is ~32 kHz mono WAV chunks. Rough estimate: **~1.9 MB per
  15 s chunk**; 10 hours of source ≈ 2400 chunks ≈ **~4.5 GB**.
- **Duration:** roughly real-time-ish per track for load+normalise+chunk;
  captioning adds tempo/key extraction (~0.5–2 s per chunk).

## Stage 2 — MusicGen LoRA (GPU)

| Model | Params | VRAM (fp16 + LoRA + grad-ckpt) | Fits T4? |
|---|---|---|---|
| `musicgen-small` | 300 M | ~5–7 GB | ✅ comfortably |
| `musicgen-medium` | 1.5 B | ~10–14 GB | ✅ (batch 1, 15 s chunks) |
| `musicgen-large` | 3.3 B | > 16 GB | ❌ not on a single T4 |

- **Settings that keep medium within 16 GB:** `batch_size: 1`,
  `gradient_checkpointing: true`, `mixed_precision: fp16`, `train_seconds: 15`,
  and gradient accumulation for an effective larger batch.
- **One-time cost:** EnCodec code caching (~fast; a few ms per clip on GPU).
- **Duration:** ~**1.5–3.0 it/s** for medium at batch 1 on a T4. 2000 optimiser
  steps × `grad_accum_steps: 8` ≈ 16k forward/backward passes ≈ **~2–4 h**.

## Stage 3 — Lyrics QLoRA (GPU)

| Model | Params | VRAM (4-bit + LoRA) | Fits T4? |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct` | 1.5 B | ~5–7 GB | ✅ easily |
| `Llama-3.2-3B-Instruct` | 3 B | ~7–9 GB | ✅ |
| 7–8 B instruct | 7–8 B | ~11–14 GB | ⚠️ tight; reduce `max_seq_length` |

- **Duration:** small datasets (tens–hundreds of songs) fine-tune in
  **~15–45 min** for 3 epochs at `max_seq_length: 1024`.

## Stage 4 — Vocals (GPU)

| Component | GPU VRAM | Disk | Notes |
|---|---|---|---|
| Demucs isolation | ~3–5 GB | model ~ 300 MB | segment long tracks; RAM-friendly |
| RVC pretrained weights | — | ~ 0.9 GB total | hubert ~180 MB, rmvpe ~180 MB, pretrained_v2 D/G ~540 MB |
| RVC training | ~4–8 GB | dataset + logs | batch 8, 40 kHz, ~200 epochs |
| RVC inference | ~2–4 GB | — | per-utterance conversion |
| Piper TTS | CPU/low | voice ~ 60 MB | fast, spoken-cadence |
| FX chain | CPU | — | numpy/scipy, no model |

- **Duration:** RVC training on ~10–20 min of vocals ≈ **1–3 h** for 200 epochs on
  a T4; Demucs isolation is a few× real-time per track.
- **Disk watch-out:** Demucs + RVC weights + datasets are the biggest consumers of
  the ~20 GB `/kaggle/working`. Keep raw refs in a read-only Kaggle Dataset and
  cap `max_minutes_per_speaker`.

## Stage 5 — Assembly (CPU)

- **GPU:** none for stitch/mix/master (only instrumental *generation* reuses Stage
  2's GPU cost). **RAM:** low. **Disk:** a few hundred MB per song.
- **Duration:** stitch/mix/master a full song in **seconds**; generating the
  instrumental sections dominates (Stage 2 generation cost × number of sections).

## Rules of thumb

- Keep **checkpoints + caches** together under `/kaggle/working` and set
  `keep_last_checkpoints` to 2 to avoid filling the ~20 GB disk.
- If you hit OOM, in order: lower `train_seconds`/`max_seq_length`, ensure
  gradient checkpointing is on, drop `batch_size` to 1, then switch to a smaller
  base model. See `docs/TROUBLESHOOTING.md`.
