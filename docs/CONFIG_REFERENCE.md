# Configuration reference

Every tunable key across the five stage configs, with its type, default, and
effect. Each `configs/*.yaml` maps 1:1 to a dataclass in the matching
`*/config.py` (loaded via `metalcore.config.load_config`). Keys not listed in a
dataclass are ignored, so unknown keys fail silently — check spelling here.

CLI flags always override config values for the run.

---

## `configs/dataset.yaml` — Stage 1 (`DatasetConfig`)

| Key | Type | Default | Effect |
|---|---|---|---|
| `sample_rate` | int | `32000` | Output chunk sample rate. **Keep at 32000** to match MusicGen's EnCodec front-end. |
| `chunk_seconds` | float | `15.0` | Length of each training chunk. Longer = more context but more VRAM in Stage 2. |
| `chunk_overlap` | float | `0.0` | Overlap between consecutive chunks (seconds). Raise for more augmentation-like coverage. |
| `target_lufs` | float | `-14.0` | Loudness-normalisation target (ITU-R BS.1770). Streaming-style reference. |
| `min_duration` | float | `2.0` | Files shorter than this (s) are rejected in validation. |
| `silence_rms_db` | float | `-50.0` | Tracks with overall RMS quieter than this (dBFS) are rejected as silent. Lower (e.g. −55) to keep quiet ambient intros. |
| `keep_tail_ratio` | float | `0.5` | Keep a trailing partial chunk only if ≥ this fraction of a full chunk; kept tails are zero-padded. |
| `val_ratio` | float | `0.1` | Fraction of source **tracks** held out for validation (grouped, no leakage). Needs ≥2 tracks. |
| `seed` | int | `42` | RNG seed for the split. |
| `style_tags` | list[str] | *(4 metalcore tags)* | Prepended to every auto-caption. **Edit these to steer the fine-tuned sound.** |
| `audio_exts` | list[str] | `.wav .mp3 .flac .ogg .m4a` | Accepted input extensions (recursive search). |

---

## `configs/music_lora.yaml` — Stage 2 (`MusicLoRAConfig`)

| Key | Type | Default | Effect |
|---|---|---|---|
| `model_id` | str | `facebook/musicgen-medium` | Base model. `-small` (~4–6 GB) / `-medium` (~10–14 GB on T4). `-large` won't fit a single T4. |
| `lora_r` | int | `16` | LoRA rank. Higher = more capacity + VRAM. |
| `lora_alpha` | int | `32` | LoRA scaling (commonly 2×`r`). |
| `lora_dropout` | float | `0.05` | Dropout on LoRA layers. |
| `target_modules` | list[str] | `q/k/v/out_proj` | Decoder attention projections to adapt. |
| `learning_rate` | float | `2e-4` | AdamW LR. |
| `weight_decay` | float | `0.0` | AdamW weight decay. |
| `warmup_steps` | int | `50` | LR warmup steps. |
| `max_steps` | int | `2000` | Total optimiser steps (× `grad_accum_steps` forward passes). |
| `batch_size` | int | `1` | Per-step batch. Keep at 1 on T4; scale with grad accumulation. |
| `grad_accum_steps` | int | `8` | Gradient accumulation → effective batch size. |
| `max_grad_norm` | float | `1.0` | Gradient clipping. |
| `guidance_dropout` | float | `0.1` | Probability of dropping the text condition (classifier-free guidance training). |
| `mixed_precision` | str | `fp16` | `fp16` \| `bf16` \| `no`. Use `fp16` on T4 (no bf16). |
| `gradient_checkpointing` | bool | `true` | Trade compute for VRAM. **Keep on** to fit medium. |
| `train_seconds` | float | `15.0` | Chunk length used for training (≤ dataset `chunk_seconds`). Lower first on OOM. |
| `save_every` | int | `200` | Checkpoint cadence (steps). |
| `val_every` | int | `500` | Validation-generation cadence (steps). |
| `keep_last_checkpoints` | int | `2` | Rolling checkpoint retention (disk guard). |
| `val_prompt` | str | *(breakdown prompt)* | Prompt used for periodic validation audio. |
| `val_seconds` | float | `8.0` | Length of validation clips. |
| `seed` | int | `42` | RNG seed. |

---

## `configs/lyrics_lora.yaml` — Stage 3 (`LyricsLoRAConfig`)

| Key | Type | Default | Effect |
|---|---|---|---|
| `model_id` | str | `Qwen/Qwen2.5-1.5B-Instruct` | Base **instruct** model (needs a chat template). Swap to `Llama-3.2-3B-Instruct` for higher quality. |
| `load_in_4bit` | bool | `true` | Enable bitsandbytes 4-bit (QLoRA). Requires a CUDA GPU. |
| `bnb_4bit_quant_type` | str | `nf4` | 4-bit quant type (`nf4` recommended). |
| `bnb_4bit_use_double_quant` | bool | `true` | Nested quantisation (saves a little more VRAM). |
| `bnb_4bit_compute_dtype` | str | `bfloat16` | Compute dtype; **auto-falls back to fp16 on T4**. |
| `lora_r` / `lora_alpha` / `lora_dropout` | int/int/float | `16` / `32` / `0.05` | LoRA hyper-parameters. |
| `target_modules` | list[str] | all attn + MLP proj | Modules to adapt. |
| `learning_rate` | float | `2e-4` | AdamW LR. |
| `weight_decay` | float | `0.0` | Weight decay. |
| `warmup_ratio` | float | `0.03` | Warmup as a fraction of total steps. |
| `num_train_epochs` | int | `3` | Epochs over the lyric corpus. |
| `batch_size` | int | `2` | Per-device batch. |
| `grad_accum_steps` | int | `4` | Gradient accumulation. |
| `max_grad_norm` | float | `0.3` | Gradient clipping. |
| `max_seq_length` | int | `1024` | Max tokens per example. Lower first on OOM. |
| `save_steps` | int | `100` | Checkpoint cadence. |
| `logging_steps` | int | `10` | Log cadence. |
| `keep_last_checkpoints` | int | `2` | Checkpoint retention. |
| `val_ratio` | float | `0.1` | Fraction of lyric files held out. |
| `sections` | list[str] | `Verse … Final Chorus` | Song structure enforced at generation time. |
| `themes` | list[str] | *(8 themes)* | Allowed themes for conditioning + keyword inference. |
| `seed` | int | `42` | RNG seed. |

---

## `configs/rvc.yaml` — Stage 4 (`RVCConfig`)

| Key | Type | Default | Effect |
|---|---|---|---|
| `demucs_model` | str | `htdemucs` | Demucs model for vocal isolation; keeps the `vocals` stem. |
| `prep_sample_rate` | int | `40000` | RVC v2 training rate (`40000` or `48000`). |
| `segment_seconds` | float | `4.0` | Length of prepared training clips. |
| `min_segment_seconds` | float | `1.2` | Drop clips shorter than this. |
| `silence_top_db` | float | `30.0` | librosa split threshold (dB below peak). |
| `min_rms_db` | float | `-45.0` | Drop near-silent segments below this RMS. |
| `max_minutes_per_speaker` | float | `15.0` | Cap per vocalist to balance a blend. |
| `blend_mode` | str | `merged` | `merged` = one blended voice from all vocalists; `per_speaker` = one model each. |
| `rvc_repo` | str | `/kaggle/working/…WebUI` | Path to the RVC-Project clone. |
| `rvc_python` | str | `python` | Python used to invoke RVC scripts. |
| `f0_method` | str | `rmvpe` | Pitch extraction: `rmvpe` (best) \| `crepe` \| `harvest` \| `pm`. |
| `epochs` | int | `200` | RVC training epochs. |
| `batch_size` | int | `8` | RVC training batch. |
| `save_every_epoch` | int | `50` | RVC checkpoint cadence. |
| `cache_in_gpu` | bool | `false` | Cache dataset in VRAM (only if you have headroom). |
| `transpose` | int | `0` | Semitone shift of the TTS input before conversion. |
| `index_rate` | float | `0.75` | Timbre-vs-articulation blend from the feature index (0–1). |
| `protect` | float | `0.33` | Protect voiceless consonants (0–0.5). |
| `piper_voice` | str | `…/en_US-ljspeech-high.onnx` | Piper voice model path. |
| `tts_sample_rate` | int | `22050` | Piper output rate. |
| `vocal_style` | str | `harsh` | FX preset after RVC: `clean` \| `harsh` \| `scream`. |
| `fx_dry_wet` | float | `1.0` | FX mix (0 = dry, 1 = fully processed). |

---

## `configs/assembly.yaml` — Stage 5 (`AssemblyConfig`)

| Key | Type | Default | Effect |
|---|---|---|---|
| `work_sample_rate` | int | `32000` | Internal assembly rate (matches MusicGen output). |
| `export_sample_rate` | int | `44100` | Final WAV/MP3 rate. |
| `sections` | list[obj] | *(7 sections)* | Arrangement; each has `name`, `prompt`, `seconds`. Reorder/duplicate freely. |
| `crossfade_seconds` | float | `1.0` | Equal-power cross-fade between sections. 0.5–1.5 is the sweet spot. |
| `instrumental_gain_db` | float | `0.0` | Instrumental level trim. |
| `vocal_gain_db` | float | `-1.0` | Vocal level trim. |
| `stereo_width` | float | `0.2` | Haas-style widening (0 = mono, ~0.6 = wide). |
| `target_lufs` | float | `-9.0` | Master loudness (loud metalcore master). Raise toward −12 if squashed. |
| `export_mp3` | bool | `true` | Also export MP3 (needs ffmpeg; WAV always written). |
| `mp3_bitrate` | str | `320k` | MP3 bitrate. |
| `guidance_scale` | float | `3.0` | CFG scale for section generation. |
| `seed` | int | `42` | RNG seed for section generation. |

---

### Tuning cheat-sheet

- **OOM in Stage 2** → lower `train_seconds`, confirm `gradient_checkpointing: true`, keep `batch_size: 1`, then `model_id: facebook/musicgen-small`.
- **OOM in Stage 3** → lower `max_seq_length`, then `batch_size`.
- **Generated audio is noise** → more steps/data; raise generation `--guidance-scale` to 3–5.
- **Master too quiet/squashed** → adjust `target_lufs` (−9 loud … −14 dynamic).
- **Clicks between sections** → `crossfade_seconds` 0.5–1.5.

See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for the full symptom → fix list.
