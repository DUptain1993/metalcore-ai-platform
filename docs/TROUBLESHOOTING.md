# Troubleshooting guide

## Installation

**`bitsandbytes` import error / "CUDA Setup failed".**
bitsandbytes needs a CUDA GPU. Enable the GPU accelerator in the Kaggle notebook
(Settings → Accelerator → GPU T4 x2). On CPU-only machines, Stage 3 cannot run.

**Reinstalling `torch` breaks CUDA on Kaggle.**
Do **not** `pip install torch` on Kaggle — use the preinstalled build. The
`requirements-*.txt` files deliberately omit torch. If you already clobbered it,
factory-reset the notebook (or "Restart & Clear Cell Outputs" + re-run without
installing torch).

**`librosa`/`soundfile` cannot read MP3/M4A.**
Install ffmpeg. On Kaggle it is preinstalled; locally run
`sudo apt-get install -y ffmpeg`.

**`numpy` ABI / version conflicts.** These pins expect `numpy<2.0`. If a
different library pulls in numpy 2.x, reinstall with `pip install "numpy<2.0"`
and restart the kernel.

## Stage 1 — dataset

**"No audio files matched extensions".** Check `--input` points at the folder
containing your audio (searched recursively) and that extensions are listed in
`configs/dataset.yaml::audio_exts`.

**Everything gets rejected as "silent" or "too_short".** Lower
`silence_rms_db` (e.g. -55) or `min_duration` in the config. Very quiet ambient
intros can trip the silence check.

**Validation split is empty.** You need **≥ 2 source tracks** for a val split
(the splitter groups by track to prevent leakage). Add more tracks.

## Stage 2 — MusicGen

**CUDA out of memory.** In order:
1. Confirm `gradient_checkpointing: true` and `mixed_precision: fp16`.
2. Lower `train_seconds` (15 → 10).
3. Keep `batch_size: 1`; raise `grad_accum_steps` for effective batch size.
4. Switch `model_id` to `facebook/musicgen-small`.

**`outputs.loss` is `None` / loss not computed.** The trainer relies on
`transformers==4.44.2` computing the MusicGen training loss from `labels`. If you
changed the transformers version and hit this, reinstall the pinned version
(`pip install "transformers==4.44.2"`).

**Loss does not decrease in the overfit test.** Sanity-check by pointing training
at a tiny dataset (2–3 tracks) with a high `max_steps`; loss should fall quickly.
If not, verify captions are non-empty in `metadata.jsonl` and that the code cache
rebuilt (delete `outputs/music/cache` to force a rebuild).

**Generated audio is noise.** LoRA needs enough steps/data; also try raising the
generation `--guidance-scale` (3–5) and confirm you passed the correct
`--adapter` directory (the one containing `adapter_config.json`).

**Resume starts from step 0.** Ensure `--resume` is passed and that
`outputs/music/checkpoints/latest.txt` exists and points to a real `step_*` dir.

## Stage 3 — Lyrics

**`eval_strategy` unknown argument.** You're on an older transformers; either
upgrade to the pinned `4.44.2` or change `eval_strategy` →
`evaluation_strategy` in `lyric_training/train.py`.

**bf16 error on T4.** T4 has no bf16. The trainer auto-detects this and falls
back to fp16; if you forced bf16 elsewhere, set
`bnb_4bit_compute_dtype: float16` in the config.

**Chat template error / model has no chat template.** Use an *instruct* model
(the defaults do). Base (non-instruct) models lack `apply_chat_template`.

**Generation ignores the section structure.** With little data the style
transfers but structure adherence varies; raise data quantity, lower
`temperature` (0.7–0.85), and keep the explicit section list in the prompt.

## Stage 4 — Vocals

**`setup` clone/download fails.** Needs internet (enable it in Kaggle notebook
settings) plus `git` and `wget`. Re-running `setup` is safe — it skips files that
already exist.

**RVC training script errors / wrong arguments.** The wrappers target
RVC-Project's canonical `infer/modules/train/*` interface. If you cloned a fork
with a different CLI, edit the `*_cmd` builders in `rvc_training/rvc.py` (they are
small, pure functions) or point `rvc_repo`/`rvc_python` at the right paths in
`configs/rvc.yaml`. Also run `pip install -r <rvc_repo>/requirements.txt` once.

**`piper: command not found`.** `pip install piper-tts` (in
`requirements-vocals.txt`) and confirm a voice `.onnx` exists at `piper_voice`.

**Demucs `vocals.wav` not found.** Demucs writes
`<out>/<model>/<track>/vocals.wav`; confirm the model name matches
`demucs_model` and that the input decoded (install ffmpeg for MP3/M4A).

**Screams sound weak / robotic.** Expected ceiling of TTS→RVC (see
`docs/VOCALS_GUIDE.md`). Push `vocal_style: scream`, raise `fx_dry_wet`, and feed
RVC a more expressive TTS take. It approximates, it does not synthesise true fry.

**FX output is silent/clipped.** The FX peak-normalises to −0.3 dBFS; if the input
was silent you get silence. Lower `--dry-wet` to retain more of the source.

## Stage 5 — Assembly

**MP3 not written / ffmpeg error.** Install ffmpeg (`apt-get install -y ffmpeg`;
preinstalled on Kaggle). The WAV is always written even if MP3 export fails.

**Clicks between sections.** Increase `crossfade_seconds` (0.5–1.5 s). Very short
fades can click on hard downbeats; very long fades smear transients.

**Master too quiet/loud or clipping.** Adjust `target_lufs` (default −9). Peaks
are limited to −0.1 dBFS, so clipping shouldn't occur; if a mix sounds squashed,
raise `target_lufs` toward −12.

**Vocal and instrumental lengths mismatch.** The mixer fits the vocal to the
instrumental (pad/trim). For tight alignment, generate the instrumental length to
match your vocal, or edit the vocal stem before mixing.

## Kaggle sessions

**Session timed out mid-training.** Expected. Re-run the training cell with
`--resume`; it restores adapter + optimizer + scheduler + step from the latest
checkpoint under `/kaggle/working`.

**Disk full (~20 GB).** Lower `keep_last_checkpoints`, delete stale caches under
`outputs/*/cache`, and avoid writing raw datasets to `/kaggle/working` (keep raw
audio in a read-only Kaggle Dataset under `/kaggle/input`).
