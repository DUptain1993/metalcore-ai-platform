# Dataset guide — collecting & organising data (legally)

You do not have a dataset yet — that's fine. This guide explains what to gather,
how to lay it out, and the legal/ethical guardrails to keep in mind.

## Legal & ethical guardrails (read first)

- **Goal is inspiration, not cloning.** These tools learn *style* (structure,
  breakdowns, textures, phrasing) — do not attempt to reproduce specific
  copyrighted songs.
- **Use material you have the right to use.** Prefer music you own, tracks
  released under permissive licenses (e.g. Creative Commons), royalty-free
  libraries, or your own recordings/stems.
- **Voice likeness (Stage 4).** Training a voice model on a real, identifiable
  vocalist raises likeness/publicity concerns in many jurisdictions. Keep such
  models for personal/research use and **do not present or distribute output as
  the real artists**.
- Training/fine-tuning on copyrighted material sits in a legally unsettled area
  that varies by country. When in doubt, use licensed or original material.

## Stage 1 — instrumental audio

**What to collect:** instrumental metalcore (or stems) in WAV/MP3/FLAC. Aim for
consistent production style. Even 1–3 hours lets LoRA learn a recognisable feel;
5–20 hours is better.

**Folder layout** (any nesting works — the tools search recursively):

```
data/raw/
├── artistA/
│   ├── song1.wav
│   └── song2.flac
├── artistB/
│   └── track.mp3
└── my_riffs/
    └── idea01.wav
```

**On Kaggle:** upload `data/raw/` as a **Kaggle Dataset** (it mounts read-only
under `/kaggle/input/<name>`), then point Stage 1 at it and write the processed
dataset to `/kaggle/working`.

**Tips**
- More *variety of sections* (intros, breakdowns, choruses, ambient parts) →
  richer generation. Whole songs are fine; the tools chunk them.
- The captioner auto-tags tempo/key and prepends your `style_tags` from
  `configs/dataset.yaml`. Edit those tags to steer the sound.

## Stage 3 — lyrics

**What to collect:** a folder of `.txt` files, **one song per file**, of lyrics
you have the right to use (your own writing is ideal). Themes are inferred
automatically, or you can tag them explicitly.

**File format** (front-matter is optional):

```
#theme: addiction, hope

[Verse]
...lines...

[Chorus]
...lines...
```

- `#theme:` — comma-separated themes (must be from the list in
  `configs/lyrics_lora.yaml::themes`); omit to auto-infer from keywords.
- `#title:` / `#artist:` / `#album:` lines are ignored if present.
- Section labels like `[Verse]` / `[Chorus]` are kept as-is and help the model
  learn structure.

**Folder layout:**

```
data/lyrics/
├── song_about_recovery.txt
├── betrayal_track.txt
└── ...
```

## Stage 4 — vocals (next build pass)

Vocal dataset preparation (isolation, trimming, validation) is documented in
`rvc_training/README.md` and will be implemented in the next pass. For blending
multiple vocalists, keep each source's clean vocal samples in its own subfolder.
