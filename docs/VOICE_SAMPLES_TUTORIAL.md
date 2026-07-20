# Getting voice samples for Stage 4 — a thorough tutorial

This is the practical, end-to-end guide to building the vocal dataset that
Stage 4 (RVC) trains on. It is written for the metalcore styles you're chasing —
**Currents, Fit For A King, Oceans Ate Alaska, Wind Walkers, Aviana, The Plot In
You** — and it tells you what to collect, how to collect it *legally*, how to
clean it, and how to lay it out so `rvc_training` produces the best possible
blended voice.

> **Read this alongside** [`VOCALS_GUIDE.md`](VOCALS_GUIDE.md) (RVC setup,
> blending, the FX chain) and the legal note below. This tutorial is the "how do
> I get the audio in the first place" companion to that guide.

---

## 0. First, what RVC actually learns (this changes everything)

RVC is a **timbre-conversion** model. During inference it takes a source
performance (your Piper TTS take, or any sung/spoken WAV) and re-voices it in the
*timbre* of your trained model. It learns **how a voice sounds** — its formants,
grit, and character — **not the songs, melodies, or lyrics** in your samples.

Three consequences that shape your whole dataset strategy:

1. **You are collecting a *timbre* reference, not performances.** 15–30 clean
   minutes of a voice teaches RVC that voice. You don't need whole songs; you
   need varied, clean vocal audio.
2. **Style diversity in = flexibility out.** If you want the model to be able to
   deliver both a Currents-style low bellow *and* a Fit For A King soaring clean,
   the training audio must contain both timbres — or you train **separate**
   models per style and pick per section (see §7).
3. **The scream ceiling still applies.** RVC swaps timbre; it does not synthesise
   the chaotic non-harmonic excitation of a real fry/false-cord scream. Training
   on screamed samples makes the *timbre* more aggressive, but the source take
   you feed at inference (Piper TTS → FX chain) sets the ceiling. Manage
   expectations: convincing *harsh/aggressive* vocals, approximate true screams.

---

## 1. Legal & ethical sourcing (read before you download anything)

**The goal of this project is style, not identity.** Training a voice model on a
real, identifiable vocalist — and distributing output that sounds like them —
raises likeness / right-of-publicity issues that vary by country and are being
actively litigated. So:

- ✅ **Best: record your own voice** (or a bandmate's, with permission). Full
  rights, unlimited data, exactly the styles you want. §4 is a complete recording
  guide.
- ✅ **Licensed / royalty-free vocal stems & acapella packs** you have bought or
  that are Creative-Commons / public-domain. Splice, Loopmasters, Cymatics,
  and similar sell vocal packs (including screams/growls) cleared for use.
- ✅ **Stems you legitimately own** — e.g. official multitrack/stem releases,
  remix-contest stems, or your own DI/vocal recordings.
- ⚠️ **Isolating vocals from commercial songs by these artists** to clone their
  identity and release it is the risky path. Keep any such experiment **private /
  research-only**, never present or distribute output as the real artist, and
  prefer the options above. See [`DATASET_GUIDE.md`](DATASET_GUIDE.md).

The rest of this tutorial works identically regardless of source — the isolation,
cleaning, and layout steps are the same. Choose your source responsibly.

---

## 2. The sound you're targeting — style breakdown by influence

Use this to decide **which timbres to capture** (record or source). Each band
mixes techniques; the table is the vocal palette to cover.

| Influence | Signature vocal palette | What to capture for it |
|---|---|---|
| **Currents** (Brian Wille) | Mid–low **bellowing** screams, huge dynamic swings, anguished mid screams, sparse emotive cleans | Sustained mid-range screams with power + a few raw clean/half-sung lines |
| **Fit For A King** (Ryan Kirby) | Powerful **highs and lows**, arena-sized melodic **cleans**, controlled fry | Both a high-scream timbre and strong, pitched clean singing |
| **Oceans Ate Alaska** (Jake Noakes era / current) | Very **high fry screams**, rapid articulate phrasing, technical | Bright, high, cutting fry-scream timbre; crisp consonants |
| **Wind Walkers** | Modern, **energetic mix** of highs + gang-y cleans, bright production | Bright cleans + high screams; upbeat, present tone |
| **Aviana** (Joel Holmqvist) | **Heavy, guttural lows**, downtempo weight, aggressive mids | Low, thick, guttural timbre with chest weight |
| **The Plot In You** (Landon Tewers) | **Emotive cleans**, whispered/soft passages, sudden harsh contrast, R&B-tinged phrasing | Intimate soft cleans + a contrasting harsh timbre |

**Distilled, you want to collect three timbre buckets:**

1. **Low / guttural** (Aviana, Currents lows) — chest-heavy, thick.
2. **High / fry scream** (Oceans Ate Alaska, FFAK highs, Wind Walkers) — bright,
   cutting.
3. **Clean / melodic** (FFAK, The Plot In You, Wind Walkers gang cleans) — pitched,
   emotive, from intimate to belted.

These map directly onto the per-vocalist folders in §6.

---

## 3. How much audio, and what quality?

| Target | Recommendation |
|---|---|
| **Total per voice/timbre** | **10–30 minutes** of *clean, isolated* vocal. 15–20 min is a sweet spot. |
| **Clip length** | Doesn't matter for collection — Stage 4 re-segments into ~4 s clips (`segment_seconds`). Longer raw files are fine. |
| **Sample rate / format** | Record/source at **44.1 kHz+ WAV** if you can. Stage 4 resamples to `prep_sample_rate` (40 kHz). Avoid heavily-compressed low-bitrate MP3. |
| **Dryness** | **As dry as possible.** Reverb, delay, and heavy mix bus FX confuse RVC — it learns the room, not the voice. Isolate/record dry; add FX later in Stage 4. |
| **Consistency** | One timbre per folder. Don't mix a whisper and a full scream in the same "voice" unless you *want* the blend to average them. |
| **Cleanliness** | No music bleed, no other vocalists, no long silences, no clipping. §5 covers QC. |

Quality beats quantity. 12 dry, clean minutes will out-train 40 noisy,
reverb-drenched, music-bleeding minutes every time.

---

## 4. Option A (recommended) — record your own samples

This gives you unlimited, fully-cleared audio in exactly the styles above.

### Gear (works on a budget)
- Any **dynamic mic** (e.g. SM58-style) is forgiving for screams and rejects room
  noise; a large-diaphragm condenser is great for cleans. A USB mic works.
- An audio interface (or USB mic) into any DAW (Reaper, Audacity — both free-ish).

### Session settings
- **44.1 or 48 kHz, 24-bit, mono**, WAV.
- Set gain so the **loudest scream peaks around −6 dBFS** — never clip. Screams
  are much louder than speech; soundcheck at full intensity.
- Record **completely dry** (no reverb/delay/comp on the recording path). Monitor
  with FX if you like, but print dry.
- **Pop filter** + a little distance for screams to protect the capsule and
  reduce plosives.

### What to perform (cover the three buckets)
Record several minutes of each timbre you want in the blend:

- **Lows/gutturals:** sustained low screams, chugged phrasing, held notes. (Aviana,
  Currents lows.)
- **Highs/fry:** bright high screams, fast articulate lines, held highs. (OAA,
  FFAK, Wind Walkers.)
- **Cleans:** intimate soft singing, then belted/arena cleans, a few gang-style
  shout lines. (FFAK, TPIY, Wind Walkers.)

Tips for usable material: vary pitch and intensity, leave a beat of silence
between takes (the cleaner splitter loves gaps), and **hydrate / warm up** — do
screams last and stop if it hurts. Fifteen good minutes is plenty.

### Then skip to §6 (organise) — your recordings are already isolated & dry.

---

## 5. Option B — isolate vocals from mixed tracks you're allowed to use

If your source is a full mix (your own mixed songs, licensed stems that are
premixed, or — with the §1 caveats — reference material), isolate the vocal first.

The repo ships a Demucs wrapper for exactly this:

```bash
python -m rvc_training.cli isolate \
    --input  data/refs \        # folder of full-mix tracks (wav/mp3/flac)
    --output data/vocals_raw    # Demucs writes <model>/<track>/vocals.wav
```

- Uses `htdemucs` (config `demucs_model`) and keeps the **`vocals`** stem.
- Output lands at `data/vocals_raw/htdemucs/<track>/vocals.wav`. You'll reorganise
  those into per-vocalist/timbre folders in §6.
- Isolation is imperfect: expect some instrument bleed and artefacts, especially
  under dense guitars. That's why §5.1 QC matters.

### 5.1 Clean & QC the isolated stems (do not skip)
Open each `vocals.wav` and cut/keep by ear + eye:

- ✂️ **Cut** sections with obvious **music bleed**, cymbal wash, or guitar
  artefacts — RVC will learn them as part of the "voice".
- ✂️ **Cut** long silences, breaths-only regions, and any **second vocalist** you
  don't want in that timbre bucket.
- ✂️ **Cut** clipped/distorted-by-artefact bits (isolation can add crunch that
  isn't the real timbre).
- ✅ **Keep** dry, clearly-voiced passages of the target timbre.
- If the stem is drenched in reverb/delay you can't remove, it's a weak sample —
  prefer drier sources.

Stage 4's `prepare` step (§6) does automatic silence-splitting and drops
near-silent / too-short fragments, but it **cannot** tell a clean voice from a
bleed-y one — that judgement is yours here.

---

## 6. Organise the samples for training

Lay out **one folder per timbre/voice** you want in the blend. Folder names are
free-form; the contents are what matters.

```
data/vocals_raw/
├── low_guttural/     # Aviana / Currents-low timbre  (dry vocal WAVs)
│   ├── take01.wav
│   └── ...
├── high_fry/         # OAA / FFAK-high / Wind Walkers timbre
│   └── ...
└── cleans/           # FFAK / The Plot In You / gang cleans
    └── ...
```

Then run the dataset preparer:

```bash
python -m rvc_training.cli prepare \
    --input  data/vocals_raw \
    --output data/rvc_dataset
```

What `prepare` does (see [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md) for the
knobs): silence-splits each file into ~`segment_seconds` (4 s) clips, drops clips
shorter than `min_segment_seconds` or quieter than `min_rms_db`, and caps each
folder at `max_minutes_per_speaker` to keep a blend balanced.

### Choose your blend mode (`configs/rvc.yaml → blend_mode`)
- **`merged`** (default): pools **all three folders into ONE dataset**
  (`data/rvc_dataset/merged/`) → trains a **single blended voice** that averages
  the timbres. Great for a signature "one vocalist" sound. Use
  `max_minutes_per_speaker` to keep the blend balanced (e.g. don't let 30 min of
  cleans drown 8 min of screams).
- **`per_speaker`**: keeps the folders separate → train a **separate model per
  timbre**, then pick the right one per song section in Stage 5 (screamed verse
  model, clean chorus model). More control, more training runs.

**Recommendation for your influences:** start with **`per_speaker`** so you get a
distinct low, high, and clean model — metalcore switches timbre by section, and
per-section models sound far better than one averaged voice. Move to `merged`
only if you specifically want a single fused character.

---

## 7. Train, then convert

Per the [`VOCALS_GUIDE.md`](VOCALS_GUIDE.md):

```bash
# Train one voice (merged) …
python -m rvc_training.cli train --dataset data/rvc_dataset/merged --name blend

# … or one model per timbre (per_speaker):
python -m rvc_training.cli train --dataset data/rvc_dataset/high_fry --name fry
python -m rvc_training.cli train --dataset data/rvc_dataset/low_guttural --name low
python -m rvc_training.cli train --dataset data/rvc_dataset/cleans --name clean

# Convert lyrics → vocal stem with the matching style + FX preset:
python -m rvc_training.cli vocal \
    --lyrics outputs/lyrics/song.txt \
    --model  .../logs/fry/fry.pth \
    --index  .../logs/fry/added_*.index \
    --output outputs/vocals/chorus_scream.wav
```

Set aggression with `vocal_style` (`clean|harsh|scream`) + `fx_dry_wet` in the
config, or ad-hoc with `rvc_training.cli fx`. For screams, feed RVC the most
expressive TTS take you can and push `vocal_style: scream` — but remember the
ceiling from §0.

---

## 8. Quick checklist

- [ ] Chose a **legal source** (own recordings ▸ licensed stems ▸ … see §1).
- [ ] Captured the **three timbre buckets** you need (low / high-fry / clean, §2).
- [ ] **10–30 clean, dry minutes** per voice, 44.1 kHz WAV, no clipping (§3).
- [ ] If from mixes: **isolated** with `rvc_training.cli isolate` and **QC'd** out
      bleed/silence/second-vocalists (§5).
- [ ] **One folder per timbre**, then `rvc_training.cli prepare` (§6).
- [ ] Picked **`per_speaker`** (per-section control) or **`merged`** (one fused
      voice) (§6).
- [ ] Trained, then converted with the matching `vocal_style` + FX (§7).

If a step errors, see the **Stage 4** section of
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
