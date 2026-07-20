# Assembly guide (Stage 5) — building a full song

Stage 5 turns section-length MusicGen clips + a vocal stem into a mastered track.

## The arrangement

Because MusicGen is only coherent for ~30 s, a full song is built from
**per-section** clips that are equal-power cross-faded together. Define the
arrangement in `configs/assembly.yaml`:

```yaml
sections:
  - name: intro
    prompt: "atmospheric metalcore intro, ambient clean guitar, building tension"
    seconds: 12.0
  - name: breakdown
    prompt: "heavy metalcore breakdown, downtuned chugging guitar, double bass"
    seconds: 14.0
  # ...
crossfade_seconds: 1.0
```

Each section's `prompt` is sent to the trained MusicGen LoRA (Stage 2); `seconds`
sets its length. Reorder/duplicate sections freely (e.g. verse → chorus → verse).

## Signal flow

```
generate each section  ->  cross-fade stitch  ->  full instrumental (32 kHz mono)
vocal stem (Stage 4)   ->  fit to length + gain
mix                    ->  stereo widen instrumental + centre vocal + soft bus limit
master                 ->  upsample to 44.1 kHz + loudness normalise + WAV/MP3
```

## Commands

```bash
pip install -r requirements-assembly.txt     # needs ffmpeg for MP3

# Full song (generate instrumental + mix vocal + master):
python -m inference.cli song \
    --music-config configs/music_lora.yaml \
    --adapter outputs/music/checkpoints/step_002000/adapter \
    --vocal   outputs/vocals/song_vocal.wav \
    --output  outputs/songs/track01
```

Variations:

- **Instrumental only:** omit `--vocal`.
- **Bring your own instrumental:** pass `--instrumental full_inst.wav` (skips
  generation; no `--adapter` needed).
- **Just stitch clips:** `inference.cli stitch --sections a.wav b.wav --output inst.wav`.
- **Just master a mix:** `inference.cli master --input mix.wav --output track01`.

## Mixing & mastering knobs

| Key | Effect |
|---|---|
| `instrumental_gain_db` / `vocal_gain_db` | per-stem level trim |
| `stereo_width` (0–1) | Haas-style widening of the instrumental (0 = mono) |
| `target_lufs` | master loudness (default −9 LUFS, loud metalcore master) |
| `export_mp3` / `mp3_bitrate` | MP3 export toggle + bitrate |
| `export_sample_rate` | final WAV/MP3 rate (default 44.1 kHz) |

Peaks are limited to −0.1 dBFS after loudness normalisation to avoid clipping.

## Tips

- Generate a few takes per section and pick the best before stitching (run Stage 2
  `generate` with `--num`), then use `inference.cli stitch` on your favourites.
- Keep `crossfade_seconds` around 0.5–1.5 s; longer smears transients, shorter can
  click on hard downbeats.
- If MP3 export logs an ffmpeg error, install ffmpeg (`apt-get install ffmpeg`);
  the WAV is still written regardless.
