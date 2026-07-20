#!/usr/bin/env python3
"""Self-contained DSP smoke test for Stages 4 & 5 (no GPU, no models, no data).

Exercises the parts we own end-to-end on synthetic audio:
  * Stage 4 FX chain (clean/harsh/scream) + low/high shelves
  * Stage 4 vocal dataset segmentation + multi-speaker blend prep
  * Stage 4 RVC command builders (pure functions)
  * Stage 5 section cross-fade stitching, mixing, mastering + WAV export

Requires: numpy, scipy, librosa, soundfile, pyloudnorm, pydub (see
requirements-vocals.txt + requirements-assembly.txt). ffmpeg is optional (MP3).

Usage:  PYTHONPATH=. python scripts/smoke_test_dsp.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("dsp-smoke")


def _tone(sr: int, seconds: float, f0: float = 180.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 8))
    return (0.3 * y).astype(np.float32)


def test_fx() -> None:
    from rvc_training import fx

    sr = 40000
    sig = _tone(sr, 2.0)
    for style in ("clean", "harsh", "scream"):
        out = fx.process(sig, sr, style=style, seed=1)
        assert out.shape == (1, sig.shape[0]), (style, out.shape)
        assert np.all(np.isfinite(out)) and float(np.max(np.abs(out))) <= 0.9701

    def amp(x, f):
        X = np.abs(np.fft.rfft(x))
        fr = np.fft.rfftfreq(len(x), 1 / sr)
        return float(X[np.argmin(np.abs(fr - f))])

    ls = fx.low_shelf(sig, sr, 450.0, 1.5)
    assert amp(ls, 180) / amp(ls, 540) > amp(sig, 180) / amp(sig, 540)
    print("  [ok] FX chain (clean/harsh/scream) + shelves")


def test_vocal_dataset() -> None:
    import soundfile as sf

    from rvc_training.config import RVCConfig
    from rvc_training.dataset import prepare_dataset, segment_audio

    cfg = RVCConfig()
    sr = cfg.prep_sample_rate
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = np.zeros_like(t)
    for start in (0.5, 3.0, 6.0):
        m = (t >= start) & (t < start + 2.0)
        y[m] = 0.3 * np.sin(2 * np.pi * 200 * t[m])
    clips, _ = segment_audio(y.astype(np.float32), sr, cfg)
    assert len(clips) >= 2

    src = Path(tempfile.mkdtemp())
    for spk in ("voc_a", "voc_b"):
        (src / spk).mkdir(parents=True)
        sf.write(src / spk / "take.wav", y.astype(np.float32), sr)
    out = Path(tempfile.mkdtemp())
    stats = prepare_dataset(src, out, cfg, LOG)
    assert stats.speakers == 2 and stats.clips > 0 and (out / "merged").exists()
    names = [p.name for p in (out / "merged").glob("*.wav")]
    assert any(n.startswith("voc_a__") for n in names) and any(n.startswith("voc_b__") for n in names)
    print(f"  [ok] vocal dataset: {stats.clips} clips blended from 2 speakers")


def test_rvc_commands() -> None:
    from rvc_training import rvc
    from rvc_training.config import RVCConfig

    cfg = RVCConfig()
    assert "preprocess.py" in rvc.preprocess_cmd(cfg, Path("/d"), "e")[1]
    tc = rvc.train_cmd(cfg, "e")
    assert "40k" in tc and str(cfg.epochs) in tc
    ic = rvc.infer_cmd(cfg, "m.pth", "i.index", "in.wav", "out.wav")
    assert "--index_rate" in ic and str(cfg.index_rate) in ic
    print("  [ok] RVC command builders")


def test_assembly() -> None:
    from inference import mix as mm
    from inference.config import AssemblyConfig
    from inference.master import export, loudness_normalize
    from inference.sections import equal_power_crossfade, stitch_sections

    cfg = AssemblyConfig()
    sr = cfg.work_sample_rate

    xf = equal_power_crossfade(np.ones(1000, np.float32), np.ones(1000, np.float32) * 0.5, 200)
    assert xf.shape[0] == 1800 and np.all(np.isfinite(xf))
    full = stitch_sections([np.random.randn(sr * 3).astype(np.float32) for _ in range(3)], sr, 1.0)
    assert full.shape[0] == sr * 9 - 2 * sr

    inst = _tone(sr, 5.0, 100.0)
    voc = _tone(sr, 4.0, 300.0)  # shorter -> padded
    mixed = mm.mix(inst, voc, cfg, LOG)
    assert mixed.shape == (2, inst.shape[0]) and float(np.max(np.abs(mixed))) <= 1.0001

    ln = loudness_normalize(mixed, sr, -9.0)
    assert ln.shape == mixed.shape and float(np.max(np.abs(ln))) <= 0.9901

    out_base = Path(tempfile.mkdtemp()) / "track"
    written = export(mixed, sr, out_base, cfg, LOG)
    import soundfile as sf

    wav = out_base.with_suffix(".wav")
    assert wav in written and sf.info(str(wav)).samplerate == cfg.export_sample_rate
    print(f"  [ok] assembly: stitch/mix/master -> {len(written)} file(s) @ {cfg.export_sample_rate} Hz")


def main() -> int:
    print("Stage 4/5 DSP smoke test:")
    test_fx()
    test_vocal_dataset()
    test_rvc_commands()
    test_assembly()
    print("DSP SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
