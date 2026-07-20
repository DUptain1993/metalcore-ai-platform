"""Configuration dataclass for Stage 4 vocal generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RVCConfig:
    """Knobs for vocal isolation, dataset prep, RVC training/inference and FX.

    See ``configs/rvc.yaml`` for documented defaults.
    """

    # Isolation
    demucs_model: str = "htdemucs"

    # Dataset preparation
    prep_sample_rate: int = 40000
    segment_seconds: float = 4.0
    min_segment_seconds: float = 1.2
    silence_top_db: float = 30.0
    min_rms_db: float = -45.0
    max_minutes_per_speaker: float = 15.0
    blend_mode: str = "merged"  # merged | per_speaker

    # RVC integration
    rvc_repo: str = "/kaggle/working/Retrieval-based-Voice-Conversion-WebUI"
    rvc_python: str = "python"
    f0_method: str = "rmvpe"
    epochs: int = 200
    batch_size: int = 8
    save_every_epoch: int = 50
    cache_in_gpu: bool = False

    # Inference / conversion
    transpose: int = 0
    index_rate: float = 0.75
    protect: float = 0.33

    # TTS front-end
    piper_voice: str = "/kaggle/working/piper_voices/en_US-ljspeech-high.onnx"
    tts_sample_rate: int = 22050

    # Vocal FX
    vocal_style: str = "harsh"  # clean | harsh | scream
    fx_dry_wet: float = 1.0

    def __post_init__(self) -> None:
        if self.blend_mode not in {"merged", "per_speaker"}:
            raise ValueError("blend_mode must be 'merged' or 'per_speaker'.")
        if self.vocal_style not in {"clean", "harsh", "scream"}:
            raise ValueError("vocal_style must be one of: clean, harsh, scream.")
        if not 0.0 <= self.fx_dry_wet <= 1.0:
            raise ValueError("fx_dry_wet must be in [0, 1].")
        if self.f0_method not in {"rmvpe", "crepe", "harvest", "pm"}:
            raise ValueError("f0_method must be one of: rmvpe, crepe, harvest, pm.")
