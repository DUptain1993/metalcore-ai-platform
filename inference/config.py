"""Configuration dataclass for Stage 5 song assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssemblyConfig:
    """Knobs for assembling and mastering a full song.

    See ``configs/assembly.yaml`` for documented defaults.
    """

    work_sample_rate: int = 32000
    export_sample_rate: int = 44100

    # Each section: {name: str, prompt: str, seconds: float}
    sections: List[Dict] = field(
        default_factory=lambda: [
            {"name": "intro", "prompt": "atmospheric metalcore intro", "seconds": 12.0},
            {"name": "verse", "prompt": "metalcore verse", "seconds": 16.0},
            {"name": "chorus", "prompt": "melodic metalcore chorus", "seconds": 16.0},
            {"name": "breakdown", "prompt": "heavy metalcore breakdown", "seconds": 14.0},
            {"name": "final_chorus", "prompt": "climactic metalcore final chorus", "seconds": 18.0},
            {"name": "outro", "prompt": "metalcore outro", "seconds": 10.0},
        ]
    )

    crossfade_seconds: float = 1.0

    instrumental_gain_db: float = 0.0
    vocal_gain_db: float = -1.0
    stereo_width: float = 0.2

    target_lufs: float = -9.0
    export_mp3: bool = True
    mp3_bitrate: str = "320k"

    guidance_scale: float = 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("At least one section is required.")
        for s in self.sections:
            if not {"name", "prompt", "seconds"} <= set(s):
                raise ValueError("Each section needs 'name', 'prompt' and 'seconds'.")
        if not 0.0 <= self.stereo_width <= 1.0:
            raise ValueError("stereo_width must be in [0, 1].")
        if self.crossfade_seconds < 0:
            raise ValueError("crossfade_seconds must be >= 0.")
