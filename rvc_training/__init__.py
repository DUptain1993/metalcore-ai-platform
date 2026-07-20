"""Stage 4 - vocal generation: dataset prep, RVC integration, and FX.

Self-contained parts (dataset prep, the scream/harsh/clean FX chain) import only
numpy/scipy/librosa. External integrations (Demucs, RVC-Project, Piper TTS) are
invoked via subprocess wrappers and import their heavy deps lazily.
"""

from __future__ import annotations

from rvc_training.config import RVCConfig

__all__ = ["RVCConfig"]
