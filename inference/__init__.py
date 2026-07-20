"""Stage 5 - song assembly & rendering.

Stitches MusicGen instrumental sections into a full arrangement, mixes in the
vocal stem, masters to a target loudness, and exports WAV/MP3.

The stitching/mixing/mastering DSP is self-contained (numpy/scipy/pydub) and
testable without a GPU. Instrumental *generation* reuses Stage 2
(:mod:`music_training.generate`) and is imported lazily.
"""

from __future__ import annotations

from inference.config import AssemblyConfig

__all__ = ["AssemblyConfig"]
