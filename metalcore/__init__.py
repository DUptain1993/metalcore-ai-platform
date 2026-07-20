"""Shared utilities for the Metalcore AI generation platform.

This package holds cross-stage infrastructure that every pipeline reuses:

* :mod:`metalcore.logging_utils` -- consistent stdout + rotating-file logging.
* :mod:`metalcore.config`        -- YAML -> typed dataclass configuration loading.
* :mod:`metalcore.paths`         -- Kaggle-aware workspace path resolution.
* :mod:`metalcore.audio_io`      -- robust audio load / convert / save helpers.

The package is intentionally dependency-light so it can be imported on machines
without a GPU (e.g. the authoring box) as well as inside Kaggle notebooks.
"""

from __future__ import annotations

__version__ = "0.1.0"

from metalcore.config import load_config, load_yaml
from metalcore.logging_utils import get_logger
from metalcore.paths import Workspace, resolve_workspace, is_kaggle

__all__ = [
    "__version__",
    "load_config",
    "load_yaml",
    "get_logger",
    "Workspace",
    "resolve_workspace",
    "is_kaggle",
]
