"""Stage 1 - dataset processing tools.

Pipeline: validate -> preprocess (normalise + chunk) -> caption -> metadata ->
split. Run via ``python -m dataset_tools.cli`` (see :mod:`dataset_tools.cli`).
"""

from __future__ import annotations

from dataset_tools.config import DatasetConfig

__all__ = ["DatasetConfig"]
