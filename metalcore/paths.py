"""Kaggle-aware workspace path resolution.

The same code runs on three very different environments:

* the authoring box (no GPU, tiny RAM) -- used only to write/lint code;
* a local machine with data -- used for the Stage-1 smoke test;
* Kaggle kernels -- where all heavy training/inference happens.

:func:`resolve_workspace` normalises those differences into a single
:class:`Workspace` object so callers never hard-code ``/kaggle/working``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


def is_kaggle() -> bool:
    """Return ``True`` when running inside a Kaggle kernel."""
    return Path("/kaggle").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


@dataclass(frozen=True)
class Workspace:
    """Resolved, existing directories for the current environment.

    Attributes:
        root: Base directory for the project on this machine.
        working: Writable directory that persists for the session (checkpoints,
            caches, generated audio). On Kaggle this is ``/kaggle/working``.
        input: Read-only input location. On Kaggle this is ``/kaggle/input``.
        temp: Scratch space that may be cleared between sessions.
        outputs: Convenience sub-directory of ``working`` for artefacts.
    """

    root: Path
    working: Path
    input: Path
    temp: Path
    outputs: Path


def resolve_workspace(project_root: Optional[Union[str, Path]] = None) -> Workspace:
    """Resolve and create the standard workspace directories.

    Args:
        project_root: Overrides the project root when not on Kaggle. Defaults to
            the current working directory.

    Returns:
        A :class:`Workspace` whose ``working``, ``temp`` and ``outputs`` folders
        are guaranteed to exist.
    """
    if is_kaggle():
        working = Path("/kaggle/working")
        input_dir = Path("/kaggle/input")
        temp = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
        root = working
    else:
        root = Path(project_root or Path.cwd()).resolve()
        working = root
        input_dir = root / "data"
        temp = root / ".tmp"

    outputs = working / "outputs"

    for directory in (working, temp, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    return Workspace(
        root=root,
        working=working,
        input=input_dir,
        temp=temp,
        outputs=outputs,
    )
