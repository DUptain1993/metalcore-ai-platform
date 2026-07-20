"""Consistent logging setup used across every stage of the pipeline."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

_CONFIGURED: set[str] = set()

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger.

    The logger writes to stdout and, optionally, to a rotating log file. Repeated
    calls with the same ``name`` reuse the existing handlers so importing modules
    never attach duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` or a stage identifier.
        log_file: Optional path to a log file. Parent directories are created.
        level: Logging level for the logger and its handlers.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        # Still allow attaching a file handler later if one is requested and the
        # logger previously had none pointing at that file.
        if log_file is not None and not _has_file_handler(logger, log_file):
            logger.addHandler(_build_file_handler(log_file, level))
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        logger.addHandler(_build_file_handler(log_file, level))

    # Prevent double logging through the root logger.
    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


def _build_file_handler(log_file: Union[str, Path], level: int) -> RotatingFileHandler:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def _has_file_handler(logger: logging.Logger, log_file: Union[str, Path]) -> bool:
    target = str(Path(log_file).resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            if str(Path(handler.baseFilename).resolve()) == target:
                return True
    return False
