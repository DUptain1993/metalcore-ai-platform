"""YAML -> typed dataclass configuration loading.

Every stage defines its own frozen/mutable dataclass describing its knobs and
loads it with :func:`load_config`. Unknown keys in the YAML raise a clear error
so typos never silently fall back to defaults.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import yaml

T = TypeVar("T")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file into a plain dict.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping (an empty dict for an empty file).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the top-level YAML document is not a mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {p} must contain a top-level mapping, got {type(data).__name__}."
        )
    return data


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build a dataclass instance from a mapping, rejecting unknown keys.

    Args:
        cls: A dataclass type.
        data: Mapping of field name -> value.

    Returns:
        An instance of ``cls``.

    Raises:
        TypeError: If ``cls`` is not a dataclass.
        ValueError: If ``data`` contains unknown keys or omits a required field.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass.")

    valid_fields = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(valid_fields)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_fields)}."
        )

    kwargs: Dict[str, Any] = {}
    for name, field_def in valid_fields.items():
        if name in data:
            kwargs[name] = data[name]
        elif field_def.default is MISSING and field_def.default_factory is MISSING:  # type: ignore[misc]
            raise ValueError(f"Missing required config key for {cls.__name__}: {name!r}.")

    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: Union[str, Path], cls: Type[T]) -> T:
    """Load a YAML file directly into a dataclass instance.

    Args:
        path: Path to the YAML config.
        cls: The dataclass type to populate.

    Returns:
        A populated instance of ``cls``.
    """
    return from_dict(cls, load_yaml(path))
