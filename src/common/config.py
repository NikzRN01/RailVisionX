from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        return cls(raw=load_yaml(path))
