"""
JSON file writer for per-run output folders.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from app.config import OUTPUTS_DIR


def create_run_dir(run_id: str) -> Path:
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_Encoder, default=str)
