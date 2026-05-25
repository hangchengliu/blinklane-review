from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    model_name: str
    sample_rate_fps: float


def get_settings() -> Settings:
    cwd = Path.cwd()
    data_dir = Path(os.getenv("VCY_DATA_DIR", cwd / ".vcy_data")).expanduser().resolve()
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "vcy.sqlite3",
        model_name=os.getenv("VCY_MODEL_NAME", "yolo26s.pt"),
        sample_rate_fps=float(os.getenv("VCY_SAMPLE_RATE_FPS", "5")),
    )


def ensure_data_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for child in ("clips", "exports", "keys", "annotated"):
        (settings.data_dir / child).mkdir(parents=True, exist_ok=True)
