"""Runtime settings for BlinkLane.

Installable sets (also listed in ``pyproject.toml``):

- ``pip install -e .`` — FastAPI service and OpenCV. No detector.
- ``pip install -e '.[dev]'`` — pytest, httpx, and ruff. CI uses this set.
- ``pip install -e '.[yolo]'`` — ``ultralytics>=8.4`` and ``torch>=2.5``.
- ``pip install -e '.[dev,yolo]'`` — local setup that can run detection.

The default weight file name is ``yolo26s.pt`` (override with
``BLINKLANE_MODEL_NAME``). Ultralytics downloads that file on first use.
Weights are not committed, and CI must not download them.

Environment variables use the ``BLINKLANE_`` prefix. The previous ``VCY_``
names still work. If neither data-dir variable is set, an existing
``.vcy_data`` directory is kept when ``.blinklane_data`` does not exist yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "yolo26s.pt"
DATA_DIR_NAME = ".blinklane_data"
LEGACY_DATA_DIR_NAME = ".vcy_data"
DB_NAME = "blinklane.sqlite3"
LEGACY_DB_NAME = "vcy.sqlite3"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    model_name: str
    sample_rate_fps: float
    clock_offset_minutes: float
    legacy_storage: bool
    volume_scan_enabled: bool
    volume_scan_interval_s: float
    report_url: str


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


def resolve_storage(cwd: Path | None = None) -> tuple[Path, bool]:
    """Return ``(data_dir, legacy_storage)``.

    ``BLINKLANE_DATA_DIR`` wins. Otherwise ``VCY_DATA_DIR`` keeps the old
    layout. With neither set, an existing ``.vcy_data`` directory is used
    when ``.blinklane_data`` is absent.
    """

    cwd = (cwd or Path.cwd()).resolve()
    new_env = _first_env("BLINKLANE_DATA_DIR")
    if new_env is not None:
        return Path(new_env).expanduser().resolve(), False
    old_env = _first_env("VCY_DATA_DIR")
    if old_env is not None:
        return Path(old_env).expanduser().resolve(), True
    new_dir = (cwd / DATA_DIR_NAME).resolve()
    legacy_dir = (cwd / LEGACY_DATA_DIR_NAME).resolve()
    if legacy_dir.exists() and not new_dir.exists():
        return legacy_dir, True
    return new_dir, False


def get_settings() -> Settings:
    data_dir, legacy = resolve_storage()
    model_name = _first_env("BLINKLANE_MODEL_NAME", "VCY_MODEL_NAME") or DEFAULT_MODEL_NAME
    sample_rate = float(_first_env("BLINKLANE_SAMPLE_RATE_FPS", "VCY_SAMPLE_RATE_FPS") or "5")
    offset = float(
        _first_env("BLINKLANE_CLOCK_OFFSET_MINUTES", "VCY_CLOCK_OFFSET_MINUTES") or "0"
    )
    interval = float(_first_env("BLINKLANE_VOLUME_SCAN_INTERVAL_S") or "15")
    report_url = (_first_env("BLINKLANE_REPORT_URL") or "").strip()
    db_name = LEGACY_DB_NAME if legacy else DB_NAME
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / db_name,
        model_name=model_name,
        sample_rate_fps=sample_rate,
        clock_offset_minutes=offset,
        legacy_storage=legacy,
        volume_scan_enabled=_env_flag("BLINKLANE_VOLUME_SCAN", True),
        volume_scan_interval_s=interval,
        report_url=report_url,
    )


def ensure_data_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    for child in ("clips", "exports", "keys", "annotated"):
        (settings.data_dir / child).mkdir(parents=True, exist_ok=True)
