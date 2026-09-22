"""Find Tesla Dashcam trees on removable volumes and import them."""

from __future__ import annotations

import os
import platform
import string
import threading
from pathlib import Path
from typing import Any

from .tesla import parse_tesla_filename

_lock = threading.Lock()
_state: dict[str, Any] = {"scanning": False, "volumes": []}


def candidate_volume_roots() -> list[Path]:
    """Mount points to scan: macOS volumes, Linux media trees, Windows drive roots."""

    system = platform.system()
    if system == "Darwin":
        return [Path("/Volumes")]
    if system == "Windows":
        return [Path(f"{letter}:\\") for letter in string.ascii_uppercase]
    return [Path("/media"), Path("/run/media")]


def default_volume_roots() -> list[Path]:
    return [path for path in candidate_volume_roots() if path.is_dir()]


def discover_tesla_folders(roots: list[Path], *, max_depth: int = 6) -> list[Path]:
    """Return directories that directly contain Tesla-named mp4 files."""

    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        base = root.resolve()
        for dirpath, dirnames, filenames in os.walk(base):
            current = Path(dirpath)
            depth = len(current.relative_to(base).parts)
            if depth > max_depth:
                dirnames.clear()
                continue
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            if any(parse_tesla_filename(name) for name in filenames):
                if current not in seen:
                    seen.add(current)
                    found.append(current)
    return found


def volume_snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "scanning": bool(_state["scanning"]),
            "volumes": [dict(item) for item in _state["volumes"]],
        }


def scan_and_import(roots: list[Path] | None = None) -> dict[str, Any]:
    """Scan volume roots and import each folder that contains Tesla clips."""

    from .main import import_folder_path

    scan_roots = list(default_volume_roots() if roots is None else roots)
    results: list[dict[str, Any]] = []
    _publish(results, scanning=True)
    try:
        for root in scan_roots:
            for folder in discover_tesla_folders([root]):
                entry: dict[str, Any] = {
                    "root": str(root),
                    "folder_path": str(folder),
                    "clip_count": 0,
                    "status": "importing",
                    "session_id": None,
                    "message": "导入中",
                }
                results.append(entry)
                _publish(results, scanning=True)
                try:
                    imported = import_folder_path(folder)
                    session = imported.get("session") or {}
                    entry["clip_count"] = len(imported.get("segments") or [])
                    entry["status"] = "imported"
                    entry["session_id"] = session.get("id")
                    entry["message"] = "已导入"
                except Exception as exc:
                    detail = getattr(exc, "detail", None)
                    entry["status"] = "failed"
                    entry["message"] = str(detail or exc)
                _publish(results, scanning=True)
    finally:
        _publish(results, scanning=False)
    return volume_snapshot()


def _publish(results: list[dict[str, Any]], *, scanning: bool) -> None:
    with _lock:
        _state["scanning"] = scanning
        _state["volumes"] = [dict(item) for item in results]
