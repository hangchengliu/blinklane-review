from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.volumes import (
    candidate_volume_roots,
    discover_tesla_folders,
    scan_and_import,
)


def test_volume_roots_match_the_platform(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.volumes.platform.system", lambda: "Darwin")
    assert candidate_volume_roots() == [Path("/Volumes")]

    monkeypatch.setattr("backend.app.volumes.platform.system", lambda: "Linux")
    assert candidate_volume_roots() == [Path("/media"), Path("/run/media")]

    monkeypatch.setattr("backend.app.volumes.platform.system", lambda: "Windows")
    windows_roots = candidate_volume_roots()
    assert len(windows_roots) == 26
    assert str(windows_roots[0]).startswith("A")
    assert str(windows_roots[-1]).startswith("Z")


def test_discover_finds_tesla_folders_and_skips_noise(tmp_path: Path) -> None:
    folder = tmp_path / "TESLADRIVE" / "TeslaCam" / "SavedClips" / "2026-05-25_18-30-02"
    folder.mkdir(parents=True)
    (folder / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "TESLADRIVE" / "random.mp4").write_bytes(b"nope")
    deep = tmp_path.joinpath(*[f"d{i}" for i in range(8)])
    deep.mkdir(parents=True)
    (deep / "2026-05-25_18-30-02-front.mp4").write_bytes(b"too deep")

    found = discover_tesla_folders([tmp_path / "TESLADRIVE"])

    assert found == [folder.resolve()]


def test_scan_auto_imports_and_api_lists_volumes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "volume"
    folder = root / "TeslaCam" / "RecentClips"
    folder.mkdir(parents=True)
    (folder / "2026-05-25_18-30-02-front.mp4").write_bytes(b"fake")

    first = scan_and_import([root])
    (folder / "2026-05-25_18-30-02-left_repeater.mp4").write_bytes(b"fake")
    second = scan_and_import([root])

    assert first["scanning"] is False
    assert first["volumes"][0]["status"] == "imported"
    assert first["volumes"][0]["clip_count"] == 1
    assert second["volumes"][0]["clip_count"] == 2
    assert second["volumes"][0]["session_id"] == first["volumes"][0]["session_id"]
    assert second["volumes"][0]["message"] == "已导入"

    client = TestClient(app)
    listed = client.get("/api/volumes")
    assert listed.status_code == 200
    body = listed.json()
    assert body["scanning"] is False
    assert body["volumes"][0]["folder_path"] == str(folder.resolve())
    assert body["volumes"][0]["root"] == str(root)
