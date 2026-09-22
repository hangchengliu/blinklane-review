from __future__ import annotations

from pathlib import Path

from backend.app.config import get_settings, resolve_storage


def _clear_data_env(monkeypatch) -> None:
    monkeypatch.delenv("BLINKLANE_DATA_DIR", raising=False)
    monkeypatch.delenv("VCY_DATA_DIR", raising=False)


def test_default_storage_uses_blinklane_names(monkeypatch, tmp_path: Path) -> None:
    _clear_data_env(monkeypatch)

    data_dir, legacy = resolve_storage(tmp_path)

    assert legacy is False
    assert data_dir == (tmp_path / ".blinklane_data").resolve()


def test_existing_vcy_data_is_kept_until_new_dir_exists(monkeypatch, tmp_path: Path) -> None:
    _clear_data_env(monkeypatch)
    legacy_dir = tmp_path / ".vcy_data"
    legacy_dir.mkdir()

    data_dir, legacy = resolve_storage(tmp_path)

    assert legacy is True
    assert data_dir == legacy_dir.resolve()

    (tmp_path / ".blinklane_data").mkdir()
    data_dir, legacy = resolve_storage(tmp_path)
    assert legacy is False
    assert data_dir == (tmp_path / ".blinklane_data").resolve()


def test_vcy_env_keeps_legacy_database_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BLINKLANE_DATA_DIR", raising=False)
    target = tmp_path / "legacy"
    monkeypatch.setenv("VCY_DATA_DIR", str(target))
    monkeypatch.setenv("VCY_MODEL_NAME", "custom.pt")

    settings = get_settings()

    assert settings.legacy_storage is True
    assert settings.data_dir == target.resolve()
    assert settings.db_path.name == "vcy.sqlite3"
    assert settings.model_name == "custom.pt"


def test_blinklane_env_wins_over_vcy(monkeypatch, tmp_path: Path) -> None:
    new_dir = tmp_path / "new"
    monkeypatch.setenv("VCY_DATA_DIR", str(tmp_path / "old"))
    monkeypatch.setenv("BLINKLANE_DATA_DIR", str(new_dir))
    monkeypatch.setenv("BLINKLANE_MODEL_NAME", "yolo26s.pt")

    settings = get_settings()

    assert settings.legacy_storage is False
    assert settings.data_dir == new_dir.resolve()
    assert settings.db_path.name == "blinklane.sqlite3"
    assert settings.model_name == "yolo26s.pt"
