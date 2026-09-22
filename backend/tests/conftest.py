from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_background_volume_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep TestClient from walking real mount points during the suite."""

    monkeypatch.setenv("BLINKLANE_VOLUME_SCAN", "0")
