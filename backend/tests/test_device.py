from __future__ import annotations

import sys

from backend.app.analysis.pipeline import choose_torch_device


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Mps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_device_order_is_cuda_then_mps_then_cpu(monkeypatch) -> None:
    class Torch:
        cuda = _Cuda(True)
        backends = type("Backends", (), {"mps": _Mps(True)})()

    monkeypatch.setitem(sys.modules, "torch", Torch)
    assert choose_torch_device() == "cuda"

    Torch.cuda = _Cuda(False)
    assert choose_torch_device() == "mps"

    Torch.backends = type("Backends", (), {"mps": _Mps(False)})()
    assert choose_torch_device() == "cpu"


def test_device_is_cpu_when_torch_is_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert choose_torch_device() == "cpu"
