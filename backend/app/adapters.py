"""Upload adapters.

``submit`` accepts an evidence package for a user-confirmed event and returns
a result. The default adapter writes nothing off the machine. City websites
are not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Evidence:
    event_id: str
    clip: str | None
    screenshot: str | None
    absolute_time: str | None
    place: str
    plate: str
    confirm_status: str
    package_dir: str
    zip_path: str


@dataclass(frozen=True)
class SubmissionResult:
    ok: bool
    adapter: str
    message: str
    location: str


class UploadAdapter(Protocol):
    name: str

    def submit(self, evidence: Evidence) -> SubmissionResult:
        """Submit one confirmed evidence package."""


class LocalExportAdapter:
    """Default adapter: the evidence zip already on disk is the submission."""

    name = "local_export"

    def submit(self, evidence: Evidence) -> SubmissionResult:
        if evidence.confirm_status != "confirmed":
            return SubmissionResult(
                ok=False,
                adapter=self.name,
                message="Only confirmed events can be submitted.",
                location="",
            )
        zip_path = Path(evidence.zip_path)
        if not zip_path.is_file():
            return SubmissionResult(
                ok=False,
                adapter=self.name,
                message="Evidence package is missing.",
                location="",
            )
        return SubmissionResult(
            ok=True,
            adapter=self.name,
            message="证据包已保存在本机。",
            location=str(zip_path),
        )


def get_upload_adapter() -> UploadAdapter:
    return LocalExportAdapter()
