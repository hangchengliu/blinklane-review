from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ReviewStatus = Literal["pending", "confirmed", "dismissed"]

# Only simple local weight filenames like ``yolo26s.pt``: a bare basename,
# no path separators, no ``..``, no URLs.
MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.pt")


class ImportRequest(BaseModel):
    folder_path: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    session_id: str
    model_name: str | None = None
    sample_rate_fps: float | None = Field(default=None, gt=0, le=30)

    @field_validator("model_name")
    @classmethod
    def _model_name_is_a_plain_weight_file(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if ".." in value or not MODEL_NAME_RE.fullmatch(value):
            raise ValueError(
                "model_name must be a plain weight filename like 'yolo26s.pt' "
                "(no paths, no URLs)."
            )
        return value


class ReviewRequest(BaseModel):
    review_status: ReviewStatus
    location: str = ""
    note: str = ""
    plate: str = ""


class HealthResponse(BaseModel):
    ok: bool
    ffmpeg_available: bool
    yolo_available: bool
    data_dir: str
    messages: list[str]
