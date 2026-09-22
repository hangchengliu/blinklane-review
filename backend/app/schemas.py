from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReviewStatus = Literal["pending", "confirmed", "dismissed"]


class ImportRequest(BaseModel):
    folder_path: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    session_id: str
    model_name: str | None = None
    sample_rate_fps: float | None = Field(default=None, gt=0, le=30)


class ReviewRequest(BaseModel):
    review_status: ReviewStatus
    location: str = ""
    note: str = ""
    plate: str = ""


class HealthResponse(BaseModel):
    ok: bool
    ffmpeg_available: bool
    yolo_available: bool
    cv2_available: bool
    data_dir: str
    messages: list[str]
