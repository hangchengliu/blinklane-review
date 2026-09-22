from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .adapters import Evidence, get_upload_adapter
from .analysis.pipeline import SegmentAnalysis, analyze_segment, yolo_available
from .config import ensure_data_dirs, get_settings
from .db import connect, init_db, row_to_dict, rows_to_dicts
from .exporter import evidence_fields, export_evidence_package
from .schemas import AnalyzeRequest, HealthResponse, ImportRequest, ReviewRequest
from .tesla import TeslaClipInfo, find_tesla_clips, new_id, probe_video
from .video import ffmpeg_available
from .volumes import scan_and_import, volume_snapshot

INTERRUPTED_JOB_MESSAGE = "Interrupted before completion; marked failed on startup."


async def _volume_scan_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.to_thread(scan_and_import)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=get_settings().volume_scan_interval_s)
        except TimeoutError:
            continue


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ensure_data_dirs()
    init_db()
    fail_interrupted_jobs()
    stop = asyncio.Event()
    scanner: asyncio.Task[None] | None = None
    if get_settings().volume_scan_enabled:
        scanner = asyncio.create_task(_volume_scan_loop(stop))
    try:
        yield
    finally:
        stop.set()
        if scanner is not None:
            scanner.cancel()
            try:
                await scanner
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="BlinkLane API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    cv2_available = importlib.util.find_spec("cv2") is not None
    messages: list[str] = []
    if not ffmpeg_available():
        messages.append("ffmpeg 未安装：可分析，但导出原始片段会使用较慢的 OpenCV 兜底。")
    if not yolo_available():
        messages.append("Ultralytics YOLO 未安装：请运行 python -m pip install -e '.[yolo]'。")
    return HealthResponse(
        ok=cv2_available,
        ffmpeg_available=ffmpeg_available(),
        yolo_available=yolo_available(),
        cv2_available=cv2_available,
        data_dir=str(settings.data_dir),
        messages=messages,
    )


@app.get("/api/volumes")
def list_volumes() -> dict[str, Any]:
    return volume_snapshot()


@app.post("/api/import")
def import_folder(payload: ImportRequest) -> dict[str, Any]:
    folder = Path(payload.folder_path).expanduser().resolve()
    return import_folder_path(folder)


def import_folder_path(folder: Path) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist or is not a directory.")

    init_db()
    now = datetime.now(UTC).isoformat()
    clips = find_tesla_clips(folder)
    with connect() as conn:
        existing = conn.execute(
            "SELECT * FROM sessions WHERE folder_path = ?",
            (str(folder),),
        ).fetchone()
        if existing is None and not clips:
            raise HTTPException(status_code=400, detail="No Tesla Dashcam MP4 files found.")
        if existing is None:
            session_id = new_id()
            conn.execute(
                "INSERT INTO sessions (id, folder_path, created_at) VALUES (?, ?, ?)",
                (session_id, str(folder), now),
            )
            _insert_segments(conn, session_id, clips, now)
            reused = False
        else:
            session_id = str(existing["id"])
            changed = _sync_segments(conn, session_id, clips, now)
            reused = not changed

        session = row_to_dict(
            conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        )
        segments = rows_to_dicts(
            conn.execute(
                "SELECT * FROM video_segments WHERE session_id = ? ORDER BY starts_at, camera",
                (session_id,),
            ).fetchall()
        )
    return {"session": session, "segments": segments, "reused": reused}


def _insert_segments(conn: Any, session_id: str, clips: list[TeslaClipInfo], now: str) -> None:
    for clip in clips:
        _insert_segment(conn, session_id, clip, now)


def _insert_segment(conn: Any, session_id: str, clip: TeslaClipInfo, now: str) -> None:
    probe = probe_video(clip.path)
    conn.execute(
        """
        INSERT INTO video_segments (
          id, session_id, camera, starts_at, path, duration_s,
          fps, width, height, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id(),
            session_id,
            clip.camera,
            clip.starts_at.isoformat(),
            str(clip.path),
            probe.duration_s,
            probe.fps,
            probe.width,
            probe.height,
            now,
        ),
    )


def _sync_segments(conn: Any, session_id: str, clips: list[TeslaClipInfo], now: str) -> bool:
    """Rescan a folder into an existing session. Return whether the file set changed."""

    rows = conn.execute(
        "SELECT * FROM video_segments WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    by_path = {row["path"]: row for row in rows}
    seen: set[str] = set()
    changed = False
    for clip in clips:
        path = str(clip.path)
        seen.add(path)
        starts_at = clip.starts_at.isoformat()
        probe = probe_video(clip.path)
        current = by_path.get(path)
        if current is None:
            changed = True
            _insert_segment(conn, session_id, clip, now)
            continue
        if current["camera"] != clip.camera or current["starts_at"] != starts_at:
            changed = True
        conn.execute(
            """
            UPDATE video_segments
            SET camera = ?, starts_at = ?, duration_s = ?, fps = ?, width = ?, height = ?
            WHERE id = ?
            """,
            (
                clip.camera,
                starts_at,
                probe.duration_s,
                probe.fps,
                probe.width,
                probe.height,
                current["id"],
            ),
        )
    for path, row in by_path.items():
        if path not in seen:
            changed = True
            conn.execute("DELETE FROM video_segments WHERE id = ?", (row["id"],))
    return changed


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    settings = get_settings()
    model_name = payload.model_name or settings.model_name
    sample_rate = payload.sample_rate_fps or settings.sample_rate_fps
    now = datetime.now(UTC).isoformat()
    job_id = new_id()
    with connect() as conn:
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (payload.session_id,),
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        conn.execute(
            """
            INSERT INTO analysis_jobs (
              id, session_id, status, progress, message, model_name,
              sample_rate_fps, created_at, updated_at
            )
            VALUES (?, ?, 'queued', 0, 'Queued', ?, ?, ?, ?)
            """,
            (job_id, payload.session_id, model_name, sample_rate, now, now),
        )
    background_tasks.add_task(run_analysis_job, job_id)
    return get_job(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with connect() as conn:
        job = row_to_dict(
            conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        )
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        event_count = conn.execute(
            "SELECT COUNT(*) AS count FROM events WHERE session_id = ?",
            (job["session_id"],),
        ).fetchone()["count"]
    job["event_count"] = event_count
    return job


@app.get("/api/events")
def list_events(session_id: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM events"
    params: tuple[Any, ...] = ()
    if session_id:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " ORDER BY confidence DESC, created_at DESC"
    with connect() as conn:
        events = rows_to_dicts(conn.execute(query, params).fetchall())
    return [with_media_urls(event) for event in events]


@app.patch("/api/events/{event_id}/review")
def update_review(event_id: str, payload: ReviewRequest) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    review_id = new_id()
    with connect() as conn:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found.")
        conn.execute(
            """
            UPDATE events
            SET review_status = ?, location = ?, note = ?, plate = ?
            WHERE id = ?
            """,
            (payload.review_status, payload.location, payload.note, payload.plate, event_id),
        )
        conn.execute(
            """
            INSERT INTO event_reviews (id, event_id, status, location, note, plate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                event_id,
                payload.review_status,
                payload.location,
                payload.note,
                payload.plate,
                now,
            ),
        )
        updated = row_to_dict(
            conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        )
    return with_media_urls(updated or {})


@app.post("/api/events/{event_id}/export")
def export_event(event_id: str) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        event = row_to_dict(
            conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        )
        if not event:
            raise HTTPException(status_code=404, detail="Event not found.")
        if event["review_status"] != "confirmed":
            raise HTTPException(status_code=400, detail="Only confirmed events can be exported.")
        segment = row_to_dict(
            conn.execute(
                "SELECT * FROM video_segments WHERE id = ?",
                (event["segment_id"],),
            ).fetchone()
        )
        if not segment:
            raise HTTPException(status_code=404, detail="Segment not found.")
        package_dir, zip_path = export_evidence_package(event, segment)
        fields = evidence_fields(event, segment)
        evidence = Evidence(
            event_id=str(event["id"]),
            clip=fields["clip"],
            screenshot=fields["screenshot"],
            absolute_time=fields["absolute_time"],
            place=str(fields["place"]),
            plate=str(fields["plate"]),
            confirm_status=str(fields["confirm_status"]),
            package_dir=str(package_dir),
            zip_path=str(zip_path),
        )
        submission = get_upload_adapter().submit(evidence)
        if not submission.ok:
            raise HTTPException(status_code=400, detail=submission.message)
        export_id = new_id()
        conn.execute(
            "INSERT INTO exports (id, event_id, path, zip_path, created_at) VALUES (?, ?, ?, ?, ?)",
            (export_id, event_id, str(package_dir), str(zip_path), now),
        )
    return {
        "id": export_id,
        "event_id": event_id,
        "path": str(package_dir),
        "zip_path": str(zip_path),
        "download_url": f"/api/exports/{export_id}/download",
        "adapter": submission.adapter,
        "submission_message": submission.message,
    }


@app.get("/api/exports/{export_id}/download")
def download_export(export_id: str) -> FileResponse:
    with connect() as conn:
        export = row_to_dict(
            conn.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
        )
        if not export:
            raise HTTPException(status_code=404, detail="Export not found.")
    zip_path = Path(export["zip_path"])
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found.")
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.get("/api/media/{path:path}")
def media(path: str) -> FileResponse:
    settings = get_settings()
    target = (settings.data_dir / path).resolve()
    try:
        target.relative_to(settings.data_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid media path.") from None
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media not found.")
    return FileResponse(target)


def run_analysis_job(job_id: str) -> None:
    started = datetime.now(UTC).isoformat()
    with connect() as conn:
        job = row_to_dict(
            conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        )
        if not job:
            return
        conn.execute(
            """
            UPDATE analysis_jobs
            SET status='running', progress=0, message='Starting analysis',
                started_at=?, updated_at=?
            WHERE id=?
            """,
            (started, started, job_id),
        )
        segments = rows_to_dicts(
            conn.execute(
                """
                SELECT * FROM video_segments
                WHERE session_id = ? AND camera = 'front'
                ORDER BY starts_at
                """,
                (job["session_id"],),
            ).fetchall()
        )
    if not segments:
        fail_job(job_id, "No front camera segments found.")
        return

    warnings: list[str] = []
    try:
        for index, segment in enumerate(segments):
            base_progress = index / len(segments)

            def progress(
                partial: float,
                message: str,
                base_progress: float = base_progress,
                total_segments: int = len(segments),
            ) -> None:
                update_job(
                    job_id,
                    status="running",
                    progress=min(0.98, base_progress + partial / total_segments),
                    message=message,
                )

            analysis: SegmentAnalysis = analyze_segment(
                segment["id"],
                Path(segment["path"]),
                model_name=job["model_name"],
                sample_rate_fps=job["sample_rate_fps"],
                progress=progress,
                repeater_paths=_repeater_paths(segment),
            )
            warnings.extend(analysis.warnings)
            replace_segment_events(job["session_id"], segment, analysis.events)
            update_job(
                job_id,
                status="running",
                progress=(index + 1) / len(segments),
                message=f"Finished {Path(segment['path']).name}",
            )
        finished = datetime.now(UTC).isoformat()
        update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=_completion_message(warnings),
        )
        with connect() as conn:
            conn.execute(
                "UPDATE analysis_jobs SET finished_at = ?, updated_at = ? WHERE id = ?",
                (finished, finished, job_id),
            )
    except Exception as exc:
        detail = str(exc)
        if warnings:
            detail = f"{detail} Asset warnings: {_warning_text(warnings)}"
        fail_job(job_id, detail)


def _repeater_paths(segment: dict[str, Any]) -> dict[str, Path]:
    """Side cameras that share the front clip's filename clock."""

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT camera, path FROM video_segments
            WHERE session_id = ? AND starts_at = ?
              AND camera IN ('left_repeater', 'right_repeater')
            """,
            (segment["session_id"], segment["starts_at"]),
        ).fetchall()
    return {str(row["camera"]): Path(str(row["path"])) for row in rows}


def replace_segment_events(
    session_id: str,
    segment: dict[str, Any],
    candidates: list[Any],
) -> None:
    """Replace events for one segment so a second analysis does not stack copies."""

    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            "DELETE FROM events WHERE session_id = ? AND segment_id = ?",
            (session_id, segment["id"]),
        )
        for candidate in candidates:
            labels = list(candidate.reason_labels)
            raw_clip_path = _extract_path_label(labels, "raw_clip=")
            annotated_clip_path = _extract_path_label(labels, "annotated_clip=")
            key_frame_path = _extract_path_label(labels, "key_frame=")
            conn.execute(
                """
                INSERT INTO events (
                  id, session_id, segment_id, camera, track_id, vehicle_type,
                  start_s, end_s, key_s, lane_change_score, turn_signal_score,
                  confidence, reason_labels, raw_clip_path, annotated_clip_path,
                  key_frame_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    session_id,
                    segment["id"],
                    segment["camera"],
                    candidate.track_id,
                    candidate.vehicle_type,
                    candidate.start_s,
                    candidate.end_s,
                    candidate.key_s,
                    candidate.lane_change_score,
                    candidate.turn_signal_score,
                    candidate.confidence,
                    json.dumps(labels, ensure_ascii=False),
                    raw_clip_path,
                    annotated_clip_path,
                    key_frame_path,
                    now,
                ),
            )


def update_job(job_id: str, *, status: str, progress: float, message: str) -> None:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            """
            UPDATE analysis_jobs
            SET status = ?, progress = ?, message = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, progress, message, now, job_id),
        )


def fail_job(job_id: str, message: str) -> None:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            """
            UPDATE analysis_jobs
            SET status='failed', progress=1, message=?, finished_at=?, updated_at=?
            WHERE id=?
            """,
            (message, now, now, job_id),
        )


def fail_interrupted_jobs() -> int:
    """Mark queued or running jobs failed after a process restart."""

    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE analysis_jobs
            SET status = 'failed',
                message = ?,
                finished_at = COALESCE(finished_at, ?),
                updated_at = ?
            WHERE status IN ('queued', 'running')
            """,
            (INTERRUPTED_JOB_MESSAGE, now, now),
        )
        return int(cursor.rowcount or 0)


def with_media_urls(event: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    for key, url_key in (
        ("raw_clip_path", "raw_clip_url"),
        ("annotated_clip_path", "annotated_clip_url"),
        ("key_frame_path", "key_frame_url"),
    ):
        value = event.get(key)
        if value:
            try:
                rel = Path(value).resolve().relative_to(settings.data_dir)
                event[url_key] = f"/api/media/{rel.as_posix()}"
            except ValueError:
                event[url_key] = None
        else:
            event[url_key] = None
    return event


def _extract_path_label(labels: list[str], prefix: str) -> str | None:
    for label in labels:
        if label.startswith(prefix):
            path = label.removeprefix(prefix).strip()
            if path:
                return path
    return None


def _warning_text(warnings: list[str]) -> str:
    detail = " | ".join(warnings)
    if len(detail) > 1800:
        return detail[:1800] + "…"
    return detail


def _completion_message(warnings: list[str]) -> str:
    if not warnings:
        return "Analysis completed"
    return f"Analysis completed with asset warnings: {_warning_text(warnings)}"
