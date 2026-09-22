from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import get_settings

PACKAGE_FILES = (
    "raw_clip.mp4",
    "annotated_clip.mp4",
    "key_frame.jpg",
    "metadata.json",
    "report.txt",
)


def export_evidence_package(event: dict[str, Any], segment: dict[str, Any]) -> tuple[Path, Path]:
    settings = get_settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    package_dir = settings.data_dir / "exports" / f"event_{event['id']}_{stamp}"
    package_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(event.get("raw_clip_path"), package_dir / "raw_clip.mp4")
    copy_if_exists(event.get("annotated_clip_path"), package_dir / "annotated_clip.mp4")
    copy_if_exists(event.get("key_frame_path"), package_dir / "key_frame.jpg")

    absolute_time = absolute_time_window(segment, event)
    metadata = {
        "event_id": event["id"],
        "session_id": event["session_id"],
        "segment_id": event["segment_id"],
        "source_video": segment["path"],
        "camera": event["camera"],
        "track_id": event["track_id"],
        "vehicle_type": event["vehicle_type"],
        "start_s": event["start_s"],
        "end_s": event["end_s"],
        "key_s": event["key_s"],
        "lane_change_score": event["lane_change_score"],
        "turn_signal_score": event["turn_signal_score"],
        "confidence": event["confidence"],
        "reason_labels": event["reason_labels"],
        "review_status": event["review_status"],
        "location": event["location"],
        "note": event["note"],
        "absolute_time": absolute_time,
        "clock_basis": (
            "Filename clock is local wall time. "
            "BLINKLANE_CLOCK_OFFSET_MINUTES is applied when the clip is imported."
        ),
        "exported_at": datetime.now(UTC).isoformat(),
        "disclaimer": (
            "This package is an aid for manual review and does not assert a legal violation."
        ),
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (package_dir / "report.txt").write_text(build_report(metadata), encoding="utf-8")

    zip_path = package_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in PACKAGE_FILES:
            file_path = package_dir / filename
            if file_path.exists():
                archive.write(file_path, arcname=filename)
    return package_dir, zip_path


def copy_if_exists(source: str | None, destination: Path) -> None:
    if not source:
        return
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        shutil.copy2(source_path, destination)


def absolute_time_window(segment: dict[str, Any], event: dict[str, Any]) -> str | None:
    """Wall-clock window: filename clock (plus import offset) + offsets in the file."""

    raw = segment.get("starts_at")
    if not raw:
        return None
    try:
        start = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    abs_start = start + timedelta(seconds=float(event["start_s"]))
    abs_end = start + timedelta(seconds=float(event["end_s"]))
    return (
        f"{abs_start.strftime('%Y-%m-%d %H:%M:%S')} 至 {abs_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def build_report(metadata: dict[str, Any]) -> str:
    signal_text = (
        "未观察到稳定转向灯闪烁"
        if metadata["turn_signal_score"] < 0.25
        else "转向灯情况不确定，需人工复核"
    )
    absolute_time = metadata.get("absolute_time") or "未知（缺少片段起始时间）"
    return "\n".join(
        [
            "疑似变道未观察到转向灯证据说明（人工复核草稿）",
            "",
            f"地点：{metadata.get('location') or '请填写具体道路、方向、附近参照物'}",
            f"绝对时间：{absolute_time}",
            f"时间段：源视频 {metadata['start_s']:.1f}s - {metadata['end_s']:.1f}s",
            f"摄像头：{metadata['camera']}",
            f"车辆类型：{metadata['vehicle_type']}",
            f"系统判断：车辆轨迹疑似变道；{signal_text}。",
            f"算法置信度：{metadata['confidence']:.2f}",
            "",
            "人工备注：",
            metadata.get("note") or "请补充现场情况、车道方向、车牌等信息。",
            "",
            "声明：本材料由本地工具辅助生成，仅供人工复核和整理证据，不代表自动认定违法。",
            "时间说明：文件名时钟按本地墙钟理解，不是 UTC；导入时会加上可配置的分钟偏移。",
        ]
    )
