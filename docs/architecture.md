# Architecture / 架构

BlinkLane has two local parts:

- FastAPI backend: import, SQLite storage, YOLO tracking, event scoring, export package generation.
- React frontend: folder input, analysis status, suspected-event list, review form, evidence export.

## Data Flow

1. User pastes a Tesla Dashcam folder path.
2. Backend scans Tesla-style MP4 names and stores video segments in SQLite.
3. User starts analysis.
4. Backend analyzes front-camera clips, tracks vehicles, scores lane-change and turn-signal evidence.
5. Frontend displays suspected events for manual review.
6. Confirmed events can be exported as a local evidence package.

## Storage

All local artifacts are written under `.vcy_data/` by default.
