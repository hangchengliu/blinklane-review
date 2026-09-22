# Architecture / 架构

BlinkLane has two local parts:

- FastAPI backend: import, SQLite storage, YOLO tracking, event scoring, export package generation.
- React frontend: folder input, analysis status, suspected-event list, review form, evidence export.

## Layout

| Path | Role |
| --- | --- |
| `backend/app/` | FastAPI, `volumes.py`, `adapters.py` (`LocalExportAdapter` only), `analysis/` |
| `backend/tests/` | pytest, including `synthetic.py` |
| `frontend/src/` | Review UI |
| `docs/` | Architecture and privacy notes |

Agent-oriented run notes and env defaults: [AGENTS.md](../AGENTS.md). Fixture commands: [README Development](../README.md#development--开发).

## Data Flow

1. BlinkLane scans mounted volumes for Tesla clips and imports them. A pasted folder path remains the fallback.
2. Backend scans Tesla-style MP4 names and stores video segments in SQLite.
3. User starts analysis.
4. Backend analyzes front-camera clips, tracks vehicles, and scores lane-change evidence. Turn-signal scoring looks at front-camera light regions and the time-aligned `left_repeater` / `right_repeater` clips.
5. Frontend displays suspected events for manual review.
6. Confirmed events can be exported as a local evidence package. `LocalExportAdapter.submit` accepts only those confirmed events. City-specific adapters are not part of this build.

## Environment

| Variable | Default | Legacy |
| --- | --- | --- |
| `BLINKLANE_DATA_DIR` | `.blinklane_data` (see Storage) | `VCY_DATA_DIR` keeps `vcy.sqlite3` |
| `BLINKLANE_MODEL_NAME` | `yolo26s.pt` | `VCY_MODEL_NAME` |
| `BLINKLANE_SAMPLE_RATE_FPS` | `5` | `VCY_SAMPLE_RATE_FPS` |
| `BLINKLANE_CLOCK_OFFSET_MINUTES` | `0` (filename clock is local wall time, not UTC) | `VCY_CLOCK_OFFSET_MINUTES` |
| `BLINKLANE_VOLUME_SCAN` | on (`0` / `false` / `no` / `off` disables) | none |
| `BLINKLANE_VOLUME_SCAN_INTERVAL_S` | `15` | none |

## Storage

All local artifacts are written under `.blinklane_data/` by default
(`blinklane.sqlite3`). If `VCY_DATA_DIR` is set, or `.vcy_data` already exists
and `.blinklane_data` does not, BlinkLane keeps using that legacy directory and
`vcy.sqlite3`.
