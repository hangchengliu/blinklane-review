# AGENTS

## Purpose

Local Tesla Dashcam review. Find clips where a lane change happened and no turn signal was observed, let a person confirm, then export an evidence package that stays on this machine. The software does not decide that something is illegal and does not auto-report.

## Layout

- `backend/app/` — FastAPI app, `volumes.py` (mount scan), `adapters.py` (`LocalExportAdapter` only), `analysis/`
- `backend/tests/` — pytest, including synthetic Tesla folders
- `frontend/src/` — React review UI
- `docs/` — architecture and privacy notes

Default data directory: `.blinklane_data/blinklane.sqlite3` (gitignored).

## Run

```bash
python -m pip install -e ".[dev,yolo]"
npm install
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal: `npm run dev`, then open `http://127.0.0.1:5173/`.

CI installs `.[dev]` only and must not download `yolo26s.pt`.

## Test

```bash
python -m pytest
python -m ruff check .
npm run build
```

Fixture commands live in [README.md](README.md#development--开发). Do not duplicate that block here.

## Environment

From `backend/app/config.py` (`get_settings`). Legacy `VCY_*` names still work.

| Variable | Default | Legacy |
| --- | --- | --- |
| `BLINKLANE_DATA_DIR` | `.blinklane_data` (see fallback below) | `VCY_DATA_DIR` keeps `vcy.sqlite3` |
| `BLINKLANE_MODEL_NAME` | `yolo26s.pt` | `VCY_MODEL_NAME` |
| `BLINKLANE_SAMPLE_RATE_FPS` | `5` | `VCY_SAMPLE_RATE_FPS` |
| `BLINKLANE_CLOCK_OFFSET_MINUTES` | `0` (filename clock is local wall time, not UTC) | `VCY_CLOCK_OFFSET_MINUTES` |
| `BLINKLANE_VOLUME_SCAN` | on (`0` / `false` / `no` / `off` disables) | none |
| `BLINKLANE_VOLUME_SCAN_INTERVAL_S` | `15` | none |
| `BLINKLANE_REPORT_URL` | empty | Optional handoff URL opened after confirm; no auto POST |

If the data directory is unset and `.vcy_data` already exists while `.blinklane_data` does not, keep the legacy directory.

## Private Tesla folder validation

To try your own SavedClips folder locally, install detection extras (`python -m pip install -e '.[yolo]'`), point the UI at the folder, and review results yourself. Do not commit real dashcam footage. Detection is not accuracy-guaranteed until tuned on real samples.

## Hard limits

- Do not bypass CAPTCHA or login.
- Do not add city-site adapters in this repository.
- Do not commit real dashcam footage.
- `submit` accepts only user-confirmed events.

The default adapter is `LocalExportAdapter`. City websites are not in this repo.
