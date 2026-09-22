# BlinkLane / 变道灯光复核助手

**Public Preview v0.1.0**  
Local Tesla Dashcam reviewer for detecting suspected lane changes without an observed turn signal.

BlinkLane 是一个本地运行的行车视频复核工具。它读取 Tesla Dashcam 文件夹，使用 YOLO 和车辆轨迹分析找出“疑似变道但未观察到转向灯”的片段，并生成可人工复核的证据包。

> Public Preview notice: this is an early testing build. It is useful for experiments and manual review, but it does not make legal conclusions and should not be used as an automated enforcement system.
>
> 公开预览版说明：这是早期测试版，只做辅助发现和人工复核，不自动举报、不自动认定违法。

## What It Does / 功能

- Import Tesla Dashcam folders with `front`, `back`, `left_repeater`, and `right_repeater` clips.
- Detect and track vehicles in front-camera footage with YOLO.
- Estimate lane-change candidates from vehicle trajectory. When lane lines are visible, the score uses position relative to those lines so the ego vehicle's own turn is less likely to be counted as another car changing lanes.
- Check front-camera light regions and the time-aligned `left_repeater` / `right_repeater` clips for amber blinking near the lane-change window.
- Let users review suspected events, add location/notes, and mark them as confirmed, dismissed, or pending.
- Export a local evidence package with raw clip, annotated clip, key frame, `metadata.json`, `report.txt`, and zip.

## What It Does Not Do / 边界

- It does not automatically report anything to a traffic authority.
- It does not bypass login, CAPTCHA, platform restrictions, or upload private videos.
- It does not assert that a traffic violation happened.
- It is not affiliated with Tesla, Inc., traffic police, or any government platform.

## Quick Start / 快速开始

```bash
cd /path/to/blinklane-review
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,yolo]"
npm install
```

Install sets:

| Extra | What you get | When |
| --- | --- | --- |
| (none) | FastAPI and OpenCV | API only |
| `.[dev]` | pytest, httpx, ruff | CI and tests. Does not install YOLO or download weights |
| `.[yolo]` | `ultralytics>=8.4`, `torch>=2.5` | Detection |
| `.[dev,yolo]` | both extras | Local full setup |

The default weight file name is `yolo26s.pt`. Ultralytics downloads it on first analysis. CI mocks the detector and must not download those weights.

Optional but recommended for reliable video clipping:

```bash
brew install ffmpeg
```

Run the backend:

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend in another terminal:

```bash
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

Plug in the Tesla USB drive. BlinkLane scans mount points for Tesla-named MP4 trees and imports them:

- macOS: `/Volumes`
- Linux: `/media` and `/run/media`
- Windows: drive roots such as `D:\`

The page lists discovered volumes and shows when an import is in progress. Set `BLINKLANE_VOLUME_SCAN=0` to turn the background scan off, or `BLINKLANE_VOLUME_SCAN_INTERVAL_S` to change the interval (default 15 seconds).

You can still paste a folder path by hand. For example:

```text
/Volumes/TESLADRIVE/TeslaCam/SavedClips/2026-05-25_18-30-02
```

The folder should contain Tesla-style MP4 names such as:

```text
2026-05-25_18-30-02-front.mp4
2026-05-25_18-30-02-left_repeater.mp4
2026-05-25_18-30-02-right_repeater.mp4
2026-05-25_18-30-02-back.mp4
```

## Development / 开发

```bash
python -m pytest
python -m ruff check .
npm run build
```

`backend/tests/test_e2e_fixture.py` walks import → analyze → confirm → zip with synthetic clips and a mocked detector. It does not download YOLO weights and does not use real Tesla footage. Later detection changes should keep that event list comparable.

Data and generated artifacts stay local under:

```text
.blinklane_data/
```

The database file there is `blinklane.sqlite3`.

### Names and clock / 命名与时间

Older builds used `.vcy_data/`, `vcy.sqlite3`, and `VCY_*` environment variables. Those still work:

- `BLINKLANE_DATA_DIR` sets the data directory. If it is unset and `VCY_DATA_DIR` is set, the legacy directory is used and the database stays `vcy.sqlite3`.
- If neither variable is set, an existing `.vcy_data` directory is kept when `.blinklane_data` does not exist yet. Otherwise the default is `.blinklane_data`.
- `BLINKLANE_MODEL_NAME` (legacy `VCY_MODEL_NAME`) defaults to `yolo26s.pt`.
- `BLINKLANE_SAMPLE_RATE_FPS` (legacy `VCY_SAMPLE_RATE_FPS`) defaults to `5`.

Tesla filenames look like `2026-05-25_18-30-02-front.mp4`. That clock is **local wall time**, not UTC. `BLINKLANE_CLOCK_OFFSET_MINUTES` (legacy `VCY_CLOCK_OFFSET_MINUTES`, default `0`) is added to the filename clock when a folder is imported. Evidence reports include the resulting absolute time.

Importing the same folder again rescans it and updates segments. Running analysis again replaces events for the segments it analyzes. Jobs left `queued` or `running` when the process stops are marked failed on the next startup. Clip or key-frame failures are written into the job message instead of being stored as empty paths.

## Project Status / 项目状态

This repository is in **Public Preview**:

- v0.1 focuses on the local review flow and evidence package export.
- Accuracy is not guaranteed before real-world Tesla Dashcam samples are used to tune thresholds.
- Contributions are welcome, especially around lane-change heuristics, turn-signal detection, privacy-safe sample generation, and Chinese documentation.

## License / 许可证

This project is released under **GNU AGPL-3.0-only**. See [LICENSE](LICENSE).

Ultralytics YOLO is offered under AGPL-3.0 or Enterprise licensing by Ultralytics. If you plan to use BlinkLane in a commercial or closed-source setting, review Ultralytics licensing carefully: https://www.ultralytics.com/license
