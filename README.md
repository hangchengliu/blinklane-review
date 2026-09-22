# BlinkLane / 变道灯光复核助手

这是一个公益项目：工具被用得越广，真实的道路才越安全。BlinkLane 在你自己的电脑上复核 Tesla 行车记录，标出变道时没有观察到转向灯的片段；由人确认之后，证据包留在这台机器上。它不认定某件事违法，不自动举报，默认不上传，也与 Tesla 或警方无关。以 AGPL-3.0-only 发布，别人可以运行，也可以再分享。

A local tool for safer roads (AGPL-3.0-only): review Tesla dashcam clips for lane changes with no observed turn signal, confirm them yourself, and keep the evidence package on the machine.

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
# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg

# Windows (winget)
winget install --id Gyan.FFmpeg -e
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

Synthetic Tesla folders come from `backend/tests/synthetic.py` (`write_tesla_folder`). The end-to-end fixture walks import → analyze → confirm → zip with a mocked detector (no YOLO weights, no real Tesla footage). The analysis smoke test is what CI runs on its own. Keep this block as the only copy of these commands; other docs link here.

```bash
python -m pytest backend/tests/test_e2e_fixture.py -q
python -m pytest backend/tests/test_analysis_smoke.py -q
```

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
- `BLINKLANE_REPORT_URL` (default empty): after you **confirm** an event, the API returns this URL together with the local evidence zip path. The UI opens the URL in a new tab so you can submit through that site’s own page. BlinkLane does not POST to the site, log in, or bypass CAPTCHA.

Tesla filenames look like `2026-05-25_18-30-02-front.mp4`. That clock is **local wall time**, not UTC. `BLINKLANE_CLOCK_OFFSET_MINUTES` (legacy `VCY_CLOCK_OFFSET_MINUTES`, default `0`) is added to the filename clock when a folder is imported. Evidence reports include the resulting absolute time.

Importing the same folder again rescans it and updates segments. Running analysis again replaces events for the segments it analyzes. Jobs left `queued` or `running` when the process stops are marked failed on the next startup. Clip or key-frame failures are written into the job message instead of being stored as empty paths.

## Evidence and upload / 证据包与上传

A confirmed event exports a city-agnostic package: clip, screenshot, absolute time, place, plate, and confirm status. Upload goes through `submit(evidence) -> SubmissionResult`. That call accepts only events the user has confirmed. The default `LocalExportAdapter` leaves the zip on this machine. Optional `BLINKLANE_REPORT_URL` returns that URL plus the local zip path after confirm so you can open the site yourself; BlinkLane does not POST, log in, or bypass CAPTCHA.

已确认的事件会导出城市无关的证据包：片段、截图、绝对时间、地点、号牌、确认状态。上传接口是 `submit(evidence) -> SubmissionResult`，只接受人工确认过的事件。默认的 `LocalExportAdapter` 把压缩包留在本机。可选环境变量 `BLINKLANE_REPORT_URL` 会在确认后把举报入口 URL 与 zip 路径一并返回，由你在浏览器里自行打开网站并上传；这里不代填表单、不登录、不过验证码。

### Validate on your own Tesla folder / 用私有行车记录自测

Install detection extras (`python -m pip install -e '.[yolo]'`), import a private SavedClips folder path in the UI, and treat every hit as unverified until you review it. Do not commit real footage to the repository. Thresholds are not tuned for accuracy on real dashcam video yet.

## Project Status / 项目状态

This repository is in **Public Preview**:

- v0.1 focuses on the local review flow and evidence package export.
- Accuracy is not guaranteed before real-world Tesla Dashcam samples are used to tune thresholds.
- Contributions are welcome, especially around lane-change heuristics, turn-signal detection, privacy-safe sample generation, and Chinese documentation.

## Docs / 文档

- [AGENTS.md](AGENTS.md) — agent entry: layout, run, env, hard limits
- [DISCLAIMER.md](DISCLAIMER.md)
- [ROADMAP.md](ROADMAP.md)
- [SECURITY.md](SECURITY.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/privacy-and-safety.md](docs/privacy-and-safety.md)

## License / 许可证

BlinkLane / 变道灯光复核助手 is released under **GNU AGPL-3.0-only** (`SPDX-License-Identifier: AGPL-3.0-only`). See [LICENSE](LICENSE) for the full AGPL text.

Ultralytics YOLO is an optional dependency, offered under AGPL-3.0 or Enterprise licensing by Ultralytics. If you plan to use BlinkLane in a commercial or closed-source setting, review Ultralytics licensing carefully: https://www.ultralytics.com/license
