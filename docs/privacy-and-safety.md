# Privacy and Safety / 隐私与安全

BlinkLane is designed as a local-first tool.

BlinkLane 默认本地优先运行。

## Local Data

- Imported videos are read from a pasted local path or from volume scan. Scan roots are defined in `backend/app/volumes.py` (`/Volumes`, `/media`, `/run/media`, and Windows drive letters). Set `BLINKLANE_VOLUME_SCAN=0` to disable background scanning.
- Generated clips, frames, metadata, and exports are stored under `.blinklane_data/` (or an existing `.vcy_data/` directory).
- The app does not require an account or cloud backend.

## Public Issue Hygiene

Do not upload real dashcam files to GitHub issues. Use synthetic clips, blurred screenshots, or textual descriptions.

不要把真实行车记录仪视频上传到 GitHub issue。请使用合成视频、打码截图或文字描述。

## Reporting Workflow

The project intentionally stops at evidence preparation and human review. Any actual report should be submitted manually by the user after checking local rules.
