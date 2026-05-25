# Contributing / 参与贡献

Thanks for helping BlinkLane. This project is in public preview, so practical testing and careful bug reports are especially valuable.

欢迎参与 BlinkLane。当前是公开预览版，真实使用反馈、阈值调优建议、文档改进都很有价值。

## Good First Areas

- Tesla Dashcam folder parsing edge cases.
- Lane-change heuristics and false-positive reduction.
- Turn-signal blink detection under night/rain/reflection conditions.
- Privacy-safe synthetic test videos.
- Chinese documentation and UI copy.
- Export package structure for different local reporting workflows.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,yolo]"
npm install
python -m pytest
python -m ruff check .
npm run build
```

## Pull Request Rules

- Do not commit real dashcam videos, license plates, faces, GPS traces, or other private evidence.
- Keep generated files out of git: `.vcy_data/`, `.venv/`, `node_modules/`, `dist/`, and model weights.
- Make safety wording explicit when adding automation around reports or exports.
- Add or update tests when changing parsing, scoring, export, or API behavior.

## Licensing

By contributing, you agree that your contribution is licensed under AGPL-3.0-only.
