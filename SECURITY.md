# Security Policy / 安全政策

## Supported Versions

BlinkLane is currently a public preview. Security fixes target the latest commit on `main`.

## Threat Model

- The API is unauthenticated and binds to loopback by convention. Never run it with `--host 0.0.0.0`; anyone who can reach the port can read review data and trigger exports.
- Media endpoints only serve paths registered in the local database (imported clip paths and event assets). `/api/media/source` rejects unregistered absolute paths, and `/api/media/{path}` is confined to the data directory.
- Model weight names are validated before reaching the detector so a request cannot point Ultralytics at arbitrary paths or URLs.

## Reporting a Vulnerability

Please do not open public issues for vulnerabilities that expose private video, local paths, or personal data.

If this repository has GitHub private vulnerability reporting enabled, use that. Otherwise, open a minimal issue that says you have a security report without including sensitive details.

## Privacy Expectations

- Do not attach real dashcam footage to public issues.
- Blur license plates, faces, road signs near home/work, and GPS traces before sharing screenshots.
- Prefer synthetic videos or tiny reproducible fixtures.
