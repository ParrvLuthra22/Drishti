# Drishti

Crowd perception, anomaly detection and intelligence pipeline.

## Setup

```
uv sync
```

## Dashboard

A Next.js command center polls a small FastAPI service that the live pipeline pushes to.

```
uv run python scripts/run_dashboard_api.py                      # API on :8002
(cd dashboard && npm install && npm run dev)                    # UI on :3000
uv run python scripts/demo_yolo_webcam.py --source 1            # pipeline -> dashboard
```

Open http://localhost:3000. Set `NEXT_PUBLIC_API_URL` to point the UI at a different API host.
