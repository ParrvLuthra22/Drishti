# Project Drishti 🎯

> AI-powered situational awareness system —
> real-time crowd analysis, fight detection,
> and anomaly detection on live camera feeds.

## What it does

Drishti watches a camera feed, finds and tracks every person, and measures how the crowd is behaving: how dense it is, whether it is moving calmly or chaotically, and where it is bottlenecking. A VideoMAE model looks for fights and a convolutional autoencoder flags scenes that stop looking normal. A LangGraph agent fuses these signals into a 0–100 risk score and, when an alert is warranted, asks an LLM for a short incident summary and a recommended action. Everything streams to a live Next.js command-center dashboard.

## Architecture

```
Camera → Detector → Tracker → CrowdAnalyser
 (OpenCV) (YOLOv8n) (ByteTrack) (density · flow · zones · heatmap)
                                    │
                                    ▼
              VideoMAE ───────→ RiskAgent (LangGraph) ──→ Claude Haiku summary
              (fight prob.)         ▲        │                (rule-based fallback)
              Autoencoder ──────────┘        ▼
              (scene anomaly)         Dashboard API (FastAPI :8002)
                                             │
                                             ▼
                                    Dashboard (Next.js :3000)
```

The perception loop (detect → track → analyse) runs on the main thread at about 15 FPS. Fight inference, risk assessment, the LLM call and the dashboard push run on a background thread once per second, so they never stall the video.

## Results

| Component | Metric | Value |
|-----------|--------|-------|
| Person detection | mAP@0.5 | YOLOv8n, COCO-pretrained (not re-evaluated here) |
| Fight detection | Val F1 (macro) | 88.74% on the 400 RWF-2000 validation clips |
| Anomaly detection | False positive rate | 0% on 255 live frames of one scene, right after calibration (see limitations) |
| Perception | Live FPS | ~15 fps (Apple Silicon, MPS); ~11–12 with `--fight` |

Fight-detection details are in [`docs/phase4_results.md`](docs/phase4_results.md). The model is published at `Parrv/drishti-fight-detector` on the Hugging Face Hub.

## Setup

```bash
git clone https://github.com/ParrvLuthra22/drishti
cd drishti
uv sync
cp .env.example .env
# Add your API keys to .env
```

- `ANTHROPIC_API_KEY` enables LLM incident summaries. Without it the agent uses a rule-based summary.
- `LANGSMITH_API_KEY` enables LangSmith evaluation and tracing. If you are not using LangSmith, set `LANGCHAIN_TRACING_V2=false` in `.env`.
- The anomaly model is trained on your own camera and is not in the repo: run `uv run python scripts/train_anomaly.py --source 1` once (about 4 minutes of normal scene). Without it, anomaly scoring is skipped.
- `--fight` downloads the fight model (~345 MB) from the Hugging Face Hub on first use.

## Run

```bash
# Start Redis (optional: only needed for the Redis event publisher,
# which the demo does not use yet)
docker compose up redis -d

# Start dashboard API
uv run python scripts/run_dashboard_api.py

# Start Next.js dashboard
cd dashboard && npm install && npm run dev

# Start perception pipeline
uv run python scripts/demo_yolo_webcam.py \
  --source 1 --fight
```

Open http://localhost:3000. `--source` is a webcam index, a video file or an RTSP URL (default `1`, the laptop webcam on the development machine). Add `--heatmap` to blend the crowd heatmap onto the video, and `--no-visualize` to run headless.

## Tech stack

| Component | Technology |
|-----------|------------|
| Language and tooling | Python 3.12, uv, ruff, pytest |
| ML runtime | PyTorch with Apple MPS, torchvision, OpenCV, NumPy |
| Person detection | YOLOv8n (Ultralytics) |
| Tracking | ByteTrack (supervision) |
| Crowd analytics | Track-based density, flow and bottleneck analysis, heatmap (NumPy, OpenCV) |
| Fight detection | VideoMAE-base fine-tuned on RWF-2000 (Hugging Face Transformers) |
| Anomaly detection | Convolutional autoencoder (PyTorch) |
| Risk fusion | LangGraph, Claude Haiku 4.5 via langchain-anthropic |
| Evaluation | LangSmith |
| Backend | FastAPI, Uvicorn, Pydantic, Redis Streams event publisher |
| Dashboard | Next.js 16, React 19, Tailwind CSS 4, Recharts |
| Infrastructure | Docker Compose (Redis) |

## Project layout

```
src/drishti/
  perception/    detector, tracker, publisher (Redis), PerceptionWorker loop
  crowd/         CrowdAnalyser (density, flow, zones, bottlenecks), heatmap
  actions/       RWF-2000 dataset, VideoMAE trainer, FightDetector
  anomaly/       autoencoder, trainer, scorer
  intelligence/  RiskAgent (LangGraph), LangSmith evaluator
  api/           inference service, dashboard service
dashboard/       Next.js command center
scripts/         demos, training and verification scripts
notebooks/       Colab notebook used to fine-tune VideoMAE
```

## Phase progress

- [x] Phase 1 — Foundation (PyTorch, MPS, YOLO)
- [x] Phase 2 — Perception (YOLOv8 + ByteTrack + Redis)
- [x] Phase 3 — Crowd intelligence (density, flow, heatmap)
- [x] Phase 4 — Action recognition (VideoMAE 88.74% F1)
- [x] Phase 5 — Anomaly detection (Conv-Autoencoder)
- [x] Phase 6 — Intelligence layer (LangGraph + Dashboard)

## Known limitations

- **Anomaly detector is weak.** It reconstructs frames of the scene it was trained on, so lighting, framing or who is in view changes it easily. The 0% false-positive figure was measured on a single scene right after calibration. In later runs it flagged the scene as anomalous almost constantly, which adds 10 points to the risk score. It needs a longer, more varied training capture.
- **Fight detection is untested on live fights.** It scores 91.7% on validation clips, but one person waving their arms at the webcam does not register (probability stayed at 0). Live frames are sampled every 0.3 s to match training, so `--fight` only suits live cameras, not video files.
- **`--fight` costs frame rate** (about 15 to 11–12 FPS) because it shares the GPU with the detector.
- **The dashboard API is unauthenticated** and accepts any origin; it listens on `127.0.0.1` only.
- The Redis event publisher exists but is not wired into the demo pipeline.

## Test suite

76 tests, all passing.
`uv run pytest tests/ -v`
