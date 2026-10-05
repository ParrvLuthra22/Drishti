![CI](https://github.com/ParrvLuthra22/Drishti/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Project Drishti 🎯

> AI-powered situational awareness system —
> real-time crowd analysis, fight detection,
> and anomaly detection on live camera feeds.

<!-- Demo GIF coming soon -->

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

| Component | Metric | Value | Context |
|-----------|--------|-------|---------|
| Person detection | Speed | 15 FPS live | YOLOv8n on Apple M3 MPS |
| Fight detection | Val F1 | **88.74%** | Fine-tuned VideoMAE-base on RWF-2000; SOTA with larger models ~95% |
| Fight detection | Training time | 62.5 min | Free Colab T4 GPU |
| Tracking | ID persistence | ByteTrack | Maintains person IDs across full video duration |
| Anomaly detection | Threshold | Held-out calibration | mean + 3σ on 200 unseen frames |
| System | Test coverage | 76 tests | Unit + integration, all passing |

Fight-detection details are in [`docs/phase4_results.md`](docs/phase4_results.md) and the [model card](docs/model_card.md). The model is published at `Parrv/drishti-fight-detector` on the Hugging Face Hub.

## Technical challenges

**Two-phase VideoMAE fine-tuning**
Directly fine-tuning all 86M parameters on 1600 clips destroys pre-trained features. Phase A freezes the backbone and trains only the 2-class head for 1 epoch. Phase B unfreezes everything with a 100× lower learning rate for the backbone (1e-5) vs head (1e-3) and cosine annealing. F1 jumps from 0.62 → 0.86 between epoch 1 and 2.

**Frame sampling for temporal alignment**
VideoMAE was trained on 16 frames spanning ~5 seconds. Feeding 16 consecutive webcam frames (~1 second) gives 50% accuracy — equivalent to random. Sampling one frame every 0.3 seconds to span the correct temporal window restores full model performance.

**Anomaly threshold calibration**
Training an autoencoder on consecutive frames causes memorization. Reconstruction error on training frames (~0.0004) is 7× lower than on fresh frames from the same scene (~0.003). Solution: compute threshold on a held-out set of 200 frames never seen during training. Formula: mean + 3σ of held-out error.

**MPS + multiprocessing constraint**
PyTorch's MPS backend on Apple Silicon doesn't support DataLoader workers (num_workers > 0 causes silent hangs). All DataLoaders in the project use num_workers=0, with the pipeline loop compensating via async background threads for the slower operations (fight inference, risk assessment, LLM calls, dashboard push).

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

## Known limitations and production gaps

**Anomaly detector needs longer training data**
The autoencoder was trained on 2000 frames of a single scene. Production deployment requires 10,000+ frames captured across varied lighting, camera angles, and times of day, with augmentation and early stopping to prevent memorization. The current model is a proof-of-concept for the anomaly scoring pipeline.

**Fight detection is domain-specific**
The model achieves 88.74% F1 on RWF-2000 surveillance clips but does not generalize to close-up webcam footage — a single person waving does not trigger detection. This is expected: the training distribution is overhead CCTV footage of multiple people. Deployment on overhead cameras would match the training domain.

**Dashboard API has no authentication**
The FastAPI service is intentionally unauthenticated for local development. Production deployment requires API key auth, rate limiting, and HTTPS.

**Redis event publisher not wired into demo**
The publisher exists and is tested in isolation but is not connected to the live demo pipeline. Connecting it is a one-line change; it was excluded to keep the demo dependencies minimal.

## Test suite

76 tests, all passing.
`uv run pytest tests/ -v`

## Phase progress

- [x] Phase 1 — Foundation (PyTorch, MPS, YOLO)
- [x] Phase 2 — Perception (YOLOv8 + ByteTrack + Redis)
- [x] Phase 3 — Crowd intelligence (density, flow, heatmap)
- [x] Phase 4 — Action recognition (VideoMAE 88.74% F1)
- [x] Phase 5 — Anomaly detection (Conv-Autoencoder)
- [x] Phase 6 — Intelligence layer (LangGraph + Dashboard)
