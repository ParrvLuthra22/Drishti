import base64
import binascii
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from drishti.perception.detector import Detector
from drishti.perception.publisher import DetectionEvent
from drishti.perception.tracker import Tracker

CAMERA_ID = "api"


class DetectRequest(BaseModel):
    image_b64: str  # base64-encoded JPEG


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.detector = Detector()
    app.state.tracker = Tracker()
    app.state.frame_id = 0
    # The detector and tracker are stateful and shared across requests.
    app.state.lock = threading.Lock()
    yield


app = FastAPI(title="Drishti Perception Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_image(image_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64") from exc
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="image_b64 is not a decodable image")
    return frame


@app.post("/detect", response_model=DetectionEvent)
def detect(request: DetectRequest) -> DetectionEvent:
    frame = decode_image(request.image_b64)

    # Sync endpoint: FastAPI runs it in a worker thread, so inference doesn't block the event loop.
    with app.state.lock:
        start = time.perf_counter()
        detections = app.state.detector.detect(frame)
        tracks = app.state.tracker.update(detections, frame)
        elapsed = time.perf_counter() - start

        frame_id = app.state.frame_id
        app.state.frame_id += 1

    return DetectionEvent.from_tracks(
        camera_id=CAMERA_ID,
        frame_id=frame_id,
        timestamp=time.time(),
        tracks=tracks,
        fps=1.0 / elapsed if elapsed > 0 else 0.0,
    )


@app.get("/health")
def health() -> dict[str, str]:
    detector: Detector = app.state.detector
    return {
        "status": "ok",
        "device": detector.device,
        "model": Path(detector.model_path).stem,
    }
