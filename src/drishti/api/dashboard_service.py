import threading
import time
from collections import deque
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HISTORY_LEN = 60
ALERT_LEN = 20
ALERT_COOLDOWN_S = 30.0  # a sustained alert is re-logged at most this often (escalations log at once)
RISK_RANK = {"low": 0, "moderate": 1, "high": 2, "critical": 3}

# Fields served by /api/status, taken from the risk agent's state, with the crowd metrics as fallback.
STATUS_DEFAULTS: dict[str, Any] = {
    "camera_id": "",
    "frame_id": 0,
    "timestamp": 0.0,
    "risk_score": 0.0,
    "risk_level": "low",
    "alert_required": False,
    "incident_summary": "",
    "recommended_action": "",
    "signals": [],
    "track_count": 0,
    "density_level": "low",
    "density_score": 0.0,
    "flow_direction": "calm",
    "flow_score": 0.0,
    "bottleneck_detected": False,
    "fight_probability": 0.0,
    "anomaly_level": "normal",
    "anomaly_score": 0.0,
}
METRIC_FALLBACKS = {"track_count": "total_tracks"}  # status field -> CrowdMetrics field


class StateStore:
    """In-memory current state, risk history and alert log. Thread-safe."""

    def __init__(
        self,
        history_len: int = HISTORY_LEN,
        alert_len: int = ALERT_LEN,
        alert_cooldown_s: float = ALERT_COOLDOWN_S,
    ) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = dict(STATUS_DEFAULTS, fps=0.0)
        self._updated_at: float | None = None  # monotonic time of the last update
        self._history: deque[dict[str, Any]] = deque(maxlen=history_len)
        self._alerts: deque[dict[str, Any]] = deque(maxlen=alert_len)
        self._alert_cooldown_s = alert_cooldown_s
        self._alert_active = False
        self._last_alert_level = "low"
        self._last_alert_at = 0.0

    def update(self, risk_state: dict[str, Any], metrics: dict[str, Any], fps: float) -> None:
        state: dict[str, Any] = {}
        for key, default in STATUS_DEFAULTS.items():
            if key in risk_state:
                state[key] = risk_state[key]
            elif METRIC_FALLBACKS.get(key, key) in metrics:
                state[key] = metrics[METRIC_FALLBACKS.get(key, key)]
            else:
                state[key] = default
        state["fps"] = float(fps)

        now = time.monotonic()
        with self._lock:
            self._state = state
            self._updated_at = now
            self._history.append(
                {"t": int(state["timestamp"] or time.time()), "score": float(state["risk_score"])}
            )
            self._record_alert(state, now)

    def _record_alert(self, state: dict[str, Any], now: float) -> None:
        """Log an alert when one starts, escalates, or has been going on for a while."""
        if not state["alert_required"]:
            self._alert_active = False
            return

        escalated = RISK_RANK.get(state["risk_level"], 0) > RISK_RANK.get(self._last_alert_level, 0)
        due = now - self._last_alert_at >= self._alert_cooldown_s
        if not self._alert_active or escalated or due:
            self._alerts.append(
                {
                    "timestamp": state["timestamp"] or time.time(),
                    "camera_id": state["camera_id"],
                    "risk_score": state["risk_score"],
                    "risk_level": state["risk_level"],
                    "summary": state["incident_summary"],
                    "signals": list(state["signals"]),
                }
            )
            self._last_alert_level = state["risk_level"]
            self._last_alert_at = now
        self._alert_active = True

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = None if self._updated_at is None else round(time.monotonic() - self._updated_at, 1)
            return {**self._state, "age_s": age}

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def alerts(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._alerts))  # newest first

    def reset_alerts(self) -> None:
        with self._lock:
            self._alerts.clear()
            self._alert_active = False
            self._last_alert_level = "low"
            self._last_alert_at = 0.0


class UpdatePayload(BaseModel):
    risk_state: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    fps: float = 0.0


app = FastAPI(title="Drishti Dashboard API")
app.state.store = StateStore()
# The dashboard dev server runs on a different origin (localhost:3000).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/update")
def update(payload: UpdatePayload) -> dict[str, str]:
    app.state.store.update(payload.risk_state, payload.metrics, payload.fps)
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return app.state.store.status()


@app.get("/api/history")
def history() -> dict[str, list[dict[str, Any]]]:
    return {"history": app.state.store.history()}


@app.get("/api/alerts")
def alerts() -> list[dict[str, Any]]:
    return app.state.store.alerts()


@app.post("/api/reset_alerts")
def reset_alerts() -> dict[str, str]:
    app.state.store.reset_alerts()
    return {"status": "ok"}
