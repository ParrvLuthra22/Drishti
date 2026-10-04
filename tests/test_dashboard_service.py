import pytest
from fastapi.testclient import TestClient

from drishti.api.dashboard_service import StateStore, app


@pytest.fixture
def client() -> TestClient:
    app.state.store = StateStore(alert_cooldown_s=30.0)
    return TestClient(app)


def risk_state(score: float = 10.0, level: str = "low", alert: bool = False, **extra) -> dict:
    return {
        "camera_id": "cam01",
        "frame_id": 3,
        "timestamp": 1000.0,
        "risk_score": score,
        "risk_level": level,
        "alert_required": alert,
        "incident_summary": "Fight at gate A." if alert else "",
        "recommended_action": "Dispatch security." if alert else "Continue monitoring.",
        "signals": ["Possible fight detected (90% confidence)"] if alert else [],
        **extra,
    }


def push(client: TestClient, state: dict, metrics: dict | None = None, fps: float = 15.0) -> None:
    response = client.post("/api/update", json={"risk_state": state, "metrics": metrics or {}, "fps": fps})
    assert response.status_code == 200


def test_status_before_any_update_is_idle(client: TestClient) -> None:
    body = client.get("/api/status").json()
    assert body["risk_level"] == "low" and body["risk_score"] == 0.0
    assert body["age_s"] is None
    assert client.get("/api/history").json() == {"history": []}
    assert client.get("/api/alerts").json() == []


def test_update_is_served_by_status(client: TestClient) -> None:
    push(client, risk_state(42.5, "moderate", track_count=4, anomaly_level="suspicious"), fps=14.2)
    body = client.get("/api/status").json()
    assert body["risk_score"] == 42.5 and body["risk_level"] == "moderate"
    assert body["track_count"] == 4 and body["anomaly_level"] == "suspicious"
    assert body["fps"] == 14.2
    assert body["age_s"] is not None and body["age_s"] < 5


def test_metrics_fill_fields_missing_from_the_risk_state(client: TestClient) -> None:
    push(client, {"risk_score": 5.0}, metrics={"total_tracks": 7, "density_level": "moderate"})
    body = client.get("/api/status").json()
    assert body["track_count"] == 7 and body["density_level"] == "moderate"


def test_history_keeps_the_last_60(client: TestClient) -> None:
    for i in range(70):
        push(client, risk_state(float(i)))
    history = client.get("/api/history").json()["history"]
    assert len(history) == 60
    assert history[0]["score"] == 10.0 and history[-1]["score"] == 69.0
    assert set(history[0]) == {"t", "score"} and isinstance(history[0]["t"], int)


def test_sustained_alert_is_logged_once(client: TestClient) -> None:
    for _ in range(10):
        push(client, risk_state(60.0, "high", alert=True))
    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 1
    assert set(alerts[0]) == {"timestamp", "camera_id", "risk_score", "risk_level", "summary", "signals"}
    assert alerts[0]["summary"] == "Fight at gate A."


def test_new_alert_after_calm_and_on_escalation(client: TestClient) -> None:
    push(client, risk_state(60.0, "high", alert=True))
    push(client, risk_state(85.0, "critical", alert=True))  # escalation
    push(client, risk_state(5.0))  # calm again
    push(client, risk_state(60.0, "high", alert=True))  # a fresh alert
    alerts = client.get("/api/alerts").json()
    assert [a["risk_level"] for a in alerts] == ["high", "critical", "high"]  # newest first


def test_alerts_capped_at_20_and_reset(client: TestClient) -> None:
    for _ in range(25):
        push(client, risk_state(60.0, "high", alert=True))
        push(client, risk_state(5.0))
    assert len(client.get("/api/alerts").json()) == 20

    assert client.post("/api/reset_alerts").status_code == 200
    assert client.get("/api/alerts").json() == []


def test_cors_allows_the_dashboard_origin(client: TestClient) -> None:
    response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"
