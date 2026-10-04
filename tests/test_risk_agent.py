from types import SimpleNamespace
from unittest.mock import patch

import pytest

from drishti.crowd.analyser import CrowdMetrics
from drishti.intelligence.risk_agent import (
    LLM_MODEL,
    RiskAgent,
    _parse_summary,
)

CHATANTHROPIC = "drishti.intelligence.risk_agent.ChatAnthropic"
ANOMALY = {"smoothed_score": 0.02, "anomaly_level": "anomaly", "is_anomaly": True}


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach the real API, even if the developer has a key in their environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def make_metrics(
    tracks: int = 1,
    density: str = "low",
    density_score: float = 0.05,
    flow: str = "calm",
    bottleneck_zones: list[str] | None = None,
) -> CrowdMetrics:
    return CrowdMetrics(
        camera_id="cam01",
        timestamp=1000.0,
        frame_id=7,
        total_tracks=tracks,
        density_level=density,
        density_score=density_score,
        avg_speed=0.0,
        flow_direction=flow,
        flow_score=0.0,
        bottleneck_detected=bool(bottleneck_zones),
        bottleneck_zones=bottleneck_zones or [],
    )


def assess(metrics: CrowdMetrics, **kwargs):
    return RiskAgent().assess("cam01", 7, 1000.0, metrics, **kwargs)


def high_risk_metrics(bottleneck: bool = True) -> CrowdMetrics:
    # 15 people is density 0.75 -> "high" (18 pts); chaotic flow is 20 pts.
    return make_metrics(
        tracks=15,
        density="high",
        density_score=0.75,
        flow="chaotic",
        bottleneck_zones=["gate_a"] if bottleneck else None,
    )


def test_low_risk_scenario() -> None:
    state = assess(make_metrics())
    assert state["risk_score"] < 25
    assert state["risk_level"] == "low"
    assert state["alert_required"] is False
    assert state["incident_summary"] == ""
    assert state["recommended_action"] == "Continue monitoring."


def test_high_risk_scenario() -> None:
    # 18 density + 20 chaotic + 15 bottleneck + 25.5 fight + 10 anomaly = 88.5
    state = assess(high_risk_metrics(), fight_probability=0.85, anomaly_result=ANOMALY)
    assert state["risk_score"] >= 75
    assert state["risk_level"] in ("high", "critical")
    assert state["alert_required"] is True
    assert state["incident_summary"]  # rule-based fallback, since the API key is unset
    assert "Bottleneck at: gate_a" in state["signals"]


def test_risk_score_is_the_weighted_sum() -> None:
    # Same scenario without the bottleneck: 18 + 20 + 25.5 + 10 = 73.5, which is only "high".
    state = assess(high_risk_metrics(bottleneck=False), fight_probability=0.85, anomaly_result=ANOMALY)
    assert state["risk_score"] == 73.5
    assert state["risk_level"] == "high"


def test_fight_alone_triggers_alert() -> None:
    state = assess(make_metrics(), fight_probability=0.9)
    assert state["risk_score"] == 27.0  # fight alone is 27 points, below the score threshold of 50
    assert state["risk_level"] == "moderate"
    assert state["alert_required"] is True
    assert state["fight_detected"] is True
    assert any("Possible fight detected (90% confidence)" in s for s in state["signals"])


def test_weak_fight_probability_does_not_alert() -> None:
    state = assess(make_metrics(), fight_probability=0.4)
    assert state["alert_required"] is False
    assert state["fight_detected"] is False


def test_summary_generation_uses_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reply = (
        "1) Possible fight with chaotic movement near gate A.\n"
        "2) Recommended immediate action: Dispatch two officers to gate A."
    )
    with patch(CHATANTHROPIC) as chat:
        chat.return_value.invoke.return_value = SimpleNamespace(content=reply)
        state = assess(high_risk_metrics(), fight_probability=0.85, anomaly_result=ANOMALY)

    chat.assert_called_once()
    assert chat.call_args.kwargs["model"] == LLM_MODEL
    assert state["incident_summary"] == "Possible fight with chaotic movement near gate A."
    assert state["recommended_action"] == "Dispatch two officers to gate A."

    # The prompt sent to the model carries the camera, score and signals.
    human = chat.return_value.invoke.call_args.args[0][1].content
    assert "Camera: cam01" in human and "Bottleneck at: gate_a" in human


def test_fallback_when_api_key_missing() -> None:
    with patch(CHATANTHROPIC) as chat:
        state = assess(high_risk_metrics(), fight_probability=0.85, anomaly_result=ANOMALY)  # must not raise
    chat.assert_not_called()
    assert "cam01" in state["incident_summary"]
    assert "Signals:" in state["incident_summary"]
    assert state["recommended_action"].startswith("Monitor situation")


def test_fallback_when_llm_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch(CHATANTHROPIC) as chat:
        chat.return_value.invoke.side_effect = TimeoutError("API timed out")
        state = assess(high_risk_metrics(), fight_probability=0.85, anomaly_result=ANOMALY)
    assert "cam01" in state["incident_summary"]
    assert state["alert_required"] is True


def test_no_llm_call_when_no_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch(CHATANTHROPIC) as chat:
        assess(make_metrics())
    chat.assert_not_called()


@pytest.mark.parametrize(
    ("reply", "summary", "action"),
    [
        ("Fight at gate A.\nEvacuate the area.", "Fight at gate A.", "Evacuate the area."),
        (
            "**Summary:** Fight at gate A. **Recommended action:** Send security.",
            "Fight at gate A.",
            "Send security.",
        ),
        ("1) Fight at gate A. 2) Recommended: Send security.", "Fight at gate A.", "Send security."),
        ("Fight at gate A.", "Fight at gate A.", ""),
    ],
)
def test_parse_summary_formats(reply: str, summary: str, action: str) -> None:
    assert _parse_summary(reply) == (summary, action)
