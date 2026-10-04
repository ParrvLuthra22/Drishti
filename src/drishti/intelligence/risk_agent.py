import logging
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from drishti.crowd.analyser import CrowdMetrics

# Read once at import so a local .env supplies ANTHROPIC_API_KEY; later changes to the environment win.
load_dotenv()

logger = logging.getLogger(__name__)

LLM_MODEL = "claude-haiku-4-5"
FIGHT_ALERT_PROBABILITY = 0.5  # above this a fight alerts on its own (and adds a signal)
ALERT_RISK_SCORE = 50.0
DEFAULT_ACTION = "Monitor situation and alert supervisor if conditions worsen."

SYSTEM_PROMPT = """You are a security operations AI for Project Drishti. Generate concise, \
actionable incident summaries for security personnel. Be direct and factual. Under 50 words."""


class DrishtiState(TypedDict):
    # Inputs (populated before graph runs)
    camera_id: str
    timestamp: float
    frame_id: int

    # From perception
    track_count: int

    # From crowd analyser
    density_level: str  # low/moderate/high/critical
    density_score: float
    flow_direction: str  # calm/directional/chaotic
    flow_score: float
    bottleneck_detected: bool
    bottleneck_zones: list[str]

    # From action recognition (VideoMAE)
    fight_probability: float  # 0.0 to 1.0
    fight_detected: bool

    # From anomaly detector
    anomaly_score: float
    anomaly_level: str  # normal/suspicious/anomaly
    is_anomaly: bool

    # Computed by agent
    risk_score: float  # 0.0 to 100.0
    risk_level: str  # low/moderate/high/critical
    alert_required: bool
    incident_summary: str
    recommended_action: str
    signals: list[str]  # human-readable list of what triggered


# ── nodes ───────────────────────────────────────────────────────────────────


def compute_risk_score(state: DrishtiState) -> dict:
    score = 0.0
    signals: list[str] = []

    # Density contribution (max 25 points)
    density_points = {"low": 0, "moderate": 8, "high": 18, "critical": 25}
    score += density_points[state["density_level"]]
    if state["density_level"] in ("high", "critical"):
        signals.append(f"Crowd density: {state['density_level']}")

    # Flow contribution (max 20 points)
    flow_points = {"calm": 0, "directional": 5, "chaotic": 20}
    score += flow_points[state["flow_direction"]]
    if state["flow_direction"] == "chaotic":
        signals.append("Chaotic crowd movement detected")

    # Bottleneck contribution (max 15 points)
    if state["bottleneck_detected"]:
        score += 15
        signals.append(f"Bottleneck at: {', '.join(state['bottleneck_zones'])}")

    # Fight detection contribution (max 30 points)
    score += state["fight_probability"] * 30
    if state["fight_probability"] > FIGHT_ALERT_PROBABILITY:
        signals.append(f"Possible fight detected ({state['fight_probability']:.0%} confidence)")

    # Anomaly contribution (max 10 points)
    anomaly_points = {"normal": 0, "suspicious": 5, "anomaly": 10}
    score += anomaly_points[state["anomaly_level"]]
    if state["is_anomaly"]:
        signals.append("Scene anomaly detected")

    # Clamp and classify
    score = min(score, 100.0)
    if score < 25:
        risk_level = "low"
    elif score < 50:
        risk_level = "moderate"
    elif score < 75:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "risk_score": round(score, 1),
        "risk_level": risk_level,
        # A fight is 30 points at most, so on the score alone it could never reach the alert
        # threshold; a likely fight must raise an alert by itself.
        "alert_required": score >= ALERT_RISK_SCORE or state["fight_probability"] > FIGHT_ALERT_PROBABILITY,
        "signals": signals,
    }


def _message_text(content: object) -> str:
    """Chat model content is a string, or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", "")) for block in content
        )
    return str(content)


def _clean(text: str) -> str:
    text = text.replace("**", "").strip()
    text = re.sub(r"^\s*(?:\d+[.)]\s*)?(?:(?:incident\s+)?summary\s*:\s*)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\d+[.)]\s*$", "", text)  # a dangling "2)" left before the marker
    return text.strip()


def _parse_summary(text: str) -> tuple[str, str]:
    """Split a model reply into (incident summary, recommended action).

    Handles a "Recommended ...:" marker ("2) Recommended immediate action: ..."), or falls back to
    first line / remaining lines.
    """
    text = text.replace("**", "").strip()
    marker = re.search(r"recommended[^:\n]*:", text, flags=re.IGNORECASE)
    if marker:
        summary, action = text[: marker.start()], text[marker.end() :]
    else:
        lines = [line for line in text.splitlines() if line.strip()]
        summary, action = (lines[0], " ".join(lines[1:])) if lines else ("", "")
    return _clean(summary), _clean(action)


def generate_summary(state: DrishtiState) -> dict:
    human = (
        f"Camera: {state['camera_id']}\n"
        f"Risk Score: {state['risk_score']}/100 ({state['risk_level']})\n"
        f"Active signals: {', '.join(state['signals'])}\n"
        f"People tracked: {state['track_count']}\n\n"
        "Generate: 1) One-sentence incident summary\n"
        "          2) Recommended immediate action"
    )

    try:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        llm = ChatAnthropic(model=LLM_MODEL, max_tokens=200, temperature=0, timeout=15, max_retries=1)
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)])
        incident_summary, recommended_action = _parse_summary(_message_text(response.content))
        if not incident_summary:
            raise ValueError("model returned an empty summary")
        recommended_action = recommended_action or DEFAULT_ACTION
    except Exception as exc:  # noqa: BLE001 - an LLM outage must never stop risk scoring
        logger.warning("LLM summary unavailable (%s); using rule-based fallback", exc)
        incident_summary = (
            f"Risk level {state['risk_level']} detected at {state['camera_id']}. "
            f"Signals: {', '.join(state['signals']) or 'none'}"
        )
        recommended_action = DEFAULT_ACTION

    return {"incident_summary": incident_summary, "recommended_action": recommended_action}


def skip_summary(state: DrishtiState) -> dict:
    return {"incident_summary": "", "recommended_action": "Continue monitoring."}


# ── routing and graph ───────────────────────────────────────────────────────


def route_after_risk(state: DrishtiState) -> str:
    if state["alert_required"]:
        return "generate_summary"
    return "skip_summary"


builder = StateGraph(DrishtiState)
builder.add_node("compute_risk", compute_risk_score)
builder.add_node("generate_summary", generate_summary)
builder.add_node("skip_summary", skip_summary)
builder.set_entry_point("compute_risk")
builder.add_conditional_edges(
    "compute_risk",
    route_after_risk,
    {"generate_summary": "generate_summary", "skip_summary": "skip_summary"},
)
builder.add_edge("generate_summary", END)
builder.add_edge("skip_summary", END)

risk_graph = builder.compile()


# ── public API ──────────────────────────────────────────────────────────────

NORMAL_ANOMALY = {"smoothed_score": 0.0, "anomaly_level": "normal", "is_anomaly": False}


class RiskAgent:
    def __init__(self) -> None:
        self.graph = risk_graph

    def assess(
        self,
        camera_id: str,
        frame_id: int,
        timestamp: float,
        crowd_metrics: CrowdMetrics,
        fight_probability: float = 0.0,
        anomaly_result: dict | None = None,
    ) -> DrishtiState:
        """Fuse crowd, fight and anomaly signals into a risk assessment.

        `anomaly_result` is the dict from AnomalyScorer.score_frame(); None means all normal.
        """
        anomaly = anomaly_result if anomaly_result is not None else NORMAL_ANOMALY
        fight_probability = min(max(float(fight_probability), 0.0), 1.0)

        initial: DrishtiState = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "track_count": crowd_metrics.total_tracks,
            "density_level": crowd_metrics.density_level,
            "density_score": crowd_metrics.density_score,
            "flow_direction": crowd_metrics.flow_direction,
            "flow_score": crowd_metrics.flow_score,
            "bottleneck_detected": crowd_metrics.bottleneck_detected,
            "bottleneck_zones": list(crowd_metrics.bottleneck_zones),
            "fight_probability": fight_probability,
            "fight_detected": fight_probability > FIGHT_ALERT_PROBABILITY,
            "anomaly_score": float(anomaly.get("smoothed_score", anomaly.get("raw_score", 0.0))),
            "anomaly_level": anomaly["anomaly_level"],
            "is_anomaly": bool(anomaly["is_anomaly"]),
            "risk_score": 0.0,
            "risk_level": "low",
            "alert_required": False,
            "incident_summary": "",
            "recommended_action": "",
            "signals": [],
        }
        return self.graph.invoke(initial)
