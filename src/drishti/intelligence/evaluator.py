import logging
import os
import threading
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_trees import RunTree

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "drishti-evals"
PLACEHOLDER_KEY = "your_key_here"  # the value shipped in .env.example


class DrishtiEvaluator:
    """Records a monitoring session in LangSmith: one run per session, one child run per assessment.

    Does nothing (and sends nothing) unless LANGSMITH_API_KEY is set to a real key. LangSmith
    problems are logged and swallowed, since evaluation must never interrupt the live pipeline.
    """

    def __init__(self, client: Client | None = None) -> None:
        api_key = os.environ.get("LANGSMITH_API_KEY", "")
        self.project = os.environ.get("LANGCHAIN_PROJECT", DEFAULT_PROJECT)
        self.enabled = client is not None or bool(api_key and api_key != PLACEHOLDER_KEY)
        self.client: Client | None = client
        if self.enabled and self.client is None:
            self.client = Client(api_key=api_key)
        self.run_id = None
        self._root: RunTree | None = None
        self._lock = threading.Lock()  # assessments are logged from the pipeline's worker thread

    def start_session(self, camera_id: str) -> None:
        if not self.enabled:
            logger.info("LangSmith evaluation disabled (LANGSMITH_API_KEY not set)")
            return
        try:
            root = RunTree(
                name=f"drishti-{camera_id}",
                run_type="chain",
                inputs={"camera_id": camera_id},
                project_name=self.project,
                client=self.client,
            )
            root.post()
        except Exception as exc:  # noqa: BLE001 - evaluation must never stop the pipeline
            logger.warning("could not start LangSmith session: %s", exc)
            return
        with self._lock:
            self._root = root
            self.run_id = root.id

    def log_assessment(self, risk_state: dict[str, Any]) -> None:
        with self._lock:
            root = self._root
        if root is None:
            return
        try:
            child = root.create_child(
                name="risk_assessment",
                run_type="chain",
                inputs={
                    "fight_probability": risk_state.get("fight_probability"),
                    "density_level": risk_state.get("density_level"),
                    "anomaly_level": risk_state.get("anomaly_level"),
                    "track_count": risk_state.get("track_count"),
                },
            )
            child.end(
                outputs={
                    "risk_score": risk_state.get("risk_score"),
                    "risk_level": risk_state.get("risk_level"),
                    "alert_required": risk_state.get("alert_required"),
                    "incident_summary": risk_state.get("incident_summary"),
                }
            )
            child.post()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not log assessment to LangSmith: %s", exc)

    def end_session(self, total_frames: int, total_alerts: int) -> None:
        with self._lock:
            root, self._root, self.run_id = self._root, None, None
        if root is None:
            return
        try:
            root.end(
                outputs={
                    "total_frames": total_frames,
                    "total_alerts": total_alerts,
                    "alert_rate": total_alerts / total_frames if total_frames else 0.0,
                }
            )
            root.patch()
            self.client.flush()  # queued runs are sent by a background thread; make sure they are out
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not end LangSmith session: %s", exc)
