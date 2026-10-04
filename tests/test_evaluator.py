from unittest.mock import MagicMock

import pytest

from drishti.intelligence.evaluator import DrishtiEvaluator

RISK_STATE = {
    "fight_probability": 0.85,
    "density_level": "high",
    "anomaly_level": "anomaly",
    "track_count": 15,
    "risk_score": 88.5,
    "risk_level": "critical",
    "alert_required": True,
    "incident_summary": "Possible fight at gate A.",
    "signals": ["not logged"],
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_PROJECT"):
        monkeypatch.delenv(name, raising=False)


def runs_created(client: MagicMock) -> list[dict]:
    """Every run posted through client.create_run, as keyword dicts."""
    return [call.kwargs for call in client.create_run.call_args_list]


def test_disabled_without_an_api_key() -> None:
    evaluator = DrishtiEvaluator()
    assert evaluator.enabled is False and evaluator.client is None
    evaluator.start_session("cam01")
    evaluator.log_assessment(RISK_STATE)
    evaluator.end_session(100, 5)  # all no-ops, nothing raised
    assert evaluator.run_id is None


def test_placeholder_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "your_key_here")
    assert DrishtiEvaluator().enabled is False


def test_project_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DrishtiEvaluator().project == "drishti-evals"
    monkeypatch.setenv("LANGCHAIN_PROJECT", "my-project")
    assert DrishtiEvaluator().project == "my-project"


def test_start_session_creates_a_named_chain_run() -> None:
    client = MagicMock()
    evaluator = DrishtiEvaluator(client=client)
    evaluator.start_session("cam01")

    (root,) = runs_created(client)
    assert root["name"] == "drishti-cam01"
    assert root["run_type"] == "chain"
    assert root["inputs"] == {"camera_id": "cam01"}
    assert evaluator.run_id is not None and str(evaluator.run_id) == str(root["id"])


def test_log_assessment_is_a_child_of_the_session_run() -> None:
    client = MagicMock()
    evaluator = DrishtiEvaluator(client=client)
    evaluator.start_session("cam01")
    evaluator.log_assessment(RISK_STATE)

    root, child = runs_created(client)
    assert child["name"] == "risk_assessment"
    assert str(child["parent_run_id"]) == str(root["id"])
    assert str(child["trace_id"]) == str(root["id"])  # same trace as the session
    assert child["inputs"] == {
        "fight_probability": 0.85,
        "density_level": "high",
        "anomaly_level": "anomaly",
        "track_count": 15,
    }
    assert child["outputs"] == {
        "risk_score": 88.5,
        "risk_level": "critical",
        "alert_required": True,
        "incident_summary": "Possible fight at gate A.",
    }


def test_log_assessment_without_a_session_does_nothing() -> None:
    client = MagicMock()
    DrishtiEvaluator(client=client).log_assessment(RISK_STATE)
    client.create_run.assert_not_called()


def test_end_session_reports_totals_and_alert_rate() -> None:
    client = MagicMock()
    evaluator = DrishtiEvaluator(client=client)
    evaluator.start_session("cam01")
    evaluator.end_session(total_frames=200, total_alerts=5)

    update = client.update_run.call_args
    assert update.kwargs["outputs"] == {"total_frames": 200, "total_alerts": 5, "alert_rate": 0.025}
    client.flush.assert_called_once()
    assert evaluator.run_id is None

    evaluator.log_assessment(RISK_STATE)  # the session is over, so this is ignored
    assert len(runs_created(client)) == 1


def test_end_session_with_zero_frames_does_not_divide_by_zero() -> None:
    client = MagicMock()
    evaluator = DrishtiEvaluator(client=client)
    evaluator.start_session("cam01")
    evaluator.end_session(total_frames=0, total_alerts=0)
    assert client.update_run.call_args.kwargs["outputs"]["alert_rate"] == 0.0


def test_langsmith_failures_never_propagate() -> None:
    client = MagicMock()
    client.create_run.side_effect = ConnectionError("LangSmith is down")
    client.update_run.side_effect = ConnectionError("LangSmith is down")
    evaluator = DrishtiEvaluator(client=client)
    evaluator.start_session("cam01")  # must not raise
    evaluator.log_assessment(RISK_STATE)
    evaluator.end_session(10, 1)
