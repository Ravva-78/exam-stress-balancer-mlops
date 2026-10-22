"""Tests for the FastAPI serving layer (JSON API + health)."""

import pytest
from fastapi.testclient import TestClient

from src.api import main

CLIENT_PAYLOAD = {
    "stress_level": "high", "hours_studied": 6,
    "days_until_exam": 3, "current_performance": 0.4,
}
ML_PAYLOAD = {"fatigue": 40, "stress": 50, "retention": 0.5, "days_left": 7, "difficulty": "medium"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Keep session history out of the repo's data/ directory.
    main._HISTORY_FILE = tmp_path_factory.mktemp("data") / "session_history.json"
    with TestClient(main.app) as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert body["version"] == main.APP_VERSION
        assert body["models_loaded"] is (body["status"] == "ok")


@pytest.mark.parametrize("endpoint", ["/api/predict", "/api/explain"])
class TestPredictEndpoints:
    @pytest.mark.parametrize("payload", [CLIENT_PAYLOAD, ML_PAYLOAD])
    def test_both_schemas_accepted(self, client, endpoint, payload):
        r = client.post(endpoint, json=payload)
        assert r.status_code == 200
        assert r.json()["recommended_action"] in {"Study", "Revise", "Sleep/Break"}

    def test_action_id_matches_label(self, client, endpoint):
        body = client.post(endpoint, json=ML_PAYLOAD).json()
        assert main.ACTION_LABELS[body["action_id"]] == body["recommended_action"]

    def test_adaptive_weights_sum_to_one(self, client, endpoint):
        w = client.post(endpoint, json=CLIENT_PAYLOAD).json()["adaptive_weights"]
        assert w["q_learning"] + w["sarsa"] == pytest.approx(1.0)

    def test_critical_stress_forces_rest(self, client, endpoint):
        r = client.post(endpoint, json={**CLIENT_PAYLOAD, "stress_level": "critical"})
        assert r.json()["recommended_action"] == "Sleep/Break"

    def test_stress_95_forces_rest(self, client, endpoint):
        r = client.post(endpoint, json={**ML_PAYLOAD, "stress": 95, "fatigue": 0})
        assert r.json()["recommended_action"] == "Sleep/Break"

    @pytest.mark.parametrize("bad", [
        {**CLIENT_PAYLOAD, "stress_level": "banana"},
        {**CLIENT_PAYLOAD, "stress_level": 5},
        {**CLIENT_PAYLOAD, "hours_studied": -1},
        {**CLIENT_PAYLOAD, "current_performance": 1.5},
        {**ML_PAYLOAD, "difficulty": "insane"},
        {**ML_PAYLOAD, "stress": 101},
        {"fatigue": 10},
        {},
    ])
    def test_invalid_input_returns_422(self, client, endpoint, bad):
        assert client.post(endpoint, json=bad).status_code == 422


class TestExplain:
    def test_explain_fields(self, client):
        body = client.post("/api/explain", json=CLIENT_PAYLOAD).json()
        assert body["explanation"]
        assert set(body["q_values"]) == {"q_learning", "sarsa", "combined"}

    def test_input_state_excludes_client_fields(self, client):
        state = client.post("/api/explain", json=CLIENT_PAYLOAD).json()["input_state"]
        assert set(state) == {"fatigue", "stress", "retention", "days_left", "difficulty"}

    def test_client_schema_mapping(self, client):
        state = client.post("/api/explain", json=CLIENT_PAYLOAD).json()["input_state"]
        assert state["stress"] == 85 and state["fatigue"] == 60
        assert state["days_left"] == 3 and state["retention"] == 0.4

    def test_override_documented_in_explanation(self, client):
        body = client.post("/api/explain", json={**CLIENT_PAYLOAD, "stress_level": "critical"}).json()
        assert "Safety override" in body["explanation"]


class TestAnalyticsEndpoints:
    def test_study_plan(self, client):
        body = client.post("/api/study-plan", json=ML_PAYLOAD).json()
        assert len(body["plan"]) == 7 and "summary" in body

    def test_study_plan_capped_at_14_days(self, client):
        body = client.post("/api/study-plan", json={**ML_PAYLOAD, "days_left": 60}).json()
        assert len(body["plan"]) == 14

    def test_burnout(self, client):
        assert "burnout_risk" in client.post("/api/burnout", json=ML_PAYLOAD).json()

    def test_wellness(self, client):
        body = client.post("/api/wellness", json=ML_PAYLOAD).json()
        assert {"productivity", "wellness", "burnout_risk"} <= set(body)


class TestHistory:
    def test_history_roundtrip(self, client):
        assert client.get("/api/history").status_code == 200
        assert client.delete("/api/history").json()["status"] == "ok"
        assert client.get("/api/history").json()["total"] == 0

    @pytest.mark.parametrize("limit", [0, -1, 51])
    def test_invalid_limit_rejected(self, client, limit):
        assert client.get(f"/api/history?limit={limit}").status_code == 422


def test_home_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
