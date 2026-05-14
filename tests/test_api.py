"""
Pytest test suite for the FastAPI serving layer.

Run:
    pytest tests/test_api.py -v
"""

import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_model_store():
    """Provide a pre-populated model store so tests don't need real artefacts."""
    return {
        "q_table":      [[0.1, 0.2, 0.3, 0.4]] * 4,
        "metadata":     {
            "version":    "1.0.0",
            "created_at": "2024-01-01T00:00:00+00:00",
            "framework":  "custom-rl",
        },
        "loaded":       True,
        "startup_time": 0.0,
    }


@pytest.fixture(scope="module")
def client(mock_model_store, tmp_path_factory):
    """
    Create a TestClient with a mocked model store and fake artefact files.
    """
    # Create minimal artefact files so load_model() won't crash
    artifacts_dir = tmp_path_factory.mktemp("models")
    q_table_path  = artifacts_dir / "q_table.pkl"
    metadata_path = artifacts_dir / "metadata.json"

    with open(q_table_path, "wb") as fh:
        pickle.dump([[0.1, 0.2, 0.3, 0.4]] * 4, fh)
    with open(metadata_path, "w") as fh:
        json.dump(mock_model_store["metadata"], fh)

    # Patch config paths and the model store atomically
    with (
        patch("src.config.Q_TABLE_PATH",        q_table_path),
        patch("src.config.MODEL_METADATA_PATH",  metadata_path),
        patch("src.api.main._model_store",       mock_model_store),
    ):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ═══════════════════════════════════════════════════════════════════════════════
# /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "uptime_sec" in data
        assert "version" in data

    def test_health_uptime_non_negative(self, client):
        data = client.get("/health").json()
        assert data["uptime_sec"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# /model-info
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_fields(self, client):
        data = client.get("/model-info").json()
        assert "model_version" in data
        assert "created_at"    in data
        assert "framework"     in data
        assert "extra"         in data

    def test_model_info_version(self, client):
        data = client.get("/model-info").json()
        assert data["model_version"] == "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# /predict
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredict:
    VALID_PAYLOAD = {
        "stress_level":          "high",
        "hours_studied":          4.0,
        "days_until_exam":        5,
        "current_performance":    0.6,
    }

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert "action_index"  in data
        assert "action_label"  in data
        assert "confidence"    in data
        assert "reasoning"     in data
        assert "request_state" in data

    def test_predict_action_index_range(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert 0 <= data["action_index"] <= 3

    def test_predict_action_label_valid(self, client):
        valid_labels = {"rest", "light_study", "moderate_study", "intense_study"}
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert data["action_label"] in valid_labels

    def test_predict_all_stress_levels(self, client):
        for level in ["low", "medium", "high", "critical"]:
            payload = {**self.VALID_PAYLOAD, "stress_level": level}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for stress_level={level}"

    def test_predict_invalid_stress_level(self, client):
        payload = {**self.VALID_PAYLOAD, "stress_level": "extreme"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_hours_out_of_range(self, client):
        payload = {**self.VALID_PAYLOAD, "hours_studied": 25.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_days(self, client):
        payload = {**self.VALID_PAYLOAD, "days_until_exam": -1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_required_field(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "stress_level"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_request_state_echoed(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert data["request_state"]["stress_level"] == "high"
        assert data["request_state"]["hours_studied"] == 4.0

    def test_predict_confidence_in_range(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_reasoning_non_empty(self, client):
        data = client.post("/predict", json=self.VALID_PAYLOAD).json()
        assert len(data["reasoning"]) > 0

    def test_predict_boundary_hours_zero(self, client):
        payload = {**self.VALID_PAYLOAD, "hours_studied": 0.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_boundary_hours_max(self, client):
        payload = {**self.VALID_PAYLOAD, "hours_studied": 24.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_predict_boundary_days_zero(self, client):
        payload = {**self.VALID_PAYLOAD, "days_until_exam": 0}
        assert client.post("/predict", json=payload).status_code == 200
