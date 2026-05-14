"""
Pytest test suite for model artefacts, training, and evaluation pipelines.

Run:
    pytest tests/test_model.py -v
"""

import json
import pickle
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.helpers import (
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    build_metadata,
    stress_level_to_index,
    index_to_action,
    validate_state,
    file_checksum,
    Timer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestPickleIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        obj = {"key": [1, 2, 3], "nested": {"a": True}}
        path = tmp_path / "test.pkl"
        save_pickle(obj, path)
        loaded = load_pickle(path)
        assert loaded == obj

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "nonexistent.pkl")

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "data.pkl"
        save_pickle([1, 2, 3], deep_path)
        assert deep_path.exists()


class TestJsonIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"project": "exam-stress", "score": 0.95, "tags": ["rl", "ml"]}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "ghost.json")

    def test_json_pretty_printed(self, tmp_path):
        path = tmp_path / "pretty.json"
        save_json({"a": 1}, path, indent=4)
        raw = path.read_text()
        assert "\n" in raw   # pretty-printed contains newlines


class TestMetadata:
    def test_build_metadata_keys(self):
        meta = build_metadata()
        assert "project"    in meta
        assert "version"    in meta
        assert "created_at" in meta
        assert "framework"  in meta

    def test_build_metadata_merge(self):
        meta = build_metadata(extra={"custom": "value"})
        assert meta["custom"] == "value"

    def test_build_metadata_project_name(self):
        meta = build_metadata()
        assert meta["project"] == "exam-stress-balancer"


class TestStressHelpers:
    def test_stress_to_index_all_levels(self):
        assert stress_level_to_index("low")      == 0
        assert stress_level_to_index("medium")   == 1
        assert stress_level_to_index("high")     == 2
        assert stress_level_to_index("critical") == 3

    def test_stress_to_index_case_insensitive(self):
        assert stress_level_to_index("HIGH")   == 2
        assert stress_level_to_index("Medium") == 1

    def test_stress_to_index_invalid_raises(self):
        with pytest.raises(ValueError):
            stress_level_to_index("extreme")

    def test_index_to_action_all(self):
        assert index_to_action(0) == "rest"
        assert index_to_action(1) == "light_study"
        assert index_to_action(2) == "moderate_study"
        assert index_to_action(3) == "intense_study"

    def test_index_to_action_out_of_range(self):
        with pytest.raises(ValueError):
            index_to_action(4)
        with pytest.raises(ValueError):
            index_to_action(-1)


class TestValidateState:
    VALID_STATE = {
        "stress_level":    "high",
        "hours_studied":   5.0,
        "days_until_exam": 3,
    }

    def test_valid_state_passes(self):
        validate_state(self.VALID_STATE)   # should not raise

    def test_missing_field_raises(self):
        state = {k: v for k, v in self.VALID_STATE.items() if k != "stress_level"}
        with pytest.raises(ValueError, match="stress_level"):
            validate_state(state)

    def test_invalid_stress_level_raises(self):
        state = {**self.VALID_STATE, "stress_level": "panic"}
        with pytest.raises(ValueError):
            validate_state(state)

    def test_hours_out_of_range_raises(self):
        state = {**self.VALID_STATE, "hours_studied": 30}
        with pytest.raises(ValueError):
            validate_state(state)

    def test_negative_days_raises(self):
        state = {**self.VALID_STATE, "days_until_exam": -5}
        with pytest.raises(ValueError):
            validate_state(state)


class TestTimer:
    def test_elapsed_positive(self):
        import time
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_elapsed_attribute_set(self):
        with Timer("t") as t:
            pass
        assert hasattr(t, "elapsed")
        assert t.elapsed >= 0


class TestChecksum:
    def test_checksum_consistent(self, tmp_path):
        p = tmp_path / "file.bin"
        p.write_bytes(b"hello world")
        c1 = file_checksum(p)
        c2 = file_checksum(p)
        assert c1 == c2

    def test_checksum_different_files(self, tmp_path):
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert file_checksum(p1) != file_checksum(p2)

    def test_checksum_length(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"test")
        assert len(file_checksum(p)) == 64   # SHA-256 hex = 64 chars


# ═══════════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingPipeline:
    def test_run_training_logic_returns_dict(self):
        from src.training.train_rl import run_training_logic
        from src.config import TRAINING_CONFIG

        config = {**TRAINING_CONFIG, "episodes": 10}   # fast
        result = run_training_logic(config)

        assert isinstance(result, dict)
        assert "q_table"         in result
        assert "episode_rewards" in result
        assert "final_epsilon"   in result
        assert "metrics"         in result

    def test_episode_rewards_length(self):
        from src.training.train_rl import run_training_logic
        config = {"episodes": 20, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995}
        result = run_training_logic(config)
        assert len(result["episode_rewards"]) == 20

    def test_metrics_keys(self):
        from src.training.train_rl import run_training_logic
        config = {"episodes": 5, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.99}
        result = run_training_logic(config)
        for key in ("avg_reward", "best_reward", "final_epsilon", "total_episodes"):
            assert key in result["metrics"], f"Missing metric: {key}"

    def test_full_training_creates_artefacts(self, tmp_path):
        """Integration test: run_training() should write model files."""
        q_path   = tmp_path / "q_table.pkl"
        meta_path = tmp_path / "metadata.json"

        with (
            patch("src.training.train_rl.Q_TABLE_PATH",        q_path),
            patch("src.training.train_rl.MODEL_METADATA_PATH",  meta_path),
            patch("src.training.train_rl.MLFLOW_TRACKING_URI",
                  f"sqlite:///{tmp_path / 'mlflow.db'}"),
        ):
            from src.training.train_rl import run_training
            from src.config import TRAINING_CONFIG
            config = {**TRAINING_CONFIG, "episodes": 5}
            run_training(config)

        assert q_path.exists(),    "q_table.pkl not created"
        assert meta_path.exists(), "metadata.json not created"


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationPipeline:
    def _make_artefacts(self, tmp_path: Path):
        q_path   = tmp_path / "q_table.pkl"
        meta_path = tmp_path / "metadata.json"
        save_pickle([[0.1, 0.2, 0.3, 0.4]] * 4, q_path)
        save_json({"version": "1.0.0", "created_at": "2024-01-01", "framework": "stub"}, meta_path)
        return q_path, meta_path

    def test_load_model_returns_dict(self, tmp_path):
        q_path, meta_path = self._make_artefacts(tmp_path)
        with (
            patch("src.evaluation.evaluate_rl.Q_TABLE_PATH",        q_path),
            patch("src.evaluation.evaluate_rl.MODEL_METADATA_PATH",  meta_path),
        ):
            from src.evaluation.evaluate_rl import load_model
            model = load_model()

        assert "q_table"  in model
        assert "metadata" in model

    def test_load_model_missing_raises(self, tmp_path):
        with (
            patch("src.evaluation.evaluate_rl.Q_TABLE_PATH",
                  tmp_path / "nope.pkl"),
            patch("src.evaluation.evaluate_rl.MODEL_METADATA_PATH",
                  tmp_path / "nope.json"),
        ):
            from src.evaluation.evaluate_rl import load_model
            with pytest.raises(FileNotFoundError):
                load_model()

    def test_predict_action_returns_valid_index(self, tmp_path):
        q_path, meta_path = self._make_artefacts(tmp_path)
        with (
            patch("src.evaluation.evaluate_rl.Q_TABLE_PATH",        q_path),
            patch("src.evaluation.evaluate_rl.MODEL_METADATA_PATH",  meta_path),
        ):
            from src.evaluation.evaluate_rl import predict_action
            q_table = [[0.1, 0.2, 0.3, 0.4]] * 4
            state   = {"stress_level": "high", "hours_studied": 3.0, "days_until_exam": 5}
            action  = predict_action(q_table, state)
        assert 0 <= action <= 3

    def test_full_evaluation_creates_report(self, tmp_path):
        q_path, meta_path = self._make_artefacts(tmp_path)
        report_path = tmp_path / "evaluation_report.json"

        with (
            patch("src.evaluation.evaluate_rl.Q_TABLE_PATH",           q_path),
            patch("src.evaluation.evaluate_rl.MODEL_METADATA_PATH",     meta_path),
            patch("src.evaluation.evaluate_rl.EVALUATION_REPORT_PATH",  report_path),
        ):
            from src.evaluation.evaluate_rl import run_evaluation
            from src.config import EVALUATION_CONFIG
            config = {**EVALUATION_CONFIG, "num_episodes": 10}
            run_evaluation(config)

        assert report_path.exists(), "evaluation_report.json not created"
        report = load_json(report_path)
        assert "aggregate_metrics" in report
        assert "episode_rewards"   in report

    def test_report_aggregate_keys(self, tmp_path):
        q_path, meta_path = self._make_artefacts(tmp_path)
        report_path = tmp_path / "evaluation_report.json"

        with (
            patch("src.evaluation.evaluate_rl.Q_TABLE_PATH",           q_path),
            patch("src.evaluation.evaluate_rl.MODEL_METADATA_PATH",     meta_path),
            patch("src.evaluation.evaluate_rl.EVALUATION_REPORT_PATH",  report_path),
        ):
            from src.evaluation.evaluate_rl import run_evaluation
            run_evaluation({"num_episodes": 5, "max_steps": 10, "seed": 0})

        report = load_json(report_path)
        for key in ("mean_reward", "std_reward", "min_reward", "max_reward",
                    "success_rate", "total_episodes"):
            assert key in report["aggregate_metrics"], f"Missing: {key}"
