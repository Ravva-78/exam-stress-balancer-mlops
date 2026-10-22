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
# RL core
# ═══════════════════════════════════════════════════════════════════════════════

from src.rl.agent.q_learning_agent import QLearningAgent
from src.rl.agent.state_discretizer import discretize_state
from src.rl.student_environment import StudentEnvironment


class TestDiscretizer:
    def test_bins(self):
        s = {"fatigue": 10, "stress": 50, "retention": 0.9, "days_left": 1, "difficulty": "hard"}
        assert discretize_state(s) == ("LOW", "MEDIUM", "HIGH", "HIGH", "HARD")

    def test_urgency_bins(self):
        base = {"fatigue": 0, "stress": 0, "retention": 0, "difficulty": "easy"}
        assert discretize_state({**base, "days_left": 4})[3] == "MEDIUM"
        assert discretize_state({**base, "days_left": 10})[3] == "LOW"


class TestQLearningAgent:
    def test_update_moves_q_towards_reward(self):
        agent = QLearningAgent([0, 1, 2], alpha=0.5, gamma=0.9, epsilon=0.0)
        agent.update("s", 1, 10.0, "s2")
        assert agent.q_table["s"][1] == pytest.approx(5.0)

    def test_greedy_picks_best_action(self):
        agent = QLearningAgent([0, 1, 2], epsilon=0.0)
        agent.q_table["s"] = {0: 0.0, 1: 3.0, 2: 1.0}
        assert agent.choose_action("s") == 1


class TestEnvironment:
    def test_episode_terminates(self):
        random.seed(0)
        env = StudentEnvironment(total_days=5)
        env.reset()
        done, steps = False, 0
        while not done and steps < 100:
            _, _, done = env.step(0)
            steps += 1
        assert done and steps <= 100

    def test_state_stays_in_bounds(self):
        random.seed(1)
        env = StudentEnvironment(total_days=10)
        env.reset()
        for _ in range(10):
            state, _, done = env.step(random.choice([0, 1, 2]))
            assert 0 <= state["fatigue"] <= 100 and 0 <= state["stress"] <= 100
            assert 0.0 <= state["retention"] <= 1.0
            if done:
                break


class TestTrainers:
    def test_q_learning_trains(self):
        from src.rl.training.train_q_learning import train_q_learning
        q_table, metrics = train_q_learning({"episodes": 5})
        assert q_table and metrics["total_episodes"] == 5

    def test_sarsa_trains(self):
        from src.rl.training.train_sarsa import train_sarsa
        q_table, metrics = train_sarsa({"episodes": 5})
        assert q_table and metrics["total_episodes"] == 5
