import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analytics.wellness import (
    compute_burnout_risk,
    compute_productivity_score,
    compute_wellness_score,
    compute_adaptive_weights,
    classify_urgency,
    get_recovery_suggestions,
)


# ── Burnout Risk Tests ─────────────────────────────────────

def test_burnout_critical_scenario():
    state = {"fatigue": 85, "stress": 85, "retention": 0.2, "days_left": 2}
    result = compute_burnout_risk(state)
    assert result.level in ["HIGH", "CRITICAL"]
    assert result.score >= 70
    assert len(result.factors) >= 2
    assert result.recovery_tip is not None

def test_burnout_low_healthy_student():
    state = {"fatigue": 20, "stress": 20, "retention": 0.85, "days_left": 10}
    result = compute_burnout_risk(state)
    assert result.level == "LOW"
    assert result.score < 35

def test_burnout_medium_range():
    state = {"fatigue": 50, "stress": 50, "retention": 0.5, "days_left": 6}
    result = compute_burnout_risk(state)
    assert result.level == "MEDIUM"
    assert 35 <= result.score < 60

def test_burnout_score_range():
    state = {"fatigue": 60, "stress": 70, "retention": 0.3, "days_left": 3}
    result = compute_burnout_risk(state)
    assert 0 <= result.score <= 100


# ── Productivity Tests ─────────────────────────────────────

def test_productivity_peak_conditions():
    state = {"fatigue": 10, "stress": 35, "retention": 0.9, "days_left": 7}
    result = compute_productivity_score(state)
    assert result.score >= 65
    assert result.label == "Peak"

def test_productivity_poor_high_fatigue():
    state = {"fatigue": 92, "stress": 30, "retention": 0.7, "days_left": 5}
    result = compute_productivity_score(state)
    assert result.score < 30
    assert result.label == "Poor"

def test_productivity_high_stress_kills_score():
    state = {"fatigue": 30, "stress": 90, "retention": 0.6, "days_left": 4}
    result = compute_productivity_score(state)
    assert result.score < 25

def test_productivity_score_always_valid():
    state = {"fatigue": 50, "stress": 50, "retention": 0.5, "days_left": 5}
    result = compute_productivity_score(state)
    assert 0 <= result.score <= 100
    assert result.label in ["Peak", "Good", "Declining", "Poor"]


# ── Wellness Tests ─────────────────────────────────────────

def test_wellness_thriving():
    state = {"fatigue": 10, "stress": 10, "retention": 0.95, "days_left": 14}
    result = compute_wellness_score(state)
    assert result.score >= 70
    assert result.label == "Thriving"

def test_wellness_at_risk():
    state = {"fatigue": 88, "stress": 85, "retention": 0.15, "days_left": 1}
    result = compute_wellness_score(state)
    assert result.score < 30
    assert result.label == "At Risk"

def test_wellness_score_always_valid():
    state = {"fatigue": 40, "stress": 40, "retention": 0.6, "days_left": 7}
    result = compute_wellness_score(state)
    assert 0 <= result.score <= 100
    assert result.label in ["Thriving", "Stable", "Strained", "At Risk"]


# ── Adaptive Weights Tests ─────────────────────────────────

def test_adaptive_weights_safety_mode_high_stress():
    state = {"fatigue": 85, "stress": 80, "retention": 0.5, "days_left": 5}
    q_w, s_w = compute_adaptive_weights(state)
    assert s_w > q_w  # SARSA dominates under high stress/fatigue

def test_adaptive_weights_performance_mode_urgency():
    state = {"fatigue": 30, "stress": 30, "retention": 0.3, "days_left": 2}
    q_w, s_w = compute_adaptive_weights(state)
    assert q_w > s_w  # Q-Learning dominates under urgency

def test_adaptive_weights_default_balanced():
    state = {"fatigue": 50, "stress": 50, "retention": 0.5, "days_left": 7}
    q_w, s_w = compute_adaptive_weights(state)
    assert q_w == 0.60
    assert s_w == 0.40

def test_adaptive_weights_always_sum_to_one():
    state = {"fatigue": 70, "stress": 65, "retention": 0.4, "days_left": 3}
    q_w, s_w = compute_adaptive_weights(state)
    assert round(q_w + s_w, 5) == 1.0


# ── Urgency Classification Tests ───────────────────────────

def test_urgency_critical_one_day():
    result = classify_urgency(1, 0.5)
    assert result["level"] == "CRITICAL"

def test_urgency_relaxed_long_time():
    result = classify_urgency(20, 0.7)
    assert result["level"] == "RELAXED"

def test_urgency_retention_warning_triggered():
    result = classify_urgency(3, 0.3)
    assert result["retention_warning"] is not None

def test_urgency_no_warning_high_retention():
    result = classify_urgency(3, 0.8)
    assert result["retention_warning"] is None


# ── Recovery Suggestions Tests ─────────────────────────────

def test_recovery_suggestions_returned():
    state = {"fatigue": 70, "stress": 75, "retention": 0.4, "days_left": 3}
    tips = get_recovery_suggestions(state, action=2)
    assert len(tips) >= 1
    assert len(tips) <= 4

def test_recovery_suggestions_healthy_student():
    state = {"fatigue": 20, "stress": 20, "retention": 0.8, "days_left": 10}
    tips = get_recovery_suggestions(state, action=1)
    assert len(tips) >= 1