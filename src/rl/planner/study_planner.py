"""
Multi-Day Adaptive Study Planner
Exam Stress Balancer — v3.0

Uses the trained hybrid RL model to simulate and generate
an intelligent day-by-day study plan for the remaining exam preparation period.

Each plan day includes:
- Recommended action (Study / Revise / Sleep)
- Predicted student state
- Burnout risk
- Productivity score
- Day-specific advice
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analytics.wellness import BurnoutRisk, ProductivityScore


ACTION_LABELS = {0: "Study", 1: "Revise", 2: "Sleep/Break"}

ACTION_ICONS = {0: "📖", 1: "🔄", 2: "💤"}

ACTION_COLORS = {0: "#6366f1", 1: "#059669", 2: "#d97706"}

DAY_ADVICE = {
    0: [  # Study
        "Focus on your weakest topics first while your mind is fresh.",
        "Use active recall: close the book and write what you remember.",
        "Work in 45-minute focused blocks (Pomodoro technique).",
        "Tackle one difficult chapter or concept per session.",
    ],
    1: [  # Revise
        "Create summary cards for key concepts and formulas.",
        "Revisit material from 2–3 days ago to strengthen long-term memory.",
        "Practice past exam questions for 20 minutes.",
        "Mind-map connections between topics you've already studied.",
    ],
    2: [  # Sleep/Break
        "Use this recovery window to consolidate memory through quality sleep.",
        "A short walk or light exercise will reset your cognitive state.",
        "Do something enjoyable — regulated recovery prevents burnout.",
        "Prepare your study space and materials for tomorrow.",
    ],
}


def _pick_advice(action: int, day_index: int) -> str:
    pool = DAY_ADVICE.get(action, DAY_ADVICE[1])
    return pool[day_index % len(pool)]


def _simulate_state_transition(state: dict, action: int) -> dict:
    """
    Lightweight deterministic state transition for planning (no noise).
    Mirrors the StudentEnvironment logic but without randomness.
    """
    s = copy.deepcopy(state)
    fatigue   = s["fatigue"]
    stress    = s["stress"]
    retention = s["retention"]

    diff = s.get("difficulty", "medium")
    if diff == "easy":
        diff_mult = 1.3
    elif diff == "hard":
        diff_mult = 0.7
    else:
        diff_mult = 1.0

    if action == 0:  # Study
        gain = 0.16 * (1 - fatigue / 100) * diff_mult
        s["retention"] = min(1.0, retention + gain)
        s["fatigue"]   = min(100, fatigue + 12)
        s["stress"]    = min(100, stress + 8)

    elif action == 1:  # Revise
        s["retention"] = min(1.0, retention + 0.06)
        s["fatigue"]   = max(0, fatigue - 3)
        s["stress"]    = max(0, stress - 2)

    elif action == 2:  # Sleep/Break
        s["fatigue"] = max(0, fatigue - 20)
        s["stress"]  = max(0, stress - 18)
        # Slight retention decay during rest
        s["retention"] = max(0.0, retention - 0.005)

    # Natural forgetting
    s["retention"] = max(0.0, s["retention"] - 0.01)
    s["days_left"] = max(0, s["days_left"] - 1)

    return s


def generate_study_plan(
    initial_state: dict,
    hybrid_predict_fn,  # callable(state) -> (action, debug)
    max_days: int = 14,
) -> dict:
    """
    Simulate max_days ahead using the hybrid RL model.

    Returns a structured plan dict with per-day entries.
    """
    from src.analytics.wellness import (
        compute_burnout_risk,
        compute_productivity_score,
        compute_wellness_score,
    )

    days_left = min(int(initial_state.get("days_left", 7)), max_days)
    current_state = copy.deepcopy(initial_state)

    plan_days = []
    cumulative_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}

    for day_idx in range(days_left):
        day_num = day_idx + 1

        # Get RL recommendation
        action, debug = hybrid_predict_fn(current_state)

        # Analytics for this state
        burnout   = compute_burnout_risk(current_state)
        prod      = compute_productivity_score(current_state)
        wellness  = compute_wellness_score(current_state)

        # Estimate reward (simplified)
        reward = _estimate_reward(current_state, action)
        cumulative_reward += reward

        action_counts[action] = action_counts.get(action, 0) + 1

        # Build combined Q-value confidence (0–100)
        combined = debug.get("combined", {})
        if combined:
            all_vals = list(combined.values())
            val_min, val_max = min(all_vals), max(all_vals)
            val_range = val_max - val_min if val_max != val_min else 1
            best_val = combined.get(action, 0)
            confidence = round(((best_val - val_min) / val_range) * 100, 1) if val_range > 0 else 50.0
        else:
            confidence = 65.0   # rule-based fallback

        plan_days.append({
            "day":            day_num,
            "days_remaining": current_state["days_left"],
            "action_id":      action,
            "action":         ACTION_LABELS[action],
            "action_icon":    ACTION_ICONS[action],
            "action_color":   ACTION_COLORS[action],
            "advice":         _pick_advice(action, day_idx),
            "confidence":     confidence,
            "state_snapshot": {
                "fatigue":   round(current_state["fatigue"], 1),
                "stress":    round(current_state["stress"], 1),
                "retention": round(current_state["retention"], 3),
            },
            "burnout_level":      burnout.level,
            "burnout_score":      burnout.score,
            "productivity_score": prod.score,
            "productivity_label": prod.label,
            "wellness_score":     wellness.score,
            "reward_estimate":    round(reward, 2),
        })

        # Advance state
        current_state = _simulate_state_transition(current_state, action)

    # Summary analytics
    study_days   = action_counts.get(0, 0)
    revise_days  = action_counts.get(1, 0)
    rest_days    = action_counts.get(2, 0)

    final_burnout  = compute_burnout_risk(current_state)
    final_prod     = compute_productivity_score(current_state)
    final_wellness = compute_wellness_score(current_state)

    balance_score = _compute_balance_score(study_days, revise_days, rest_days, days_left)

    return {
        "plan": plan_days,
        "summary": {
            "total_days":         days_left,
            "study_days":         study_days,
            "revise_days":        revise_days,
            "rest_days":          rest_days,
            "balance_score":      balance_score,
            "cumulative_reward":  round(cumulative_reward, 2),
            "projected_retention": round(current_state["retention"], 3),
            "final_burnout_level": final_burnout.level,
            "final_burnout_score": final_burnout.score,
            "final_productivity":  final_prod.score,
            "final_wellness":      final_wellness.score,
        },
    }


def _estimate_reward(state: dict, action: int) -> float:
    fatigue   = state["fatigue"]
    stress    = state["stress"]
    retention = state["retention"]
    days_left = state.get("days_left", 7)

    if action == 0:
        gain = 0.16 * (1 - fatigue / 100)
        r = gain * 20 - 0.04 * fatigue - 0.03 * stress
        if retention < 0.4:
            r += 5
    elif action == 1:
        r = 0.06 * 14 - 0.02 * fatigue
        if 0.4 <= retention <= 0.8:
            r += 2
    else:
        r = 1.0
        if fatigue > 60 or stress > 60:
            r += 5

    if fatigue > 85 or stress > 85:
        r -= 12

    return r


def _compute_balance_score(study: int, revise: int, rest: int, total: int) -> float:
    """
    Ideal balance: ~40% study, ~35% revise, ~25% rest (for >5 days).
    Returns 0–100.
    """
    if total == 0:
        return 0.0

    s = study / total
    rv = revise / total
    rt = rest / total

    # Target ratios
    ts, tr, tt = 0.40, 0.35, 0.25

    deviation = abs(s - ts) + abs(rv - tr) + abs(rt - tt)
    score = max(0.0, 1.0 - deviation) * 100

    return round(score, 1)
