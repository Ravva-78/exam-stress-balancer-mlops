"""
Wellness & Burnout Analytics Module
Exam Stress Balancer — v3.0

Provides:
- Burnout risk scoring (LOW / MEDIUM / HIGH) with numeric score 0–100
- Productivity score (composite of fatigue, stress, retention)
- Wellness score (holistic student health indicator)
- Smart recovery suggestions
- Exam urgency classification
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal


# ── Types ──────────────────────────────────────────────────────────────────────

BurnoutLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
UrgencyLevel = Literal["RELAXED", "MODERATE", "URGENT", "CRITICAL"]


# ── Burnout Risk ───────────────────────────────────────────────────────────────

@dataclass
class BurnoutRisk:
    level: BurnoutLevel
    score: float          # 0–100, higher = more at risk
    factors: list[str]    # human-readable contributing factors
    recovery_tip: str     # immediate actionable advice

    def to_dict(self) -> dict:
        return asdict(self)


def compute_burnout_risk(state: dict) -> BurnoutRisk:
    """
    Compute a burnout risk score from the student state.

    Formula weights:
      - fatigue: 40%
      - stress:  40%
      - retention gap (low retention = higher risk): 20%
    """
    fatigue   = float(state.get("fatigue", 50))
    stress    = float(state.get("stress", 50))
    retention = float(state.get("retention", 0.5))
    days_left = int(state.get("days_left", 7))

    # Retention gap: 1.0 − retention → 0.0 means perfect, 1.0 means totally forgotten
    retention_gap = 1.0 - retention

    # Base burnout score (0–100)
    score = round(
        0.40 * fatigue
        + 0.40 * stress
        + 0.20 * retention_gap * 100,
        1,
    )

    # Urgency amplifier: if few days remain AND score is elevated, push higher
    if days_left <= 3 and score >= 55:
        score = min(100.0, score + 8)

    # Classify
    if score >= 80:
        level: BurnoutLevel = "CRITICAL"
    elif score >= 60:
        level = "HIGH"
    elif score >= 35:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Identify contributing factors
    factors: list[str] = []
    if fatigue >= 70:
        factors.append("Severe physical fatigue")
    elif fatigue >= 45:
        factors.append("Moderate fatigue accumulation")

    if stress >= 70:
        factors.append("High psychological stress")
    elif stress >= 45:
        factors.append("Elevated anxiety levels")

    if retention_gap >= 0.6:
        factors.append("Low material retention increasing study pressure")
    elif retention_gap >= 0.35:
        factors.append("Moderate retention gap")

    if days_left <= 2:
        factors.append("Extreme time urgency (≤2 days remaining)")
    elif days_left <= 5:
        factors.append("High time pressure")

    if not factors:
        factors.append("No significant risk indicators detected")

    # Recovery tip
    if level == "CRITICAL":
        tip = "Immediate rest required. Stop studying for 2–3 hours. Hydrate, breathe deeply, and sleep 7–8 hours tonight."
    elif level == "HIGH":
        tip = "Take a 30–45 min break now. Light walk, music, or meditation. Avoid heavy new material today."
    elif level == "MEDIUM":
        tip = "Schedule short breaks every 45 minutes. Prioritise revision over new study. Keep sleep above 7 hours."
    else:
        tip = "You're in a healthy zone. Keep your current routine and maintain regular sleep and breaks."

    return BurnoutRisk(level=level, score=score, factors=factors, recovery_tip=tip)


# ── Productivity Score ─────────────────────────────────────────────────────────

@dataclass
class ProductivityScore:
    score: float           # 0–100, higher = more productive
    label: str             # "Peak", "Good", "Declining", "Poor"
    insight: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_productivity_score(state: dict) -> ProductivityScore:
    """
    Estimates current study productivity potential.

    High retention + low fatigue + moderate stress = productive.
    """
    fatigue   = float(state.get("fatigue", 50))
    stress    = float(state.get("stress", 50))
    retention = float(state.get("retention", 0.5))

    # Cognitive availability: inversely related to fatigue
    cognitive_available = max(0, 1 - fatigue / 100)

    # Stress sweet spot: moderate stress (20–50) enhances performance (Yerkes-Dodson)
    if 20 <= stress <= 50:
        stress_factor = 1.0
    elif stress < 20:
        stress_factor = 0.85   # too relaxed → less focused
    elif stress <= 70:
        stress_factor = max(0.3, 1 - (stress - 50) / 80)
    else:
        stress_factor = 0.2    # high stress destroys productivity

    # Retention acts as a multiplier — higher base → easier to build on
    retention_boost = 0.6 + 0.4 * retention

    raw = cognitive_available * stress_factor * retention_boost * 100
    score = round(min(100.0, max(0.0, raw)), 1)

    if score >= 75:
        label, insight = "Peak", "Excellent conditions for deep study or complex problem-solving."
    elif score >= 50:
        label, insight = "Good", "Solid study potential. Focus on consolidating core topics."
    elif score >= 25:
        label, insight = "Declining", "Reduced effectiveness. Favour revision over new material."
    else:
        label, insight = "Poor", "Low productivity window. Rest will provide more return than studying now."

    return ProductivityScore(score=score, label=label, insight=insight)


# ── Wellness Score ─────────────────────────────────────────────────────────────

@dataclass
class WellnessScore:
    score: float      # 0–100, higher = healthier
    label: str
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_wellness_score(state: dict) -> WellnessScore:
    """
    Holistic student wellness: low fatigue + low stress + high retention = good wellness.
    """
    fatigue   = float(state.get("fatigue", 50))
    stress    = float(state.get("stress", 50))
    retention = float(state.get("retention", 0.5))

    score = round(
        (1 - fatigue / 100) * 35
        + (1 - stress / 100) * 35
        + retention * 30,
        1,
    )

    if score >= 70:
        label, summary = "Thriving", "You are in great shape for exam prep. Keep this balance."
    elif score >= 50:
        label, summary = "Stable", "Manageable wellness. Minor adjustments will improve performance."
    elif score >= 30:
        label, summary = "Strained", "Your wellbeing is under pressure. Prioritise recovery."
    else:
        label, summary = "At Risk", "Significant wellness deficit. Immediate rest and self-care needed."

    return WellnessScore(score=score, label=label, summary=summary)


# ── Exam Urgency ───────────────────────────────────────────────────────────────

def classify_urgency(days_left: int, retention: float) -> dict:
    """Return urgency level + message."""
    if days_left <= 1:
        level: UrgencyLevel = "CRITICAL"
        msg = "Exam is tomorrow. Focus only on highest-yield topics and ensure good sleep tonight."
    elif days_left <= 3:
        level = "URGENT"
        msg = f"{days_left} days left. Prioritise weak areas and revise key concepts daily."
    elif days_left <= 7:
        level = "MODERATE"
        msg = f"{days_left} days left. Balance new study and revision while protecting sleep."
    else:
        level = "RELAXED"
        msg = f"{days_left} days remaining. Build strong foundations before shifting to revision."

    retention_warning = None
    if days_left <= 5 and retention < 0.5:
        retention_warning = f"Warning: retention ({retention:.0%}) is low with only {days_left} days left."

    return {"level": level, "message": msg, "retention_warning": retention_warning}


# ── Adaptive Hybrid Weight ─────────────────────────────────────────────────────

def compute_adaptive_weights(state: dict) -> tuple[float, float]:
    """
    Dynamically adjust Q-Learning / SARSA fusion weights based on student state.

    Logic:
    - When stress or fatigue is HIGH → lean toward SARSA (safer, on-policy)
    - When urgency is HIGH (few days, low retention) → lean toward Q-Learning (optimal)
    - Default: 60% Q-Learning, 40% SARSA
    """
    fatigue   = float(state.get("fatigue", 50))
    stress    = float(state.get("stress", 50))
    retention = float(state.get("retention", 0.5))
    days_left = int(state.get("days_left", 7))

    q_weight   = 0.60
    sarsa_weight = 0.40

    # High stress/fatigue → prefer safer SARSA recommendations
    if fatigue > 70 or stress > 70:
        q_weight   -= 0.15
        sarsa_weight += 0.15

    # Critical urgency + low retention → prefer aggressive Q-Learning
    elif days_left <= 3 and retention < 0.5:
        q_weight   += 0.15
        sarsa_weight -= 0.15

    # Near-exam with high retention → balanced default is fine
    q_weight   = round(max(0.2, min(0.8, q_weight)), 2)
    sarsa_weight = round(1.0 - q_weight, 2)

    return q_weight, sarsa_weight


# ── Smart Recovery Suggestions ─────────────────────────────────────────────────

def get_recovery_suggestions(state: dict, action: int) -> list[str]:
    """Return 2–4 contextual recovery/wellness suggestions."""
    fatigue   = float(state.get("fatigue", 50))
    stress    = float(state.get("stress", 50))
    days_left = int(state.get("days_left", 7))

    suggestions = []

    if fatigue > 65:
        suggestions.append("🛌 Sleep 7–9 hours tonight — memory consolidation happens during sleep.")
    if stress > 65:
        suggestions.append("🧘 Try 5 minutes of box breathing (4s inhale → 4s hold → 4s exhale → 4s hold).")
    if days_left <= 5:
        suggestions.append("📋 Write a prioritised topic list — tackle the hardest while your mind is fresh.")
    if action == 2:  # Sleep/Break recommended
        suggestions.append("☕ A 20-minute power nap + coffee (nappuccino) boosts alertness better than either alone.")
    if fatigue > 80 and stress > 80:
        suggestions.append("🚫 Avoid all screens 1 hour before bed to protect sleep quality tonight.")
    if not suggestions:
        suggestions.append("✅ Maintain your current balance — you're doing well.")
        suggestions.append("💧 Stay hydrated and take a short walk between study sessions.")

    return suggestions[:4]
