# Architecture

The Exam Stress Balancer relies on a Hybrid Reinforcement Learning architecture designed specifically to balance cognitive load, stress, and retention.

## State Space (5 Dimensions)

The student's cognitive and temporal state is mapped into 5 distinct dimensions. The `state_discretizer.py` bins these continuous values into a finite discrete state space of **243 states** (3 × 3 × 3 × 3 × 3).

| Feature | Range | Bins |
|---------|-------|------|
| **Fatigue** | 0 - 100 | LOW (<40), MEDIUM (40-70), HIGH (>70) |
| **Stress** | 0 - 100 | LOW (<40), MEDIUM (40-70), HIGH (>70) |
| **Retention** | 0.0 - 1.0 | LOW (<0.4), MEDIUM (0.4-0.8), HIGH (>0.8) |
| **Days Left** | 0 - ∞ | LOW Urgency, MEDIUM Urgency, HIGH Urgency |
| **Difficulty**| str | EASY, MEDIUM, HARD |

## Action Space (3 Actions)

At each step, the RL agent must choose one of three actions:

1. **Study (0):** Learn new material. (High cognitive load, high reward if retention is low).
2. **Revise (1):** Review existing material. (Medium cognitive load, high reward if retention is medium).
3. **Sleep/Break (2):** Recover. (Negative cognitive load, high reward if stress/fatigue is high).

## Q-Learning vs SARSA

We train two distinct RL agents that learn simultaneously:
- **Q-Learning (Off-policy):** Assumes the optimal action is always taken next. This results in aggressive "Performance-oriented" policies.
- **SARSA (On-policy):** Factors in the current exploration policy. This results in safer, "Conservative" policies that avoid high-penalty states (burnout).

## Adaptive Weighting Logic

The core of `v3.0` is the adaptive weighting engine. Rather than a static 50/50 split, the system dynamically weights the Q-values from both agents based on the student's state:

- **Safety Mode (SARSA 55%, Q-Learning 45%):** Triggered when `fatigue > 70` or `stress > 70`. Prioritizes the conservative SARSA agent to prevent pushing the student into a burnout state.
- **Performance Mode (Q-Learning 75%, SARSA 25%):** Triggered when `retention < 0.4` and `days_left <= 3`. The student is in a critical learning deficit with high urgency. Prioritizes the aggressive Q-Learning agent to maximize output.
- **Balanced (Q-Learning 60%, SARSA 40%):** Default state for nominal learning.

## Critical Safety Boundary

In `v3.1.0`, we introduced a hard safety boundary within the API layer.
If `stress >= 95`:
1. RL logic is **ignored**.
2. Action is forced to **Sleep/Break (2)**.
3. Adaptive weights are set to `q_weight: 0.0, sarsa_weight: 1.0` and `safety_override: True` is flagged.

No RL agent, regardless of performance multipliers, is allowed to recommend intense study to a critically stressed student.

## Flow

1. **Client Request** → `PredictRequest` (Dual-schema parsing via Pydantic)
2. **Feature Mapping** → Internal ML schema
3. **Safety Check** → Trigger Sleep/Break if `stress >= 95`
4. **Adaptive Weighting** → Calculate `q_w` and `s_w`
5. **Agent Query** → Fetch Q-values from both `.pkl` tables
6. **Fusion** → `argmax(q_w * Q + s_w * S)`
7. **Response Generation** → Format JSON and human-readable explanation
