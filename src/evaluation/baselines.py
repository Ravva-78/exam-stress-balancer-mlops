"""
Baseline Policies for Exam Stress Balancer.

Provides two simple baselines to compare against the trained RL agents:
  - RandomPolicy    : uniformly samples from valid actions
  - RuleBasedPolicy : hand-crafted heuristic (stress>70 → Sleep,
                      fatigue<40 → Study, else → Revise)
"""

import random


ACTIONS = [0, 1, 2]  # study=0, revise=1, sleep/break=2


class RandomPolicy:
    """Uniformly random action selection — lower-bound baseline."""

    name = "random"

    def choose_action(self, state) -> int:
        return random.choice(ACTIONS)


class RuleBasedPolicy:
    """
    Hand-crafted heuristic:
      - If stress  > 70  → Sleep  (action 2) — needs recovery
      - If fatigue < 40  → Study  (action 0) — still has capacity
      - Else             → Revise (action 1) — consolidate knowledge
    """

    name = "rule_based"

    def choose_action(self, state) -> int:
        # state can be a dict (raw) or tuple (discretised) —
        # we only get called with raw dicts from compare_baselines.py
        stress  = state["stress"]
        fatigue = state["fatigue"]

        if stress > 70:
            return 2   # sleep / break
        elif fatigue < 40:
            return 0   # study
        else:
            return 1   # revise
