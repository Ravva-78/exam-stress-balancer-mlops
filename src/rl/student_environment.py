"""
Student Environment for Exam Stress Balancer (Reinforcement Learning)

This environment simulates a student's cognitive state during exam preparation.
It is responsible for:
- Maintaining the current state (fatigue, stress, retention, etc.)
- Applying actions (study, revise, rest, sleep, ignore)
- Updating the state based on predefined transition rules
- Computing rewards for reinforcement learning agents

NOTE:
This file only defines the environment structure.
No reinforcement learning logic is implemented here.
"""

from typing import Dict, Tuple
import random

class StudentEnvironment:
    """
    Simulated environment representing a student's exam preparation process.

    v1.1 improvements:
    - Stochastic noise on state transitions (±noise_scale)
    - Penalty for sustained extreme fatigue/stress (burnout zone)
    - Fully backward-compatible API
    """
    """
    Simulated environment representing a student's exam preparation process.
    """

    def __init__(self, total_days: int = 30, difficulty: str = "medium", noise_scale: float = 3.0):

        """
        Initialize the environment with default values.

        Args:
            total_days (int): Number of days before the exam.
        """
        self.total_days = total_days
        self.difficulty = difficulty
        self.noise_scale = noise_scale  # stddev for Gaussian transition noise
        self.current_day = 0

        # Initialize the student state
        self.state: Dict[str, float | int | str] = {}

        # Define possible actions
        self.actions = {
            0: "study",
            1: "revise",
            2: "break",
            3: "sleep",
            4: "ignore"
        }

        # Reset environment to initial state
        self.reset()

    def reset(self) -> Dict[str, float | int | str]:
        """
        Reset the environment to the initial state.

        Returns:
            dict: Initial state of the environment.
        """
        self.current_day = 0
        difficulty = random.choice(["easy", "medium", "hard"])
        self.state = {
            "fatigue": 30,        # 0 to 100
            "stress": 20,         # 0 to 100
            "retention": 0.2,     # 0.0 to 1.0
            "days_left": self.total_days,
            "difficulty": self.difficulty
        }

        return self.state

    def step(self, action: int):

        learning_gain = 0.0
        reward = 0.0

        fatigue = self.state["fatigue"]
        stress = self.state["stress"]
        retention = self.state["retention"]
        difficulty = self.state["difficulty"]

        # Difficulty multiplier
        if difficulty == "easy":
            difficulty_multiplier = 1.3
        elif difficulty == "hard":
            difficulty_multiplier = 0.7
        else:
            difficulty_multiplier = 1.0

        # ================= STUDY =================
        if action == 0:

            learning_gain = 0.16 * (1 - fatigue / 100)
            learning_gain *= difficulty_multiplier

            self.state["retention"] = min(1.0, retention + learning_gain)

            fatigue_inc = 12
            stress_inc = 8

            self.state["fatigue"] = min(100, fatigue + fatigue_inc)
            self.state["stress"] = min(100, stress + stress_inc)

            reward = (
                learning_gain * 20
                - 0.04 * self.state["fatigue"]
                - 0.03 * self.state["stress"]
            )
            if retention < 0.4:
                reward += 5


            # Penalize studying when already mastered
            if self.state["retention"] > 0.9:
                reward -= 5

        # ================= REVISE =================
        elif action == 1:

            revision_gain = 0.06

            self.state["retention"] = min(1.0, retention + revision_gain)
            self.state["fatigue"] = max(0, fatigue - 3)
            self.state["stress"] = max(0, stress - 2)

            reward = (
                revision_gain * 14
                - 0.02 * self.state["fatigue"]
            )

            # Bonus only if retention moderate
            if 0.4 <= retention <= 0.8:
                reward += 2

            # Penalize over-revising when too low
            if retention < 0.3:
                reward -= 3

        # ================= BREAK =================
        elif action == 2:

            self.state["fatigue"] = max(0, fatigue - 12)
            self.state["stress"] = max(0, stress - 10)

            reward = (
                1.0
                - 0.02 * retention
            )

            # Reward break only if actually tired
            if fatigue > 60 or stress > 60:
                reward += 5
            else:
                reward -= 2

        # ================= NOISE =================
        # Small Gaussian noise on fatigue and stress (realistic variability)
        if self.noise_scale > 0:
            fatigue_noise = random.gauss(0, self.noise_scale)
            stress_noise  = random.gauss(0, self.noise_scale)
            self.state["fatigue"] = max(0, min(100, self.state["fatigue"] + fatigue_noise))
            self.state["stress"]  = max(0, min(100, self.state["stress"]  + stress_noise))

        # ================= EXTREME-STATE PENALTY =================
        # Additional penalty when both fatigue AND stress are in the danger zone
        if self.state["fatigue"] > 80 and self.state["stress"] > 80:
            reward -= 5.0  # double-danger burnout

        # ================= RETENTION DECAY =================
        # Small natural forgetting
        self.state["retention"] = max(0.0, self.state["retention"] - 0.01)

        # ================= BURNOUT =================
        if self.state["fatigue"] > 85 or self.state["stress"] > 85:
            reward -= 12
            if action == 0:
                reward -= 8

        # ================= TIME =================
        self.state["days_left"] -= 1
        self.current_day += 1
        done = self.state["days_left"] <= 0

        if done:
            reward += self.state["retention"] * 15

        # ================= URGENCY =================
        if self.state["days_left"] <= 3:
            if self.state["retention"] < 0.6 and action == 0:
                reward += 5
            if self.state["retention"] >= 0.7 and action == 1:
                reward += 3

        return self.state, reward, done


    



    def get_state(self) -> Dict[str, float | int | str]:
        """
        Get the current state of the environment.

        Returns:
            dict: Current state.
        """
        return self.state

    def get_action_meaning(self, action: int) -> str:
        """
        Get human-readable meaning of an action.

        Args:
            action (int): Action index.

        Returns:
            str: Action description.
        """
        return self.actions.get(action, "unknown")

    