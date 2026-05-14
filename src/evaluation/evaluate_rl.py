"""
Evaluation pipeline for Exam Stress Balancer.
"""

import numpy as np
import statistics

from src.rl.student_environment import StudentEnvironment
from src.rl.agent.state_discretizer import discretize_state

from src.config import (
    EVALUATION_CONFIG,
    EVALUATION_REPORT_PATH,
    MODEL_METADATA_PATH,
)

from src.utils.logger import get_logger
from src.utils.helpers import load_pickle, load_json, save_json, utc_now_iso, Timer

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (FIXED - DYNAMIC)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    logger.info("🚀 Loading trained model...")

    # 🔥 Load metadata first
    metadata = load_json(MODEL_METADATA_PATH)

    model_path = metadata["model_path"]

    # 🔥 Load correct model dynamically
    q_table = load_pickle(model_path)

    logger.info(f"✅ Model loaded from: {model_path}")

    return {"q_table": q_table, "metadata": metadata}


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_action(q_table, state):
    if state in q_table:
        return max(q_table[state], key=q_table[state].get)
    else:
        return np.random.choice([0, 1, 2])


# ═══════════════════════════════════════════════════════════════════════════════
# REAL RL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(model, config):
    logger.info("🚀 Starting REAL evaluation...")

    q_table = model["q_table"]

    env = StudentEnvironment(total_days=15)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for ep in range(config["num_episodes"]):

        state = env.reset()

        # Random initial conditions (same as training)
        env.state["fatigue"] = np.random.randint(0, 101)
        env.state["stress"] = np.random.randint(0, 101)
        env.state["retention"] = np.random.uniform(0, 1)
        env.state["difficulty"] = np.random.choice(["easy", "medium", "hard"])

        state = discretize_state(state)

        done = False
        total_reward = 0
        steps = 0

        while not done:

            action = predict_action(q_table, state)

            next_state, reward, done = env.step(action)

            state = discretize_state(next_state)

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward > 0:
            success_count += 1

    logger.info("✅ Evaluation complete")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_count": success_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(config=None):
    config = config or EVALUATION_CONFIG

    logger.info("══════════════════════════════════════════════")
    logger.info("  Exam Stress Balancer — Evaluation Pipeline")
    logger.info("══════════════════════════════════════════════")

    model = load_model()

    with Timer("evaluation") as timer:
        raw = evaluate_policy(model, config)

    rewards = raw["episode_rewards"]
    lengths = raw["episode_lengths"]

    aggregate = {
        "mean_reward": round(statistics.mean(rewards), 4),
        "std_reward": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "mean_ep_length": round(statistics.mean(lengths), 2),
        "success_rate": round(raw["success_count"] / config["num_episodes"], 4),
        "total_episodes": config["num_episodes"],
    }

    report = {
        "project": "exam-stress-balancer",
        "evaluated_at": utc_now_iso(),
        "model_version": model["metadata"].get("version", "unknown"),
        "model_created_at": model["metadata"].get("created_at", "unknown"),
        "algorithm": model["metadata"].get("algorithm", "unknown"),
        "evaluation_config": config,
        "aggregate_metrics": aggregate,
        "duration_sec": round(timer.elapsed, 3),
        "episode_rewards": rewards,
    }

    save_json(report, EVALUATION_REPORT_PATH)

    logger.info("📊 Evaluation Summary:")
    for k, v in aggregate.items():
        logger.info(f"{k}: {v}")

    logger.info(f"📁 Report saved → {EVALUATION_REPORT_PATH}")


# CLI
if __name__ == "__main__":
    run_evaluation()