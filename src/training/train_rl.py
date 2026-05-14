"""
Training pipeline for Exam Stress Balancer.
Supports Q-Learning and SARSA (config-based via params.yaml)
"""

import mlflow
import yaml

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    Q_TABLE_PATH,
    MODEL_METADATA_PATH,
)

from src.utils.logger import get_logger
from src.utils.helpers import save_pickle, save_json, build_metadata, Timer

# 🔥 RL TRAINING FUNCTIONS
from src.rl.training.train_q_learning import train_q_learning
from src.rl.training.train_sarsa import train_sarsa

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PARAMS FROM params.yaml (DVC CONTROL)
# ═══════════════════════════════════════════════════════════════════════════════

def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["training"]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_training_logic(config: dict) -> dict:
    algo = config.get("algorithm", "q_learning")

    logger.info(f"🚀 Training using: {algo}")

    if algo == "q_learning":
        q_table, metrics = train_q_learning(config)

    elif algo == "sarsa":
        q_table, metrics = train_sarsa(config)

    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    logger.info("✅ RL training completed")

    return {
        "q_table": q_table,
        "episode_rewards": metrics.get("episode_rewards", []),
        "final_epsilon": metrics["final_epsilon"],
        "metrics": {
            "avg_reward": metrics["avg_reward_last_100"],
            "final_epsilon": metrics["final_epsilon"],
            "total_episodes": metrics["total_episodes"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_training() -> None:

    # 🔥 Always load from params.yaml
    config = load_params()

    logger.info("══════════════════════════════════════════════")
    logger.info("  Exam Stress Balancer — Training Pipeline")
    logger.info("══════════════════════════════════════════════")
    logger.info("Config: %s", config)

    import os
    tracking_uri = "file:///" + os.getcwd().replace("\\", "/") + "/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    try:
        with mlflow.start_run(run_name="train_rl") as run:
            run_id = run.info.run_id
            logger.info("MLflow run_id: %s", run_id)

            # ✅ Log params
            mlflow.log_params(config)
            mlflow.log_param("algorithm", config.get("algorithm"))

            # 🔥 TRAIN
            with Timer("training") as timer:
                result = run_training_logic(config)

            # ✅ Log metrics
            mlflow.log_metric("avg_reward", result["metrics"]["avg_reward"])
            mlflow.log_metric("final_epsilon", result["metrics"]["final_epsilon"])
            mlflow.log_metric("total_episodes", result["metrics"]["total_episodes"])
            mlflow.log_metric("training_duration_sec", round(timer.elapsed, 3))

            # 📊 Log reward curve
            rewards = result["episode_rewards"]
            if rewards:
                sample_step = max(1, len(rewards) // 50)
                for i in range(0, len(rewards), sample_step):
                    mlflow.log_metric("episode_reward", rewards[i], step=i)

            # 💾 Save model based on algorithm
            algo = config.get("algorithm")

            if algo == "q_learning":
                model_path = Q_TABLE_PATH.parent / "q_learning.pkl"

            elif algo == "sarsa":
                model_path = Q_TABLE_PATH.parent / "sarsa.pkl"

            else:
                raise ValueError("Invalid algorithm")

            save_pickle(result["q_table"], model_path)
            mlflow.log_artifact(str(model_path))

            # 📦 Metadata
            metadata = build_metadata(
                extra={
                    "mlflow_run_id": run_id,
                    "algorithm": algo,
                    "training_config": config,
                    "metrics": result["metrics"],
                    "final_epsilon": result["final_epsilon"],
                    "model_path": str(model_path),
                }
            )

            save_json(metadata, MODEL_METADATA_PATH)
            mlflow.log_artifact(str(MODEL_METADATA_PATH))

            logger.info("✔ Training complete.")
            logger.info("Model → %s", model_path)
            logger.info("Meta → %s", MODEL_METADATA_PATH)
            logger.info("Run ID → %s", run_id)

    except Exception as e:
        logger.exception("❌ Training failed")
        raise e


# CLI
if __name__ == "__main__":
    run_training()