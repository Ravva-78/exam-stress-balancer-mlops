"""
Baseline Comparison Pipeline for Exam Stress Balancer.

Evaluates 5 policies:
  1. Q-Learning (trained)
  2. SARSA (trained)
  3. Hybrid RL (60% Q-Learning + 40% SARSA)
  4. Random Policy
  5. Rule-Based Policy

Outputs a comparison table to stdout and saves results to
  results/baseline_comparison.json
"""

import statistics
import numpy as np
from pathlib import Path

from src.rl.student_environment import StudentEnvironment
from src.rl.agent.state_discretizer import discretize_state
from src.config import MODEL_METADATA_PATH, EVALUATION_CONFIG
from src.utils.helpers import load_pickle, load_json, save_json, utc_now_iso
from src.utils.logger import get_logger
from src.evaluation.baselines import RandomPolicy, RuleBasedPolicy

logger = get_logger(__name__)

RESULTS_PATH = Path("results") / "baseline_comparison.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_models():
    base = Path("models")
    q_path = base / "q_learning.pkl"
    s_path = base / "sarsa.pkl"

    models = {}
    if q_path.exists():
        models["q_learning"] = load_pickle(q_path)
    else:
        logger.warning("q_learning.pkl not found — skipping Q-Learning and Hybrid")

    if s_path.exists():
        models["sarsa"] = load_pickle(s_path)
    else:
        logger.warning("sarsa.pkl not found — skipping SARSA and Hybrid")

    return models


def _predict_rl(q_table, state_dict, fallback_actions=None):
    """Greedy action from a Q-table given a raw state dict."""
    if fallback_actions is None:
        fallback_actions = [0, 1, 2]
    d_state = discretize_state(state_dict)
    if d_state in q_table:
        return max(q_table[d_state], key=q_table[d_state].get)
    return np.random.choice(fallback_actions)


def _predict_hybrid(q_table_q, q_table_s, state_dict):
    """60/40 weighted hybrid of Q-Learning and SARSA."""
    actions = [0, 1, 2]
    d_state = discretize_state(state_dict)
    qv_q = q_table_q.get(d_state, {a: 0.0 for a in actions})
    qv_s = q_table_s.get(d_state, {a: 0.0 for a in actions})
    combined = {a: 0.6 * qv_q[a] + 0.4 * qv_s[a] for a in actions}
    return max(combined, key=combined.get)


# ─── Episode Runner ──────────────────────────────────────────────────────────

def _run_episodes(policy_fn, num_episodes, max_steps, seed=0):
    """
    Run *num_episodes* episodes with *policy_fn(state_dict) -> int*.
    Returns dict with episode_rewards, episode_lengths, success_count.
    """
    rng = np.random.default_rng(seed)
    env = StudentEnvironment(total_days=15)

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for _ in range(num_episodes):
        state = env.reset()
        env.state["fatigue"]    = int(rng.integers(0, 101))
        env.state["stress"]     = int(rng.integers(0, 101))
        env.state["retention"]  = float(rng.uniform(0, 1))
        env.state["difficulty"] = rng.choice(["easy", "medium", "hard"])

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = policy_fn(env.state)
            _, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0:
            success_count += 1

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_count": success_count,
    }


def _aggregate(raw, num_episodes):
    r = raw["episode_rewards"]
    l = raw["episode_lengths"]
    return {
        "mean_reward":     round(statistics.mean(r), 4),
        "std_reward":      round(statistics.stdev(r) if len(r) > 1 else 0.0, 4),
        "success_rate":    round(raw["success_count"] / num_episodes, 4),
        "mean_ep_length":  round(statistics.mean(l), 2),
        "min_reward":      round(min(r), 4),
        "max_reward":      round(max(r), 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_baseline_comparison(config=None):
    config = config or EVALUATION_CONFIG
    num_ep   = config["num_episodes"]
    max_step = config["max_steps"]
    seed     = config.get("seed", 0)

    logger.info("══════════════════════════════════════════════")
    logger.info("  Exam Stress Balancer — Baseline Comparison")
    logger.info("══════════════════════════════════════════════")

    models = _load_models()

    policies = {}

    # Random baseline
    _rand = RandomPolicy()
    policies["random"] = lambda s: _rand.choose_action(s)

    # Rule-based baseline
    _rule = RuleBasedPolicy()
    policies["rule_based"] = lambda s: _rule.choose_action(s)

    # RL policies (only if models loaded)
    if "q_learning" in models:
        qt_q = models["q_learning"]
        policies["q_learning"] = lambda s, qt=qt_q: _predict_rl(qt, s)

    if "sarsa" in models:
        qt_s = models["sarsa"]
        policies["sarsa"] = lambda s, qt=qt_s: _predict_rl(qt, s)

    if "q_learning" in models and "sarsa" in models:
        qt_q2 = models["q_learning"]
        qt_s2 = models["sarsa"]
        policies["hybrid"] = lambda s, q=qt_q2, ss=qt_s2: _predict_hybrid(q, ss, s)

    results = {}
    for name, fn in policies.items():
        logger.info(f"Evaluating: {name} ...")
        raw = _run_episodes(fn, num_ep, max_step, seed=seed)
        results[name] = _aggregate(raw, num_ep)

    # ─── Print Table ─────────────────────────────────────────────────────────
    col_w = 16
    metrics = ["mean_reward", "success_rate", "mean_ep_length"]
    header = f"{'Policy':<{col_w}}" + "".join(f"{m:>{col_w}}" for m in metrics)

    print("\n" + "═" * len(header))
    print("  Baseline Comparison Results")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for policy_name, agg in results.items():
        row = f"{policy_name:<{col_w}}"
        for m in metrics:
            row += f"{agg[m]:>{col_w}}"
        print(row)

    print("═" * len(header) + "\n")

    # ─── Save JSON ───────────────────────────────────────────────────────────
    output = {
        "project": "exam-stress-balancer",
        "compared_at": utc_now_iso(),
        "evaluation_config": config,
        "policies_compared": list(results.keys()),
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, RESULTS_PATH)
    logger.info(f"✅ Baseline comparison saved → {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_baseline_comparison()
