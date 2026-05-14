"""
Visualization for Exam Stress Balancer.

Generates the following plots (saved to results/plots/):
  1. reward_curve.png          — Episode rewards over training (from evaluation report)
  2. baseline_comparison.png   — Bar chart comparing all 5 policies
  3. stress_fatigue_sim.png    — Stress & fatigue over a sample episode
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

PLOTS_DIR              = Path("results") / "plots"
EVALUATION_REPORT_PATH = Path("evaluations") / "evaluation_report.json"
BASELINE_REPORT_PATH   = Path("results") / "baseline_comparison.json"


# ─── Style ───────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#4F8EF7",
    "secondary": "#F7774F",
    "accent":    "#4FF79E",
    "neutral":   "#AAAAAA",
    "bg":        "#0F1117",
    "fg":        "#E8E8E8",
    "grid":      "#2A2A3A",
}

POLICY_COLORS = {
    "random":     "#AAAAAA",
    "rule_based": "#F7C74F",
    "q_learning": "#4F8EF7",
    "sarsa":      "#F7774F",
    "hybrid":     "#4FF79E",
}

def _apply_dark_style(fig, ax_list):
    """Apply a clean dark theme to a figure + axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in ax_list:
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["fg"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["fg"])
        ax.yaxis.label.set_color(PALETTE["fg"])
        ax.title.set_color(PALETTE["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.7)


# ─── Plot 1: Reward Curve ────────────────────────────────────────────────────

def plot_reward_curve():
    if not EVALUATION_REPORT_PATH.exists():
        logger.warning(f"Evaluation report not found: {EVALUATION_REPORT_PATH}")
        return

    with open(EVALUATION_REPORT_PATH) as f:
        report = json.load(f)

    rewards = report.get("episode_rewards", [])
    if not rewards:
        logger.warning("No episode_rewards in evaluation report")
        return

    episodes = list(range(1, len(rewards) + 1))

    # Smoothed reward (rolling mean window=10)
    window = max(1, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_x = list(range(window, len(rewards) + 1))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_dark_style(fig, [ax])

    ax.plot(episodes, rewards, color=PALETTE["primary"], alpha=0.25, linewidth=0.8, label="Episode Reward")
    ax.plot(smooth_x, smoothed, color=PALETTE["accent"], linewidth=2.0, label=f"Smoothed (w={window})")

    algo = report.get("algorithm", "RL")
    ax.set_title(f"Reward vs Episodes — {algo.upper()}", fontsize=13, pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["fg"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    out = PLOTS_DIR / "reward_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved: {out}")


# ─── Plot 2: Baseline Comparison ─────────────────────────────────────────────

def plot_baseline_comparison():
    if not BASELINE_REPORT_PATH.exists():
        logger.warning(f"Baseline report not found: {BASELINE_REPORT_PATH}")
        return

    with open(BASELINE_REPORT_PATH) as f:
        report = json.load(f)

    results = report.get("results", {})
    if not results:
        logger.warning("No results in baseline comparison report")
        return

    policies      = list(results.keys())
    mean_rewards  = [results[p]["mean_reward"]  for p in policies]
    success_rates = [results[p]["success_rate"] for p in policies]
    colors        = [POLICY_COLORS.get(p, PALETTE["primary"]) for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _apply_dark_style(fig, [ax1, ax2])

    # Mean reward
    bars1 = ax1.bar(policies, mean_rewards, color=colors, edgecolor=PALETTE["grid"], linewidth=0.6)
    ax1.set_title("Mean Reward per Policy", fontsize=12, pad=10)
    ax1.set_xlabel("Policy")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_xticklabels(policies, rotation=20, ha="right")
    for bar, val in zip(bars1, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom",
                 color=PALETTE["fg"], fontsize=9)

    # Success rate
    bars2 = ax2.bar(policies, [r * 100 for r in success_rates],
                    color=colors, edgecolor=PALETTE["grid"], linewidth=0.6)
    ax2.set_title("Success Rate per Policy", fontsize=12, pad=10)
    ax2.set_xlabel("Policy")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 110)
    ax2.set_xticklabels(policies, rotation=20, ha="right")
    for bar, val in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", va="bottom",
                 color=PALETTE["fg"], fontsize=9)

    fig.suptitle("Baseline Policy Comparison", fontsize=14, color=PALETTE["fg"], y=1.02)
    fig.tight_layout()
    out = PLOTS_DIR / "baseline_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved: {out}")


# ─── Plot 3: Stress & Fatigue over a Sample Episode ──────────────────────────

def plot_stress_fatigue_sim():
    """
    Run a short deterministic episode with the rule-based policy and
    plot how stress and fatigue evolve over time.
    """
    try:
        from src.rl.student_environment import StudentEnvironment
        from src.evaluation.baselines import RuleBasedPolicy
    except ImportError as e:
        logger.warning(f"Cannot import environment: {e}")
        return

    env = StudentEnvironment(total_days=15, noise_scale=0)   # no noise for clarity
    random.seed(0)
    state = env.reset()
    env.state.update({"fatigue": 30, "stress": 20, "retention": 0.2, "difficulty": "medium"})

    policy = RuleBasedPolicy()

    stresses  = [env.state["stress"]]
    fatigues  = [env.state["fatigue"]]
    retentions = [env.state["retention"]]
    days = [0]

    done = False
    step = 0
    while not done and step < 20:
        action = policy.choose_action(env.state)
        state, _, done = env.step(action)
        stresses.append(env.state["stress"])
        fatigues.append(env.state["fatigue"])
        retentions.append(env.state["retention"])
        days.append(step + 1)
        step += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _apply_dark_style(fig, [ax1, ax2])

    ax1.plot(days, stresses,  color=PALETTE["secondary"], linewidth=2, label="Stress",  marker="o", markersize=4)
    ax1.plot(days, fatigues,  color=PALETTE["primary"],   linewidth=2, label="Fatigue", marker="s", markersize=4)
    ax1.axhline(70, color=PALETTE["neutral"], linestyle="--", linewidth=0.8, alpha=0.6, label="Danger threshold (70)")
    ax1.set_title("Stress & Fatigue over a Sample Episode", fontsize=12, pad=10)
    ax1.set_ylabel("Level (0–100)")
    ax1.set_ylim(0, 110)
    ax1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["fg"])

    ax2.plot(days, [r * 100 for r in retentions],
             color=PALETTE["accent"], linewidth=2, label="Retention (%)", marker="^", markersize=4)
    ax2.set_title("Knowledge Retention over Episode", fontsize=12, pad=10)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Retention (%)")
    ax2.set_ylim(0, 110)
    ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["fg"])

    fig.tight_layout()
    out = PLOTS_DIR / "stress_fatigue_sim.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved: {out}")


# ─── Master Runner ───────────────────────────────────────────────────────────

def generate_all_plots():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("📊 Generating visualizations...")
    plot_reward_curve()
    plot_baseline_comparison()
    plot_stress_fatigue_sim()
    logger.info(f"✅ All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
