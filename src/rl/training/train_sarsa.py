"""
SARSA training pipeline (MLOps compatible)
"""

import numpy as np

from src.rl.student_environment import StudentEnvironment
from src.rl.agent.sarsa_agent import SarsaAgent
from src.rl.agent.state_discretizer import discretize_state


def train_sarsa(config: dict):
    """
    Train SARSA agent and return model + metrics
    """

    EPISODES = config.get("episodes", 500)
    alpha = config.get("learning_rate", 0.1)
    gamma = config.get("discount_factor", 0.95)
    epsilon = config.get("epsilon", 1.0)
    epsilon_min = config.get("epsilon_min", 0.05)
    epsilon_decay = config.get("epsilon_decay", 0.997)

    actions = [0, 1, 2]

    env = StudentEnvironment(total_days=15)
    agent = SarsaAgent(actions, alpha=alpha, gamma=gamma, epsilon=epsilon)

    episode_rewards = []

    for episode in range(EPISODES):

        state = env.reset()

        # Randomize initial state
        env.state["fatigue"] = np.random.randint(0, 101)
        env.state["stress"] = np.random.randint(0, 101)
        env.state["retention"] = np.random.uniform(0, 1)
        env.state["difficulty"] = np.random.choice(["easy", "medium", "hard"])

        state = discretize_state(state)

        action = agent.choose_action(state)

        total_reward = 0
        done = False

        while not done:

            next_state, reward, done = env.step(action)
            next_state = discretize_state(next_state)

            next_action = agent.choose_action(next_state)

            # SARSA update
            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            total_reward += reward

            if done:
                break

        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        episode_rewards.append(total_reward)

    # Metrics
    avg_reward_last_100 = float(np.mean(episode_rewards[-100:]))

    metrics = {
        "avg_reward_last_100": avg_reward_last_100,
        "final_epsilon": agent.epsilon,
        "total_episodes": EPISODES,
        "episode_rewards": episode_rewards,
    }

    return agent.q_table, metrics