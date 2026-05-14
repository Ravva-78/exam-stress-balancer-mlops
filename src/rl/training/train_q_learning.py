from src.rl.student_environment import StudentEnvironment
from src.rl.agent.state_discretizer import discretize_state
from src.rl.agent.q_learning_agent import QLearningAgent

import numpy as np


def train_q_learning(config: dict = None):
    """
    Q-Learning training function for MLOps pipeline.

    Returns:
        q_table (dict): Learned Q-table
        metrics (dict): Training metrics (avg reward, rewards list)
    """

    EPISODES = config.get("episodes", 3000) if config else 3000
    actions = [0, 1, 2]  # study, revise, sleep

    env = StudentEnvironment(total_days=15)
    agent = QLearningAgent(actions)

    episode_rewards = []

    for episode in range(EPISODES):

        state = env.reset()

        # Randomize initial conditions
        env.state["fatigue"] = np.random.randint(0, 101)
        env.state["stress"] = np.random.randint(0, 101)
        env.state["retention"] = np.random.uniform(0, 1)
        env.state["difficulty"] = np.random.choice(["easy", "medium", "hard"])

        state = discretize_state(state)

        env.current_day = 0

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)
            next_state = discretize_state(next_state)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        # epsilon decay
        agent.epsilon = max(0.05, agent.epsilon * 0.997)

        episode_rewards.append(total_reward)

    avg_reward = float(np.mean(episode_rewards[-100:]))

    metrics = {
        "avg_reward_last_100": avg_reward,
        "total_episodes": EPISODES,
        "final_epsilon": agent.epsilon,
    }

    print("Training Complete")
    print("Average Reward:", avg_reward)
    print("Total learned states:", len(agent.q_table))

    return agent.q_table, metrics