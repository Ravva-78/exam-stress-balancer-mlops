import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = {}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):

        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        current_q = self.q_table[state][action]

        max_next_q = max(self.q_table[next_state].values())

        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )
