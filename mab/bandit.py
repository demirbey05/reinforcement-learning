import numpy as np
from tqdm import trange


class Bandit:
    def __init__(self, actions, epsilon,initial_value=0,step_size=None,ucb=None,gradient=None) -> None:
        self.actions = actions
        self.epsilon = epsilon
        self.N = np.zeros(len(actions))
        self.Q = np.zeros(len(actions)) + initial_value
        self.initial_value = initial_value
        self.true_optimal_action = np.argmax([action.mean_reward for action in self.actions])
        self.ucb = ucb
        self.time = 0
        self.gradient = gradient
        self.step_size = step_size

    def reset(self) -> None:
        self.N = np.zeros(len(self.actions))
        self.Q = np.zeros(len(self.actions)) + self.initial_value
        self.time = 0


    def act(self) -> int:

        if self.ucb is not None:
            ucb = self.Q + self.ucb * np.sqrt(np.log(self.time + 1) / (self.N + 1e-5))
            return np.random.choice(np.where(ucb == ucb.max())[0])

        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))        
        q_best = np.max(self.Q)
        return np.random.choice(np.where(self.Q == q_best)[0])

    def iterate(self,action_index) -> float:
        # Apply only the selected action
        reward = self.actions[action_index].apply()
        # Update Q-values
        self.N[action_index] += 1
        if self.step_size is None:
            step_size = 1 / self.N[action_index]
        else:
            step_size = self.step_size
        self.Q[action_index] += ((reward - self.Q[action_index]) * step_size)
        self.true_optimal_action = np.argmax([action.mean_reward for action in self.actions])
        self.time += 1

        return reward

def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.iterate(action)
                rewards[i, r, t] = reward
                if action == bandit.true_optimal_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards
