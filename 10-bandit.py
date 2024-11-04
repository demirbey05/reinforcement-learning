import numpy as np
import matplotlib.pyplot as plt
from action import Action
from typing import List


class Bandit:
    def __init__(self, actions: List[Action], epsilon: float) -> None:
        self.actions = actions
        self.epsilon = epsilon
        self.N = np.zeros(len(actions))
        self.Q = np.zeros(len(actions))
        self.true_action = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self.average_rewards = []
        self.true_action_ratio = []
        self.true_optimal_action = np.argmax([action.mean_reward for action in actions])

    def iterate(self) -> None:
        self.total_steps += 1
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(len(self.actions))
        else:
            q_best = np.max(self.Q)
            action_index= np.random.choice(np.where(self.Q == q_best)[0])


        # Apply only the selected action
        reward = self.actions[action_index].apply()

        # Update total reward and average reward
        self.total_reward += reward
        self.average_rewards.append(self.total_reward / self.total_steps)

        # Update Q-values
        self.N[action_index] += 1
        self.Q[action_index] += (reward - self.Q[action_index]) / self.N[action_index]

        # Check if the selected action is the true optimal action
        if action_index == self.true_optimal_action:
            self.true_action += 1
        self.true_action_ratio.append(self.true_action / self.total_steps)

# Rest of your code remains the same
num_actions = 10
num_iterations = 1000
epsilons = [0, 0.01, 0.1]
all_action_means = [np.random.normal(0, 1)* 5 for _ in range(num_actions)]
optimal_action_ratios = {}
average_rewards = {}

for epsilon in epsilons:
    bandit = Bandit(
        actions=[Action(mean=m) for m in all_action_means],
        epsilon=epsilon)
    for _ in range(num_iterations):
        bandit.iterate()
    optimal_action_ratios[epsilon] = bandit.true_action_ratio
    average_rewards[epsilon] = bandit.average_rewards

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for epsilon in epsilons:
    plt.plot(optimal_action_ratios[epsilon], label=f'Epsilon = {epsilon}')
plt.xlabel('Iterations')
plt.ylabel('Optimal Action Ratio')
plt.title('Optimal Action Ratio over Time')
plt.legend()

plt.subplot(2, 1, 2)
for epsilon in epsilons:
    plt.plot(average_rewards[epsilon], label=f'Epsilon = {epsilon}')
plt.xlabel('Iterations')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.legend()

plt.tight_layout()
plt.show()