import numpy as np

class StationaryAction:
    def __init__(self, mean: float, std_dev: float = 1.0) -> None:
        self.mean_reward = mean
        self.std_dev = std_dev

    def apply(self) -> float:
        return np.random.normal(self.mean_reward, self.std_dev)

    
class NonStationaryAction:
    def __init__(self, initial_mean: float, initial_std_dev: float = 1.0) -> None:
        self.mean_reward = initial_mean
        self.std_dev = initial_std_dev

    def apply(self) -> float:
        self.mean_reward += self._random_walk()
        return np.random.normal(self.mean_reward, self.std_dev)

    def _random_walk(self) -> float:
        return np.random.normal(0, 0.01)