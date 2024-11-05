import numpy as np

class StationaryAction:
    def __init__(self, mean: float, std_dev: float = 1.0) -> None:
        self.mean_reward = mean
        self.std_dev = std_dev

    def apply(self) -> float:
        return np.random.normal(self.mean_reward, self.std_dev)

    
class NonStationaryAction:
    def apply(self) -> float:
        return np.random.normal(np.random.randn(), np.random.randn())