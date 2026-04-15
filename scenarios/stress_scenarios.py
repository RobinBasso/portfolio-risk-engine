import numpy as np
from scenarios.base import Scenario


class EquityCrashScenario(Scenario):
    """
    Instant equity crash at t=0.
    """

    def __init__(self, crash_size: float = -0.2):
        self.crash_size = crash_size

    def apply(self, returns: np.ndarray) -> np.ndarray:
        shocked = returns.copy()
        shocked[0, 0] += self.crash_size  # assume asset 0 = equities
        return shocked


class VolatilitySpikeScenario(Scenario):
    """
    Increase volatility by scaling returns.
    """

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def apply(self, returns: np.ndarray) -> np.ndarray:
        return returns * self.factor


class CombinedStressScenario(Scenario):
    """
    Combine multiple stresses.
    """

    def apply(self, returns: np.ndarray) -> np.ndarray:
        shocked = returns.copy()
        shocked[0, 0] -= 0.2  # equity crash
        shocked *= 1.5        # volatility increase
        return shocked