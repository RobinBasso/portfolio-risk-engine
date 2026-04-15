import numpy as np


class Scenario:
    """
    Base class for stress scenarios.
    """

    def apply(self, returns: np.ndarray) -> np.ndarray:
        """
        Apply scenario to returns.

        Args:
            returns: shape (n_steps, n_assets)

        Returns:
            modified returns
        """
        raise NotImplementedError("Scenario must implement apply method.")