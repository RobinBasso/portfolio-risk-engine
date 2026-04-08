import numpy as np
from typing import Optional


class ReturnsModel:
    """
    Multivariate returns model using a Gaussian distribution.

    Attributes:
        mu (np.ndarray): Expected returns vector
        cov (np.ndarray): Covariance matrix
    """

    def __init__(self, mu, cov):
        self.mu = np.array(mu, dtype=float)
        self.cov = np.array(cov, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate dimensions and properties."""
        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("Covariance matrix must be square.")

        if self.cov.shape[0] != len(self.mu):
            raise ValueError("Mu and covariance dimension mismatch.")

        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix must be symmetric.")

    def simulate(
        self,
        n_steps: int,
        n_sims: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate asset returns.

        Args:
            n_steps (int): number of time steps
            n_sims (int): number of simulation paths
            seed (int, optional): random seed

        Returns:
            np.ndarray: shape (n_sims, n_steps, n_assets)
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.multivariate_normal(
            mean=self.mu,
            cov=self.cov,
            size=(n_sims, n_steps)
        )

    def __repr__(self) -> str:
        return f"ReturnsModel(n_assets={len(self.mu)})"