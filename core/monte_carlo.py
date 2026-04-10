import numpy as np

from core.portfolio import Portfolio
from core.returns_model import ReturnsModel


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for portfolio risk analysis.
    """

    def __init__(self, portfolio: Portfolio, returns_model: ReturnsModel):
        self.portfolio = portfolio
        self.returns_model = returns_model

    def simulate(
        self,
        n_steps: int,
        n_sims: int,
        seed: int = None
    ) -> np.ndarray:
        """
        Simulate portfolio value paths.

        Returns:
            np.ndarray: shape (n_sims, n_steps)
        """
        asset_returns = self.returns_model.simulate(
            n_steps=n_steps,
            n_sims=n_sims,
            seed=seed
        )

        portfolio_paths = []

        portfolio_paths = np.array([
            self.portfolio.portfolio_path(sim)
            for sim in asset_returns
            ])

        return np.array(portfolio_paths)

    def final_values(
        self,
        n_steps: int,
        n_sims: int,
        seed: int = None
    ) -> np.ndarray:
        """
        Return final portfolio values across simulations.
        """
        paths = self.simulate(n_steps, n_sims, seed)
        return paths[:, -1]