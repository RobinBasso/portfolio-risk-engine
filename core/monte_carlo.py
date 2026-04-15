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
        seed: int = None,
        scenario=None
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

        for sim in asset_returns:
            if scenario is not None:
                sim = scenario.apply(sim)
                
            path = self.portfolio.portfolio_path(sim)
            portfolio_paths.append(path)

        return np.array(portfolio_paths)

    def final_values(
        self,
        n_steps: int,
        n_sims: int,
        seed: int = None,
        scenario=None
    ) -> np.ndarray:
        """
        Return final portfolio values across simulations.
        """
        paths = self.simulate(n_steps, n_sims, seed, scenario)
        return paths[:, -1]