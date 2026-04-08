import numpy as np
from typing import List


class Portfolio:
    """
    Represents a portfolio of assets with fixed weights.

    Attributes:
        assets (List[str]): Names of the assets
        weights (np.ndarray): Portfolio weights (must sum to 1)
        initial_value (float): Initial portfolio value
    """

    def __init__(
        self,
        assets: List[str],
        weights: List[float],
        initial_value: float = 1_000_000
    ):
        self.assets = assets
        self.weights = np.array(weights, dtype=float)
        self.initial_value = float(initial_value)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate portfolio inputs."""
        if len(self.assets) != len(self.weights):
            raise ValueError(
                f"Assets ({len(self.assets)}) and weights ({len(self.weights)}) must have same length."
            )

        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError(
                f"Weights must sum to 1. Current sum: {np.sum(self.weights):.4f}"
            )

        if np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative.")

    def portfolio_return(self, asset_returns: np.ndarray) -> float:
        """
        Compute portfolio return from asset returns.

        Args:
            asset_returns (np.ndarray): shape (n_assets,)

        Returns:
            float: portfolio return
        """
        if asset_returns.shape[0] != len(self.weights):
            raise ValueError("Asset returns dimension mismatch.")

        return float(np.dot(self.weights, asset_returns))

    def portfolio_path(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Compute portfolio value path over time.

        Args:
            returns_matrix (np.ndarray): shape (n_steps, n_assets)

        Returns:
            np.ndarray: portfolio value over time
        """
        if returns_matrix.shape[1] != len(self.weights):
            raise ValueError("Returns matrix dimension mismatch.")

        portfolio_returns = returns_matrix @ self.weights
        return self.initial_value * np.cumprod(1 + portfolio_returns)

    def __repr__(self) -> str:
        return (
            f"Portfolio(n_assets={len(self.assets)}, "
            f"initial_value={self.initial_value:,.0f})"
        )