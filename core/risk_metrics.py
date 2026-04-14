import numpy as np


class RiskMetrics:
    """
    Collection of portfolio risk metrics.
    """

    @staticmethod
    def var(final_values: np.ndarray, alpha: float = 0.95) -> float:
        """
        Value at Risk (VaR).

        Args:
            final_values: simulated final portfolio values
            alpha: confidence level (e.g. 0.95)

        Returns:
            VaR as a loss (positive number)
        """
        initial_value = np.mean(final_values)  # approximation

        percentile = np.percentile(final_values, (1 - alpha) * 100)

        return initial_value - percentile

    @staticmethod
    def cvar(final_values: np.ndarray, alpha: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        """
        var_threshold = np.percentile(final_values, (1 - alpha) * 100)

        tail_losses = final_values[final_values <= var_threshold]

        return np.mean(np.mean(final_values) - tail_losses)

    @staticmethod
    def max_drawdown(path: np.ndarray) -> float:
        """
        Maximum drawdown of a portfolio path.

        Args:
            path: portfolio value over time

        Returns:
            max drawdown (positive number)
        """
        peak = np.maximum.accumulate(path)
        drawdowns = (peak - path) / peak
        return np.max(drawdowns)