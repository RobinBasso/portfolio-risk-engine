from core.portfolio import Portfolio
from core.returns_model import ReturnsModel
from core.monte_carlo import MonteCarloEngine

import numpy as np


def test_monte_carlo():
    # Portfolio
    assets = ["Equity", "Bonds"]
    weights = [0.6, 0.4]
    portfolio = Portfolio(assets, weights)

    # Returns model
    mu = [0.0005, 0.0002]
    cov = [
        [0.0001, 0.00002],
        [0.00002, 0.00008]
    ]
    model = ReturnsModel(mu, cov)

    # Engine
    engine = MonteCarloEngine(portfolio, model)

    # Simulate
    paths = engine.simulate(n_steps=50, n_sims=1000, seed=42)

    print("Paths shape:", paths.shape)

    final_vals = engine.final_values(n_steps=50, n_sims=1000, seed=42)
    print("Final values sample:", final_vals[:5])
    print("Mean final value:", np.mean(final_vals))
    print("Min:", np.min(final_vals))
    print("Max:", np.max(final_vals))


if __name__ == "__main__":
    test_monte_carlo()