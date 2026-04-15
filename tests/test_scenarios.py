from core.portfolio import Portfolio
from core.returns_model import ReturnsModel
from core.monte_carlo import MonteCarloEngine
from scenarios.stress_scenarios import EquityCrashScenario

import numpy as np


def test_scenarios():
    portfolio = Portfolio(["Equity", "Bonds"], [0.6, 0.4])

    mu = [0.0005, 0.0002]
    cov = [
        [0.0001, 0.00002],
        [0.00002, 0.00008]
    ]
    model = ReturnsModel(mu, cov)

    engine = MonteCarloEngine(portfolio, model)

    # Baseline
    base_vals = engine.final_values(n_steps=50, n_sims=3000, seed=42)

    # Scenario
    scenario = EquityCrashScenario()
    stressed_vals = engine.final_values(
        n_steps=50,
        n_sims=3000,
        seed=42,
        scenario=scenario
    )

    print("\n--- Scenario Comparison ---")
    print("Base mean:", np.mean(base_vals))
    print("Stressed mean:", np.mean(stressed_vals))

    print("Base min:", np.min(base_vals))
    print("Stressed min:", np.min(stressed_vals))

    print("\nImpact:")
    print(f"Mean loss: {np.mean(base_vals) - np.mean(stressed_vals):,.0f} €")


if __name__ == "__main__":
    test_scenarios()