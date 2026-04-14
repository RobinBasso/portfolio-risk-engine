from core.portfolio import Portfolio
from core.returns_model import ReturnsModel
from core.monte_carlo import MonteCarloEngine
from core.risk_metrics import RiskMetrics


def test_risk_metrics():
    # Portfolio
    portfolio = Portfolio(["Equity", "Bonds"], [0.6, 0.4])

    # Model
    mu = [0.0005, 0.0002]
    cov = [
        [0.0001, 0.00002],
        [0.00002, 0.00008]
    ]
    model = ReturnsModel(mu, cov)

    # Engine
    engine = MonteCarloEngine(portfolio, model)

    final_vals = engine.final_values(n_steps=50, n_sims=5000, seed=42)

    var_95 = RiskMetrics.var(final_vals, 0.95)
    cvar_95 = RiskMetrics.cvar(final_vals, 0.95)

    # Drawdown on one path
    path = engine.simulate(n_steps=50, n_sims=1, seed=42)[0]
    mdd = RiskMetrics.max_drawdown(path)

    print("\n--- Risk Summary ---")
    print(f"VaR (95%): {var_95:,.0f} €")
    print(f"CVaR (95%): {cvar_95:,.0f} €")
    print(f"Max Drawdown: {mdd:.2%}")


if __name__ == "__main__":
    test_risk_metrics()