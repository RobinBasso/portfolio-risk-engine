from core.portfolio import Portfolio
import numpy as np


def test_portfolio():
    assets = ["Equity", "Bonds"]
    weights = [0.6, 0.4]

    p = Portfolio(assets, weights)

    # Single return
    r = np.array([0.01, -0.002])
    pr = p.portfolio_return(r)
    print("Portfolio return:", pr)

    # Path simulation
    returns_matrix = np.array([
        [0.01, -0.002],
        [-0.005, 0.001],
        [0.002, 0.0]
    ])

    path = p.portfolio_path(returns_matrix)
    print("Portfolio path:", path)
    print(p)


if __name__ == "__main__":
    test_portfolio()