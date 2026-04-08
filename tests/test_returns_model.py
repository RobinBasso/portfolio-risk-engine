from core.returns_model import ReturnsModel


def test_returns_model():
    mu = [0.0005, 0.0002]
    cov = [
        [0.0001, 0.00002],
        [0.00002, 0.00008]
    ]

    model = ReturnsModel(mu, cov)

    sims = model.simulate(n_steps=5, n_sims=3, seed=42)

    print("Shape:", sims.shape)
    print("Sample simulation:\n", sims[0])


if __name__ == "__main__":
    test_returns_model()