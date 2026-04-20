import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core.portfolio import Portfolio
from core.returns_model import ReturnsModel
from core.monte_carlo import MonteCarloEngine
from core.risk_metrics import RiskMetrics
from scenarios.stress_scenarios import EquityCrashScenario


st.title("Portfolio Risk & Scenario Engine")

st.markdown("Evaluate portfolio risk under normal and stressed market conditions.")

# --- Inputs ---
st.sidebar.header("Portfolio Settings")

equity_weight = st.sidebar.slider("Equity Weight", 0.0, 1.0, 0.6)
bond_weight = 1.0 - equity_weight

n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
n_steps = st.sidebar.slider("Time Steps", 10, 252, 50)

use_crash = st.sidebar.checkbox("Apply Equity Crash Scenario")

# --- Portfolio ---
portfolio = Portfolio(["Equity", "Bonds"], [equity_weight, bond_weight])

# --- Returns model ---
mu = [0.0005, 0.0002]
cov = [
    [0.0001, 0.00002],
    [0.00002, 0.00008]
]

model = ReturnsModel(mu, cov)
engine = MonteCarloEngine(portfolio, model)

# --- Scenario ---
scenario = EquityCrashScenario() if use_crash else None

# --- Simulation ---
final_vals = engine.final_values(
    n_steps=n_steps,
    n_sims=n_sims,
    seed=42,
    scenario=scenario
)

# --- Risk metrics ---
var_95 = RiskMetrics.var(final_vals)
cvar_95 = RiskMetrics.cvar(final_vals)

# --- Display ---
st.subheader("Portfolio Distribution")

fig, ax = plt.subplots()
ax.hist(final_vals, bins=50)
ax.set_title("Final Portfolio Value Distribution")
st.pyplot(fig)

# --- Metrics ---
st.subheader("Risk Metrics")

st.metric("VaR (95%)", f"{var_95:,.0f} €")
st.metric("CVaR (95%)", f"{cvar_95:,.0f} €")

# --- Scenario info ---
if use_crash:
    st.warning("Equity crash scenario applied (-20%)")