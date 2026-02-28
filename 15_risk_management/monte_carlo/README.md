# Module 36: Monte Carlo Risk Simulation

**Target:** 10,000+ portfolio path simulations | VaR/CVaR with confidence intervals

## Overview

Monte Carlo simulation for complete portfolio risk analysis. Simulates thousands
of possible futures to estimate Value at Risk (VaR), Conditional VaR (CVaR), and
confidence intervals using bootstrap methodology.

## Why This Matters

- **reliable Risk Estimates:** 10,000 simulations vs analytical formulas
- **Tail Risk:** Captures extreme scenarios that historical data misses
- **Stress Testing:** "What if" scenario analysis
- **Confidence Intervals:** Quantify uncertainty in risk estimates
- **Regulatory:** Basel III recommends Monte Carlo for VaR

## Key Features

- ✅ Geometric Brownian motion simulation
- ✅ Bootstrap confidence intervals for VaR
- ✅ Multiple stress test scenarios
- ✅ CVaR (Expected Shortfall) calculation
- ✅ Probability distributions of outcomes

## Usage

```python
from module36_monte_carlo_risk import MonteCarloRiskSimulator

# Initialize simulator
sim = MonteCarloRiskSimulator(n_simulations=10000)

# Simulate portfolio paths
paths = sim.simulate_portfolio_paths(
    initial_value=1_000_000,
    expected_return=0.10,
    volatility=0.20,
    n_days=252
)

# Calculate VaR with confidence interval
final_values = paths[:, -1]
var_95, ci = sim.calculate_var(final_values, confidence_level=0.95)

print(f"95% VaR: ${var_95:,.0f}")
print(f"95% CI: [${ci[0]:,.0f}, ${ci[1]:,.0f}]")

# Calculate CVaR
cvar_95 = sim.calculate_cvar(final_values, confidence_level=0.95)
print(f"95% CVaR: ${cvar_95:,.0f}")
```

## Run Demo

```bash
python module36_monte_carlo_risk.py
```

## Technical Details

- **Algorithm:** Geometric Brownian Motion
- **Simulations:** 10,000 paths (configurable)
- **Bootstrap:** 1,000 iterations for confidence intervals
- **Performance:** <5 seconds for 252-day horizon
- **Dependencies:** NumPy, Pandas, SciPy

## Interview Insight

**Q (Goldman Sachs Risk):** "Your VaR uses historical simulation. Why not Monte Carlo?"

**A:** Monte Carlo is forward-looking (historical = past only), captures tail risk better 
(10,000 sims vs 252 historical days), provides confidence intervals (VaR = $1M ± $200K), 
and supports stress scenarios. Trade-off: computationally expensive (5 sec vs 0.1 sec). 
We use both: historical for daily monitoring, Monte Carlo for official risk reports.

## Real-World Applications

- **Risk Reporting:** Banks use Monte Carlo for Basel III VaR
- **Stress Testing:** Fed's CCAR requires stress scenario analysis
- **Portfolio Management:** Quantify downside risk for investor reporting
- **Option Pricing:** American options via Monte Carlo
- **Capital Allocation:** Determine reserves for tail events

## Performance

- 10,000 simulations: ~5 seconds
- Trade-off: More sims = better accuracy but slower
- Production: Parallelize with multiprocessing

## References

- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
- Hull, J. (2018). *Risk Management and Financial Institutions*
- Basel Committee (2016). *Minimum capital requirements for market risk*

---

**Module 36 complete. Monte Carlo = gold standard for risk management.**
