# Module 37: Statistical Arbitrage & Pairs Trading

**Target:** Market-neutral returns | Sharpe 2.0+ | Cointegration-based

## Overview

Statistical arbitrage strategy using pairs trading. Identifies cointegrated stock pairs 
and trades mean reversion of the spread. Market-neutral strategy that profits from 
relative value differences.

## Why This Matters

- **Market Neutral:** Zero beta to market, uncorrelated returns
- **Mean Reversion:** Exploits temporary deviations from equilibrium
- **Quantifiable:** Statistical tests (cointegration) for pair selection
- **Scalable:** Can run hundreds of pairs simultaneously
- **Proven:** Used by Renaissance, DE Shaw, Two Sigma

## Key Features

- ✅ Engle-Granger cointegration testing
- ✅ Hedge ratio calculation via OLS
- ✅ Z-score normalization for signals
- ✅ Complete backtesting framework
- ✅ Sharpe ratio & drawdown metrics

## Usage

```python
from module37_stat_arb import PairsTradingStrategy

# Initialize strategy
strategy = PairsTradingStrategy(lookback=60)

# Find cointegrated pairs
pairs = strategy.find_cointegrated_pairs(prices_df, threshold=0.05)

# Calculate spread
z_score = strategy.calculate_spread(price1, price2)

# Generate signals
signals = strategy.generate_signals(z_score, entry=2.0, exit=0.0)

# Backtest
results = strategy.backtest_pair(price1, price2)
print(f"Sharpe: {results['sharpe']:.2f}")
```

## Run Demo

```bash
python module37_stat_arb.py
```

## Strategy Logic

1. **Pair Selection:** Test for cointegration (p-value < 0.05)
2. **Spread Construction:** `spread = stock1 - β × stock2`
3. **Normalization:** `z-score = (spread - μ) / σ`
4. **Signal Generation:**
   - Long when z < -2 (spread too low)
   - Short when z > +2 (spread too high)
   - Exit when |z| < 0 (mean reversion)

## Interview Insight

**Q (Renaissance Technologies):** "Why pairs trading over single-stock alpha?"

**A:** Pairs trading is market-neutral (beta = 0, survives crashes), has high Sharpe 
(2.0+ vs 1.5 for long-only), exploits mean reversion (well-documented), and statistical 
tests provide quantifiable edge. Trade-off: lower capacity than directional strategies.

## Real-World Applications

- Hedge funds (core strategy at market-neutral funds)
- Proprietary trading desks
- Cross-asset arbitrage (futures, FX, commodities)

## References

- Engle, R. & Granger, C. (1987). *Co-integration and Error Correction*
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods*
- Gatev et al. (2006). *Pairs Trading: Performance of a Relative-Value Strategy*
