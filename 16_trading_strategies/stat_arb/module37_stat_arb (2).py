"""
Module 37: Statistical Arbitrage & Pairs Trading
=================================================
Find cointegrated pairs and trade mean reversion.
Target: Sharpe 2.0+, market-neutral returns.
"""

import numpy as np
import pandas as pd
from scipy import stats


class PairsTradingStrategy:
    """Statistical arbitrage via pairs trading."""
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.pairs = []
    
    def find_cointegrated_pairs(self, prices: pd.DataFrame, 
                               threshold: float = 0.05) -> list:
        """Find cointegrated stock pairs using Engle-Granger test."""
        from statsmodels.tsa.stattools import coint
        
        pairs = []
        n = len(prices.columns)
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = prices.iloc[:, i]
                stock2 = prices.iloc[:, j]
                
                # Cointegration test
                score, pvalue, _ = coint(stock1, stock2)
                
                if pvalue < threshold:
                    pairs.append({
                        'stock1': prices.columns[i],
                        'stock2': prices.columns[j],
                        'pvalue': pvalue,
                        'score': score
                    })
        
        self.pairs = pairs
        return pairs
    
    def calculate_spread(self, price1: pd.Series, 
                        price2: pd.Series) -> pd.Series:
        """Calculate normalized spread between two prices."""
        # Hedge ratio via OLS
        slope = np.cov(price1, price2)[0,1] / np.var(price2)
        spread = price1 - slope * price2
        
        # Normalize
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        return z_score
    
    def generate_signals(self, z_score: pd.Series, 
                        entry: float = 2.0, 
                        exit: float = 0.0) -> pd.Series:
        """Generate trading signals from z-score."""
        signals = pd.Series(index=z_score.index, data=0)
        
        # Long when z < -entry (spread too low)
        signals[z_score < -entry] = 1
        
        # Short when z > entry (spread too high)
        signals[z_score > entry] = -1
        
        # Exit when z crosses exit threshold
        signals[abs(z_score) < exit] = 0
        
        return signals
    
    def backtest_pair(self, price1: pd.Series, price2: pd.Series) -> dict:
        """Backtest pairs trading strategy."""
        z_score = self.calculate_spread(price1, price2)
        signals = self.generate_signals(z_score)
        
        # Calculate returns
        ret1 = price1.pct_change()
        ret2 = price2.pct_change()
        
        # Strategy returns: long stock1, short stock2 (or vice versa)
        strategy_returns = signals.shift(1) * (ret1 - ret2)
        
        # Metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_dd = (strategy_returns.cumsum().cummax() - 
                 strategy_returns.cumsum()).max()
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': (signals.diff() != 0).sum()
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 37: STATISTICAL ARBITRAGE")
    print("=" * 70)
    
    # Generate synthetic cointegrated pairs
    np.random.seed(42)
    n_days = 500
    
    # Cointegrated pair
    x = np.cumsum(np.random.randn(n_days)) + 100
    y = 0.8 * x + np.random.randn(n_days) * 2
    
    price1 = pd.Series(x, name='STOCK_A')
    price2 = pd.Series(y, name='STOCK_B')
    
    # Strategy
    strategy = PairsTradingStrategy(lookback=60)
    
    # Calculate spread
    z_score = strategy.calculate_spread(price1, price2)
    
    print(f"\nPair: {price1.name} / {price2.name}")
    print(f"Spread mean: {z_score.mean():.2f}")
    print(f"Spread std: {z_score.std():.2f}")
    
    # Backtest
    results = strategy.backtest_pair(price1, price2)
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results['total_return']:.1%}")
    print(f"  Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"  Number of Trades: {results['n_trades']}")
    
    print(f"\n✓ Pairs trading complete")
    print(f"  Key: Market-neutral strategy, profits from mean reversion")
