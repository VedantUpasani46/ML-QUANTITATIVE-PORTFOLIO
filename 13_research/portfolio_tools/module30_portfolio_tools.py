"""
Module 30: Portfolio Tools
==========================
Risk analytics, factor attribution, performance reporting
"""

import numpy as np
import pandas as pd


class PortfolioTools:
    """Portfolio analysis and risk tools."""
    
    def __init__(self):
        pass
    
    def calculate_sharpe(self, returns, rf=0.02):
        """Calculate Sharpe ratio."""
        excess_returns = returns - rf/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def factor_attribution(self, portfolio_returns, factor_returns):
        """Attribute portfolio returns to factors."""
        # Simple regression-based attribution
        betas = {}
        for factor_name, factor_rets in factor_returns.items():
            cov = np.cov(portfolio_returns, factor_rets)[0, 1]
            var = np.var(factor_rets)
            betas[factor_name] = cov / var if var > 0 else 0
        
        return betas
    
    def risk_metrics(self, returns):
        """Calculate comprehensive risk metrics."""
        return {
            'sharpe': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'volatility': np.std(returns) * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'skewness': pd.Series(returns).skew(),
            'kurtosis': pd.Series(returns).kurt()
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 30: PORTFOLIO TOOLS")
    print("=" * 70)
    
    tools = PortfolioTools()
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003
    
    print(f"\n── Risk Metrics ──")
    metrics = tools.risk_metrics(returns)
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  VaR (95%): {metrics['var_95']*100:.2f}%")
    print(f"  Skewness: {metrics['skewness']:.2f}")
    
    # Factor attribution
    print(f"\n── Factor Attribution ──")
    factor_returns = {
        'market': np.random.randn(252) * 0.012,
        'value': np.random.randn(252) * 0.008,
        'momentum': np.random.randn(252) * 0.006
    }
    betas = tools.factor_attribution(returns, factor_returns)
    for factor, beta in betas.items():
        print(f"  {factor.capitalize()} Beta: {beta:.3f}")
    
    print(f"\n✓ Module 30 complete")
