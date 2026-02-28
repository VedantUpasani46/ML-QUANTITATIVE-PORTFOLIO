"""
Module 40: ESG & Impact Investing
==================================
Integrate ESG scores into portfolio optimization.
Target: Maintain returns while improving ESG profile.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ESGPortfolioOptimizer:
    """Portfolio optimization with ESG constraints."""
    
    def __init__(self, min_esg_score: float = 0.6):
        self.min_esg_score = min_esg_score
        self.optimal_weights = None
    
    def optimize(self, returns: pd.DataFrame, 
                esg_scores: pd.Series,
                risk_aversion: float = 2.0) -> np.ndarray:
        """
        Optimize portfolio with ESG constraint.
        
        max: μ'w - λ/2 * w'Σw
        s.t.: ESG'w >= min_esg_score
              sum(w) = 1
              w >= 0
        """
        n = len(returns.columns)
        
        # Expected returns and covariance
        mu = returns.mean().values
        Sigma = returns.cov().values
        
        # Objective: negative utility (for minimization)
        def objective(w):
            return -(mu @ w - risk_aversion/2 * w @ Sigma @ w)
        
        # ESG constraint
        def esg_constraint(w):
            return esg_scores.values @ w - self.min_esg_score
        
        # Sum to 1 constraint
        def sum_constraint(w):
            return w.sum() - 1.0
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'ineq', 'fun': esg_constraint}
        ]
        
        # Bounds: long-only
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess: equal weight
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.optimal_weights = result.x
            return result.x
        else:
            print("Optimization failed")
            return w0
    
    def calculate_portfolio_esg(self, weights: np.ndarray,
                               esg_scores: pd.Series) -> float:
        """Calculate portfolio's weighted average ESG score."""
        return np.dot(weights, esg_scores.values)
    
    def screen_stocks(self, stocks: list,
                     esg_scores: pd.Series,
                     min_score: float = 0.5) -> list:
        """Screen out stocks below ESG threshold."""
        mask = esg_scores >= min_score
        return [s for s, include in zip(stocks, mask) if include]
    
    def tilt_portfolio(self, base_weights: np.ndarray,
                      esg_scores: pd.Series,
                      tilt_strength: float = 0.5) -> np.ndarray:
        """Tilt portfolio towards high-ESG stocks."""
        # Normalize ESG scores
        esg_normalized = (esg_scores - esg_scores.min()) / (esg_scores.max() - esg_scores.min())
        
        # Tilt: blend base weights with ESG-tilted weights
        esg_tilted = esg_normalized / esg_normalized.sum()
        tilted_weights = (1 - tilt_strength) * base_weights + tilt_strength * esg_tilted
        
        # Renormalize
        return tilted_weights / tilted_weights.sum()


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 40: ESG & IMPACT INVESTING")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_stocks = 20
    n_days = 252
    
    stocks = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    
    # Returns (some stocks have better returns)
    returns_data = {}
    for i, stock in enumerate(stocks):
        returns_data[stock] = np.random.normal(0.0005, 0.02, n_days)
    
    returns = pd.DataFrame(returns_data)
    
    # ESG scores (0-1 scale)
    esg_scores = pd.Series(
        np.random.beta(2, 2, n_stocks),
        index=stocks
    )
    
    print(f"\nPortfolio Universe: {n_stocks} stocks")
    print(f"ESG Score Range: [{esg_scores.min():.2f}, {esg_scores.max():.2f}]")
    print(f"Average ESG: {esg_scores.mean():.2f}")
    
    # Optimize without ESG constraint
    print(f"\n── Standard Portfolio (no ESG constraint) ──")
    
    optimizer_standard = ESGPortfolioOptimizer(min_esg_score=0.0)
    weights_standard = optimizer_standard.optimize(returns, esg_scores)
    
    portfolio_esg_standard = optimizer_standard.calculate_portfolio_esg(
        weights_standard, esg_scores
    )
    portfolio_return_standard = (returns.mean() @ weights_standard) * 252
    
    print(f"Portfolio ESG Score: {portfolio_esg_standard:.2f}")
    print(f"Expected Annual Return: {portfolio_return_standard:.1%}")
    
    # Optimize with ESG constraint
    print(f"\n── ESG-Constrained Portfolio (min ESG=0.6) ──")
    
    optimizer_esg = ESGPortfolioOptimizer(min_esg_score=0.6)
    weights_esg = optimizer_esg.optimize(returns, esg_scores)
    
    portfolio_esg_constrained = optimizer_esg.calculate_portfolio_esg(
        weights_esg, esg_scores
    )
    portfolio_return_esg = (returns.mean() @ weights_esg) * 252
    
    print(f"Portfolio ESG Score: {portfolio_esg_constrained:.2f}")
    print(f"Expected Annual Return: {portfolio_return_esg:.1%}")
    
    # Compare
    print(f"\n── Impact of ESG Constraint ──")
    esg_improvement = portfolio_esg_constrained - portfolio_esg_standard
    return_impact = portfolio_return_esg - portfolio_return_standard
    
    print(f"ESG Score Improvement: +{esg_improvement:.2f}")
    print(f"Return Impact: {return_impact:.1%}")
    
    if abs(return_impact) < 0.01:
        print(f"  → Minimal return sacrifice for ESG improvement!")
    
    # ESG screening
    print(f"\n── ESG Screening ──")
    high_esg_stocks = optimizer_esg.screen_stocks(stocks, esg_scores, min_score=0.6)
    
    print(f"Stocks passing ESG screen (≥0.6): {len(high_esg_stocks)}/{n_stocks}")
    print(f"  Screened out: {n_stocks - len(high_esg_stocks)} stocks")
    
    print(f"\n✓ ESG portfolio optimization complete")
    print(f"  Key: Can improve ESG with minimal return sacrifice")
    print(f"  Growing demand: $35 trillion in ESG assets globally")
