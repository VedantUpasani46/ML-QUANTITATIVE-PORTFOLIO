"""
Module 36: Monte Carlo Risk Simulation
=======================================
Simulate 10,000+ portfolio paths for robust risk estimation.
Target: VaR, CVaR, confidence intervals, stress testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class MonteCarloRiskSimulator:
    """Monte Carlo portfolio risk simulator."""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        np.random.seed(42)
    
    def simulate_portfolio_paths(self, initial_value: float, 
                                 expected_return: float, 
                                 volatility: float,
                                 n_days: int = 252) -> np.ndarray:
        """Simulate portfolio paths using geometric Brownian motion."""
        mu_daily = expected_return / 252
        sigma_daily = volatility / np.sqrt(252)
        
        daily_returns = np.random.normal(mu_daily, sigma_daily, 
                                        (self.n_simulations, n_days))
        paths = initial_value * np.cumprod(1 + daily_returns, axis=1)
        return paths
    
    def calculate_var(self, final_values: np.ndarray, 
                     confidence: float = 0.95) -> Tuple[float, Tuple]:
        """Calculate VaR with bootstrap confidence interval."""
        initial = final_values.mean()
        losses = initial - final_values
        var = np.percentile(losses, confidence * 100)
        
        # Bootstrap CI
        var_bootstrap = []
        for _ in range(1000):
            sample = np.random.choice(losses, len(losses), replace=True)
            var_bootstrap.append(np.percentile(sample, confidence * 100))
        
        ci = (np.percentile(var_bootstrap, 2.5), 
              np.percentile(var_bootstrap, 97.5))
        return var, ci
    
    def calculate_cvar(self, final_values: np.ndarray, 
                      confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        initial = final_values.mean()
        losses = initial - final_values
        var = np.percentile(losses, confidence * 100)
        return losses[losses >= var].mean()
    
    def stress_test(self, initial_value: float, scenarios: Dict) -> Dict:
        """Run stress test scenarios."""
        results = {}
        for name, params in scenarios.items():
            paths = self.simulate_portfolio_paths(
                initial_value,
                params['return'],
                params['volatility']
            )
            final = paths[:, -1]
            var, ci = self.calculate_var(final)
            cvar = self.calculate_cvar(final)
            
            results[name] = {
                'mean': final.mean(),
                'var_95': var,
                'var_ci': ci,
                'cvar_95': cvar,
                'worst': initial_value - final.min()
            }
        return results


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 36: MONTE CARLO RISK SIMULATION")
    print("=" * 70)
    
    sim = MonteCarloRiskSimulator(n_simulations=10000)
    
    # Simulate
    initial = 1_000_000
    paths = sim.simulate_portfolio_paths(initial, 0.10, 0.20)
    final = paths[:, -1]
    
    # Risk metrics
    var_95, ci = sim.calculate_var(final, 0.95)
    cvar_95 = sim.calculate_cvar(final, 0.95)
    
    print(f"\nPortfolio: ${initial:,}")
    print(f"Simulations: {sim.n_simulations:,}")
    print(f"\n95% VaR: ${var_95:,.0f}")
    print(f"95% CI: [${ci[0]:,.0f}, ${ci[1]:,.0f}]")
    print(f"95% CVaR: ${cvar_95:,.0f}")
    
    # Stress test
    scenarios = {
        'Base': {'return': 0.10, 'volatility': 0.20},
        'Crash': {'return': -0.30, 'volatility': 0.50}
    }
    
    results = sim.stress_test(initial, scenarios)
    
    print(f"\nStress Test:")
    for name, res in results.items():
        print(f"  {name}: VaR=${res['var_95']:,.0f}, CVaR=${res['cvar_95']:,.0f}")
    
    print(f"\n✓ Monte Carlo simulation complete")
    print(f"  Key: 10,000 sims capture tail risk better than analytics")
