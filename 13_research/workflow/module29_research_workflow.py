"""
Module 29: Research Workflow
============================
Systematic alpha research framework
"""

import pandas as pd
import numpy as np
from datetime import datetime


class AlphaResearch:
    """Systematic framework for alpha research."""
    
    def __init__(self):
        self.experiments = []
    
    def log_experiment(self, hypothesis, data, results):
        """Log research experiment."""
        experiment = {
            'timestamp': datetime.now(),
            'hypothesis': hypothesis,
            'ic': results.get('ic', 0),
            'sharpe': results.get('sharpe', 0),
            'status': 'pass' if results.get('ic', 0) > 0.05 else 'fail'
        }
        self.experiments.append(experiment)
        return experiment
    
    def backtest_signal(self, signal, returns):
        """Backtest alpha signal."""
        # Calculate IC (Information Coefficient)
        ic = np.corrcoef(signal[:-1], returns[1:])[0, 1]
        
        # Calculate strategy returns
        strategy_returns = signal[:-1] * returns[1:]
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        return {
            'ic': ic,
            'sharpe': sharpe,
            'mean_return': np.mean(strategy_returns),
            'volatility': np.std(strategy_returns)
        }
    
    def generate_report(self):
        """Generate research report."""
        df = pd.DataFrame(self.experiments)
        summary = {
            'total_experiments': len(df),
            'passed': (df['status'] == 'pass').sum(),
            'avg_ic': df['ic'].mean(),
            'best_ic': df['ic'].max()
        }
        return summary


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 29: RESEARCH WORKFLOW")
    print("=" * 70)
    
    research = AlphaResearch()
    
    # Simulate research experiments
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    
    print(f"\n── Experiment 1: Momentum Signal ──")
    momentum_signal = np.roll(returns, 20)
    results = research.backtest_signal(momentum_signal, returns)
    exp1 = research.log_experiment("Momentum (20-day)", None, results)
    print(f"  IC: {results['ic']:.4f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Status: {exp1['status']}")
    
    print(f"\n── Experiment 2: Mean Reversion ──")
    mr_signal = -np.roll(returns, 5)
    results = research.backtest_signal(mr_signal, returns)
    exp2 = research.log_experiment("Mean Reversion (5-day)", None, results)
    print(f"  IC: {results['ic']:.4f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Status: {exp2['status']}")
    
    print(f"\n── Research Summary ──")
    summary = research.generate_report()
    print(f"  Total Experiments: {summary['total_experiments']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Average IC: {summary['avg_ic']:.4f}")
    
    print(f"\n✓ Module 29 complete")
