"""
Advanced Risk Models: CVaR & Expected Shortfall
================================================
Better risk metrics than VaR for tail risk management.
Target: Accurately measure and hedge tail risk.
"""
import numpy as np

class AdvancedRiskMetrics:
    def __init__(self):
        pass
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        
        Average of losses beyond VaR. Better than VaR because:
        - Coherent risk measure
        - Captures tail distribution
        - Used by Basel III
        """
        var_threshold = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var_threshold].mean()
        return cvar
    
    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Ratio of right tail (gains) to left tail (losses)."""
        right_tail = np.abs(np.percentile(returns, 95))
        left_tail = np.abs(np.percentile(returns, 5))
        return right_tail / (left_tail + 1e-8)

if __name__ == "__main__":
    print("Module 35: Advanced Risk - CVaR, Expected Shortfall")
    
    # Generate returns with fat tails
    np.random.seed(42)
    returns = np.random.standard_t(df=3, size=1000) * 0.02
    
    risk = AdvancedRiskMetrics()
    
    var_95 = risk.calculate_var(returns, 0.95)
    cvar_95 = risk.calculate_cvar(returns, 0.95)
    max_dd = risk.calculate_max_drawdown(returns)
    tail_ratio = risk.calculate_tail_ratio(returns)
    
    print(f"95% VaR: {var_95:.2%}")
    print(f"95% CVaR (ES): {cvar_95:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Tail Ratio: {tail_ratio:.2f}")
