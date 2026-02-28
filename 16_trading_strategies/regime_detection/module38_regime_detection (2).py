"""
Module 38: Advanced Regime Detection
=====================================
Detect market regimes: Bull, Bear, High-Vol, Low-Vol.
Target: 80%+ regime classification accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class RegimeDetector:
    """Market regime detection using Hidden Markov Models."""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = ['Bull Low-Vol', 'Bull High-Vol', 
                             'Bear High-Vol', 'Sideways']
    
    def fit(self, returns: pd.Series, volatility: pd.Series):
        """Fit regime detection model."""
        # Features: returns and volatility
        X = np.column_stack([returns, volatility])
        
        # Gaussian Mixture Model (simplified HMM)
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        
        self.model.fit(X)
        
        # Assign regime labels based on means
        means = self.model.means_
        self._assign_regime_labels(means)
    
    def _assign_regime_labels(self, means):
        """Assign interpretable labels to regimes."""
        labels = []
        for mean_ret, mean_vol in means:
            if mean_ret > 0.001 and mean_vol < 0.02:
                labels.append('Bull Low-Vol')
            elif mean_ret > 0.001 and mean_vol > 0.02:
                labels.append('Bull High-Vol')
            elif mean_ret < -0.001 and mean_vol > 0.02:
                labels.append('Bear High-Vol')
            else:
                labels.append('Sideways')
        
        self.regime_labels = labels
    
    def predict(self, returns: pd.Series, 
               volatility: pd.Series) -> np.ndarray:
        """Predict regime for each time period."""
        X = np.column_stack([returns, volatility])
        regime_ids = self.model.predict(X)
        return regime_ids
    
    def get_regime_probabilities(self, returns: pd.Series,
                                volatility: pd.Series) -> np.ndarray:
        """Get probability of each regime."""
        X = np.column_stack([returns, volatility])
        return self.model.predict_proba(X)
    
    def current_regime(self, returns: pd.Series,
                      volatility: pd.Series) -> str:
        """Get most recent regime."""
        regime_id = self.predict(returns, volatility)[-1]
        return self.regime_labels[regime_id]


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 38: REGIME DETECTION")
    print("=" * 70)
    
    # Generate synthetic market data with regime shifts
    np.random.seed(42)
    n_days = 500
    
    returns = []
    vols = []
    
    # Bull low-vol (days 0-150)
    returns.extend(np.random.normal(0.001, 0.01, 150))
    vols.extend([0.01] * 150)
    
    # Bear high-vol (days 150-250)
    returns.extend(np.random.normal(-0.002, 0.04, 100))
    vols.extend([0.04] * 100)
    
    # Bull high-vol (days 250-400)
    returns.extend(np.random.normal(0.001, 0.03, 150))
    vols.extend([0.03] * 150)
    
    # Sideways (days 400-500)
    returns.extend(np.random.normal(0, 0.015, 100))
    vols.extend([0.015] * 100)
    
    returns = pd.Series(returns)
    volatility = pd.Series(vols)
    
    # Fit model
    detector = RegimeDetector(n_regimes=4)
    detector.fit(returns, volatility)
    
    # Predict regimes
    regimes = detector.predict(returns, volatility)
    
    print(f"\nDetected {detector.n_regimes} regimes")
    print(f"Regime labels: {detector.regime_labels}")
    
    # Regime statistics
    print(f"\nRegime Distribution:")
    for i, label in enumerate(detector.regime_labels):
        pct = (regimes == i).mean()
        print(f"  {label}: {pct:.1%} of time")
    
    # Current regime
    current = detector.current_regime(returns, volatility)
    print(f"\nCurrent Regime: {current}")
    
    # Probabilities
    probs = detector.get_regime_probabilities(returns, volatility)[-1]
    print(f"Regime Probabilities:")
    for label, prob in zip(detector.regime_labels, probs):
        print(f"  {label}: {prob:.1%}")
    
    print(f"\n✓ Regime detection complete")
    print(f"  Key: Adapt strategy based on market conditions")
