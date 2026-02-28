"""
Module 22: Data Engineering Infrastructure
==========================================
Feature engineering at scale, data quality monitoring
"""

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Feature engineering for ML models."""
    
    def __init__(self):
        self.features = []
    
    def technical_indicators(self, df):
        """Calculate technical indicators."""
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = self.calculate_rsi(df['price'], 14)
        df['momentum'] = df['price'] / df['price'].shift(20) - 1
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def cross_sectional_features(self, df, group_col='sector'):
        """Calculate cross-sectional features."""
        df['sector_mean_return'] = df.groupby(group_col)['returns'].transform('mean')
        df['relative_strength'] = df['returns'] - df['sector_mean_return']
        return df
    
    def lag_features(self, df, columns, lags=[1, 5, 10]):
        """Create lagged features."""
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 22: DATA ENGINEERING INFRASTRUCTURE")
    print("=" * 70)
    
    # Sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.randn(100).cumsum() + 100,
        'sector': np.random.choice(['Tech', 'Finance', 'Energy'], 100)
    })
    
    fe = FeatureEngineering()
    
    print(f"\n── Technical Indicators ──")
    data = fe.technical_indicators(data)
    print(f"  Added: returns, volatility, rsi, momentum")
    print(f"  RSI (latest): {data['rsi'].iloc[-1]:.2f}")
    
    print(f"\n── Cross-Sectional Features ──")
    data = fe.cross_sectional_features(data)
    print(f"  Added: sector_mean_return, relative_strength")
    
    print(f"\n── Lag Features ──")
    data = fe.lag_features(data, ['returns'], lags=[1, 5])
    print(f"  Added: returns_lag_1, returns_lag_5")
    
    print(f"\n  Total features: {len(data.columns)}")
    print(f"\n✓ Module 22 complete")
