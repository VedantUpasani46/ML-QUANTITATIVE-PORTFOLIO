"""
Module 39: Order Flow Toxicity (VPIN)
======================================
Detect toxic order flow using Volume-Synchronized Probability of Informed Trading.
Target: Early warning system for adverse selection.
"""

import numpy as np
import pandas as pd


class OrderFlowToxicity:
    """VPIN - Volume-Synchronized Probability of Informed Trading."""
    
    def __init__(self, bucket_volume: float = 50000):
        self.bucket_volume = bucket_volume
        self.vpin_history = []
    
    def classify_trades(self, prices: np.ndarray, 
                       volumes: np.ndarray) -> tuple:
        """Classify trades as buys or sells using tick rule."""
        price_changes = np.diff(prices)
        
        # Tick rule: up-tick = buy, down-tick = sell
        buys = volumes[1:][(price_changes > 0)]
        sells = volumes[1:][(price_changes < 0)]
        
        return buys.sum(), sells.sum()
    
    def calculate_vpin(self, prices: pd.Series, 
                      volumes: pd.Series,
                      n_buckets: int = 50) -> pd.Series:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
        
        High VPIN = toxic order flow (informed traders active)
        Low VPIN = benign order flow
        """
        vpins = []
        
        # Create volume buckets
        cumulative_volume = volumes.cumsum()
        bucket_boundaries = np.arange(0, cumulative_volume.iloc[-1], 
                                     self.bucket_volume)
        
        for i in range(len(bucket_boundaries) - n_buckets):
            # Get trades in window of n_buckets
            start_vol = bucket_boundaries[i]
            end_vol = bucket_boundaries[i + n_buckets]
            
            mask = (cumulative_volume >= start_vol) & (cumulative_volume < end_vol)
            window_prices = prices[mask].values
            window_volumes = volumes[mask].values
            
            if len(window_prices) < 2:
                continue
            
            # Classify trades
            buy_volume, sell_volume = self.classify_trades(
                window_prices, window_volumes
            )
            
            # VPIN = |Buy - Sell| / Total
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                vpin = abs(buy_volume - sell_volume) / total_volume
                vpins.append(vpin)
            else:
                vpins.append(0)
        
        return pd.Series(vpins)
    
    def detect_toxic_flow(self, vpin: pd.Series, 
                         threshold: float = 0.7) -> np.ndarray:
        """Detect periods of toxic order flow."""
        return (vpin > threshold).values
    
    def calculate_order_imbalance(self, prices: pd.Series,
                                  volumes: pd.Series) -> pd.Series:
        """Calculate simple order imbalance ratio."""
        price_changes = prices.diff()
        
        # Volume-weighted order flow
        buy_volume = volumes[price_changes > 0].rolling(20).sum()
        sell_volume = volumes[price_changes < 0].rolling(20).sum()
        
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        return imbalance.fillna(0)


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 39: ORDER FLOW TOXICITY (VPIN)")
    print("=" * 70)
    
    # Generate synthetic trading data
    np.random.seed(42)
    n_trades = 1000
    
    # Simulate price process with informed trading periods
    prices = [100]
    volumes = []
    
    for i in range(n_trades):
        # Informed trading in middle period (trades 400-600)
        if 400 < i < 600:
            # Informed: larger moves, higher volume
            price_change = np.random.normal(0.02, 0.05)
            volume = np.random.exponential(1000)
        else:
            # Uninformed: smaller moves, lower volume
            price_change = np.random.normal(0, 0.01)
            volume = np.random.exponential(500)
        
        prices.append(prices[-1] * (1 + price_change))
        volumes.append(volume)
    
    prices = pd.Series(prices[:-1])
    volumes = pd.Series(volumes)
    
    # Calculate VPIN
    detector = OrderFlowToxicity(bucket_volume=25000)
    vpin = detector.calculate_vpin(prices, volumes, n_buckets=50)
    
    print(f"\nAnalyzing {n_trades} trades")
    print(f"Average VPIN: {vpin.mean():.3f}")
    print(f"Max VPIN: {vpin.max():.3f}")
    
    # Detect toxic flow
    toxic_periods = detector.detect_toxic_flow(vpin, threshold=0.6)
    pct_toxic = toxic_periods.mean()
    
    print(f"\nToxic Flow Detection:")
    print(f"  Threshold: 0.6")
    print(f"  Toxic periods: {pct_toxic:.1%} of time")
    
    # Order imbalance
    imbalance = detector.calculate_order_imbalance(prices, volumes)
    
    print(f"\nOrder Flow Imbalance:")
    print(f"  Mean: {imbalance.mean():.3f}")
    print(f"  Std: {imbalance.std():.3f}")
    
    print(f"\n✓ VPIN analysis complete")
    print(f"  Key: High VPIN = informed traders, adjust spreads wider")
    print(f"  Application: Market makers use VPIN to avoid adverse selection")
