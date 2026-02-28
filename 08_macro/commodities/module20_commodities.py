"""
Module 20: Commodity Trading
============================
Oil futures, contango/backwardation, roll yield strategies
"""

import numpy as np


class CommodityTrading:
    """Commodity futures trading and roll yield strategies."""
    
    def __init__(self):
        pass
    
    def calculate_roll_yield(self, front_month, back_month):
        """Calculate annualized roll yield."""
        return (front_month - back_month) / back_month * 12  # Annualized
    
    def identify_market_structure(self, futures_curve):
        """Identify contango or backwardation."""
        if futures_curve[0] < futures_curve[-1]:
            return "contango"
        else:
            return "backwardation"
    
    def optimal_roll_strategy(self, futures_curve):
        """Determine optimal rolling strategy."""
        structure = self.identify_market_structure(futures_curve)
        
        if structure == "backwardation":
            # Roll early, capture positive roll yield
            return {
                'strategy': 'roll_early',
                'reason': 'Capture positive roll yield in backwardation',
                'expected_return': (futures_curve[0] - futures_curve[1]) / futures_curve[1]
            }
        else:
            # Roll late, minimize negative roll yield
            return {
                'strategy': 'roll_late',
                'reason': 'Minimize negative roll yield in contango',
                'expected_cost': (futures_curve[1] - futures_curve[0]) / futures_curve[0]
            }
    
    def spread_trade(self, front, back):
        """Calendar spread trading strategy."""
        spread = front - back
        mean_spread = np.mean([front, back])
        spread_pct = spread / mean_spread * 100
        
        return {
            'spread': spread,
            'spread_pct': spread_pct,
            'signal': 'buy_spread' if spread_pct < -5 else 'sell_spread' if spread_pct > 5 else 'neutral'
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 20: COMMODITY TRADING")
    print("=" * 70)
    
    ct = CommodityTrading()
    
    # Oil futures curve (backwardation)
    print(f"\n── Scenario 1: Oil in Backwardation ──")
    oil_curve = [75.0, 73.5, 72.0, 71.0]  # Front to back
    structure = ct.identify_market_structure(oil_curve)
    print(f"  Market Structure: {structure}")
    print(f"  Front Month: ${oil_curve[0]:.2f}")
    print(f"  Back Month: ${oil_curve[-1]:.2f}")
    
    strategy = ct.optimal_roll_strategy(oil_curve)
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Expected Return: {strategy['expected_return']*100:.2f}%")
    
    # Natural gas (contango)
    print(f"\n── Scenario 2: Natural Gas in Contango ──")
    gas_curve = [3.0, 3.2, 3.4, 3.5]
    structure = ct.identify_market_structure(gas_curve)
    print(f"  Market Structure: {structure}")
    
    strategy = ct.optimal_roll_strategy(gas_curve)
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Expected Cost: {strategy['expected_cost']*100:.2f}%")
    
    # Spread trade
    print(f"\n── Calendar Spread Trade ──")
    spread = ct.spread_trade(oil_curve[0], oil_curve[1])
    print(f"  Front-Back Spread: ${spread['spread']:.2f} ({spread['spread_pct']:.2f}%)")
    print(f"  Signal: {spread['signal']}")
    
    print(f"\n✓ Module 20 complete")
