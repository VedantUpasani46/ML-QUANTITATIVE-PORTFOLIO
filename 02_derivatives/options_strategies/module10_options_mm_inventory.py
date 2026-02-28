"""
Module 10: Options Market Making & Inventory Management
=======================================================
Market making with inventory risk and Greeks hedging
"""

import numpy as np
from scipy.stats import norm


class OptionsMarketMaker:
    """Options market maker with inventory management."""
    
    def __init__(self, r=0.05, max_gamma=1000, max_vega=5000):
        self.r = r
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.inventory = {}
    
    def black_scholes(self, S, K, T, sigma, option_type='call'):
        """Black-Scholes pricing."""
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def calculate_greeks(self, S, K, T, sigma, option_type='call'):
        """Calculate option Greeks."""
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - self.r * K * np.exp(-self.r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2))
        
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}
    
    def quote_spread(self, S, K, T, sigma, option_type='call', base_spread=0.05):
        """Generate bid-ask spread based on inventory."""
        fair_value = self.black_scholes(S, K, T, sigma, option_type)
        greeks = self.calculate_greeks(S, K, T, sigma, option_type)
        
        # Adjust spread based on gamma risk
        current_gamma = abs(self.inventory.get('gamma', 0))
        gamma_adjustment = (current_gamma / self.max_gamma) * 0.10
        
        # Adjust spread based on vega risk
        current_vega = abs(self.inventory.get('vega', 0))
        vega_adjustment = (current_vega / self.max_vega) * 0.05
        
        total_spread = base_spread + gamma_adjustment + vega_adjustment
        half_spread = fair_value * total_spread / 2
        
        return {
            'bid': fair_value - half_spread,
            'ask': fair_value + half_spread,
            'fair_value': fair_value,
            'spread': total_spread * 100,  # in percentage
            'greeks': greeks
        }
    
    def update_inventory(self, quantity, greeks):
        """Update inventory after trade."""
        for key in ['delta', 'gamma', 'vega']:
            self.inventory[key] = self.inventory.get(key, 0) + quantity * greeks[key]
    
    def hedge_delta(self, S):
        """Calculate shares needed to hedge delta."""
        current_delta = self.inventory.get('delta', 0)
        shares_to_hedge = -current_delta
        hedge_cost = shares_to_hedge * S
        
        return {
            'current_delta': current_delta,
            'shares_to_hedge': shares_to_hedge,
            'hedge_cost': hedge_cost
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 10: OPTIONS MARKET MAKING & INVENTORY")
    print("=" * 70)
    
    mm = OptionsMarketMaker(r=0.05, max_gamma=1000, max_vega=5000)
    S, K, T, sigma = 100, 100, 0.25, 0.25
    
    print(f"\nMarket: Stock=${S}, Strike=${K}, Vol={sigma*100:.0f}%")
    
    # Initial quote
    print(f"\n── Initial Quote (Empty Inventory) ──")
    quote = mm.quote_spread(S, K, T, sigma, 'call')
    print(f"  Bid: ${quote['bid']:.2f}")
    print(f"  Ask: ${quote['ask']:.2f}")
    print(f"  Spread: {quote['spread']:.2f}%")
    
    # Sell 10 calls
    print(f"\n── After Selling 10 Calls ──")
    mm.update_inventory(-10, quote['greeks'])
    print(f"  Inventory Delta: {mm.inventory['delta']:.2f}")
    print(f"  Inventory Gamma: {mm.inventory['gamma']:.4f}")
    
    # New quote with inventory
    quote2 = mm.quote_spread(S, K, T, sigma, 'call')
    print(f"  New Spread: {quote2['spread']:.2f}% (wider due to risk)")
    
    # Hedge calculation
    hedge = mm.hedge_delta(S)
    print(f"\n── Delta Hedge ──")
    print(f"  Need to buy {hedge['shares_to_hedge']:.0f} shares")
    print(f"  Hedge cost: ${hedge['hedge_cost']:,.0f}")
    
    print(f"\n✓ Module 10 complete")
