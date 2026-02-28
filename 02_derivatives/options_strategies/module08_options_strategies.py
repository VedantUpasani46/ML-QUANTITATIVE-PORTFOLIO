"""
Module 8: Options Trading Strategies
====================================
Iron condors, butterflies, straddles for volatility trading
Target: Positive theta, limited risk
"""

import numpy as np
from scipy.stats import norm


class OptionsStrategies:
    """Options trading strategies for volatility and income generation."""
    
    def __init__(self, r=0.05):
        self.r = r
    
    def black_scholes(self, S, K, T, sigma, option_type='call'):
        """Black-Scholes option pricing."""
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def iron_condor(self, S, K1, K2, K3, K4, T, sigma):
        """Iron Condor strategy."""
        short_put = self.black_scholes(S, K2, T, sigma, 'put')
        long_put = self.black_scholes(S, K1, T, sigma, 'put')
        put_spread_credit = short_put - long_put
        
        short_call = self.black_scholes(S, K3, T, sigma, 'call')
        long_call = self.black_scholes(S, K4, T, sigma, 'call')
        call_spread_credit = short_call - long_call
        
        total_credit = put_spread_credit + call_spread_credit
        max_profit = total_credit
        max_loss = (K2 - K1) - total_credit
        
        be_lower = K2 - total_credit
        be_upper = K3 + total_credit
        
        return {
            'total_credit': total_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_lower': be_lower,
            'breakeven_upper': be_upper
        }
    
    def long_straddle(self, S, K, T, sigma):
        """Long Straddle strategy."""
        call_price = self.black_scholes(S, K, T, sigma, 'call')
        put_price = self.black_scholes(S, K, T, sigma, 'put')
        
        total_cost = call_price + put_price
        be_lower = K - total_cost
        be_upper = K + total_cost
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'total_cost': total_cost,
            'max_loss': total_cost,
            'breakeven_lower': be_lower,
            'breakeven_upper': be_upper,
            'needs_move': total_cost / S * 100
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 8: OPTIONS TRADING STRATEGIES")
    print("=" * 70)
    
    strat = OptionsStrategies(r=0.05)
    S, sigma, T = 100, 0.25, 0.25
    
    print(f"\nMarket: Stock=${S}, Vol={sigma*100:.0f}%, Days={T*365:.0f}")
    
    # Iron Condor
    print(f"\n── Iron Condor ──")
    ic = strat.iron_condor(S, K1=90, K2=95, K3=105, K4=110, T=T, sigma=sigma)
    print(f"  Credit: ${ic['total_credit']:.2f}")
    print(f"  Max Profit: ${ic['max_profit']:.2f}")
    print(f"  Max Loss: ${ic['max_loss']:.2f}")
    print(f"  Profit Zone: ${ic['breakeven_lower']:.2f} - ${ic['breakeven_upper']:.2f}")
    
    # Long Straddle
    print(f"\n── Long Straddle ──")
    ls = strat.long_straddle(S, K=100, T=T, sigma=sigma)
    print(f"  Total Cost: ${ls['total_cost']:.2f}")
    print(f"  Breakevens: ${ls['breakeven_lower']:.2f} - ${ls['breakeven_upper']:.2f}")
    print(f"  Needs {ls['needs_move']:.1f}% move")
    
    print(f"\n✓ Module 8 complete")
