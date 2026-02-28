#!/usr/bin/env python3
"""
Script to generate all 7 complete module implementations
"""

implementations = {

# Module 8: Options Strategies
"02_derivatives/options_strategies/module08_options_strategies.py": '''"""
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
    
    print(f"\\nMarket: Stock=${S}, Vol={sigma*100:.0f}%, Days={T*365:.0f}")
    
    # Iron Condor
    print(f"\\n── Iron Condor ──")
    ic = strat.iron_condor(S, K1=90, K2=95, K3=105, K4=110, T=T, sigma=sigma)
    print(f"  Credit: ${ic['total_credit']:.2f}")
    print(f"  Max Profit: ${ic['max_profit']:.2f}")
    print(f"  Max Loss: ${ic['max_loss']:.2f}")
    print(f"  Profit Zone: ${ic['breakeven_lower']:.2f} - ${ic['breakeven_upper']:.2f}")
    
    # Long Straddle
    print(f"\\n── Long Straddle ──")
    ls = strat.long_straddle(S, K=100, T=T, sigma=sigma)
    print(f"  Total Cost: ${ls['total_cost']:.2f}")
    print(f"  Breakevens: ${ls['breakeven_lower']:.2f} - ${ls['breakeven_upper']:.2f}")
    print(f"  Needs {ls['needs_move']:.1f}% move")
    
    print(f"\\n✓ Module 8 complete")
''',

# Module 10: Options MM & Inventory
"02_derivatives/options_strategies/module10_options_mm_inventory.py": '''"""
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
    
    print(f"\\nMarket: Stock=${S}, Strike=${K}, Vol={sigma*100:.0f}%")
    
    # Initial quote
    print(f"\\n── Initial Quote (Empty Inventory) ──")
    quote = mm.quote_spread(S, K, T, sigma, 'call')
    print(f"  Bid: ${quote['bid']:.2f}")
    print(f"  Ask: ${quote['ask']:.2f}")
    print(f"  Spread: {quote['spread']:.2f}%")
    
    # Sell 10 calls
    print(f"\\n── After Selling 10 Calls ──")
    mm.update_inventory(-10, quote['greeks'])
    print(f"  Inventory Delta: {mm.inventory['delta']:.2f}")
    print(f"  Inventory Gamma: {mm.inventory['gamma']:.4f}")
    
    # New quote with inventory
    quote2 = mm.quote_spread(S, K, T, sigma, 'call')
    print(f"  New Spread: {quote2['spread']:.2f}% (wider due to risk)")
    
    # Hedge calculation
    hedge = mm.hedge_delta(S)
    print(f"\\n── Delta Hedge ──")
    print(f"  Need to buy {hedge['shares_to_hedge']:.0f} shares")
    print(f"  Hedge cost: ${hedge['hedge_cost']:,.0f}")
    
    print(f"\\n✓ Module 10 complete")
''',

# Module 20: Commodities
"08_macro/commodities/module20_commodities.py": '''"""
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
    print(f"\\n── Scenario 1: Oil in Backwardation ──")
    oil_curve = [75.0, 73.5, 72.0, 71.0]  # Front to back
    structure = ct.identify_market_structure(oil_curve)
    print(f"  Market Structure: {structure}")
    print(f"  Front Month: ${oil_curve[0]:.2f}")
    print(f"  Back Month: ${oil_curve[-1]:.2f}")
    
    strategy = ct.optimal_roll_strategy(oil_curve)
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Expected Return: {strategy['expected_return']*100:.2f}%")
    
    # Natural gas (contango)
    print(f"\\n── Scenario 2: Natural Gas in Contango ──")
    gas_curve = [3.0, 3.2, 3.4, 3.5]
    structure = ct.identify_market_structure(gas_curve)
    print(f"  Market Structure: {structure}")
    
    strategy = ct.optimal_roll_strategy(gas_curve)
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Expected Cost: {strategy['expected_cost']*100:.2f}%")
    
    # Spread trade
    print(f"\\n── Calendar Spread Trade ──")
    spread = ct.spread_trade(oil_curve[0], oil_curve[1])
    print(f"  Front-Back Spread: ${spread['spread']:.2f} ({spread['spread_pct']:.2f}%)")
    print(f"  Signal: {spread['signal']}")
    
    print(f"\\n✓ Module 20 complete")
''',

# Module 22: Data Pipelines
"09_alternative_data/pipelines/module22_data_pipelines.py": '''"""
Module 22: Data Engineering Pipelines
=====================================
ETL pipelines, data quality, feature stores
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataPipeline:
    """ETL pipeline for quantitative data."""
    
    def __init__(self):
        self.data_quality_checks = []
    
    def extract(self, source):
        """Extract data from source."""
        # Simulate data extraction
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        return data
    
    def validate_schema(self, df, required_columns):
        """Validate data schema."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return {'valid': False, 'missing': missing}
        return {'valid': True}
    
    def check_null_values(self, df):
        """Check for null values."""
        null_counts = df.isnull().sum()
        return {'null_counts': null_counts.to_dict(), 'has_nulls': null_counts.sum() > 0}
    
    def check_data_range(self, df, column, min_val, max_val):
        """Check if data is within expected range."""
        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
        return {'out_of_range': out_of_range, 'valid': out_of_range == 0}
    
    def transform(self, df):
        """Transform data."""
        df['returns'] = df['price'].pct_change()
        df['log_volume'] = np.log(df['volume'])
        df['ma_5'] = df['price'].rolling(window=5).mean()
        df['ma_20'] = df['price'].rolling(window=20).mean()
        return df
    
    def load(self, df, destination):
        """Load data to destination."""
        # Simulate loading
        print(f"  Loading {len(df)} rows to {destination}")
        return {'status': 'success', 'rows': len(df)}


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 22: DATA ENGINEERING PIPELINES")
    print("=" * 70)
    
    pipeline = DataPipeline()
    
    print(f"\\n── Extract ──")
    data = pipeline.extract('market_data')
    print(f"  Extracted {len(data)} rows")
    
    print(f"\\n── Validate ──")
    schema_check = pipeline.validate_schema(data, ['date', 'price', 'volume'])
    print(f"  Schema valid: {schema_check['valid']}")
    
    null_check = pipeline.check_null_values(data)
    print(f"  Null values: {null_check['has_nulls']}")
    
    range_check = pipeline.check_data_range(data, 'price', 50, 150)
    print(f"  Range check: {'PASS' if range_check['valid'] else 'FAIL'}")
    
    print(f"\\n── Transform ──")
    transformed = pipeline.transform(data)
    print(f"  Added features: returns, log_volume, ma_5, ma_20")
    
    print(f"\\n── Load ──")
    result = pipeline.load(transformed, 'feature_store')
    print(f"  Status: {result['status']}")
    
    print(f"\\n✓ Module 22 complete")
''',

# Module 22: Data Engineering (duplicate name, different implementation)
"09_alternative_data/pipelines/module22_data_engineering.py": '''"""
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
    
    print(f"\\n── Technical Indicators ──")
    data = fe.technical_indicators(data)
    print(f"  Added: returns, volatility, rsi, momentum")
    print(f"  RSI (latest): {data['rsi'].iloc[-1]:.2f}")
    
    print(f"\\n── Cross-Sectional Features ──")
    data = fe.cross_sectional_features(data)
    print(f"  Added: sector_mean_return, relative_strength")
    
    print(f"\\n── Lag Features ──")
    data = fe.lag_features(data, ['returns'], lags=[1, 5])
    print(f"  Added: returns_lag_1, returns_lag_5")
    
    print(f"\\n  Total features: {len(data.columns)}")
    print(f"\\n✓ Module 22 complete")
''',

# Module 29: Research Workflow
"13_research/workflow/module29_research_workflow.py": '''"""
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
    
    print(f"\\n── Experiment 1: Momentum Signal ──")
    momentum_signal = np.roll(returns, 20)
    results = research.backtest_signal(momentum_signal, returns)
    exp1 = research.log_experiment("Momentum (20-day)", None, results)
    print(f"  IC: {results['ic']:.4f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Status: {exp1['status']}")
    
    print(f"\\n── Experiment 2: Mean Reversion ──")
    mr_signal = -np.roll(returns, 5)
    results = research.backtest_signal(mr_signal, returns)
    exp2 = research.log_experiment("Mean Reversion (5-day)", None, results)
    print(f"  IC: {results['ic']:.4f}")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  Status: {exp2['status']}")
    
    print(f"\\n── Research Summary ──")
    summary = research.generate_report()
    print(f"  Total Experiments: {summary['total_experiments']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Average IC: {summary['avg_ic']:.4f}")
    
    print(f"\\n✓ Module 29 complete")
''',

# Module 30: Portfolio Tools
"13_research/portfolio_tools/module30_portfolio_tools.py": '''"""
Module 30: Portfolio Tools
==========================
Risk analytics, factor attribution, performance reporting
"""

import numpy as np
import pandas as pd


class PortfolioTools:
    """Portfolio analysis and risk tools."""
    
    def __init__(self):
        pass
    
    def calculate_sharpe(self, returns, rf=0.02):
        """Calculate Sharpe ratio."""
        excess_returns = returns - rf/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def factor_attribution(self, portfolio_returns, factor_returns):
        """Attribute portfolio returns to factors."""
        # Simple regression-based attribution
        betas = {}
        for factor_name, factor_rets in factor_returns.items():
            cov = np.cov(portfolio_returns, factor_rets)[0, 1]
            var = np.var(factor_rets)
            betas[factor_name] = cov / var if var > 0 else 0
        
        return betas
    
    def risk_metrics(self, returns):
        """Calculate comprehensive risk metrics."""
        return {
            'sharpe': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'volatility': np.std(returns) * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'skewness': pd.Series(returns).skew(),
            'kurtosis': pd.Series(returns).kurt()
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 30: PORTFOLIO TOOLS")
    print("=" * 70)
    
    tools = PortfolioTools()
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003
    
    print(f"\\n── Risk Metrics ──")
    metrics = tools.risk_metrics(returns)
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  VaR (95%): {metrics['var_95']*100:.2f}%")
    print(f"  Skewness: {metrics['skewness']:.2f}")
    
    # Factor attribution
    print(f"\\n── Factor Attribution ──")
    factor_returns = {
        'market': np.random.randn(252) * 0.012,
        'value': np.random.randn(252) * 0.008,
        'momentum': np.random.randn(252) * 0.006
    }
    betas = tools.factor_attribution(returns, factor_returns)
    for factor, beta in betas.items():
        print(f"  {factor.capitalize()} Beta: {beta:.3f}")
    
    print(f"\\n✓ Module 30 complete")
'''

}

# Write all implementations
for filepath, content in implementations.items():
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Created {filepath}")

print(f"\n{'='*70}")
print(f"  ALL 7 IMPLEMENTATIONS CREATED")
print(f"{'='*70}")
