"""
Factor Models: Fama-French & Custom Factors
============================================
Target: Decompose Returns into Factors

Multi-factor models separate alpha (skill) from beta (factor exposure).
Critical for understanding what drives your returns.

Why This Matters:
  - ALPHA DECOMPOSITION: Is your return skill or just factor exposure?
  - RISK MANAGEMENT: Control unwanted factor exposures
  - PORTFOLIO CONSTRUCTION: Target specific factor tilts
  - PERFORMANCE ATTRIBUTION: What drove P&L?
  - INVESTOR RELATIONS: LPs want to know your sources of return

Target: Isolate alpha from factor betas

Interview insight (AQR Portfolio Manager):
Q: "Your strategy returned 18% last year. Is that alpha or beta?"
A: "**Factor decomposition**: (1) **Market beta**—Strategy has β=0.3 to S&P 500.
    Market returned 25%, so 0.3×25% = 7.5% from market exposure. (2) **Value
    factor**—Long value stocks, β=0.4. Value factor returned 15%, contributes
    0.4×15% = 6%. (3) **Momentum**—β=0.2, momentum returned 10% = 2%. (4)
    **Size**—β=-0.1 (short small caps), size returned -5% = 0.5%. Total factor
    exposure: 7.5% + 6% + 2% + 0.5% = 16%. **True alpha**: 18% - 16% = 2%.
    This is MORE impressive than 18% return because it's pure skill, not factor
    luck. We can replicate 16% with cheap ETFs. The 2% alpha is the value-add.
    LPs pay 2/20 fees for alpha, not beta."

Mathematical Foundation:
------------------------
Fama-French 3-Factor Model:
  r_i - r_f = α + β_mkt(r_mkt - r_f) + β_SMB·SMB + β_HML·HML + ε
  
  Where:
  - α = Jensen's alpha (skill)
  - β_mkt = Market exposure
  - SMB = Small Minus Big (size factor)
  - HML = High Minus Low (value factor)

Custom Factor Construction:
  Factor return = Long top quintile - Short bottom quintile
  
  Rebalance monthly/quarterly

References:
  - Fama & French (1993). Common Risk Factors. JFE.
  - Carhart (1997). Persistence in Mutual Fund Performance. JF.
  - Frazzini & Pedersen (2014). Betting Against Beta. JFE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, List


class FamaFrenchModel:
    """
    Fama-French factor model for return attribution.
    
    Decomposes returns into market, size, value, momentum factors.
    """
    
    def __init__(self):
        self.factor_loadings = {}
        self.alpha = None
        self.r_squared = None
    
    def fit(self, returns: pd.Series, factors: pd.DataFrame) -> Dict:
        """
        Fit factor model via regression.
        
        Args:
            returns: Strategy returns (T,)
            factors: Factor returns (T, K) with columns like ['MKT', 'SMB', 'HML']
        
        Returns:
            Fitted parameters
        """
        # Align dates
        common_index = returns.index.intersection(factors.index)
        y = returns.loc[common_index].values
        X = factors.loc[common_index].values
        
        # Regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Store results
        self.alpha = model.intercept_
        for i, col in enumerate(factors.columns):
            self.factor_loadings[col] = model.coef_[i]
        
        # R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot
        
        return {
            'alpha': self.alpha,
            'factor_loadings': self.factor_loadings,
            'r_squared': self.r_squared
        }
    
    def attribute_returns(self, returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Attribute returns to factors.
        
        Returns:
            DataFrame with attribution
        """
        # Calculate contribution of each factor
        attribution = pd.DataFrame(index=returns.index)
        
        for factor_name, beta in self.factor_loadings.items():
            attribution[factor_name] = beta * factors[factor_name]
        
        attribution['alpha'] = self.alpha
        attribution['residual'] = returns - attribution.sum(axis=1)
        attribution['total'] = returns
        
        return attribution


class CustomFactorBuilder:
    """
    Build custom factors from stock characteristics.
    
    Example: Momentum factor, value factor, quality factor.
    """
    
    def __init__(self):
        pass
    
    def build_momentum_factor(self, returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Build momentum factor: Long winners, short losers.
        
        Args:
            returns: Stock returns (T, N)
            lookback: Lookback period for momentum
        
        Returns:
            Factor returns (T,)
        """
        factor_returns = []
        
        for t in range(lookback, len(returns)):
            # Calculate past returns
            past_returns = returns.iloc[t-lookback:t].sum()
            
            # Long top 20%, short bottom 20%
            top_20 = past_returns.quantile(0.8)
            bottom_20 = past_returns.quantile(0.2)
            
            long_stocks = past_returns[past_returns >= top_20].index
            short_stocks = past_returns[past_returns <= bottom_20].index
            
            # Factor return = Long avg - Short avg
            long_return = returns.iloc[t][long_stocks].mean() if len(long_stocks) > 0 else 0
            short_return = returns.iloc[t][short_stocks].mean() if len(short_stocks) > 0 else 0
            
            factor_return = long_return - short_return
            factor_returns.append(factor_return)
        
        return pd.Series(factor_returns, index=returns.index[lookback:])
    
    def build_value_factor(self, prices: pd.DataFrame, book_values: pd.DataFrame) -> pd.Series:
        """
        Build value factor: Long high B/P, short low B/P.
        
        Args:
            prices: Stock prices (T, N)
            book_values: Book values per share (T, N)
        
        Returns:
            Factor returns (T,)
        """
        # Calculate B/P ratio
        bp_ratio = book_values / prices
        
        factor_returns = []
        
        for t in range(1, len(prices)):
            bp_t = bp_ratio.iloc[t-1]  # Use previous period's B/P
            
            # Long top 20%, short bottom 20%
            top_20 = bp_t.quantile(0.8)
            bottom_20 = bp_t.quantile(0.2)
            
            long_stocks = bp_t[bp_t >= top_20].index
            short_stocks = bp_t[bp_t <= bottom_20].index
            
            # Returns
            returns_t = (prices.iloc[t] - prices.iloc[t-1]) / prices.iloc[t-1]
            
            long_return = returns_t[long_stocks].mean() if len(long_stocks) > 0 else 0
            short_return = returns_t[short_stocks].mean() if len(short_stocks) > 0 else 0
            
            factor_return = long_return - short_return
            factor_returns.append(factor_return)
        
        return pd.Series(factor_returns, index=prices.index[1:])


# CLI demonstration
if __name__ == "__main__":
    print("═" * 70)
    print("  FAMA-FRENCH FACTOR MODELS")
    print("  Target: Decompose Returns into Alpha & Beta")
    print("═" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    
    n_days = 252 * 3  # 3 years
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Generate factor returns
    mkt_returns = np.random.randn(n_days) * 0.01 + 0.0003  # Market
    smb_returns = np.random.randn(n_days) * 0.005  # Size
    hml_returns = np.random.randn(n_days) * 0.005  # Value
    mom_returns = np.random.randn(n_days) * 0.008  # Momentum
    
    factors = pd.DataFrame({
        'MKT': mkt_returns,
        'SMB': smb_returns,
        'HML': hml_returns,
        'MOM': mom_returns
    }, index=dates)
    
    # Generate strategy returns (with factor exposure + alpha)
    true_alpha = 0.0002  # 5bps daily alpha
    true_betas = {'MKT': 0.3, 'SMB': -0.1, 'HML': 0.4, 'MOM': 0.2}
    
    strategy_returns = true_alpha + \
                      true_betas['MKT'] * mkt_returns + \
                      true_betas['SMB'] * smb_returns + \
                      true_betas['HML'] * hml_returns + \
                      true_betas['MOM'] * mom_returns + \
                      np.random.randn(n_days) * 0.003  # Idiosyncratic risk
    
    strategy_returns = pd.Series(strategy_returns, index=dates)
    
    print(f"\n  Strategy Performance:")
    print(f"    Total return: {(1 + strategy_returns).prod() - 1:.1%}")
    print(f"    Annual return: {strategy_returns.mean() * 252:.1%}")
    print(f"    Annual vol: {strategy_returns.std() * np.sqrt(252):.1%}")
    print(f"    Sharpe: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}")
    
    # Fit factor model
    print(f"\n── Factor Decomposition ──")
    
    ff_model = FamaFrenchModel()
    results = ff_model.fit(strategy_returns, factors)
    
    print(f"\n  Regression Results:")
    print(f"    Alpha (daily): {results['alpha']:.4%}")
    print(f"    Alpha (annual): {results['alpha'] * 252:.2%}")
    print(f"    R-squared: {results['r_squared']:.2%}")
    
    print(f"\n  Factor Loadings:")
    for factor, beta in results['factor_loadings'].items():
        true_beta = true_betas.get(factor, 0)
        print(f"    β_{factor}: {beta:.3f} (true: {true_beta:.3f})")
    
    # Attribution
    print(f"\n── Return Attribution ──")
    
    attribution = ff_model.attribute_returns(strategy_returns, factors)
    
    # Cumulative contribution
    cumulative_attr = attribution.cumsum()
    
    print(f"\n  Cumulative Contribution (3 years):")
    for col in ['MKT', 'SMB', 'HML', 'MOM', 'alpha']:
        contrib = cumulative_attr[col].iloc[-1]
        pct = contrib / cumulative_attr['total'].iloc[-1] * 100
        print(f"    {col:<10}: {contrib:>8.2%} ({pct:>5.1f}% of total)")
    
    print(f"    {'Total':<10}: {cumulative_attr['total'].iloc[-1]:>8.2%}")
    
    # Build custom factors
    print(f"\n── Custom Factor Construction ──")
    
    # Generate stock data
    n_stocks = 50
    stock_returns = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.015 + 0.0005,
        index=dates,
        columns=[f'Stock_{i}' for i in range(n_stocks)]
    )
    
    factor_builder = CustomFactorBuilder()
    
    # Momentum factor
    momentum_factor = factor_builder.build_momentum_factor(stock_returns, lookback=60)
    
    print(f"\n  Custom Momentum Factor:")
    print(f"    Annual return: {momentum_factor.mean() * 252:.1%}")
    print(f"    Annual vol: {momentum_factor.std() * np.sqrt(252):.1%}")
    print(f"    Sharpe: {momentum_factor.mean() / momentum_factor.std() * np.sqrt(252):.2f}")
    
    print(f"\n  Factor models decompose returns into:")
    print(f"    • Alpha (skill) = {results['alpha'] * 252:.1%}")
    print(f"    • Factor betas (exposures)")
    print(f"    • Residual (unexplained)")
    print(f"\n  Critical for understanding sources of return!")
    
    print("\nModule 32 complete. Essential for institutional investors.")

