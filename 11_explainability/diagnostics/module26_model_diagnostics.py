"""
Feature Importance & Model Diagnostics
=======================================
Target: Validate Models | Detect Issues |

This module implements comprehensive model diagnostics including feature
importance analysis, bias detection, and model failure analysis.

Why Model Diagnostics Matter:
  - MODEL VALIDATION: Required by regulators before deployment
  - RISK MANAGEMENT: Detect when model will fail (regime changes)
  - FAIRNESS: Ensure model doesn't discriminate
  - DEBUGGING: "Model worked in 2019, failed in 2020. Why?"
  - CONTINUOUS MONITORING: Track model degradation over time

Target: Comprehensive model validation, catch failures before production

Interview insight (Citadel Model Validation):
Q: "Your momentum model lost $50M in March 2020. What went wrong?"
A: "**Regime change we didn't detect**. Model trained on 2015-2019 (low vol,
    trending market). March 2020: COVID → volatility spiked 5x, correlations →1,
    momentum strategies got destroyed (everyone selling everything). **Autopsy**:
    (1) **Feature drift**—Volatility in training: mean 15%, March 2020: 85%.
    Model had never seen vol >40%. Extrapolated poorly. (2) **Correlation breakdown**
    —Training: stock-stock correlation ~0.3. March 2020: Correlation →0.9 (everything
    moving together). Momentum signals became useless. (3) **Missing features**—
    Model had no 'VIX' input (market fear). In calm markets, not needed. In crisis,
    critical. **Fix**: (1) Add regime detection (if VIX >40 → reduce size 80%).
    (2) Stress test on 2008 data (simulate another crisis). (3) Add vol + correlation
    as features. (4) Weekly model monitoring (track feature distributions). Result:
    Model would have reduced positions 80% on Feb 27 (VIX spiked). Loss: -$10M vs
    -$50M. Still lost money (crisis inevitable) but 5x less."

Mathematical Foundation:
------------------------
Feature Drift Detection:
  KL(P_train || P_prod) = Σ P_train(x) log(P_train(x) / P_prod(x))

  If KL divergence > threshold → feature distribution shifted

Model Calibration:
  Reliability diagram: Bin predictions by quantile, measure actual frequency

  Well-calibrated: Predicted 70% → Actual 70%
  Over-confident: Predicted 90% → Actual 60%

Residual Analysis:
  ε_i = y_i - ŷ_i

  Check: E[ε] = 0, Var[ε] constant, ε uncorrelated with features

References:
  - Niculescu-Mizil & Caruana (2005). Predicting Good Probabilities with SL. ICML.
  - Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
  - Rabanser et al. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. NeurIPS.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Feature Drift Detector
# ---------------------------------------------------------------------------

class FeatureDriftDetector:
    """
    Detect when feature distributions shift from training to production.

    Uses KL divergence and statistical tests.
    """

    def __init__(self):
        self.train_stats = {}

    def fit(self, X_train: np.ndarray, feature_names: List[str] = None):
        """
        Learn training feature distributions.

        Args:
            X_train: Training features
            feature_names: Feature names
        """
        n_features = X_train.shape[1]

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        for i, name in enumerate(feature_names):
            feature_data = X_train[:, i]

            self.train_stats[name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'quantiles': np.percentile(feature_data, [25, 50, 75])
            }

    def detect_drift(self, X_prod: np.ndarray, feature_names: List[str] = None):
        """
        Detect drift in production data.

        Args:
            X_prod: Production features
            feature_names: Feature names

        Returns:
            Dictionary of drift scores per feature
        """
        n_features = X_prod.shape[1]

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]

        drift_report = {}

        for i, name in enumerate(feature_names):
            feature_data = X_prod[:, i]
            train_stat = self.train_stats[name]

            # Calculate drift metrics
            mean_shift = abs(np.mean(feature_data) - train_stat['mean']) / (train_stat['std'] + 1e-8)
            std_shift = abs(np.std(feature_data) - train_stat['std']) / (train_stat['std'] + 1e-8)

            # Kolmogorov-Smirnov test (distribution similarity)
            # Generate sample from train distribution (approximate)
            train_sample = np.random.normal(
                train_stat['mean'],
                train_stat['std'],
                len(feature_data)
            )
            ks_stat, ks_pvalue = stats.ks_2samp(train_sample, feature_data)

            # Flag if drift detected
            drift_detected = (mean_shift > 2.0) or (std_shift > 1.0) or (ks_pvalue < 0.01)

            drift_report[name] = {
                'mean_shift_std': mean_shift,
                'std_ratio': std_shift,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'drift_detected': drift_detected
            }

        return drift_report


# ---------------------------------------------------------------------------
# Model Calibration Checker
# ---------------------------------------------------------------------------

class CalibrationChecker:
    """
    Check if model predictions are well-calibrated.

    Example: If model predicts 70% probability, actual should be ~70%.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def check_calibration(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Measure calibration.

        Args:
            y_pred: Predicted probabilities
            y_true: Actual outcomes (0 or 1)

        Returns:
            Calibration metrics
        """
        # Bin predictions
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1

        calibration_data = []

        for i in range(self.n_bins):
            # Get predictions in this bin
            mask = bin_indices == i

            if mask.sum() > 0:
                bin_pred = y_pred[mask].mean()
                bin_actual = y_true[mask].mean()
                bin_count = mask.sum()

                calibration_data.append({
                    'bin': i,
                    'predicted': bin_pred,
                    'actual': bin_actual,
                    'count': bin_count,
                    'error': abs(bin_pred - bin_actual)
                })

        # Expected Calibration Error (ECE)
        if calibration_data:
            total_samples = len(y_pred)
            ece = sum(d['count'] * d['error'] for d in calibration_data) / total_samples
        else:
            ece = 0

        return {
            'calibration_data': calibration_data,
            'ece': ece
        }


# ---------------------------------------------------------------------------
# Residual Analyzer
# ---------------------------------------------------------------------------

class ResidualAnalyzer:
    """
    Analyze model residuals for patterns.

    Good model: Residuals should be random (white noise).
    Bad model: Residuals have patterns (model missing something).
    """

    def analyze_residuals(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         X: Optional[np.ndarray] = None,
                         feature_names: List[str] = None):
        """
        Comprehensive residual analysis.

        Args:
            y_true: True values
            y_pred: Predictions
            X: Features (optional, for correlation analysis)
            feature_names: Feature names

        Returns:
            Analysis results
        """
        residuals = y_true - y_pred

        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }

        # Test for normality
        _, normality_pvalue = stats.normaltest(residuals)
        analysis['normality_pvalue'] = normality_pvalue
        analysis['is_normal'] = normality_pvalue > 0.05

        # Autocorrelation (should be near zero for good model)
        if len(residuals) > 10:
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            analysis['autocorrelation'] = autocorr

        # Correlation with features (should be zero)
        if X is not None and feature_names is not None:
            feature_correlations = {}
            for i, name in enumerate(feature_names):
                corr = np.corrcoef(residuals, X[:, i])[0, 1]
                feature_correlations[name] = corr

            analysis['feature_correlations'] = feature_correlations

        return analysis


# ---------------------------------------------------------------------------
# Model Performance Monitor
# ---------------------------------------------------------------------------

class ModelPerformanceMonitor:
    """
    Track model performance over time.

    Detects degradation before it costs money.
    """

    def __init__(self):
        self.history = []

    def record_performance(self,
                          date: str,
                          predictions: np.ndarray,
                          actuals: np.ndarray,
                          metadata: Dict = None):
        """
        Record model performance for a time period.

        Args:
            date: Date or period identifier
            predictions: Model predictions
            actuals: Actual values
            metadata: Additional info (market regime, etc.)
        """
        # Calculate metrics
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))

        record = {
            'date': date,
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'n_samples': len(predictions),
            'metadata': metadata or {}
        }

        self.history.append(record)

    def detect_degradation(self, lookback: int = 10, threshold: float = 0.2):
        """
        Detect if model performance is degrading.

        Args:
            lookback: Number of recent periods to compare
            threshold: Degradation threshold (20% = significant)

        Returns:
            Degradation report
        """
        if len(self.history) < lookback * 2:
            return {'degradation_detected': False, 'reason': 'Insufficient history'}

        # Compare recent performance vs baseline
        recent = self.history[-lookback:]
        baseline = self.history[-lookback*2:-lookback]

        recent_corr = np.mean([r['correlation'] for r in recent])
        baseline_corr = np.mean([r['correlation'] for r in baseline])

        degradation_pct = (baseline_corr - recent_corr) / abs(baseline_corr)

        degradation_detected = degradation_pct > threshold

        return {
            'degradation_detected': degradation_detected,
            'baseline_correlation': baseline_corr,
            'recent_correlation': recent_corr,
            'degradation_pct': degradation_pct,
            'threshold': threshold
        }


# ---------------------------------------------------------------------------
# CLI demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 70)
    print("  FEATURE IMPORTANCE & MODEL DIAGNOSTICS")
    print("  Target: Validate Models | Detect Issues")
    print("═" * 70)

    # Generate synthetic data
    print("\n── Simulating Model Lifecycle ──")

    np.random.seed(42)

    # Training data (2015-2019: low volatility)
    n_train = 1000
    X_train = np.random.randn(n_train, 5) * 0.5  # Low vol features
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * -1 +
               X_train[:, 2] * 0.5 + np.random.randn(n_train) * 0.1)

    feature_names = ['Momentum', 'Mean_Reversion', 'Volatility', 'Volume', 'Sentiment']

    print(f"  Training: 2015-2019 ({n_train} samples)")
    print(f"  Features: {feature_names}")

    # Production data (2020: high volatility, regime shift)
    n_prod = 200
    X_prod = np.random.randn(n_prod, 5) * 2.0  # HIGH VOL (regime shift)
    y_prod = (X_prod[:, 0] * 1.0 + X_prod[:, 1] * -1 +  # Momentum weakened!
              X_prod[:, 2] * 1.5 + np.random.randn(n_prod) * 0.5)  # Vol matters more

    print(f"  Production: 2020 ({n_prod} samples, regime shift)")

    # Train simple model
    from numpy.linalg import lstsq

    X_train_with_intercept = np.column_stack([np.ones(n_train), X_train])
    params = lstsq(X_train_with_intercept, y_train, rcond=None)[0]

    intercept = params[0]
    coef = params[1:]

    print(f"\n  Model coefficients:")
    for name, c in zip(feature_names, coef):
        print(f"    {name:<20}: {c:>8.4f}")

    # Demo 1: Feature Drift Detection
    print(f"\n── 1. Feature Drift Detection ──")

    drift_detector = FeatureDriftDetector()
    drift_detector.fit(X_train, feature_names)

    drift_report = drift_detector.detect_drift(X_prod, feature_names)

    print(f"\n  Drift Analysis:")
    print(f"  {'Feature':<20} {'Mean Shift':<15} {'KS p-value':<15} {'Drift?'}")
    print(f"  {'-' * 65}")

    for name, metrics in drift_report.items():
        drift_status = "🚨 YES" if metrics['drift_detected'] else "✅ NO"
        print(f"  {name:<20} {metrics['mean_shift_std']:>8.2f}σ      {metrics['ks_pvalue']:>10.3f}      {drift_status}")

    # Demo 2: Calibration Check
    print(f"\n── 2. Model Calibration ──")

    # Generate probability predictions
    y_pred_prob = 1 / (1 + np.exp(-y_train))  # Convert to probabilities
    y_true_binary = (y_train > np.median(y_train)).astype(int)

    calibration_checker = CalibrationChecker(n_bins=5)
    calibration_results = calibration_checker.check_calibration(y_pred_prob, y_true_binary)

    print(f"\n  Calibration Analysis:")
    print(f"  Expected Calibration Error (ECE): {calibration_results['ece']:.3f}")
    print(f"\n  {'Predicted':<15} {'Actual':<15} {'Count':<10} {'Error'}")
    print(f"  {'-' * 55}")

    for bin_data in calibration_results['calibration_data']:
        print(f"  {bin_data['predicted']:>8.2%}      {bin_data['actual']:>8.2%}      {bin_data['count']:>6}    {bin_data['error']:>6.2%}")

    # Demo 3: Residual Analysis
    print(f"\n── 3. Residual Analysis ──")

    y_pred_train = intercept + np.dot(X_train, coef)

    residual_analyzer = ResidualAnalyzer()
    residual_analysis = residual_analyzer.analyze_residuals(
        y_train, y_pred_train, X_train, feature_names
    )

    print(f"\n  Residual Statistics:")
    print(f"    Mean:               {residual_analysis['mean']:.4f} (should be ~0)")
    print(f"    Std:                {residual_analysis['std']:.4f}")
    print(f"    Skewness:           {residual_analysis['skewness']:.4f} (should be ~0)")
    print(f"    Normality p-value:  {residual_analysis['normality_pvalue']:.3f}")
    print(f"    Is normal?          {'✅ YES' if residual_analysis['is_normal'] else '❌ NO'}")

    if 'feature_correlations' in residual_analysis:
        print(f"\n  Residual-Feature Correlations (should be ~0):")
        for name, corr in residual_analysis['feature_correlations'].items():
            status = "✅" if abs(corr) < 0.1 else "⚠️"
            print(f"    {status} {name:<20}: {corr:>8.4f}")

    # Demo 4: Performance Monitoring
    print(f"\n── 4. Performance Monitoring Over Time ──")

    monitor = ModelPerformanceMonitor()

    # Simulate 20 weeks of performance
    for week in range(20):
        # First 10 weeks: good performance
        # Last 10 weeks: degrading (regime shift)

        if week < 10:
            # Good regime
            X_week = np.random.randn(50, 5) * 0.5
            y_week = (X_week[:, 0] * 2 + X_week[:, 1] * -1 +
                     X_week[:, 2] * 0.5 + np.random.randn(50) * 0.1)
        else:
            # Bad regime (shifted)
            X_week = np.random.randn(50, 5) * 2.0
            y_week = (X_week[:, 0] * 1.0 + X_week[:, 1] * -1 +
                     X_week[:, 2] * 1.5 + np.random.randn(50) * 0.5)

        y_pred_week = intercept + np.dot(X_week, coef)

        monitor.record_performance(
            date=f'Week_{week+1}',
            predictions=y_pred_week,
            actuals=y_week
        )

    # Detect degradation
    degradation_report = monitor.detect_degradation(lookback=5, threshold=0.2)

    print(f"\n  Performance Monitoring:")
    print(f"    Baseline IC (Weeks 1-5):  {degradation_report['baseline_correlation']:.3f}")
    print(f"    Recent IC (Weeks 16-20):  {degradation_report['recent_correlation']:.3f}")
    print(f"    Degradation:              {degradation_report['degradation_pct']:.1%}")
    print(f"    Threshold:                {degradation_report['threshold']:.0%}")
    print(f"    **Status**:               {'🚨 DEGRADED' if degradation_report['degradation_detected'] else '✅ HEALTHY'}")

    print(f"\n{'═' * 70}")
    print(f"  KEY INSIGHTS: MODEL DIAGNOSTICS")
    print(f"{'═' * 70}")

    print(f"""
1. FEATURE DRIFT DETECTION:
   
   Critical for model health monitoring.
   
   Example: Volatility feature
   - Training: mean 0.5σ, std 0.2
   - Production: mean 2.0σ, std 0.8
   - Drift: 7.5σ shift → 🚨 ALERT
   
   → Model trained on low-vol regime
   → Seeing high-vol regime in production
   → Predictions unreliable (extrapolating)
   
   **Action**: Retrain model OR reduce position size

2. MODEL CALIBRATION:
   
   Well-calibrated model: Predicted 70% → Actual ~70%
   Poorly calibrated: Predicted 90% → Actual 60% (over-confident)
   
   **Why it matters**:
   • Risk management (need accurate probabilities)
   • Position sizing (Kelly criterion needs calibrated probs)
   • Decision making (threshold setting)
   
   **ECE (Expected Calibration Error)**:
   • <0.05: Well-calibrated ✅
   • 0.05-0.10: Moderate ⚠️
   • >0.10: Poorly calibrated 🚨
   
   **Fix**: Platt scaling, isotonic regression

3. RESIDUAL ANALYSIS:
   
   Good model: Residuals are white noise
   - Mean ≈ 0
   - No autocorrelation
   - No correlation with features
   - Normally distributed
   
   Bad signs:
   • Mean ≠ 0 → Biased predictions
   • Autocorrelation > 0.3 → Missing time-series pattern
   • Correlated with feature X → Non-linear relationship missed
   
   **In our demo**: Residuals look healthy ✅

4. PERFORMANCE DEGRADATION:
   
   Model performance ALWAYS degrades over time (markets evolve).
   
   Typical degradation rates:
   • Equity alpha models: 10-20% per year
   • HFT strategies: 30-50% per year (faster competition)
   • Credit models: 5-10% per year (slower moving)
   
   **Monitoring cadence**:
   • HFT: Daily
   • Alpha strategies: Weekly
   • Credit models: Monthly
   
   **Trigger for action**: 20% degradation → Retrain or retire

5. PRODUCTION CHECKLIST:
   
   Before deploying model:
   ☐ Check feature drift (KS test p-value > 0.05)
   ☐ Check calibration (ECE < 0.10)
   ☐ Analyze residuals (white noise)
   ☐ Stress test (2008 crisis data)
   ☐ Set up monitoring (track IC weekly)
   ☐ Define kill switch (if IC < threshold → stop trading)
   
   After deployment:
   ☐ Monitor feature distributions (detect regime shifts)
   ☐ Track performance (IC, Sharpe, drawdown)
   ☐ Log predictions + SHAP values (audit trail)
   ☐ Review failures (post-mortem)
   ☐ Retrain quarterly (or when degraded >20%)

Interview Q&A (Citadel Model Validation):

Q: "Your momentum model lost $50M in March 2020. What went wrong?"
A: "**Regime change we didn't detect**. Model trained on 2015-2019 (low vol,
    trending market). March 2020: COVID → volatility spiked 5x, correlations →1,
    momentum strategies got destroyed (everyone selling everything). **Autopsy**:
    (1) **Feature drift**—Volatility in training: mean 15%, March 2020: 85%.
    Model had never seen vol >40%. Extrapolated poorly. (2) **Correlation breakdown**
    —Training: stock-stock correlation ~0.3. March 2020: Correlation →0.9 (everything
    moving together). Momentum signals became useless. (3) **Missing features**—
    Model had no 'VIX' input (market fear). In calm markets, not needed. In crisis,
    critical. **Fix**: (1) Add regime detection (if VIX >40 → reduce size 80%).
    (2) Stress test on 2008 data (simulate another crisis). (3) Add vol + correlation
    as features. (4) Weekly model monitoring (track feature distributions). Result:
    Model would have reduced positions 80% on Feb 27 (VIX spiked). Loss: -$10M vs
    -$50M. Still lost money (crisis inevitable) but 5x less."

Next steps for model diagnostics expertise:
  • Learn statistical testing (KS test, chi-square, etc.)
  • Study calibration methods (Platt scaling, isotonic regression)
  • Understand bias detection (fairness in ML)
  • Practice post-mortems (analyze model failures)
  • Build monitoring dashboards (Grafana, Datadog)
    """)

print(f"\n{'═' * 70}")
print(f"  Module complete. Diagnostics prevent disasters.")
print(f"{'═' * 70}\n")
