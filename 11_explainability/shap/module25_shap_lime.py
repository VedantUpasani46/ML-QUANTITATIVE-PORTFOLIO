"""
Explainable AI: SHAP & LIME for Model Interpretation
=====================================================
Target: Explain Any Model | Regulatory Compliance |

This module implements SHAP (SHapley Additive exPlanations) and LIME (Local
Interpretable Model-agnostic Explanations) for understanding black-box models.

Why Explainability Matters:
  - REGULATORY: Basel III, MiFID II require model interpretability
  - RISK MANAGEMENT: Need to understand model decisions before deployment
  - DEBUGGING: "Why did model fail on 2020-03-15?" → SHAP shows which features
  - TRUST: Investors/clients want to know "why" not just "what"
  - MODEL IMPROVEMENT: Feature importance guides data collection

Target: Explain any model prediction, satisfy regulators

Interview insight (JPMorgan Model Risk):
Q: "Your XGBoost credit model got rejected by regulators. Why?"
A: "**Lack of explainability**. Model had 92% accuracy (excellent), but when
    regulator asked 'Why did you deny loan #12345?', we couldn't explain beyond
    'model said 0.82 default probability'. Regulator: 'What drove that 0.82?
    Was it income? Debt ratio? Credit score?' We had no answer. **Solution**:
    Implemented SHAP values. Now for loan #12345: 'Denied because: (1) DTI ratio
    45% contributed +0.30 to default prob (high impact), (2) Credit score 620
    contributed +0.25 (medium impact), (3) Recent delinquency +0.15 (medium).
    Income $80K contributed -0.05 (slightly protective). Net: 0.82 default prob.'
    Regulator approved. Lesson: Accuracy without explainability = useless for
    regulated industries. We now produce SHAP values for EVERY prediction. Adds
    50ms latency but required for compliance."

Mathematical Foundation:
------------------------
SHAP Value (Shapley Value from Game Theory):
  φ_i = Σ_{S⊆F\{i}} |S|!(|F|-|S|-1)! / |F|! × [f(S∪{i}) - f(S)]
  
  Where:
  - φ_i: SHAP value for feature i
  - F: All features
  - S: Subset of features
  - f: Model prediction function
  
  Interpretation: φ_i is feature i's contribution to prediction

LIME (Local Linear Approximation):
  g(z) = β_0 + Σ β_i z_i
  
  Where g approximates f locally:
  min Σ π(x, x') [f(x') - g(x')]²
  
  π(x, x'): Proximity kernel (closer samples weighted higher)

References:
  - Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NIPS.
  - Ribeiro et al. (2016). "Why Should I Trust You?": Explaining Predictions. KDD.
  - Molnar (2020). Interpretable Machine Learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# SHAP Value Calculator (Simplified)
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Calculate SHAP values for model predictions.
    
    Simplified implementation for demonstration.
    Production: Use official shap library.
    """
    
    def __init__(self, model, background_data: np.ndarray):
        """
        Initialize explainer.
        
        Args:
            model: Trained model with predict() method
            background_data: Representative sample (for baseline)
        """
        self.model = model
        self.background_data = background_data
        self.baseline_prediction = np.mean(model.predict(background_data))
    
    def explain_prediction(self, instance: np.ndarray, feature_names: List[str] = None):
        """
        Calculate SHAP values for single instance.
        
        Args:
            instance: Single data point to explain
            feature_names: Names of features
        
        Returns:
            SHAP values for each feature
        """
        n_features = len(instance)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Simplified SHAP calculation (not exact)
        # Real SHAP uses KernelExplainer or TreeExplainer
        
        shap_values = np.zeros(n_features)
        
        # For each feature, measure marginal contribution
        for i in range(n_features):
            # Prediction with this feature
            instance_with_feature = instance.copy()
            pred_with = self.model.predict(instance_with_feature.reshape(1, -1))[0]
            
            # Prediction without this feature (replace with baseline mean)
            instance_without_feature = instance.copy()
            instance_without_feature[i] = np.mean(self.background_data[:, i])
            pred_without = self.model.predict(instance_without_feature.reshape(1, -1))[0]
            
            # SHAP value ≈ difference
            shap_values[i] = pred_with - pred_without
        
        return dict(zip(feature_names, shap_values))
    
    def plot_explanation(self, shap_values: Dict, prediction: float):
        """
        Display SHAP explanation (text format).
        
        Args:
            shap_values: Dictionary of feature → SHAP value
            prediction: Model prediction
        """
        print(f"\n  Prediction: {prediction:.3f}")
        print(f"  Baseline: {self.baseline_prediction:.3f}")
        print(f"  Difference: {prediction - self.baseline_prediction:.3f}")
        
        print(f"\n  Feature Contributions (SHAP values):")
        print(f"  {'Feature':<25} {'SHAP Value':<15} {'Impact'}")
        print(f"  {'-' * 60}")
        
        # Sort by absolute value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, value in sorted_features:
            impact = "🔴 Negative" if value < 0 else "🟢 Positive"
            print(f"  {feature:<25} {value:>10.3f}      {impact}")


# ---------------------------------------------------------------------------
# LIME Explainer (Simplified)
# ---------------------------------------------------------------------------

class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations.
    
    Fits simple linear model locally around prediction.
    """
    
    def __init__(self, model):
        self.model = model
    
    def explain_prediction(self, 
                          instance: np.ndarray,
                          feature_names: List[str] = None,
                          n_samples: int = 100):
        """
        Explain prediction using LIME.
        
        Args:
            instance: Instance to explain
            feature_names: Feature names
            n_samples: Number of perturbed samples
        
        Returns:
            Linear coefficients (local feature importance)
        """
        n_features = len(instance)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Generate perturbed samples around instance
        # Add noise to create local neighborhood
        perturbations = np.random.randn(n_samples, n_features) * 0.1
        perturbed_instances = instance + perturbations
        
        # Get predictions for perturbed instances
        predictions = self.model.predict(perturbed_instances)
        
        # Calculate distances (for weighting)
        distances = np.linalg.norm(perturbations, axis=1)
        weights = np.exp(-distances)  # Closer samples weighted higher
        
        # Fit linear model (weighted least squares)
        from numpy.linalg import lstsq
        
        # Add intercept
        X = np.column_stack([np.ones(n_samples), perturbations])
        
        # Weighted least squares
        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ predictions
        
        coefficients = np.linalg.solve(XtWX, XtWy)
        
        # Extract feature coefficients (skip intercept)
        feature_coefficients = coefficients[1:]
        
        return dict(zip(feature_names, feature_coefficients))


# ---------------------------------------------------------------------------
# Feature Importance Analyzer
# ---------------------------------------------------------------------------

class FeatureImportanceAnalyzer:
    """
    Analyze global feature importance across entire dataset.
    """
    
    def __init__(self, model):
        self.model = model
    
    def permutation_importance(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              feature_names: List[str] = None,
                              n_repeats: int = 10):
        """
        Calculate permutation feature importance.
        
        Measures how much accuracy drops when feature is randomly shuffled.
        
        Args:
            X: Features
            y: Targets
            feature_names: Feature names
            n_repeats: Number of permutation repeats
        
        Returns:
            Importance scores for each feature
        """
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Baseline accuracy
        baseline_pred = self.model.predict(X)
        baseline_acc = np.corrcoef(baseline_pred, y)[0, 1]  # Use correlation as "accuracy"
        
        importances = {}
        
        for i in range(n_features):
            importance_scores = []
            
            for _ in range(n_repeats):
                # Shuffle feature i
                X_permuted = X.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Measure accuracy drop
                permuted_pred = self.model.predict(X_permuted)
                permuted_acc = np.corrcoef(permuted_pred, y)[0, 1]
                
                # Importance = drop in accuracy
                importance = baseline_acc - permuted_acc
                importance_scores.append(importance)
            
            # Average over repeats
            importances[feature_names[i]] = np.mean(importance_scores)
        
        return importances


# ---------------------------------------------------------------------------
# Simple Model for Demo
# ---------------------------------------------------------------------------

class SimpleLinearModel:
    """Simple linear model for demonstration."""
    
    def __init__(self):
        self.coef = None
        self.intercept = None
    
    def fit(self, X, y):
        """Fit linear model."""
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Least squares
        params = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        self.intercept = params[0]
        self.coef = params[1:]
    
    def predict(self, X):
        """Predict."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return self.intercept + np.dot(X, self.coef)


# ---------------------------------------------------------------------------
# CLI demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 70)
    print("  EXPLAINABLE AI: SHAP & LIME")
    print("  Target: Explain Any Model | Regulatory Compliance")
    print("═" * 70)
    
    # Generate synthetic credit risk data
    print("\n── Credit Risk Model Example ──")
    
    np.random.seed(42)
    
    n_samples = 1000
    
    # Features
    income = np.random.lognormal(11, 0.5, n_samples)  # $60K mean
    debt = income * np.random.uniform(0.1, 0.8, n_samples)
    credit_score = np.random.normal(700, 80, n_samples).clip(300, 850)
    age = np.random.normal(40, 12, n_samples).clip(18, 80)
    
    # Target: Default probability
    # Higher debt-to-income → higher default
    # Lower credit score → higher default
    dti = debt / income
    default_prob = (
        0.1 +
        0.3 * dti +
        -0.0005 * credit_score +
        -0.002 * age +
        np.random.randn(n_samples) * 0.05
    ).clip(0, 1)
    
    # Create feature matrix
    X = np.column_stack([income, debt, credit_score, age])
    y = default_prob
    
    feature_names = ['Income', 'Debt', 'Credit_Score', 'Age']
    
    print(f"  Dataset: {n_samples} loan applications")
    print(f"  Features: {feature_names}")
    
    # Train model
    print(f"\n  Training linear model...")
    
    model = SimpleLinearModel()
    model.fit(X, y)
    
    print(f"  Model coefficients:")
    for name, coef in zip(feature_names, model.coef):
        print(f"    {name:<15}: {coef:>8.4f}")
    
    # SHAP Explanation
    print(f"\n── SHAP Explanation for Single Loan ──")
    
    # Select example loan application
    example_idx = 42
    example_instance = X[example_idx]
    example_prediction = model.predict(example_instance)[0]
    
    print(f"\n  Loan Application #{example_idx}:")
    for name, value in zip(feature_names, example_instance):
        print(f"    {name:<15}: {value:>10.2f}")
    
    # Calculate SHAP values
    shap_explainer = SHAPExplainer(model, X[:100])  # Use first 100 as background
    shap_values = shap_explainer.explain_prediction(example_instance, feature_names)
    
    shap_explainer.plot_explanation(shap_values, example_prediction)
    
    # LIME Explanation
    print(f"\n── LIME Explanation (Local Linear Approximation) ──")
    
    lime_explainer = LIMEExplainer(model)
    lime_coefficients = lime_explainer.explain_prediction(example_instance, feature_names, n_samples=100)
    
    print(f"\n  Local Linear Coefficients:")
    for name, coef in sorted(lime_coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<15}: {coef:>8.4f}")
    
    # Global Feature Importance
    print(f"\n── Global Feature Importance (Permutation) ──")
    
    importance_analyzer = FeatureImportanceAnalyzer(model)
    importances = importance_analyzer.permutation_importance(X[:500], y[:500], feature_names, n_repeats=5)
    
    print(f"\n  Feature Importance (correlation drop when shuffled):")
    for name, imp in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<15}: {imp:>8.4f}")
    
    print(f"\n{'═' * 70}")
    print(f"  KEY INSIGHTS: EXPLAINABLE AI")
    print(f"{'═' * 70}")
    
    print(f"""
1. SHAP VALUES INTERPRETATION:
   
   For Loan #{example_idx}:
   - Prediction: {example_prediction:.3f} default probability
   - Baseline: {shap_explainer.baseline_prediction:.3f}
   
   Each SHAP value shows feature's contribution to final prediction.
   
   Example: Debt SHAP = +0.15 means:
   → This borrower's debt level INCREASES default prob by 0.15
   → If debt was average, prediction would be 0.15 lower
   
   **Use case**: "Why did you deny this loan?"
   → "High debt-to-income ratio (+0.15), low credit score (+0.08)"

2. SHAP vs LIME:
   
   **SHAP** (Shapley values):
   ✅ Theoretically sound (game theory)
   ✅ Consistent (same features → same SHAP values)
   ✅ Additive (SHAP values sum to prediction - baseline)
   ❌ Computationally expensive (exponential in features)
   
   **LIME** (Local linear):
   ✅ Fast (just fit linear model)
   ✅ Interpretable (linear coefficients)
   ❌ Less consistent (random sampling introduces variance)
   ❌ Less theoretically rigorous
   
   **In practice**: Use SHAP for production (more reliable)
   Use LIME for quick debugging (faster)

3. REGULATORY REQUIREMENTS:
   
   **Basel III** (banks):
   → Models must be "explainable and auditable"
   → Need to explain individual predictions
   → SHAP values satisfy this requirement
   
   **MiFID II** (investment firms):
   → Algorithm trading must be explainable
   → Document model logic + feature importance
   → SHAP + permutation importance covers this
   
   **SR 11-7** (Federal Reserve):
   → Model risk management framework
   → Validate models, explain limitations
   → Explainability is KEY validation step

4. WHEN EXPLAINABILITY MATTERS:
   
   ✅ **Critical**:
   • Credit scoring (fair lending laws)
   • Market risk models (Basel III)
   • Algorithmic trading (MiFID II)
   • Healthcare (life-or-death decisions)
   
   ⚠️  **Important**:
   • Hiring models (discrimination lawsuits)
   • Insurance pricing (regulatory review)
   • Investment recommendations (fiduciary duty)
   
   ❌ **Less critical**:
   • Internal research (no regulatory scrutiny)
   • Backtesting (exploratory analysis)
   • Low-stakes decisions

5. PRODUCTION WORKFLOW:
   
   **Development**:
   1. Train model (XGBoost, Neural Net, etc.)
   2. Validate on holdout set (accuracy, IC, etc.)
   3. Calculate SHAP values for validation set
   4. Review feature importance (does it make sense?)
   5. Document findings for regulators
   
   **Production**:
   1. Model makes prediction
   2. Calculate SHAP values (adds 10-50ms latency)
   3. Log SHAP values + prediction
   4. If prediction extreme (>95th percentile), alert human
   5. Human reviews SHAP values, approves/rejects
   
   **Cost**: Adds 10-50ms latency, worth it for compliance

Interview Q&A (JPMorgan Model Risk):

Q: "Your XGBoost credit model got rejected by regulators. Why?"
A: "**Lack of explainability**. Model had 92% accuracy (excellent), but when
    regulator asked 'Why did you deny loan #12345?', we couldn't explain beyond
    'model said 0.82 default probability'. Regulator: 'What drove that 0.82?
    Was it income? Debt ratio? Credit score?' We had no answer. **Solution**:
    Implemented SHAP values. Now for loan #12345: 'Denied because: (1) DTI ratio
    45% contributed +0.30 to default prob (high impact), (2) Credit score 620
    contributed +0.25 (medium impact), (3) Recent delinquency +0.15 (medium).
    Income $80K contributed -0.05 (slightly protective). Net: 0.82 default prob.'
    Regulator approved. Lesson: Accuracy without explainability = useless for
    regulated industries. We now produce SHAP values for EVERY prediction. Adds
    50ms latency but required for compliance."

Q: "SHAP values. How do you handle computationally expensive models?"
A: "**Three optimizations**: (1) **TreeExplainer**—For tree models (XGBoost,
    RandomForest), use fast SHAP algorithm. Exploits tree structure. O(TLD²)
    where T=trees, L=leaves, D=max depth. 1000 trees → 50ms vs 10 seconds naive.
    (2) **KernelExplainer with sampling**—For deep learning, approximate SHAP
    with 100 background samples (vs full 10,000). Accuracy 95%, speed 100x faster.
    (3) **Precompute for batch**—For credit cards (1M applications/month), compute
    SHAP offline in batch (overnight). Store in database. Latency: <1ms lookup vs
    50ms compute. Trade-off: Staleness (SHAP computed on yesterday's model) but
    acceptable for batch decisions."

Q: "Feature importance via permutation. Isn't this expensive?"
A: "**Yes**, but worth it for model validation. **Cost**: If dataset has N=1M
    samples, K=100 features, R=10 repeats → 1M × 100 × 10 = 1B predictions.
    At 1ms/prediction → 1M seconds = 11 days compute. **Optimizations**: (1)
    **Sample data**—Use N=10K samples (1% of data) for importance. Still accurate.
    Reduces compute to 2.7 hours. (2) **Parallel**—Run 100 features in parallel
    on 100 CPUs. Time: 1.6 minutes. (3) **Tree-specific shortcuts**—XGBoost has
    built-in feature_importances_ (zero cost). Use as first pass. **We do**:
    XGBoost importances (fast) + permutation on top 20 features (detailed). Best
    of both worlds."

Next steps for explainability expertise:
  • Study official SHAP library (shap.TreeExplainer, shap.DeepExplainer)
  • Learn LIME package (lime.lime_tabular)
  • Understand regulatory frameworks (Basel III, MiFID II)
  • Practice explaining models to non-technical stakeholders
  • Build audit trails (log SHAP values for every prediction)
    """)

print(f"\n{'═' * 70}")
print(f"  Module complete. Explainability = regulatory requirement.")
print(f"{'═' * 70}\n")
