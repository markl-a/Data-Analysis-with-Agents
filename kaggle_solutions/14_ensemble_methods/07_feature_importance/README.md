# Feature Importance Across Ensembles

## Overview
Comprehensive analysis of feature importance methods across different ensemble models, comparing built-in and permutation importance.

## Problem Description
This example explores feature importance:
- Built-in importance (Gini, gain-based)
- Permutation importance
- Consistency across ensemble methods
- Importance by feature type
- Method comparison and validation

## Dataset
- **Source**: Synthetic binary classification with **known** feature importance
- **Samples**: 2000
- **Features**: 30
  - 10 informative (truly predictive)
  - 10 redundant (correlated with informative)
  - 5 repeated (exact duplicates)
  - 5 noise (pure random)
- **Classes**: 2 (imbalanced: 60%, 40%)
- **Difficulty**: ⭐⭐⭐⭐ Expert

## Key Features

### 1. Ensemble Models
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost

### 2. Importance Methods
- **Built-in**: Gini importance (RF, ET) / Gain-based (GB)
- **Permutation**: Model-agnostic feature shuffling
- **Comparison**: Correlation between methods

### 3. Analysis Components
- Top important features per model
- Feature importance correlation across models
- Built-in vs permutation comparison
- Importance by feature type (informative/redundant/noise)
- Consistency analysis

## Methodology

1. **Dataset Creation**: Generate data with known feature types
2. **Model Training**: Train 4 ensemble models
3. **Built-in Importance**: Extract from trained models
4. **Permutation Importance**: Calculate model-agnostic importance
5. **Comparison**: Correlate built-in vs permutation
6. **Type Analysis**: Validate against known feature types
7. **Consistency**: Analyze agreement across models

## Feature Importance Methods

### Built-in Importance
**Gini Importance (Random Forest, Extra Trees)**
- Based on total decrease in node impurity
- Fast to compute (already calculated during training)
- Can be biased toward high-cardinality features
- Works only for tree-based models

**Gain-based (Gradient Boosting)**
- Based on loss function improvement
- Similar to Gini but for boosting
- Can emphasize features used in early splits

### Permutation Importance
- Model-agnostic (works with any model)
- Shuffle feature and measure performance drop
- More computationally expensive
- Less biased, more reliable
- Better for correlated features

## Expected Results

### Top Features
All models should correctly identify informative features as most important.

### By Feature Type
- **Informative**: High importance (0.05 - 0.15)
- **Redundant**: Medium importance (0.01 - 0.05)
- **Repeated**: Low importance (0.001 - 0.01)
- **Noise**: Very low importance (< 0.001)

### Method Correlation
- Built-in vs Permutation: r = 0.85 - 0.95
- Model agreement: r > 0.80

## Visualizations

The analysis generates a 12-panel visualization:

1. **Top 15 Features - Random Forest**: Color-coded by feature type
2. **Top 15 Features - Gradient Boosting**: Importance ranking
3. **Feature Importance Heatmap**: Top 20 features across all models
4. **Importance by Feature Type**: Bar chart grouped by type
5. **Built-in vs Permutation (RF)**: Correlation scatter plot
6. **Built-in vs Permutation (GB)**: Validation of methods
7. **Model Agreement Matrix**: Correlation heatmap
8. **Coefficient of Variation**: Features with highest disagreement
9. **Average Importance**: Top 15 with error bars
10. **Permutation Importance (RF)**: With confidence intervals
11. **Feature Type Distribution**: Average ranks by type
12. **Summary & Legend**: Key findings and color legend

## Key Insights

1. **Validation**: Models correctly identify informative features
2. **Consistency**: High agreement across ensemble methods
3. **Methods**: Built-in and permutation correlate strongly
4. **Noise**: Random features correctly assigned low importance
5. **Redundancy**: Correlated features share importance

## When to Use Each Method

### Built-in Importance
✓ Use when:
- Using tree-based models
- Speed is critical
- Initial exploration
- Quick feature selection

✗ Avoid when:
- Features are highly correlated
- High-cardinality features present
- Need model-agnostic measure

### Permutation Importance
✓ Use when:
- Features are correlated
- Need reliable, unbiased estimates
- Model-agnostic analysis needed
- Final feature selection

✗ Avoid when:
- Computational resources limited
- Very large datasets
- Quick prototype needed

## Practical Recommendations

1. **Use Multiple Methods**: Built-in for speed, permutation for validation
2. **Check Consistency**: Features important in multiple models are robust
3. **Validate**: Use domain knowledge to verify importance
4. **Beware Correlation**: Redundant features split importance
5. **Use Confidence Intervals**: Especially for permutation importance

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- scikit-learn >= 0.22 (for permutation_importance)
- matplotlib
- seaborn

## Learning Objectives
- Understand feature importance methods
- Compare built-in vs permutation importance
- Analyze consistency across models
- Validate against known feature types
- Choose appropriate importance method
- Interpret feature importance correctly

## Extension Ideas
1. Add SHAP values analysis
2. Compare with LIME
3. Test on real datasets
4. Add partial dependence plots
5. Implement custom importance metrics
6. Test with categorical features
7. Analyze feature interactions
8. Add recursive feature elimination

## Common Pitfalls

1. **Over-interpretation**: Importance ≠ causation
2. **Correlation**: Redundant features split importance
3. **Scale Differences**: May need feature scaling
4. **Sample Size**: Small samples → unreliable importance
5. **Model-specific**: Different models, different importance
6. **Multicollinearity**: Can inflate/deflate importance

## References
- Breiman (2001). "Random Forests"
- Strobl et al. (2007). "Bias in random forest variable importance measures"
- Altmann et al. (2010). "Permutation importance: a corrected feature importance measure"
- scikit-learn Feature Importance documentation
- "Interpretable Machine Learning" - Christoph Molnar
