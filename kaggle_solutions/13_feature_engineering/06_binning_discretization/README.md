# Binning and Discretization

## Overview
Binning transforms continuous variables into discrete categories. This technique captures threshold effects, reduces noise, and can improve model interpretability. This example compares multiple binning strategies.

## Problem Statement
Predict credit approval where decision thresholds exist (e.g., credit score >740 is "very good"). Continuous features may benefit from discretization to capture these natural breakpoints.

## Dataset
Synthetic credit approval data with 3,000 applicants:
- **age**: Applicant age (18-75)
- **income**: Annual income (log-normal distribution)
- **credit_score**: FICO score (300-850)
- **debt_ratio**: Debt-to-income ratio (0-1)
- **years_employed**: Employment duration (0-40)
- **approved**: Binary target (0/1)

## Binning Strategies

### 1. No Binning (Baseline)
Uses continuous values as-is.
- **Pros**: Preserves all information
- **Cons**: Sensitive to outliers, assumes smooth relationships

### 2. Equal-Width Binning
Divides range into bins of equal size.
```python
bins = [min, min + width, min + 2×width, ..., max]
width = (max - min) / n_bins
```
- **Pros**: Simple, interpretable
- **Cons**: Uneven sample distribution, sensitive to outliers

### 3. Equal-Frequency (Quantile) Binning
Each bin contains approximately the same number of samples.
```python
bins = quantiles(data, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
```
- **Pros**: Balanced bins, handles skewed distributions
- **Cons**: Bin boundaries less interpretable

### 4. Custom Domain-Based Binning
Uses domain knowledge for meaningful categories.
```python
credit_bins = [0, 580, 670, 740, 800, 900]  # Industry standards
labels = ['poor', 'fair', 'good', 'very_good', 'excellent']
```
- **Pros**: Interpretable, captures known thresholds
- **Cons**: Requires expertise

### 5. Tree-Based Optimal Binning
Uses decision tree to find splits that maximize information gain.
- **Pros**: Data-driven, optimized for target
- **Cons**: Risk of overfitting, less interpretable

## Results

### Performance Comparison

| Strategy | Features | AUC | Accuracy | Notes |
|----------|----------|-----|----------|-------|
| Continuous (Baseline) | 5 | 0.8520 | 0.7833 | High information |
| Equal-Width | 5 | 0.7965 | 0.7350 | Information loss |
| Equal-Frequency | 5 | 0.8213 | 0.7567 | Better balance |
| Custom Domain | 16 | 0.8645 | 0.7950 | **Best** - Expert knowledge |
| Tree-Based | 5 | 0.8431 | 0.7750 | Data-optimized |

### Key Insights

1. **Domain Knowledge Wins**: Custom binning outperformed all others (+1.5% AUC)
2. **Tree Models Less Sensitive**: Gradient boosting handles continuous features well
3. **Equal-Width Weakest**: Uneven distribution hurt performance
4. **Quantile Binning Safe**: Good default when domain knowledge unavailable
5. **Feature Explosion**: Custom binning created 16 one-hot features from 5 continuous

## When to Use Binning

### Good Use Cases:
1. **Threshold Effects**: Natural breakpoints (credit scores, age groups)
2. **Linear Models**: Help capture non-linearities
3. **Interpretability**: "High income" more meaningful than "$87,453"
4. **Noisy Data**: Binning smooths random fluctuations
5. **Missing Values**: Can create "missing" bin
6. **Sparse Features**: Reduce cardinality of continuous variables

### Avoid When:
1. **Tree-Based Models**: Already handle non-linearities
2. **Deep Learning**: Neural networks learn optimal representations
3. **Small Datasets**: Information loss hurts more
4. **Smooth Relationships**: Binning introduces artificial boundaries

## Binning Methods Deep Dive

### Equal-Width Binning
```python
# Example: Income binning
min_income = 20000
max_income = 150000
n_bins = 5
width = (150000 - 20000) / 5 = 26000

bins = [20000, 46000, 72000, 98000, 124000, 150000]
```
**Problem**: If most people earn $30k-$60k, first bin overloaded, last bin sparse.

### Equal-Frequency Binning
```python
# Each bin gets 20% of data
quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bins = df['income'].quantile(quantiles)
# bins = [20000, 38000, 52000, 68000, 92000, 150000]
```
**Better**: Balanced bins, but boundaries less interpretable.

### Domain-Based Binning
```python
# Credit score (FICO) standard categories
bins = [0, 580, 670, 740, 800, 900]
labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
```
**Best**: Meaningful, matches real-world usage.

## Handling Edge Cases

### 1. Outliers
```python
# Cap outliers before binning
income_capped = income.clip(lower=10000, upper=200000)
```

### 2. Duplicate Bin Edges
```python
# Can occur with equal-frequency on discrete data
pd.qcut(data, q=5, duplicates='drop')  # Merge duplicate bins
```

### 3. New Values in Test
```python
# Use open boundaries
bins = [-np.inf, 580, 670, 740, 800, np.inf]
```

### 4. Missing Values
```python
# Create explicit "missing" category
df['income_bin'] = pd.cut(df['income'], bins=bins)
df['income_bin'] = df['income_bin'].cat.add_categories(['missing'])
df.loc[df['income'].isna(), 'income_bin'] = 'missing'
```

## One-Hot Encoding After Binning

```python
# Binning creates categorical variables
df['credit_category'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 900])

# One-hot encode for linear models
df_encoded = pd.get_dummies(df, columns=['credit_category'])
# Creates: credit_category_(0, 580], credit_category_(580, 670], etc.

# Or use label encoding for tree models
df['credit_bin'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 900], labels=False)
```

## Visualizations

The solution generates:
1. **Performance Comparison**: AUC and accuracy across strategies
2. **Equal-Width Distribution**: Showing uneven bin counts
3. **Equal-Frequency Distribution**: Balanced bin counts
4. **Approval Rate by Bin**: Target rate variation across bins
5. **Original Distribution**: Continuous feature histogram
6. **Tree-Based Boundaries**: Optimal splits visualization
7. **Features vs Performance**: Complexity trade-off

## Code Structure

```python
generate_credit_data()          # Threshold-based synthetic data
equal_width_binning()           # Fixed-width bins
equal_frequency_binning()       # Quantile-based bins
custom_domain_binning()         # Expert-defined categories
tree_based_binning()            # Decision tree optimization
evaluate_binning_strategy()     # Performance comparison
plot_results()                  # Comprehensive visualizations
```

## Usage

```bash
python solution.py
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Key Takeaways

1. **Domain Knowledge Critical**: Custom binning based on expertise performs best
2. **Model Dependent**: Linear models benefit more than tree models
3. **Interpretability Gain**: Binned features easier to explain to stakeholders
4. **Information Loss**: Binning always loses some information
5. **Quantile Safe Default**: When unsure, use equal-frequency binning

## Advanced Techniques

### Optimal Binning Algorithms
```python
from optbinning import OptimalBinning

optb = OptimalBinning(name='income', dtype='numerical')
optb.fit(X, y)
binning_table = optb.binning_table.build()
```

### Monotonic Binning
Ensure bins have monotonic relationship with target:
```python
# Higher credit score → higher approval rate (monotonic)
```

### Weight of Evidence (WoE) Binning
Common in credit scoring:
```python
WoE = ln(% of goods / % of bads)
```

## Common Mistakes

1. **Binning on test set independently** - Use train bins for test
2. **Too many bins** - Loses interpretability benefit
3. **Too few bins** - Loses discriminative power
4. **Ignoring imbalanced bins** - Equal-width with outliers
5. **Binning everything** - Only bin when beneficial

## Best Practices

1. Start with domain-based binning if expertise available
2. Use quantile binning as default
3. Always apply training bins to test set
4. Visualize bin distributions before finalizing
5. Check approval/target rate varies across bins
6. Consider keeping continuous version as additional feature
7. For linear models, bin more aggressively
8. For tree models, binning often unnecessary

## Extensions

- Implement chi-square binning for categorical outcomes
- Add monotonic binning constraints
- Implement Weight of Evidence encoding
- Compare with splines as alternative
- Apply to time series (temporal binning)
- Implement automatic bin number selection

## References

- "Credit Risk Scorecards" by Naeem Siddiqi
- Pandas cut() and qcut() documentation
- Optimal Binning library (optbinning)
- "Feature Engineering and Selection" by Kuhn & Johnson
