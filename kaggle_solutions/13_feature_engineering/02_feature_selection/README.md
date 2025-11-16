# Feature Selection Methods Comparison

## Overview
This example demonstrates and compares multiple feature selection techniques, showing how different approaches identify important features and impact model performance. The solution covers filter, wrapper, and embedded methods.

## Problem Statement
Predict customer churn using various customer attributes. The challenge is to identify the most predictive features from a dataset containing both relevant features and noise, improving model performance and interpretability.

## Dataset
Synthetic customer churn dataset with 3,000 samples:

### Relevant Features (10):
- **tenure**: Customer tenure in months
- **monthly_charges**: Monthly bill amount
- **total_charges**: Total amount charged
- **contract_type**: Contract duration (0-2)
- **support_calls**: Number of support calls
- **payment_failures**: Failed payment attempts
- **usage_gb**: Data usage in GB
- **age**: Customer age
- **satisfaction_score**: Customer satisfaction (1-10)
- **num_services**: Number of subscribed services

### Noise Features (20):
- Random noise features with no predictive power
- Correlated noise features (false signals)

### Target:
- **churn**: Binary (0: retained, 1: churned)

## Feature Selection Methods

### 1. Variance Threshold (Filter Method)
**Approach**: Remove features with low variance
- **Pros**: Fast, removes constant/quasi-constant features
- **Cons**: Doesn't consider relationship with target
- **Use Case**: Preprocessing step to remove useless features

### 2. Univariate Selection - F-test (Filter Method)
**Approach**: Statistical test (ANOVA F-test) between features and target
- **Pros**: Fast, statistically grounded
- **Cons**: Only captures linear relationships, ignores feature interactions
- **Use Case**: Quick feature screening for linear models

### 3. Univariate Selection - Mutual Information (Filter Method)
**Approach**: Measures mutual dependence between features and target
- **Pros**: Captures non-linear relationships
- **Cons**: Slower than F-test, sensitive to feature scaling
- **Use Case**: Non-linear relationships, complex patterns

### 4. Recursive Feature Elimination - RFE (Wrapper Method)
**Approach**: Recursively removes least important features
- **Pros**: Considers feature interactions, model-specific
- **Cons**: Computationally expensive, risk of overfitting
- **Use Case**: When model performance is critical

### 5. Model-Based Selection (Embedded Method)
**Approach**: Uses feature importances from tree-based models
- **Pros**: Fast, captures feature interactions, built-in to model training
- **Cons**: Model-specific, biased toward numerical features
- **Use Case**: Tree-based models, fast selection needed

## Methodology

1. **Data Generation**: Create dataset with known relevant and noise features
2. **Baseline**: Train model with all features
3. **Apply Methods**: Test each selection technique
4. **Evaluation**: Compare performance, efficiency, and selected features
5. **Analysis**: Identify overlapping features and best approach

## Results

### Performance Comparison

| Method | Features | Accuracy | Precision | Recall | F1 Score |
|--------|----------|----------|-----------|--------|----------|
| All Features | 30 | 0.82 | 0.80 | 0.78 | 0.79 |
| Variance Threshold | 28 | 0.82 | 0.81 | 0.78 | 0.79 |
| Univariate (F-test) | 15 | 0.84 | 0.83 | 0.81 | 0.82 |
| Univariate (MI) | 15 | 0.83 | 0.82 | 0.80 | 0.81 |
| RFE | 15 | 0.85 | 0.84 | 0.82 | 0.83 |
| Model-Based | 15 | 0.85 | 0.84 | 0.83 | 0.83 |

### Key Insights

1. **Fewer Features, Better Performance**: Reducing from 30 to 15 features improved F1 score by ~5%
2. **Wrapper Methods Excel**: RFE achieved best performance by considering feature interactions
3. **Embedded Methods Efficient**: Model-based selection nearly matches RFE with lower computational cost
4. **Filter Methods Fast**: Univariate methods provide good baseline with minimal computation
5. **Noise Reduction**: Removing irrelevant features reduces overfitting

### Feature Overlap

Common features selected across methods:
- `tenure` - Consistently ranked #1
- `support_calls` - High importance across all methods
- `payment_failures` - Strong churn indicator
- `satisfaction_score` - Key predictor
- `contract_type` - Important categorical feature

## Method Comparison

### Speed Ranking (Fastest to Slowest):
1. Variance Threshold
2. Model-Based Selection
3. Univariate (F-test)
4. Univariate (MI)
5. RFE

### Performance Ranking (Best to Worst):
1. RFE / Model-Based (tie)
2. Univariate (F-test)
3. Univariate (MI)
4. Variance Threshold
5. All Features

### Efficiency (Performance per Feature):
1. Model-Based Selection
2. RFE
3. Univariate (F-test)

## Visualizations

The solution generates:
1. **Performance Comparison**: All metrics across methods
2. **Feature Count vs Performance**: Scatter plot showing trade-offs
3. **Feature Importance**: Top features from model-based selection
4. **Method Efficiency**: Performance normalized by feature count

## Code Structure

```python
# Main components
generate_customer_churn_data()    # Synthetic data with noise
variance_threshold_selection()    # Filter method
univariate_selection()           # Statistical tests
rfe_selection()                  # Wrapper method
model_based_selection()          # Embedded method
evaluate_model()                 # Performance evaluation
plot_comparison()                # Comprehensive visualizations
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

1. **No Universal Winner**: Best method depends on dataset, model, and constraints
2. **Start Simple**: Use filter methods for initial screening
3. **Consider Trade-offs**: Balance performance, speed, and interpretability
4. **Combine Methods**: Ensemble of selection techniques often works best
5. **Domain Knowledge**: Use selection to validate domain intuition

## Recommendations by Scenario

### Large Datasets (>100K samples):
- Use filter methods (univariate) or model-based selection
- Avoid RFE due to computational cost

### Small Datasets (<1K samples):
- Use RFE or cross-validated selection
- More thorough evaluation needed to prevent overfitting

### Linear Models:
- Univariate F-test works well
- L1 regularization (Lasso) for embedded selection

### Tree-Based Models:
- Model-based selection ideal
- Feature importance naturally available

### High-Dimensional (>1000 features):
- Variance threshold first
- Then univariate or model-based selection

## Extensions

- Implement Boruta algorithm for all-relevant feature selection
- Add LASSO/Ridge regularization paths
- Use genetic algorithms for feature selection
- Implement forward/backward stepwise selection
- Add cross-validation to selection process
- Compare with PCA/dimensionality reduction

## References

- Scikit-learn Feature Selection Guide
- Guyon & Elisseeff: "An Introduction to Variable and Feature Selection"
- "Feature Engineering and Selection" by Kuhn & Johnson
