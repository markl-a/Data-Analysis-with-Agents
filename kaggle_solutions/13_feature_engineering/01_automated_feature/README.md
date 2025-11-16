# Automated Feature Generation

## Overview
This example demonstrates automated feature engineering techniques that create new predictive features from existing ones. The solution shows how systematic feature generation can significantly improve model performance.

## Problem Statement
Predict sales revenue using customer and transaction data. The challenge is to automatically generate features that capture complex relationships between variables without manual domain expertise.

## Dataset
Synthetic sales dataset with 2,000 samples containing:
- **price**: Product price ($10-$200)
- **quantity**: Units sold (1-100)
- **discount**: Discount percentage (0-30%)
- **advertising_spend**: Marketing budget ($100-$10,000)
- **competitor_price**: Competitor pricing
- **day_of_week**: Day of week (0-6)
- **month**: Month of year (1-12)
- **store_category**: Store type (A, B, C)
- **revenue**: Target variable (complex function of features)

## Feature Engineering Techniques

### 1. Arithmetic Interactions
- **Multiplicative**: `price * quantity`, `price * discount`
- **Difference**: `price - competitor_price`
- **Ratio**: `price / competitor_price`

### 2. Mathematical Transformations
- **Power**: Squared features for non-linear relationships
- **Root**: Square root for diminishing returns
- **Logarithmic**: Log transformations for scale normalization

### 3. Binning Features
- Discretize continuous variables into categories
- Capture threshold effects

### 4. Time-Based Features
- **Cyclical encoding**: Sin/cos for month
- **Binary flags**: Weekend, holiday season
- **Derived**: Business days, quarter

### 5. Categorical Encoding
- One-hot encoding for store categories
- Interaction with numerical features

### 6. Statistical Features
- Ratios and percentages
- Per-unit calculations
- Intensity measures

## Methodology

1. **Data Generation**: Create synthetic sales data with known relationships
2. **Baseline Model**: Train on original features only
3. **Feature Engineering**: Apply automated feature generation
4. **Model Training**: Train with engineered features
5. **Comparison**: Evaluate performance improvement
6. **Analysis**: Identify most important generated features

## Results

### Performance Metrics

| Metric | Baseline | With Features | Improvement |
|--------|----------|---------------|-------------|
| RMSE   | ~150     | ~80          | ~47%        |
| MAE    | ~110     | ~60          | ~45%        |
| R²     | ~0.85    | ~0.96        | ~13%        |

### Key Insights

1. **Interaction Features**: `price_quantity` and `price_discount` are top predictors
2. **Ratio Features**: `price_ratio` captures competitive positioning
3. **Transformation Features**: Log and square root transformations capture non-linearity
4. **Time Features**: Cyclical encoding better than raw month values
5. **Feature Count**: 40+ engineered features vs 8 original features

## Feature Importance

Top generated features by importance:
1. `price_quantity` - Direct revenue driver
2. `revenue_per_unit` - Unit economics
3. `price_discount` - Effective price
4. `advertising_spend_log` - Non-linear ad effect
5. `month_sin` - Seasonal patterns

## Visualizations

The solution generates:
1. **Prediction Comparison**: Baseline vs engineered features
2. **Residual Distribution**: Error reduction visualization
3. **Feature Importance**: Top contributing features
4. **Metrics Comparison**: Performance improvements

## Code Structure

```python
# Main components
generate_sales_data()          # Synthetic data generation
create_automated_features()    # Feature engineering pipeline
train_and_evaluate()          # Model training and evaluation
plot_results()                # Visualization
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

1. **Automation**: Systematic feature generation can be automated
2. **Performance**: Significant improvement (47% RMSE reduction)
3. **Interactions**: Multiplicative features capture complex relationships
4. **Transformations**: Mathematical transformations handle non-linearity
5. **Domain Agnostic**: Techniques work across different domains

## Extensions

- Use feature-tools for deep feature synthesis
- Implement genetic algorithms for feature selection
- Add polynomial features of higher degrees
- Create lag and rolling window features
- Implement automatic feature selection (RFECV, SelectKBest)

## References

- Scikit-learn Feature Engineering Guide
- Feature Engineering for Machine Learning (Alice Zheng)
- Automated Feature Engineering using Deep Feature Synthesis
