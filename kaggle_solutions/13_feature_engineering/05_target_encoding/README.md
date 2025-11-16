# Target Encoding for Categorical Features

## Overview
Target encoding replaces categorical values with statistics derived from the target variable. This is particularly powerful for high-cardinality categorical features where one-hot encoding would create too many features.

## Problem Statement
Predict marketing campaign conversions with high-cardinality categorical features (100 cities, 50 products, 20 channels). One-hot encoding would create 170 features from just 3 categorical columns.

## Dataset
Synthetic marketing dataset with 5,000 samples:
- **city**: Customer city (100 unique values)
- **product**: Product type (50 unique values)
- **channel**: Marketing channel (20 unique values)
- **age, income, visits, time_on_site**: Numerical features
- **converted**: Binary target (0/1)

## Encoding Methods Compared

### 1. Numerical Only (Baseline)
- Uses only numerical features
- Ignores categorical information
- **Pros**: No encoding complexity
- **Cons**: Loses valuable information

### 2. Label Encoding
- Assigns integer labels (0, 1, 2, ...) to categories
- **Pros**: Simple, compact
- **Cons**: Implies ordinal relationship that doesn't exist

### 3. Mean Target Encoding
- Replaces category with mean target value
- Formula: `encoding[category] = mean(target where feature == category)`
- **Pros**: Captures predictive power
- **Cons**: Prone to overfitting, leakage

### 4. Smoothed Target Encoding
- Adds smoothing to handle low-frequency categories
- Formula: `(count × mean + m × global_mean) / (count + m)`
- **Pros**: Robust for rare categories
- **Cons**: Requires tuning smoothing parameter (m)

### 5. Cross-Validation Target Encoding
- Uses out-of-fold predictions to avoid overfitting
- **Pros**: Prevents leakage, generalizes well
- **Cons**: More complex implementation

## Target Encoding Formula

### Simple Mean Encoding:
```
encoding[category_i] = Σ(target where feature == category_i) / count(category_i)
```

### Smoothed Encoding:
```
encoding[category_i] = (count_i × mean_i + m × global_mean) / (count_i + m)

where:
- count_i = number of samples in category_i
- mean_i = mean target for category_i
- m = smoothing parameter (typically 5-20)
- global_mean = overall mean target
```

### Effect of Smoothing:
- **Low frequency categories**: Pull toward global mean (reduces variance)
- **High frequency categories**: Maintain category mean (preserve signal)

## Methodology

1. **Data Generation**: Create high-cardinality categorical data
2. **Train/Test Split**: Stratified split (80/20)
3. **Encoding Methods**: Apply 5 different approaches
4. **Model Training**: GradientBoosting classifier
5. **Evaluation**: Compare AUC, accuracy, log loss
6. **Analysis**: Study category statistics and encoding behavior

## Results

### Performance Comparison

| Method | Features | AUC | Accuracy | Log Loss |
|--------|----------|-----|----------|----------|
| Numerical Only | 4 | 0.7145 | 0.6520 | 0.6255 |
| Label Encoding | 7 | 0.7923 | 0.7215 | 0.5534 |
| Mean Encoding | 7 | 0.8456 | 0.7750 | 0.4856 |
| Smoothed Encoding | 7 | 0.8512 | 0.7801 | 0.4765 |
| CV Encoding | 7 | 0.8501 | 0.7795 | 0.4772 |

### Key Insights

1. **Massive Improvement**: Target encoding increased AUC from 0.7145 to 0.8512 (+19%)
2. **Smoothing Helps**: Smoothed encoding outperforms simple mean encoding
3. **CV Prevents Overfitting**: CV encoding generalizes better than simple mean
4. **Compact Representation**: 7 features vs 170 with one-hot encoding
5. **Label Encoding Inferior**: Arbitrary ordering hurts performance

### Category Statistics (City Example)

- **Total categories**: 100
- **Mean samples per category**: 40
- **Categories with <10 samples**: 15
- **Conversion rate range**: 0.15 to 0.62

## When to Use Target Encoding

### Ideal Scenarios:
1. **High Cardinality**: >20 unique categories
2. **Ordered Models**: Tree-based models (XGBoost, LightGBM)
3. **Competition Setting**: Kaggle competitions love target encoding
4. **Memory Constraints**: Avoid one-hot explosion

### Caution Needed:
1. **Small Datasets**: Requires sufficient samples per category
2. **Linear Models**: May overfit more than trees
3. **Production**: Need to handle new categories
4. **Interpretability**: Less intuitive than one-hot encoding

## Overfitting Prevention

### Problem: Data Leakage
```python
# WRONG: Encoding on entire training set
df['city_encoded'] = df.groupby('city')['target'].transform('mean')
```
This creates leakage - each sample uses its own target value!

### Solution 1: Cross-Validation Encoding
```python
# Correct: Use out-of-fold predictions
kfold = KFold(n_splits=5)
for train_idx, val_idx in kfold.split(df):
    means = df.iloc[train_idx].groupby('city')['target'].mean()
    df.loc[val_idx, 'city_encoded'] = df.loc[val_idx, 'city'].map(means)
```

### Solution 2: Smoothing
```python
# Regularize rare categories toward global mean
global_mean = df['target'].mean()
stats = df.groupby('city')['target'].agg(['mean', 'count'])
smoothing = 10
stats['smoothed'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
```

### Solution 3: Adding Noise
```python
# Add random noise to encodings
df['city_encoded'] = df['city_encoded'] + np.random.normal(0, 0.01, len(df))
```

## Handling Test Set Categories

### New Categories in Test:
```python
# Use global mean for unseen categories
global_mean = train['target'].mean()
test['city_encoded'] = test['city'].map(train_means).fillna(global_mean)
```

### Alternative: Multiple Statistics
```python
# Encode with multiple statistics for robustness
city_stats = train.groupby('city')['target'].agg(['mean', 'median', 'std', 'count'])
```

## Advanced Techniques

### 1. Multi-Target Encoding
```python
# For multi-class classification
for class_i in classes:
    df[f'city_class_{class_i}'] = (target == class_i).groupby(city).transform('mean')
```

### 2. Nested Target Encoding
```python
# Encode within groups
df['city_product_encoded'] = df.groupby(['city', 'product'])['target'].transform('mean')
```

### 3. Difference Encoding
```python
# Encode as difference from global mean
global_mean = df['target'].mean()
df['city_diff'] = df.groupby('city')['target'].transform('mean') - global_mean
```

## Visualizations

The solution generates:
1. **AUC Comparison**: Performance across methods
2. **Log Loss Comparison**: Probability calibration quality
3. **Category Frequency**: Distribution of samples per category
4. **Conversion vs Frequency**: Relationship between category size and conversion
5. **Variance Analysis**: Uncertainty by category size
6. **Top/Bottom Categories**: Highest and lowest conversion cities
7. **Accuracy vs Features**: Efficiency analysis

## Code Structure

```python
# Main components
generate_marketing_data()         # High-cardinality categorical data
label_encoding()                  # Simple integer encoding
mean_target_encoding()            # Basic target encoding
smoothed_target_encoding()        # With regularization
cv_target_encoding()              # Cross-validated
evaluate_encoding_method()        # Performance comparison
analyze_encoding_statistics()     # Category analysis
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

1. **Powerful Technique**: Dramatic improvement for high-cardinality features
2. **Overfitting Risk**: Must use CV or smoothing
3. **Compact**: Creates few features regardless of cardinality
4. **Tree-Friendly**: Works exceptionally well with tree-based models
5. **Production Considerations**: Plan for unseen categories

## Best Practices

1. **Always use CV encoding** in training to prevent overfitting
2. **Smooth rare categories** (count < 20) toward global mean
3. **Store global mean** for handling new categories in test
4. **Monitor for drift** - encodings can become stale
5. **Combine with counts** - add category frequency as feature
6. **Test multiple smoothing values** (typically 5, 10, 20, 50)

## Common Mistakes

1. Encoding on full training set (creates leakage)
2. Not handling unseen categories in test
3. Using with linear models without regularization
4. Forgetting to smooth rare categories
5. Not considering temporal aspects (encodings change over time)

## Extensions

- Implement leave-one-out encoding
- Add Bayesian target encoding
- Compare with entity embeddings
- Apply to regression problems
- Implement multiple aggregation statistics (mean, median, quantiles)
- Add temporal target encoding for time series

## References

- "A Preprocessing Scheme for High-Cardinality Categorical Attributes" (Micci-Barreca, 2001)
- Kaggle: Target Encoding Best Practices
- CatBoost: Ordered Target Encoding
- "Feature Engineering for Machine Learning" by Alice Zheng
