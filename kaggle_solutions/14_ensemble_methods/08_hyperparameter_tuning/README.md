# Hyperparameter Tuning for Ensembles

## Overview
Comprehensive guide to hyperparameter optimization for ensemble methods, comparing grid search and random search strategies.

## Problem Description
This example demonstrates:
- Baseline model performance with default parameters
- Grid search for exhaustive parameter exploration
- Random search for efficient optimization
- Parameter importance analysis
- Search strategy comparison

## Dataset
- **Source**: Synthetic binary classification
- **Samples**: 2500
- **Features**: 25 (18 informative, 4 redundant, 3 repeated)
- **Classes**: 2 (imbalanced: 60%, 40%)
- **Difficulty**: ⭐⭐⭐⭐ Expert

## Key Features

### 1. Models Tuned
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- **Gradient Boosting**: n_estimators, learning_rate, max_depth, min_samples_split, subsample

### 2. Search Strategies
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter distributions
- **Comparison**: Time, performance, efficiency

### 3. Analysis Components
- Baseline performance
- Grid search results
- Random search results
- Parameter importance
- Final model comparison

## Methodology

1. **Baseline**: Train with default parameters
2. **Grid Search**: Exhaustive parameter combinations
3. **Random Search**: Sample 100 random configurations
4. **Parameter Analysis**: Identify most important parameters
5. **Comparison**: Evaluate all approaches
6. **Best Model**: Select optimal configuration

## Hyperparameters Explored

### Random Forest
- **n_estimators**: 50, 100, 200, 300
- **max_depth**: 10, 20, 30, 40, None
- **min_samples_split**: 2-20
- **min_samples_leaf**: 1-10
- **max_features**: 'sqrt', 'log2', None
- **bootstrap**: True, False

### Gradient Boosting
- **n_estimators**: 50, 100, 200
- **learning_rate**: 0.01, 0.1, 0.3
- **max_depth**: 3, 5, 7
- **min_samples_split**: 2, 5, 10
- **subsample**: 0.8, 1.0

## Grid Search vs Random Search

### Grid Search
**Advantages:**
- Exhaustive (tests all combinations)
- Guaranteed to find best in grid
- Easy to parallelize
- Reproducible

**Disadvantages:**
- Computationally expensive
- Suffers from curse of dimensionality
- May miss optimal values between grid points
- Wastes time on unimportant parameters

**Best for:**
- Small parameter spaces
- Final refinement
- Critical applications
- Well-understood parameters

### Random Search
**Advantages:**
- More efficient for large spaces
- Can explore wider range
- Works well with varying parameter importance
- Easy to add more iterations

**Disadvantages:**
- Not exhaustive
- May miss optimal combination
- Less reproducible (unless seeded)
- Harder to interpret results

**Best for:**
- Large parameter spaces
- Initial exploration
- When some parameters matter more
- Limited computational budget

## Expected Results

### Baseline Models
- RF Default: ~0.84
- GB Default: ~0.85

### After Tuning
- RF Grid Search: ~0.87 (+3% improvement)
- RF Random Search: ~0.87 (+3% improvement)
- GB Grid Search: ~0.88 (+3.5% improvement)

### Search Times
- Grid Search: 60-180 seconds
- Random Search: 40-120 seconds

### Key Finding
Random search achieves similar performance to grid search in less time, especially for large parameter spaces.

## Visualizations

The analysis generates a 9-panel visualization:

1. **Model Accuracy Comparison**: Baseline vs tuned models
2. **Search Time Comparison**: Grid vs random search duration
3. **Improvement Over Baseline**: Percentage gains
4. **RF Grid Search Top Configs**: Best 10 configurations
5. **RF Random Search Distribution**: Score histogram
6. **GB Learning Rate Effect**: Parameter impact analysis
7. **Parameter Importance**: Most impactful parameters
8. **Best Parameters Summary**: Optimal configurations
9. **Summary Statistics**: Key findings and recommendations

## Parameter Importance

### Most Important (High Impact)
- **n_estimators**: More trees generally better (diminishing returns after 100-200)
- **learning_rate** (GB): Critical for boosting performance
- **max_depth**: Controls model complexity

### Moderately Important
- **min_samples_split**: Affects overfitting
- **min_samples_leaf**: Regularization parameter
- **subsample** (GB): Can improve generalization

### Less Important
- **max_features**: Often 'sqrt' works well
- **bootstrap**: Usually True for RF

## Best Practices

### 1. Start with Baseline
Always establish baseline performance before tuning.

### 2. Coarse-to-Fine
1. Random search for initial exploration
2. Identify promising regions
3. Grid search for refinement

### 3. Cross-Validation
Always use CV to avoid overfitting hyperparameters.

### 4. Computational Budget
- Limited budget → Random search
- Unlimited budget → Grid search
- Best balance → Random then grid

### 5. Parameter Ranges
Start with wide ranges, narrow based on results.

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy (for random search distributions)

## Learning Objectives
- Understand hyperparameter tuning importance
- Implement grid and random search
- Compare search strategies
- Identify important parameters
- Optimize ensemble performance
- Balance accuracy and computational cost

## Extension Ideas
1. Implement Bayesian optimization
2. Try Optuna or Hyperopt
3. Add early stopping
4. Use nested CV for unbiased estimates
5. Implement custom scoring functions
6. Add ensemble of tuned models
7. Compare with AutoML solutions
8. Implement halving search strategies

## Common Pitfalls

1. **Overfitting to Validation Set**: Use proper CV
2. **Too Many Iterations**: Diminishing returns
3. **Ignoring Defaults**: Sometimes default is best
4. **Not Enough Data**: Can't reliably tune with small datasets
5. **Correlation**: Some parameters interact
6. **Computational Cost**: Balance with benefits
7. **No Baseline**: Always compare to default

## Time-Saving Tips

1. **Use n_jobs=-1**: Parallelize search
2. **Start Small**: Use subset of data initially
3. **Random First**: Then refine with grid
4. **Reduce CV Folds**: 3-fold often sufficient
5. **Warm Start**: Use previous results
6. **Early Stopping**: Stop if no improvement

## References
- Bergstra & Bengio (2012). "Random Search for Hyper-Parameter Optimization"
- GridSearchCV documentation (scikit-learn)
- RandomizedSearchCV documentation (scikit-learn)
- "Hyperparameter Optimization" - Feurer & Hutter
- Kaggle hyperparameter tuning tutorials
