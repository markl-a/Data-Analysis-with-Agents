# Random Forest Deep Dive

## Overview
Comprehensive analysis of Random Forest classifiers with detailed parameter exploration and performance optimization.

## Problem Description
This example provides an in-depth exploration of Random Forest algorithms, analyzing:
- Effect of number of trees (n_estimators)
- Impact of tree depth (max_depth)
- Minimum sample parameters
- Feature importance analysis
- Individual tree performance
- Comparison with single decision trees

## Dataset
- **Source**: Synthetic classification dataset
- **Samples**: 2000
- **Features**: 20 (15 informative, 3 redundant, 2 repeated)
- **Classes**: 3 (imbalanced: 40%, 35%, 25%)
- **Difficulty**: ⭐⭐⭐ Advanced

## Key Features

### 1. Parameter Analysis
- **n_estimators**: Tests 1 to 500 trees
- **max_depth**: Tests depths from 3 to unlimited
- **min_samples_split**: Tests values from 2 to 100
- **min_samples_leaf**: Tests values from 2 to 100

### 2. Model Comparison
- Single Decision Tree baseline
- Random Forests with varying parameters
- Optimal configuration identification

### 3. Advanced Analysis
- Out-of-bag (OOB) scoring
- Individual tree accuracy distribution
- Feature importance ranking
- ROC curves for multi-class classification

## Methodology

1. **Data Generation**: Create synthetic multi-class dataset
2. **Baseline**: Train single decision tree for comparison
3. **Parameter Exploration**: Systematically test key parameters
4. **Optimization**: Identify optimal configuration
5. **Evaluation**: Comprehensive performance metrics
6. **Visualization**: 9-panel comprehensive visualization

## Results

### Optimal Configuration
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True
)
```

### Expected Performance
- **Single Tree Accuracy**: ~0.75
- **Random Forest Accuracy**: ~0.85-0.90
- **Improvement**: 10-15% over single tree
- **OOB Score**: ~0.84

## Visualizations

The analysis generates a comprehensive 9-panel visualization:

1. **Number of Trees Effect**: Shows how accuracy improves with more trees
2. **Max Depth Effect**: Demonstrates the bias-variance tradeoff
3. **Min Samples Effect**: Compares split vs leaf parameters
4. **Feature Importance**: Top 10 most important features
5. **Confusion Matrix**: Prediction accuracy by class
6. **Model Comparison**: Bar chart comparing different configurations
7. **ROC Curves**: One-vs-rest ROC for each class
8. **Tree Distribution**: Histogram of individual tree accuracies
9. **Summary Statistics**: Key metrics and configuration

## Key Insights

1. **Ensemble Power**: Random Forests significantly outperform single trees
2. **Diminishing Returns**: Accuracy plateaus after ~100-200 trees
3. **Depth Matters**: Unlimited depth can lead to overfitting
4. **Bootstrap Sampling**: Individual trees have varying accuracy, but ensemble is robust
5. **Feature Importance**: Some features contribute much more than others

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

## Learning Objectives
- Understand Random Forest ensemble mechanics
- Learn hyperparameter tuning strategies
- Analyze bias-variance tradeoff
- Compare ensemble vs single model performance
- Interpret feature importance
- Use OOB scoring for validation

## Extension Ideas
1. Try regression tasks with RandomForestRegressor
2. Experiment with ExtraTreesClassifier
3. Compare with GradientBoostingClassifier
4. Implement custom feature importance calculation
5. Analyze feature interactions
6. Test on imbalanced datasets with class weights

## References
- Breiman, L. (2001). "Random Forests". Machine Learning
- scikit-learn Random Forest documentation
- Understanding the Bias-Variance Tradeoff
