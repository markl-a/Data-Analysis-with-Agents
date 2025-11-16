# Stacking Ensemble

## Overview
Comprehensive analysis of stacking ensemble methods that combine multiple models through meta-learning.

## Problem Description
This example demonstrates stacking (stacked generalization):
- Training multiple diverse base models
- Using base model predictions as features for a meta-model
- Comparing different stacking configurations
- Analyzing prediction diversity and ensemble improvement

## Dataset
- **Source**: Synthetic multi-class classification
- **Samples**: 3000
- **Features**: 25 (18 informative, 4 redundant, 3 repeated)
- **Classes**: 3 (imbalanced: 50%, 30%, 20%)
- **Difficulty**: ⭐⭐⭐⭐ Expert

## Key Features

### 1. Base Models
- Random Forest
- Extra Trees
- Gradient Boosting
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree

### 2. Stacking Configurations
- **Level 1**: 3 base models (RF, ET, GB)
- **Level 2**: 5 base models (RF, ET, GB, SVM, KNN)
- **Meta-model comparison**: LR, RF, GB, SVM

### 3. Analysis Components
- Individual base model performance
- Stacking ensemble performance
- Cross-validation comparison
- Prediction diversity analysis
- Meta-model selection

## Methodology

1. **Data Preparation**: Generate and scale dataset
2. **Base Model Training**: Train 8 diverse classifiers
3. **Stacking Level 1**: Stack 3 tree-based models
4. **Stacking Level 2**: Stack 5 diverse models
5. **Meta-Model Testing**: Compare 4 meta-learners
6. **Diversity Analysis**: Measure prediction agreement
7. **Evaluation**: Comprehensive performance comparison

## Stacking Architecture

```
Layer 0: Input Features (25 features)
         ↓
Layer 1: Base Models
         ├── Random Forest
         ├── Extra Trees
         ├── Gradient Boosting
         ├── SVM
         └── KNN
         ↓
Layer 2: Meta-Model (Logistic Regression)
         ↓
Layer 3: Final Predictions
```

## Expected Results

### Base Model Performance
- Random Forest: ~0.78
- Extra Trees: ~0.76
- Gradient Boosting: ~0.80
- Logistic Regression: ~0.70
- SVM: ~0.72
- KNN: ~0.68
- Naive Bayes: ~0.65
- Decision Tree: ~0.70

### Stacking Performance
- **Stacking L1** (3 models): ~0.82 (+2-4% improvement)
- **Stacking L2** (5 models): ~0.83 (+3-5% improvement)

### Key Insights
- Stacking consistently outperforms individual models
- More diverse base models → better stacking
- Meta-model choice affects performance
- Cross-validation prevents overfitting

## Visualizations

The analysis generates a 9-panel visualization:

1. **Base Models Performance**: Accuracy comparison of all base models
2. **Stacking vs Base**: Direct comparison with best individual models
3. **Meta-Model Comparison**: Performance of different meta-learners
4. **Cross-Validation**: CV scores with error bars
5. **Diversity Heatmap**: Pairwise prediction agreement
6. **Confusion Matrix**: Best stacking model predictions
7. **Architecture Diagram**: Visual stacking structure
8. **Improvement Analysis**: Percentage gain over baseline
9. **Summary Statistics**: Comprehensive results table

## Key Concepts

### Why Stacking Works
1. **Diversity**: Combines different model types
2. **Meta-Learning**: Learns optimal combination weights
3. **Bias-Variance**: Reduces both bias and variance
4. **Generalization**: Better than simple averaging

### Design Considerations
1. **Base Model Diversity**: Use different algorithms
2. **Cross-Validation**: Prevent overfitting in meta-model
3. **Feature Engineering**: Meta-features from predictions
4. **Meta-Model Selection**: Simple models often work best

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- scikit-learn >= 0.22 (for StackingClassifier)
- matplotlib
- seaborn

## Learning Objectives
- Understand stacking ensemble methodology
- Implement multi-level ensemble models
- Analyze prediction diversity
- Select appropriate meta-models
- Evaluate ensemble improvements
- Use cross-validation in stacking

## Extension Ideas
1. Implement manual stacking (without StackingClassifier)
2. Try multi-level stacking (3+ levels)
3. Use neural networks as meta-model
4. Experiment with feature selection for meta-model
5. Implement time-series aware stacking
6. Compare with blending approach
7. Add feature engineering in meta-model

## Common Pitfalls
1. **Data Leakage**: Always use CV for meta-features
2. **Overfitting**: Meta-model can overfit to validation set
3. **Complexity**: More models ≠ better performance
4. **Correlation**: Highly correlated base models reduce benefit
5. **Computational Cost**: Training time increases significantly

## References
- Wolpert (1992). "Stacked Generalization"
- scikit-learn StackingClassifier documentation
- Kaggle Ensemble Guide
- "Ensemble Methods in Machine Learning" - Dietterich
