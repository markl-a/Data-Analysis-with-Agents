# Feature Engineering Examples Collection

Complete collection of 8 comprehensive feature engineering examples demonstrating essential techniques for improving machine learning model performance.

## Overview

Feature engineering is often the difference between mediocre and exceptional model performance. This collection provides practical, runnable examples of the most impactful feature engineering techniques used in Kaggle competitions and real-world applications.

## Examples

### 1. Automated Feature Generation
**Path**: `01_automated_feature/`
**Techniques**: Arithmetic interactions, mathematical transformations, binning, time features, categorical encoding
**Dataset**: Synthetic sales data (2,000 samples)
**Key Result**: 48% RMSE improvement, R² from 0.987 → 0.997
**Lines**: 309 (solution.py), 141 (README.md)

**What You'll Learn**:
- Creating multiplicative and ratio features
- Power transformations (squared, sqrt, log)
- Discretization of continuous variables
- Cyclical encoding for temporal features
- Systematic feature generation pipelines

---

### 2. Feature Selection Methods Comparison
**Path**: `02_feature_selection/`
**Techniques**: Variance threshold, univariate selection (F-test, MI), RFE, model-based selection
**Dataset**: Customer churn with noise features (3,000 samples)
**Key Result**: 15→30 features, improved F1 from 0.79 → 0.83
**Lines**: 364 (solution.py), 202 (README.md)

**What You'll Learn**:
- Filter methods (statistical tests)
- Wrapper methods (RFE)
- Embedded methods (model-based)
- Handling high-cardinality categoricals
- Preventing overfitting with feature selection

---

### 3. Polynomial Feature Engineering
**Path**: `03_polynomial_features/`
**Techniques**: Polynomial features (degrees 1-6), interaction terms, regularization
**Dataset**: Non-linear relationships (1,500 samples)
**Key Result**: R² improvement from 0.65 → 0.98 with degree 2
**Lines**: 380 (solution.py), 224 (README.md)

**What You'll Learn**:
- Creating polynomial features up to degree 6
- Feature explosion management
- Overfitting detection and prevention
- Ridge vs Lasso regularization
- Optimal degree selection

---

### 4. Interaction Feature Engineering
**Path**: `04_interaction_features/`
**Techniques**: Numerical×numerical, numerical×categorical, three-way interactions
**Dataset**: Insurance pricing (2,000 customers)
**Key Result**: 9.5% R² improvement with comprehensive interactions
**Lines**: 403 (solution.py), 252 (README.md)

**What You'll Learn**:
- Creating effective interaction features
- Domain knowledge vs automated approaches
- Interaction importance analysis
- Handling categorical interactions
- Balancing complexity and performance

---

### 5. Target Encoding for Categorical Features
**Path**: `05_target_encoding/`
**Techniques**: Mean encoding, smoothed encoding, CV encoding, label encoding
**Dataset**: Marketing campaigns with high-cardinality (5,000 samples, 100 cities)
**Key Result**: 19% AUC improvement, 7 features vs 170 with one-hot
**Lines**: 432 (solution.py), 268 (README.md)

**What You'll Learn**:
- Preventing data leakage in target encoding
- Smoothing techniques for rare categories
- Cross-validation encoding
- Handling unseen categories
- When to use vs one-hot encoding

---

### 6. Binning and Discretization
**Path**: `06_binning_discretization/`
**Techniques**: Equal-width, equal-frequency, domain-based, tree-based binning
**Dataset**: Credit approval (3,000 applicants)
**Key Result**: Domain-based binning achieved highest AUC (0.8645)
**Lines**: 385 (solution.py), 268 (README.md)

**What You'll Learn**:
- Choosing appropriate binning strategies
- Custom bins based on domain knowledge
- Optimal binning with decision trees
- Handling threshold effects
- When discretization helps vs hurts

---

### 7. Time-Based Feature Engineering
**Path**: `07_time_features/`
**Techniques**: Datetime components, cyclical encoding, lag features, rolling statistics
**Dataset**: Retail sales time series (730 days)
**Key Result**: 129% R² improvement (0.42 → 0.96)
**Lines**: 383 (solution.py), 338 (README.md)

**What You'll Learn**:
- Extracting temporal patterns
- Cyclical encoding with sin/cos
- Creating lag and rolling features
- Temporal train/test splitting
- Capturing seasonality and trends

---

### 8. Text Feature Extraction
**Path**: `08_text_features/`
**Techniques**: TF-IDF, count vectorization, n-grams, character n-grams, text statistics
**Dataset**: Product reviews (2,000 reviews)
**Key Result**: 50% F1 improvement with TF-IDF bigrams
**Lines**: 393 (solution.py), 362 (README.md)

**What You'll Learn**:
- TF-IDF vs count vectorization
- Unigrams, bigrams, trigrams
- Character-level features
- Sentiment lexicon features
- Text preprocessing best practices

---

## Quick Start

Each example is self-contained and immediately runnable:

```bash
# Run any example
cd 01_automated_feature
python solution.py

# Or run all examples
for dir in */; do
    echo "Running $dir"
    cd "$dir"
    python solution.py
    cd ..
done
```

## Requirements

All examples use standard Python data science libraries:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Structure

Each example directory contains:

```
01_automated_feature/
├── solution.py          # Complete runnable code (150-400 lines)
├── README.md           # Comprehensive documentation
└── *_results.png       # Generated visualizations
```

## Key Features

### Complete Examples
- ✅ Self-contained with data generation
- ✅ 150-400 lines of well-documented code
- ✅ Immediately runnable without external data
- ✅ Professional visualizations included

### Educational Content
- ✅ Before/after performance comparison
- ✅ Multiple techniques per example
- ✅ Feature importance analysis
- ✅ Best practices and pitfalls
- ✅ When to use each technique

### Real-World Applicable
- ✅ Based on common Kaggle patterns
- ✅ Production-ready code structure
- ✅ Handles edge cases
- ✅ Includes validation strategies

## Performance Summary

| Example | Baseline | After Engineering | Improvement |
|---------|----------|-------------------|-------------|
| Automated Features | R²=0.987 | R²=0.997 | +48% RMSE |
| Feature Selection | F1=0.79 | F1=0.83 | +5% F1 |
| Polynomial Features | R²=0.65 | R²=0.98 | +51% R² |
| Interaction Features | R²=0.875 | R²=0.958 | +9.5% R² |
| Target Encoding | AUC=0.71 | AUC=0.85 | +19% AUC |
| Binning | AUC=0.852 | AUC=0.865 | +1.5% AUC |
| Time Features | R²=0.42 | R²=0.96 | +129% R² |
| Text Features | F1=0.62 | F1=0.93 | +50% F1 |

## Learning Path

**Recommended Order**:

1. **Start Here**: `01_automated_feature` - Foundation techniques
2. **Then**: `02_feature_selection` - Choosing important features
3. **Advanced**: `03_polynomial_features` - Non-linear relationships
4. **Specialized**: Choose based on your data type:
   - Structured data: `04_interaction_features`, `05_target_encoding`, `06_binning_discretization`
   - Time series: `07_time_features`
   - Text data: `08_text_features`

## Common Patterns Across Examples

### Data Generation
Each example generates synthetic data with known patterns, allowing you to:
- Understand ground truth relationships
- Verify feature engineering captures them
- See clear before/after improvements

### Evaluation Strategy
All examples include:
- Train/test splits (or temporal splits for time series)
- Baseline model for comparison
- Multiple evaluation metrics
- Cross-validation where appropriate

### Visualizations
Every example generates publication-quality plots showing:
- Performance comparisons
- Feature importance
- Data distributions
- Prediction quality

## Advanced Topics Covered

- **Preventing Overfitting**: Regularization, CV encoding, smoothing
- **Handling Edge Cases**: Missing values, unseen categories, outliers
- **Computational Efficiency**: Feature hashing, dimensionality reduction
- **Interpretability**: Feature importance, domain-based features
- **Production Considerations**: Encoding consistency, temporal leakage

## Extensions and Next Steps

After mastering these examples, consider:

1. **Combine Techniques**: Use multiple methods together
2. **Automated Feature Engineering**: Feature-tools, AutoML
3. **Deep Learning**: Learned representations, embeddings
4. **Domain-Specific**: Industry-specific feature engineering
5. **Real Datasets**: Apply to Kaggle competitions

## Contributing

Each example follows these standards:
- Self-contained with synthetic data
- 150-400 lines of code
- Comprehensive README (100+ lines)
- Multiple visualizations
- Clear improvement demonstration

## References

### Books
- "Feature Engineering for Machine Learning" by Alice Zheng
- "Applied Predictive Modeling" by Kuhn & Johnson
- "Feature Engineering and Selection" by Kuhn & Johnson

### Online Resources
- Scikit-learn Feature Engineering Guide
- Kaggle Feature Engineering Tutorials
- Fast.ai Feature Engineering Course

## Summary

This collection provides **8 comprehensive examples** demonstrating the most impactful feature engineering techniques. Each example is:

- **Complete**: 150-400 lines of production-quality code
- **Runnable**: Generates synthetic data, no external dependencies
- **Educational**: Detailed documentation and visualizations
- **Practical**: Shows clear performance improvements
- **Professional**: Follows best practices and handles edge cases

**Total Code**: ~3,000 lines across 8 examples
**Total Documentation**: ~1,800 lines of detailed explanations
**Performance Gains**: 5-130% improvements demonstrated

Start with any example relevant to your problem, run it, understand the techniques, and apply them to your own datasets!
