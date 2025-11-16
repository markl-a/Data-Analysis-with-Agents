# Advanced Credit Card Fraud Detection

## Overview
This example demonstrates advanced anomaly detection techniques for identifying fraudulent credit card transactions. It compares three popular methods: Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.

## Problem Description
Credit card fraud is a critical problem in the financial industry. This solution detects fraudulent transactions by:
- Analyzing transaction patterns (amount, time, frequency)
- Identifying unusual geographic locations
- Detecting anomalous feature combinations

## Dataset
Synthetic dataset with 10,000 transactions including:
- **Normal transactions (98%)**: Regular purchasing patterns
- **Fraudulent transactions (2%)**: Unusual amounts, times, and locations

### Features
- `Time`: Transaction timestamp (seconds in day)
- `V1-V4`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Distance_from_home`: Distance from cardholder's home
- `Transaction_frequency_24h`: Number of transactions in last 24h
- `Class`: 0 (normal) or 1 (fraud)

## Methods Used

### 1. Isolation Forest
- **How it works**: Isolates observations by randomly selecting features and split values
- **Strengths**: Fast, effective for high-dimensional data
- **Parameters**: 100 estimators, 2% contamination

### 2. Local Outlier Factor (LOF)
- **How it works**: Measures local density deviation compared to neighbors
- **Strengths**: Detects local anomalies in varying density regions
- **Parameters**: 2% contamination

### 3. One-Class SVM
- **How it works**: Learns decision boundary around normal data
- **Strengths**: Effective for non-linear patterns
- **Parameters**: Auto gamma, 2% nu parameter

## Evaluation Metrics
- **Precision**: Proportion of predicted frauds that are actual frauds
- **Recall**: Proportion of actual frauds detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## Results Visualizations
1. **data_distribution.png**: Feature distributions for normal vs fraud
2. **confusion_matrices.png**: Model prediction accuracy
3. **model_comparison.png**: Performance comparison
4. **pca_visualization.png**: 2D projection of anomaly detection

## Key Insights
- Fraudulent transactions show distinct patterns in amount and frequency
- Multiple models provide complementary detection capabilities
- Ensemble approach recommended for production systems
- Feature engineering (distance, frequency) improves detection

## Usage
```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Performance Considerations
- Imbalanced dataset requires careful metric selection
- Precision/Recall tradeoff depends on business cost of false positives vs false negatives
- Real-time systems need fast inference (Isolation Forest recommended)

## Extensions
1. Add temporal features (day of week, hour patterns)
2. Implement ensemble voting across models
3. Use SMOTE for balanced training
4. Add deep learning autoencoder
5. Incorporate transaction velocity features
