# Link Prediction

## Overview
This solution predicts future connections in networks using graph-based features and machine learning.

## Problem Statement
Link prediction is essential for:
- Friend recommendations in social networks
- Product recommendations
- Collaboration prediction
- Network evolution forecasting
- Missing data imputation

## Approach

### 1. Network Generation
- Creates evolving network using Barabasi-Albert model
- Simulates realistic connection patterns
- Provides ground truth for evaluation

### 2. Train/Test Split
- Removes portion of edges for testing
- Creates negative samples (non-edges)
- Maintains balanced classes

### 3. Feature Engineering
Graph-based similarity features:
- **Common Neighbors**: Shared connections
- **Jaccard Coefficient**: Normalized common neighbors
- **Adamic-Adar Index**: Weighted common neighbors
- **Preferential Attachment**: Product of degrees
- **Resource Allocation**: Flow-based similarity
- **Node Degrees**: Connection counts
- **Clustering Coefficients**: Local density
- **Shortest Path Length**: Distance between nodes

### 4. Machine Learning Model
- Random Forest classifier
- Handles non-linear relationships
- Feature importance analysis
- ROC-AUC evaluation

### 5. Similarity Analysis
- Compares different metrics
- Links vs non-links distributions
- Identifies best predictors

## Key Algorithms

### Jaccard Coefficient
```
J(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
```

### Adamic-Adar Index
```
AA(u,v) = Σ(1 / log|N(w)|) for w in N(u) ∩ N(v)
```

### Preferential Attachment
```
PA(u,v) = |N(u)| × |N(v)|
```

## Key Findings

### Best Features
1. Adamic-Adar Index (highest importance)
2. Common neighbors count
3. Resource allocation index
4. Jaccard coefficient

### Model Performance
- AUC-ROC typically > 0.85
- Clear separation between links and non-links
- Graph features strongly predictive

### Metric Comparison
- Links have higher similarity scores
- Non-links often have zero common neighbors
- Preferential attachment less discriminative

## Visualizations
1. **Feature Importance**: Top predictive features
2. **ROC Curve**: Model discrimination ability
3. **Precision-Recall**: Performance across thresholds
4. **Metric Comparison**: Links vs non-links

## Output Files
- `link_prediction_features.csv`: Feature importance
- `link_prediction_analysis.png`: Visualizations

## Requirements
```
networkx
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Usage
```bash
python solution.py
```

## Real-World Applications
- **Social Networks**: Friend recommendations
- **E-commerce**: Product/seller recommendations
- **Research**: Collaboration prediction
- **Biology**: Protein interaction prediction
- **Security**: Fraud network detection

## Key Insights
- Graph structure is highly predictive
- Simple features work well
- Common neighbors crucial predictor
- ML improves over simple thresholds

## Extensions
- Deep learning (Graph Neural Networks)
- Temporal link prediction
- Multi-relational networks
- Dynamic network evolution
- Attributed networks
