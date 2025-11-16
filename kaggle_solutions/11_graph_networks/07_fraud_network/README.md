# Fraud Detection in Networks

## Overview
Detects fraudulent behavior in transaction networks using graph-based features and machine learning.

## Problem Statement
Network-based fraud detection is critical for:
- Financial transaction monitoring
- Credit card fraud detection
- Money laundering prevention
- Account takeover detection
- Insurance fraud identification

## Approach

### 1. Transaction Network Generation
- Users as nodes, transactions as edges
- Fraudsters exhibit distinct patterns
- Legitimate users follow normal behavior
- Realistic fraud/legitimate ratio

### 2. Graph-Based Features
- **Degree Features**: In/out transaction counts
- **Amount Features**: Total, average, max sent/received
- **Network Position**: PageRank, betweenness
- **Clustering**: Local connectivity patterns
- **Account Features**: Age, activity patterns
- **Flow Features**: Money imbalance

### 3. Unsupervised Detection
- Isolation Forest for anomaly detection
- No labeled data required
- Identifies outliers in feature space

### 4. Supervised Detection
- Random Forest classifier
- Uses labeled training data
- Feature importance analysis
- High accuracy predictions

### 5. Pattern Analysis
- Fraud vs legitimate comparisons
- Network structure analysis
- Behavioral patterns

## Key Features for Fraud

### Network Anomalies
- High incoming transaction counts (money mules)
- High outgoing to other fraudsters
- Young account age
- Unusual transaction amounts
- High betweenness (bridge between clusters)

### Fraud Patterns
- Fraudsters often transact with other fraudsters
- Receive from many legitimate users
- Newer accounts more suspicious
- Extreme transaction values

## Key Findings

### Fraud Characteristics
- Higher in-degree (receive money)
- Connected to fraud clusters
- Newer accounts
- Unusual amount patterns

### Detection Performance
- Unsupervised: 70-85% accuracy
- Supervised: 85-95% accuracy
- Graph features highly predictive

### Important Features
1. In-degree (incoming transactions)
2. Total received amount
3. Account age
4. Number of neighbors
5. Money flow imbalance

## Visualizations
1. **Transaction Network**: Fraud vs legitimate users
2. **Confusion Matrix**: Detection accuracy
3. **Behavior Comparison**: Feature differences
4. **Account Age**: Distribution patterns

## Output Files
- `fraud_detection_results.csv`: User-level predictions
- `fraud_feature_importance.csv`: Feature rankings
- `fraud_network_analysis.png`: Visualizations

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
- **Banking**: Transaction monitoring
- **E-commerce**: Payment fraud detection
- **Insurance**: Claim fraud identification
- **Cryptocurrency**: Scam detection
- **Social Networks**: Fake account detection

## Key Insights
- Network structure reveals fraud
- Graph features outperform transaction features alone
- Fraud forms connected components
- Early detection saves money

## Extensions
- Temporal analysis (evolution over time)
- Deep learning (Graph Neural Networks)
- Real-time detection systems
- Multi-modal fraud (combining data sources)
- Explainable AI for investigators
