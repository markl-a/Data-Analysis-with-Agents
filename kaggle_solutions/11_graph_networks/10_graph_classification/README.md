# Graph Classification

## Overview
Classifies entire graphs into categories using graph-level structural features and machine learning.

## Problem Statement
Graph classification is essential for:
- Molecular property prediction
- Protein function classification
- Social network type identification
- Chemical compound categorization
- Network anomaly detection

## Approach

### 1. Graph Dataset Generation
Four distinct graph classes:
- **Random (Erdos-Renyi)**: Uniform random connections
- **Small-World**: High clustering, short paths
- **Scale-Free**: Power-law degree distribution
- **Community**: Strong modular structure

### 2. Feature Extraction
Extract 20+ structural features:

#### Basic Features
- Number of nodes/edges
- Density
- Connectivity

#### Degree Features
- Average, std, max, min degree
- Degree skewness, kurtosis

#### Clustering Features
- Average clustering coefficient
- Transitivity

#### Path Features
- Diameter
- Average shortest path
- Radius

#### Advanced Features
- Assortativity
- Betweenness centrality
- Graph energy (eigenvalues)

### 3. Machine Learning
- Random Forest classifier
- Feature scaling
- Train/test split
- Multi-class classification

### 4. Feature Analysis
- Feature importance ranking
- Distribution by class
- Discriminative features

### 5. Evaluation
- Accuracy metrics
- Confusion matrix
- Per-class performance

## Graph Classes

### Random Graphs
- Uniform edge probability
- Poisson degree distribution
- Low clustering

### Small-World Graphs
- High clustering
- Short average paths
- "Six degrees of separation"

### Scale-Free Graphs
- Power-law degree distribution
- Hub nodes
- Preferential attachment

### Community Graphs
- Dense intra-community edges
- Sparse inter-community edges
- High modularity

## Key Features for Classification

### Most Discriminative
1. **Average Clustering**: Separates small-world
2. **Degree Distribution**: Identifies scale-free
3. **Assortativity**: Random vs structured
4. **Density**: Community graphs differ
5. **Diameter**: Path length varies

### Feature Patterns
- Small-world: High clustering, low diameter
- Scale-free: High degree variance
- Community: High clustering, high modularity
- Random: Moderate on all measures

## Key Findings

### Classification Performance
- Accuracy typically > 90%
- Clear separation between classes
- Structural features highly predictive

### Important Features
1. Average clustering coefficient
2. Degree statistics
3. Assortativity
4. Diameter
5. Graph density

### Class Characteristics
Each graph type has unique signature in feature space.

## Visualizations
1. **Feature Importance**: Top predictive features
2. **Confusion Matrix**: Classification accuracy
3. **Feature Distributions**: By graph class
4. **Feature Space**: 2D projection

## Output Files
- `graph_features.csv`: All extracted features
- `graph_feature_importance.csv`: Feature rankings
- `graph_classification_analysis.png`: Visualizations

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
- **Chemistry**: Molecule property prediction
- **Biology**: Protein function classification
- **Social Networks**: Community type detection
- **Cybersecurity**: Malware network patterns
- **Finance**: Transaction network analysis
- **Neuroscience**: Brain network classification

## Key Insights
- Graph structure is highly informative
- Multiple features needed for accuracy
- Different graph types have distinct signatures
- Machine learning effectively captures patterns

## Extensions
- Graph neural networks (GNNs)
- Graph kernels
- Spectral features
- Motif-based features
- Deep learning approaches
- Larger benchmark datasets (TUDataset)
- Node attributes incorporation
- Edge features
