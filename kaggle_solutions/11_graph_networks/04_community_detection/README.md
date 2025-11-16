# Community Detection

## Overview
This solution detects communities in networks using multiple algorithms including Louvain, Label Propagation, and Girvan-Newman.

## Problem Statement
Community detection is crucial for:
- Social group identification
- Market segmentation
- Protein complex detection
- Topic clustering in citation networks
- Organization structure analysis

## Approach

### 1. Network Generation
- Uses Stochastic Block Model (SBM)
- Creates known community structure
- High intra-community edges, low inter-community edges
- Provides ground truth for evaluation

### 2. Louvain Algorithm
- Greedy modularity optimization
- Fast and scalable
- Hierarchical community structure
- Industry standard

### 3. Label Propagation
- Nodes adopt majority label of neighbors
- Simple and fast
- Non-deterministic
- Good for large networks

### 4. Girvan-Newman
- Edge betweenness-based
- Removes edges with highest betweenness
- Creates hierarchical dendrogram
- Computationally expensive

### 5. Evaluation
- **Modularity**: Quality of community division
- **ARI**: Adjusted Rand Index vs ground truth
- **NMI**: Normalized Mutual Information
- Community structure analysis

## Key Metrics

### Modularity
```
Q = 1/(2m) Σ[A_ij - k_i*k_j/(2m)]δ(c_i,c_j)
```
Where:
- `m`: number of edges
- `A_ij`: adjacency matrix
- `k_i, k_j`: node degrees
- `δ(c_i,c_j)`: 1 if same community

### Adjusted Rand Index
- Measures agreement with ground truth
- Accounts for chance
- Range: -1 to 1 (1 = perfect match)

## Key Findings

### Algorithm Performance
- **Louvain**: Highest modularity, fast
- **Label Propagation**: Fast, moderate accuracy
- **Girvan-Newman**: Accurate but slow

### Community Properties
- Clear internal/external edge ratio
- High internal density
- Strong clustering within communities

### Modularity Scores
- Higher modularity = better communities
- Typical range: 0.3 to 0.7
- Depends on network structure

## Visualizations
1. **Ground Truth**: Known community structure
2. **Detected Communities**: Louvain results
3. **Size Comparison**: Community sizes
4. **Modularity Comparison**: Algorithm quality

## Output Files
- `community_statistics.csv`: Per-community metrics
- `algorithm_comparison.csv`: Performance comparison
- `community_detection_analysis.png`: Visualizations

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
- **Social Media**: Group detection, interest communities
- **Biology**: Protein complexes, gene modules
- **Marketing**: Customer segmentation
- **Organization**: Team structure analysis
- **Infrastructure**: Network partitioning

## Key Insights
- Modularity optimization works well
- Multiple algorithms provide validation
- Ground truth comparison important
- Trade-off between speed and accuracy

## Extensions
- Overlapping communities
- Dynamic community detection
- Hierarchical communities
- Attributed networks
- Multi-layer networks
