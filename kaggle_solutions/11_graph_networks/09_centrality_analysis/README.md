# Network Centrality Analysis

## Overview
Comprehensive analysis of various centrality measures to identify important nodes in networks.

## Problem Statement
Centrality analysis is fundamental for:
- Identifying key individuals in organizations
- Finding critical infrastructure nodes
- Detecting influential users
- Network vulnerability assessment
- Resource allocation optimization

## Approach

### 1. Network Generation
- Small-world network (Watts-Strogatz)
- Realistic structure
- Balanced clustering and path lengths

### 2. Centrality Measures Computed

#### Degree Centrality
- Number of connections
- Local importance
- Simple but effective

#### Betweenness Centrality
- Number of shortest paths through node
- Bridge/broker identification
- Information flow control

#### Closeness Centrality
- Average distance to all other nodes
- Access/reachability measure
- Information spread efficiency

#### Eigenvector Centrality
- Connections to well-connected nodes
- Recursive importance
- Quality over quantity

#### PageRank
- Google's algorithm
- Random walk probability
- Damped eigenvector centrality

#### Katz Centrality
- Walks of all lengths
- Parameter-controlled reach
- Generalization of eigenvector

#### Harmonic Centrality
- Sum of inverse distances
- Handles disconnected graphs
- Alternative to closeness

#### Load Centrality
- Betweenness variant
- Communication load
- Stress measure

### 3. Statistical Analysis
- Distribution properties
- Gini coefficient (inequality)
- Skewness analysis
- Summary statistics

### 4. Correlation Analysis
- Pearson correlations
- Spearman rank correlations
- Measure relationships

### 5. Key Node Identification
- Composite scoring
- Specialist detection
- Multi-metric ranking

## Centrality Formulas

### Degree Centrality
```
C_D(v) = deg(v) / (n-1)
```

### Betweenness Centrality
```
C_B(v) = Σ(σ_st(v) / σ_st)
```
Where σ_st is shortest paths s→t, σ_st(v) through v

### Closeness Centrality
```
C_C(v) = (n-1) / Σd(v,u)
```

### PageRank
```
PR(v) = (1-d)/n + d·Σ(PR(u)/deg(u))
```

## Key Findings

### Centrality Properties
- Different measures capture different aspects
- High correlation between some measures
- Specialist vs generalist nodes

### Node Types
- **Hubs**: High degree
- **Bridges**: High betweenness
- **Central**: High closeness
- **Influential**: High eigenvector/PageRank

### Distribution Patterns
- Power-law tendencies
- High inequality (Gini)
- Long-tailed distributions

### Correlations
- Degree-PageRank: High
- Degree-Betweenness: Moderate
- Closeness-Betweenness: Varies by structure

## Visualizations
1. **Network Graph**: Sized by centrality
2. **Correlation Heatmap**: Measure relationships
3. **Distribution Boxplots**: Value ranges
4. **Top Nodes Profile**: Multi-metric view

## Output Files
- `centrality_measures.csv`: All centrality values
- `normalized_centralities.csv`: Normalized scores
- `centrality_analysis.png`: Visualizations

## Requirements
```
networkx
numpy
pandas
matplotlib
seaborn
scipy
```

## Usage
```bash
python solution.py
```

## Real-World Applications
- **Organizations**: Leadership identification
- **Infrastructure**: Critical node detection
- **Social Media**: Influencer discovery
- **Biology**: Protein interaction importance
- **Transportation**: Hub identification
- **Cybersecurity**: Attack vulnerability

## Key Insights
- No single best centrality measure
- Context determines appropriate measure
- Multiple measures provide robust identification
- Correlations reveal network structure

## Extensions
- Weighted networks
- Directed networks
- Temporal centrality evolution
- Community-aware centrality
- Percolation centrality
- K-shell decomposition
