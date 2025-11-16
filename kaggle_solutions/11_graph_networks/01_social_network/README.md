# Social Network Analysis

## Overview
This solution analyzes social network connections, identifies influential users, and detects communities within a network using graph analysis techniques.

## Problem Statement
Understanding social networks is crucial for:
- Identifying influential users and opinion leaders
- Detecting communities and groups
- Understanding information flow
- Optimizing marketing and outreach strategies

## Approach

### 1. Network Generation
- Creates a scale-free network using Barabasi-Albert model
- Simulates realistic social network properties
- Generates user attributes (age, join date, etc.)

### 2. Network Analysis
- **Basic Properties**: Nodes, edges, density, connectivity
- **Degree Statistics**: Connection distribution analysis
- **Clustering**: Measures network cohesiveness
- **Path Analysis**: Diameter and average shortest paths

### 3. Influence Detection
Multiple centrality measures to identify influential users:
- **Degree Centrality**: Users with most connections
- **Betweenness Centrality**: Bridge users connecting communities
- **Closeness Centrality**: Users with shortest paths to others
- **Eigenvector Centrality**: Influence based on connected users
- **PageRank**: Google's page ranking algorithm

### 4. Community Detection
- Uses greedy modularity optimization
- Identifies natural groupings in the network
- Calculates modularity score

## Key Findings

### Network Properties
- Scale-free network with preferential attachment
- Power-law degree distribution
- Small-world properties

### Influential Users
- Different metrics identify different types of influence
- Degree centrality finds popular users
- Betweenness finds bridge connectors
- PageRank finds overall influence

### Communities
- Natural clustering of users
- High modularity indicates strong community structure
- Community sizes follow typical distributions

## Visualizations
1. **Network Graph**: Shows connections and communities
2. **Degree Distribution**: Shows connection patterns
3. **Centrality Comparison**: Compares top users
4. **Community Sizes**: Distribution of community sizes

## Output Files
- `social_network_users.csv`: User data with centrality metrics
- `social_network_analysis.png`: Comprehensive visualizations

## Requirements
```
networkx
numpy
pandas
matplotlib
seaborn
```

## Usage
```bash
python solution.py
```

## Real-World Applications
- **Social Media**: Identify influencers and communities
- **Marketing**: Target key users for campaigns
- **Organization Analysis**: Understand team structures
- **Recommendation Systems**: Friend suggestions
- **Viral Marketing**: Seed selection for campaigns

## Key Metrics
- **Network Density**: Connectivity level
- **Clustering Coefficient**: Local cohesiveness
- **Modularity**: Community structure quality
- **Centrality Measures**: User importance

## Extensions
- Temporal network analysis (evolution over time)
- Directed networks (follower/following)
- Weighted edges (interaction strength)
- Dynamic community detection
- Link prediction
