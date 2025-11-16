# PageRank Implementation

## Overview
This solution implements Google's PageRank algorithm from scratch and analyzes web page ranking using various metrics.

## Problem Statement
PageRank is fundamental to:
- Search engine result ranking
- Understanding web page importance
- Measuring authority and relevance
- Identifying high-value content

## Approach

### 1. Web Graph Generation
- Creates directed graph representing web pages and links
- Simulates realistic linking patterns
- Pages tend to link to same-topic pages
- Generates multiple topic categories

### 2. Custom PageRank Implementation
Implements PageRank from scratch:
- Random surfer model
- Damping factor (probability of following links)
- Iterative computation until convergence
- Comparison with NetworkX implementation

### 3. Algorithm Analysis
- **Convergence**: Monitors iteration count
- **Statistics**: Distribution of PageRank scores
- **Topic Analysis**: PageRank by content category
- **Verification**: Compares custom vs library implementation

### 4. Comparison with Other Metrics
Compares PageRank with:
- **In-Degree Centrality**: Simple link counting
- **HITS Algorithm**: Authority and hub scores
- **Correlation Analysis**: Relationship between metrics

### 5. Sensitivity Analysis
- Tests different damping factors (0.5 to 0.95)
- Analyzes impact on score distribution
- Calculates Gini coefficient (inequality)

## PageRank Algorithm

The PageRank formula:
```
PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))
```

Where:
- `PR(A)`: PageRank of page A
- `d`: Damping factor (typically 0.85)
- `N`: Total number of pages
- `Ti`: Pages linking to A
- `C(Ti)`: Number of outgoing links from Ti

## Key Findings

### Implementation Verification
- Custom implementation matches NetworkX (difference < 1e-10)
- Typical convergence in 20-40 iterations

### PageRank Properties
- Not simply proportional to incoming links
- Considers quality of linking pages
- Dampens link spam effects

### Damping Factor Impact
- Higher damping → More concentrated scores
- Lower damping → More uniform distribution
- Standard value (0.85) balances both

## Visualizations
1. **Web Graph**: Node sizes proportional to PageRank
2. **Score Distribution**: Histogram of PageRank values
3. **PageRank vs Links**: Scatter plot showing relationship
4. **Metric Comparison**: Top pages by different measures

## Output Files
- `pagerank_results.csv`: All pages with scores and metrics
- `pagerank_analysis.png`: Comprehensive visualizations

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
- **Search Engines**: Ranking search results
- **Social Networks**: Identifying influential users
- **Citation Analysis**: Important papers/authors
- **Recommendation Systems**: Content importance
- **Web Analytics**: Site authority measurement

## Key Insights
- PageRank captures link structure quality
- Not all links are equal
- Recursive nature prevents gaming
- Balances popularity and authority

## Extensions
- Personalized PageRank
- Topic-sensitive PageRank
- Temporal PageRank evolution
- Weighted PageRank
- TrustRank for spam detection
