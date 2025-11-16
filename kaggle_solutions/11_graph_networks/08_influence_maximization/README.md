# Influence Maximization

## Overview
Finds optimal seed nodes for maximum influence spread in social networks using various algorithms.

## Problem Statement
Influence maximization is crucial for:
- Viral marketing campaigns
- Product adoption strategies
- Information dissemination
- Public health interventions
- Social awareness campaigns

## Approach

### 1. Network Generation
- Scale-free social network
- Edge weights represent influence probabilities
- Realistic network structure

### 2. Influence Propagation Model
**Independent Cascade (IC) Model:**
- Start with seed nodes
- Each influenced node tries to influence neighbors
- Success probability based on edge weights
- Process continues until no new activations

### 3. Seed Selection Algorithms

#### Greedy Algorithm
- Iteratively select node with maximum marginal gain
- Optimal approximation (1-1/e ≈ 63%)
- Computationally expensive

#### Degree Centrality
- Select high-degree nodes
- Fast heuristic
- Often near-optimal

#### PageRank
- Select high PageRank nodes
- Captures global importance

#### Betweenness Centrality
- Select bridge nodes
- Good for information flow

#### Random (Baseline)
- Random selection for comparison

### 4. Performance Comparison
- Monte Carlo simulations
- Average influence spread
- Algorithm trade-offs

### 5. Temporal Analysis
- Track influence spread over time
- Cascade dynamics
- Saturation analysis

## Independent Cascade Model

### Process
1. Activate seed nodes at t=0
2. At each step t:
   - Newly activated nodes try to activate neighbors
   - Success with probability p(u,v)
   - Each attempt made once
3. Stop when no new activations

### Expected Influence
Average nodes influenced across many simulations.

## Key Findings

### Algorithm Performance
1. **Greedy**: Highest influence, slow
2. **Degree**: Near-optimal, very fast
3. **PageRank**: Good balance
4. **Betweenness**: Moderate performance
5. **Random**: Poor baseline

### Seed Node Properties
- High-degree nodes often optimal
- Central network positions
- Well-connected neighborhoods
- Bridge communities

### Spread Dynamics
- Rapid initial spread
- Gradual saturation
- Network structure dependent

## Visualizations
1. **Network with Seeds**: Highlighted optimal nodes
2. **Algorithm Comparison**: Performance ranking
3. **Temporal Spread**: Influence over time
4. **Degree Distribution**: Seed characteristics

## Output Files
- `influence_maximization_results.csv`: Algorithm comparison
- `influence_maximization_analysis.png`: Visualizations

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
- **Marketing**: Identify influencers for campaigns
- **Public Health**: Vaccination strategies
- **Politics**: Opinion leader identification
- **Technology**: Early adopter selection
- **Social Good**: Awareness campaign seeding

## Key Insights
- Few well-chosen seeds can influence many
- Network structure matters more than individual attributes
- Degree centrality is surprisingly effective
- Greedy gives best results but is slow

## Extensions
- Linear Threshold (LT) model
- Time-varying networks
- Budget constraints
- Competitive influence (multiple campaigns)
- Real network data analysis
- Deep learning approaches
