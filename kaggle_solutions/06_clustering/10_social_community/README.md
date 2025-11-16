# Social Network Community Detection

## Overview
This solution demonstrates community detection in social networks using clustering algorithms combined with graph-based features. Identifies groups of users with similar behavior and interaction patterns.

## Problem Statement
Social networks contain hidden community structures - groups of users who interact more frequently with each other than with outsiders. Detecting these communities helps understand network dynamics, recommend connections, target content, and identify influential users.

## Dataset Features

### User Activity Metrics
- **posts_count**: Number of posts created
- **likes_given**: Total likes given to others
- **comments_made**: Total comments made
- **followers_count**: Number of followers
- **following_count**: Number of users being followed

### Derived Engagement Features
- **engagement_ratio**: Likes per post (passive engagement)
- **interaction_ratio**: Comments per post (active engagement)
- **follower_ratio**: Followers to following ratio (influence)

### Network Topology Features
- **degree_centrality**: Normalized connection count
- **betweenness_centrality**: Bridge position in network
- **clustering_coefficient**: Tendency of connections to cluster
- **pagerank**: Influence score (Google's PageRank algorithm)

## Community Types Generated
1. **Tech Enthusiasts**: High activity, many connections
2. **Casual Users**: Moderate engagement across metrics
3. **Influencers**: Very high activity, large follower base
4. **Lurkers**: Low activity, few posts but some connections
5. **Content Creators**: High post count, active engagement

## Clustering Algorithms
1. **K-Means**: Standard clustering on feature vectors
2. **Spectral Clustering**: Graph-based method using eigenvectors
   - Excellent for detecting communities in network data
   - Uses nearest neighbors affinity
3. **Hierarchical (Ward)**: Agglomerative clustering
   - Creates community hierarchy

## Network Analysis
Uses NetworkX library for:
- Graph construction from edges
- Centrality metrics calculation
- Network visualization with spring layout
- Community structure analysis

## Evaluation Metrics
- **Silhouette Score**: How well users fit their communities
- **Davies-Bouldin Index**: Community separation quality
- **Calinski-Harabasz Score**: Variance-based evaluation
- **Network Density**: Overall connectivity
- **Average Clustering**: Tendency to form clusters

## Analysis Steps
1. Generate 200 users across 5 communities
2. Create network edges (intra-community and bridges)
3. Calculate network topology features using NetworkX
4. Combine behavioral and network features
5. Determine optimal community count
6. Compare K-Means, Spectral, and Hierarchical clustering
7. Visualize network graph with detected communities
8. Profile each community's characteristics

## Key Visualizations
- **Network Graph**: Nodes colored by community, spring layout
- **PCA Scatter**: 2D projection of feature space
- **Box Plots**: Compare metrics across communities
- **Silhouette Analysis**: Optimal community count

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
networkx
```

## Usage
```bash
python solution.py
```

## Output
1. Social network statistics (nodes, edges, density)
2. Network visualization (true communities)
3. Optimal community count plot
4. PCA visualization of detected communities
5. Detailed community profiles
6. Comparative box plots across communities
7. Network with detected communities colored

## Applications

### Social Media Platforms
- **Friend Recommendations**: Suggest connections within community
- **Content Targeting**: Tailor feeds to community interests
- **Moderation**: Identify problematic groups
- **Advertising**: Target ads to specific communities

### Network Analysis
- **Influence Identification**: Find community leaders (high PageRank)
- **Bridge Detection**: Users connecting different communities
- **Information Flow**: Understand how content spreads
- **Network Resilience**: Identify critical connectors

### Business Intelligence
- **Customer Segmentation**: Group customers by behavior
- **Churn Prediction**: Detect at-risk communities
- **Product Recommendations**: Community-based suggestions
- **Market Research**: Understand user groups

### Security
- **Bot Detection**: Identify coordinated bot networks
- **Fraud Detection**: Spot fraudulent user rings
- **Misinformation**: Track spread through communities

## Technical Advantages
- **Spectral Clustering**: Superior for graph-structured data
- **NetworkX Integration**: Powerful graph analysis tools
- **Multi-feature**: Combines behavior and topology
- **Scalable**: Efficient for thousands of users

## Community Characteristics
Each detected community typically shows:
- **Distinct activity patterns**: Posts, likes, comments
- **Network position**: Central vs peripheral
- **Engagement style**: Active creators vs passive consumers
- **Influence level**: Follower counts and PageRank

## Comparison with Other Methods
- **Louvain Algorithm**: Modularity-based community detection
- **Label Propagation**: Fast iterative method
- **Girvan-Newman**: Edge betweenness-based
- **Clustering**: Feature-based approach (this solution)

## Real-World Extensions
- Temporal community evolution
- Overlapping community detection
- Weighted edge importance
- Content-based features (text analysis)
- Demographic information integration
- Multi-platform network analysis

## Performance Notes
- NetworkX efficient for networks up to ~10K nodes
- Spectral clustering best for community structure
- K-Means faster but may miss graph patterns
- Betweenness centrality computation can be slow (sampled)

## Interpretation Tips
- High degree centrality = well-connected user
- High betweenness = bridge between communities
- High clustering coefficient = tight-knit group
- High PageRank = influential user

## Author
Kaggle Competition Solution - Network Analysis Clustering
