# Social Network Friend Recommendation System using Graph-Based Methods

## Overview
This solution implements a friend recommendation system for social networks using graph-based algorithms. It combines structural network analysis (common neighbors, Jaccard similarity, Adamic-Adar) with content-based user similarity (interests, location, demographics).

## Problem Statement
Given a social network graph and user profiles, build a recommendation system that can:
- Suggest potential friends based on network structure
- Consider user attributes and interests
- Handle network dynamics and growth
- Provide explainable recommendations
- Balance diversity and relevance

## Approach

### Graph-Based Link Prediction

The problem of friend recommendation is a **link prediction** problem: predict which edges (friendships) are likely to form in the future.

We use multiple graph-based metrics:

### 1. Common Neighbors (CN)

**Intuition**: Friends of friends are likely to become friends

```
CN(u,v) = |neighbors(u) ∩ neighbors(v)|
```

**Example**: If you and a stranger have 10 mutual friends, you're likely to befriend them.

### 2. Jaccard Similarity

**Intuition**: Normalize common neighbors by total neighbors

```
Jaccard(u,v) = |neighbors(u) ∩ neighbors(v)| / |neighbors(u) ∪ neighbors(v)|
```

**Advantage**: Handles users with different numbers of friends better than raw common neighbors.

### 3. Adamic-Adar Index

**Intuition**: Common neighbors with fewer friends provide stronger evidence

```
AA(u,v) = Σ 1/log(|neighbors(z)|) for z in common_neighbors(u,v)
```

**Rationale**: A mutual friend with 1000 friends provides less signal than one with 10 friends.

### 4. Content-Based Similarity

Uses user attributes:
- **Interests**: Jaccard similarity of interest sets
- **Location**: Same city/area
- **Age**: Similar age group

### 5. Hybrid Score

Combines all signals:
```
score = 0.3 × CN_normalized + 0.3 × AA_normalized + 0.2 × Jaccard + 0.2 × Content
```

## Key Features

1. **Multiple Algorithms**: Implements 5 different recommendation methods
2. **Homophily Modeling**: Similar users tend to connect
3. **Preferential Attachment**: Popular users attract more connections
4. **Content Integration**: Combines network and profile data
5. **Explainability**: Can explain why someone was recommended

## Data Generation

### Network Generation Process
Uses realistic network formation models:

1. **Homophily**: Users with common interests/location more likely to connect
2. **Preferential Attachment**: Popular users gain friends faster (power law distribution)
3. **Triadic Closure**: Friends of friends become friends

### User Attributes
- **Interests**: 12 categories, users have 2-4 interests
- **Location**: 10 cities
- **Age Groups**: 5 age brackets
- **Join Date**: Account creation date

### Network Statistics
- **Nodes**: 500 users
- **Edges**: ~10,000 friendships
- **Avg Degree**: ~20 friends/user
- **Distribution**: Power law (few highly connected users, many with fewer friends)

## Evaluation Metrics

1. **Precision@10**
   - Of top-10 recommendations, how many become actual friends?
   - Measures recommendation accuracy

2. **Recall@10**
   - Of all future friends, how many were in top-10?
   - Measures recommendation coverage

## Implementation Details

### Algorithm Workflow
1. Generate social network with realistic structure
2. Model friendships using homophily and preferential attachment
3. Build graph as adjacency list
4. For each user, compute scores for all non-friends
5. Rank candidates by score
6. Evaluate on held-out future friendships

### Graph Representation
- **Adjacency List**: `{user_id: {friend_ids}}`
- **Undirected**: If A is friends with B, B is friends with A
- **Unweighted**: All friendships have equal weight

### Computational Complexity
- **Common Neighbors**: O(|neighbors(u)| × |neighbors(v)|)
- **Jaccard**: O(|neighbors(u)| + |neighbors(v)|)
- **Adamic-Adar**: O(|common_neighbors| × |neighbors(z)|)
- **Full Recommendation**: O(n²) for n users (can optimize with candidate selection)

## Results

Typical performance metrics:
- **Precision@10**: ~0.15-0.30
- **Recall@10**: ~0.20-0.35

Note: Friend recommendation is inherently difficult due to many valid choices.

## Visualizations

The solution generates four visualizations:

1. **Friend Count Distribution**: Shows power law (scale-free network)
2. **Interest Distribution**: Popular vs. niche interests
3. **Location Distribution**: Geographic user distribution
4. **Age Distribution**: Demographic breakdown

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn

## Advantages of Graph-Based Methods

1. **No Cold Start for Structure**: Works if user has any friends
2. **Network Effects**: Leverages social network structure
3. **Interpretability**: "You both know Alice and Bob"
4. **Proven Effectiveness**: Strong empirical performance
5. **Scalability**: Can optimize with approximate methods

## Limitations

1. **Cold Start**: New users with 0 friends get no structural recommendations
2. **Popularity Bias**: Tends to recommend already-popular users
3. **Filter Bubble**: May reinforce existing community structure
4. **Privacy**: Reveals network structure

## Improvements and Extensions

1. **Temporal Dynamics**: Model how friendships form over time
2. **Community Detection**: Recommend within/across communities
3. **Random Walk Methods**: PageRank, SimRank
4. **Deep Learning**: Graph Neural Networks (GNN)
5. **Negative Sampling**: Learn from non-friendships
6. **Context-Aware**: Consider interaction context
7. **Mutual Friends Display**: Show which friends are mutual
8. **Social Circles**: Respect friendship circles (work, school, family)

## Business Applications

### For Social Platforms
- **Facebook**: "People You May Know"
- **LinkedIn**: Professional connection suggestions
- **Twitter**: "Who to Follow"
- **Instagram**: Account recommendations
- **TikTok**: Creator suggestions

### Use Cases
- **Network Growth**: Help users expand their networks
- **User Engagement**: Keep users active on platform
- **Community Building**: Foster communities around interests
- **Professional Networking**: Career opportunities
- **Dating Apps**: Suggest compatible matches

### Business Metrics
- **Acceptance Rate**: % of recommendations accepted
- **Network Growth**: New connections per user
- **User Retention**: Impact on user engagement
- **Session Time**: Users explore recommendations longer
- **Viral Coefficient**: How recommendations spread

## Advanced Features

1. **Mutual Interest Groups**: Recommend based on group membership
2. **Event-Based**: Recommend attendees of same events
3. **Content Interaction**: Users who liked same posts
4. **Conversation Starters**: Suggest why to connect
5. **Privacy Controls**: Don't recommend in certain contexts
6. **Diversity**: Balance similar and diverse recommendations
7. **A/B Testing**: Compare algorithm variants
8. **Notification Optimization**: When to suggest friends

## Social Network Science

### Network Properties
- **Small World**: Most users connected by short paths (6 degrees)
- **Clustering**: Friends of friends are often friends
- **Power Law Degree**: Few hubs, many with few connections
- **Homophily**: "Birds of a feather flock together"
- **Triadic Closure**: Open triangles tend to close

### Psychological Factors
- **Social Proof**: Mutual friends increase trust
- **Similarity Attraction**: We befriend similar people
- **Proximity**: Geographic closeness matters
- **Reciprocity**: Friend requests often reciprocated
- **Status**: Interest in connecting with high-status users

## Privacy and Ethics

1. **Consent**: Users should control discoverability
2. **Harassment Prevention**: Don't enable stalking
3. **Filter Bubbles**: Avoid echo chambers
4. **Discrimination**: Avoid biased recommendations
5. **Transparency**: Explain why someone was suggested
6. **Data Protection**: Secure social graph data

## Technical Considerations

1. **Scalability**: Billions of users, trillions of edges
2. **Real-Time**: Update recommendations as network changes
3. **Sampling**: Don't compute all O(n²) pairs
4. **Caching**: Pre-compute recommendations
5. **Distributed**: Use graph databases (Neo4j, etc.)
6. **Incremental**: Update efficiently on edge addition

## References

- Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. Journal of the American society for information science and technology, 58(7), 1019-1031.
- Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the web. Social networks, 25(3), 211-230.
- Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). Mining of massive datasets. Cambridge university press.
