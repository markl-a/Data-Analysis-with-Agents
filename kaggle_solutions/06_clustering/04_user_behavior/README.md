# User Behavior Clustering Analysis

## Overview
This solution demonstrates clustering analysis on user behavior data to segment users based on their interaction patterns, session duration, and engagement metrics.

## Problem Statement
Understanding user behavior is crucial for product optimization, personalized marketing, and user retention strategies. This analysis clusters users into distinct segments based on their behavioral patterns.

## Dataset Features
- **sessions_per_week**: Number of sessions per week
- **avg_session_duration**: Average duration of each session (seconds)
- **pages_per_session**: Average number of pages viewed per session
- **bounce_rate**: Percentage of single-page sessions
- **conversion_rate**: Percentage of sessions ending in conversion
- **time_on_site**: Average time spent on site (seconds)

## User Segments Generated
1. **Power Users**: High engagement, frequent sessions, low bounce rate
2. **Regular Users**: Moderate engagement and session frequency
3. **Occasional Users**: Low engagement, infrequent visits
4. **Churned/At-Risk Users**: Very low engagement, high bounce rate

## Clustering Algorithms
1. **K-Means**: Partitioning method that minimizes within-cluster variance
2. **DBSCAN**: Density-based clustering that can find arbitrarily shaped clusters
3. **Agglomerative Clustering**: Hierarchical clustering using bottom-up approach

## Evaluation Metrics
- **Silhouette Score**: Measures how similar an object is to its cluster (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)

## Analysis Steps
1. Generate realistic user behavior data (1000 users)
2. Standardize features using StandardScaler
3. Apply elbow method to find optimal number of clusters
4. Compare K-Means, DBSCAN, and Agglomerative clustering
5. Visualize clusters using PCA dimensionality reduction
6. Profile each cluster to understand characteristics

## Key Insights
- Elbow method helps determine optimal cluster count
- Different algorithms may identify different patterns
- PCA visualization shows cluster separation in 2D space
- Cluster profiling reveals distinct user behavior patterns

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

## Usage
```bash
python solution.py
```

## Output
- Elbow method plot showing optimal k
- Clustering comparison visualization
- Detailed cluster profiles with statistics
- Performance metrics for each algorithm

## Business Applications
- **Targeted Marketing**: Customize campaigns for each user segment
- **Product Development**: Prioritize features for high-value segments
- **Retention Strategies**: Identify and engage at-risk users
- **Resource Allocation**: Focus support on specific user groups
- **Personalization**: Tailor user experience based on segment

## Author
Kaggle Competition Solution - Clustering Analysis
