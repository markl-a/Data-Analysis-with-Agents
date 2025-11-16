# Document Clustering Analysis

## Overview
This solution demonstrates text clustering using TF-IDF vectorization and multiple clustering algorithms to group similar documents by topic.

## Problem Statement
Organizing large collections of text documents is challenging. This analysis automatically groups documents into topics, enabling better organization, search, and content discovery.

## Dataset
Synthetic documents generated from 5 distinct topics:
- **Technology**: Software, programming, AI, databases
- **Sports**: Football, basketball, athletes, tournaments
- **Health**: Medicine, treatment, healthcare, wellness
- **Finance**: Investment, stocks, banking, markets
- **Science**: Research, experiments, discoveries, theories

## Text Processing
- **TF-IDF Vectorization**: Converts text to numerical features
  - Term Frequency: How often a word appears in a document
  - Inverse Document Frequency: Rarity of word across all documents
- **Stop Words Removal**: Filters common English words
- **Feature Extraction**: Creates 100 most important features

## Clustering Algorithms
1. **K-Means**: Partitions documents into k clusters based on centroid distance
2. **Agglomerative Clustering**: Hierarchical clustering building clusters bottom-up

## Visualization
- **t-SNE**: Projects high-dimensional TF-IDF vectors to 2D for visualization
- Better than PCA for visualizing document clusters
- Preserves local structure and reveals cluster separation

## Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Average similarity ratio of clusters (lower is better)
- **Cluster Size Distribution**: Balance across clusters

## Analysis Steps
1. Generate 300 synthetic documents across 5 topics
2. Create TF-IDF feature matrix (100 features)
3. Determine optimal cluster count using elbow and silhouette methods
4. Apply K-Means and Agglomerative clustering
5. Visualize clusters using t-SNE dimensionality reduction
6. Extract and analyze top terms for each cluster

## Key Features
- Topic extraction from clusters using TF-IDF weights
- Sample document display for each cluster
- Comparative analysis of clustering algorithms
- Visual cluster separation analysis

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage
```bash
python solution.py
```

## Output
- Optimal cluster count visualization
- t-SNE cluster visualization for each algorithm
- Top terms for each discovered cluster
- Sample documents from each cluster
- Performance metrics comparison

## Applications
- **Document Organization**: Automatically categorize articles
- **Content Discovery**: Find similar documents
- **Topic Modeling**: Extract main themes from corpus
- **Search Enhancement**: Improve relevance by cluster
- **Archive Management**: Organize large document collections

## Technical Notes
- TF-IDF captures term importance across documents
- t-SNE provides better visualization than PCA for text
- Silhouette score helps validate cluster quality
- Top terms reveal cluster topics and themes

## Author
Kaggle Competition Solution - Text Clustering
