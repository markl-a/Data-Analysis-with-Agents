# Gene Expression Clustering Analysis

## Overview
This solution demonstrates clustering of genes based on their expression patterns across different biological conditions, a fundamental technique in bioinformatics and genomics research.

## Problem Statement
Analyzing gene expression data helps identify genes with similar functions, discover disease biomarkers, and understand biological processes. Clustering groups genes with similar expression patterns across conditions.

## Dataset
Synthetic gene expression data simulating:
- **200 genes** × **50 samples**
- 5 sample groups: Control, Disease A, Disease B, Treatment High Response, Treatment Low Response

## Gene Expression Patterns
1. **Disease A Markers**: Upregulated in Disease A samples
2. **Disease B Markers**: Upregulated in Disease B samples
3. **Treatment Response Genes**: High expression in treatment responders
4. **Housekeeping Genes**: Stable expression across all conditions
5. **Suppressed Genes**: Downregulated in disease states

## Clustering Algorithms
1. **K-Means**: Partitions genes into k clusters based on expression similarity
2. **Hierarchical (Ward)**: Creates tree structure showing gene relationships
3. **Spectral Clustering**: Uses graph theory for non-linear cluster shapes

## Key Visualizations
- **Heatmap**: Expression levels across all genes and samples
- **PCA Plot**: 2D projection showing cluster separation
- **Dendrogram**: Hierarchical tree structure of gene relationships
- **Expression Profiles**: Average patterns for each cluster

## Evaluation Metrics
- **Silhouette Score**: Measures how well genes fit their assigned cluster
- **Davies-Bouldin Index**: Average similarity ratio (lower is better)
- **Calinski-Harabasz Score**: Ratio of between to within cluster dispersion

## Analysis Steps
1. Generate realistic gene expression matrix (200 genes × 50 samples)
2. Standardize expression values (z-score normalization)
3. Apply three clustering algorithms (K-Means, Hierarchical, Spectral)
4. Visualize clusters using PCA and heatmaps
5. Create hierarchical dendrogram showing gene relationships
6. Analyze expression patterns for each cluster

## Technical Details
- **Standardization**: Z-score normalization ensures fair comparison
- **Ward Linkage**: Minimizes variance when merging clusters
- **PCA**: Reduces dimensionality while preserving variance
- **Distance Metric**: Euclidean distance for expression similarity

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
1. Expression heatmap with hierarchical clustering
2. PCA visualization of gene clusters
3. Dendrogram showing gene relationships
4. Cluster-wise expression profiles
5. Statistical analysis per cluster
6. Top variable genes in each cluster

## Biological Interpretation
- **Co-expressed genes** often share functions
- **Disease markers** show distinct patterns
- **Housekeeping genes** cluster together (stable expression)
- **Treatment response** genes identify therapeutic targets

## Applications
- **Disease Biomarker Discovery**: Identify diagnostic markers
- **Drug Target Identification**: Find genes for therapeutic intervention
- **Pathway Analysis**: Group genes by biological function
- **Patient Stratification**: Classify patients by expression profiles
- **Functional Annotation**: Predict gene function from expression

## Research Context
Gene expression clustering is used in:
- Cancer subtype classification
- Drug response prediction
- Developmental biology studies
- Comparative genomics
- Personalized medicine

## Author
Kaggle Competition Solution - Bioinformatics Clustering
