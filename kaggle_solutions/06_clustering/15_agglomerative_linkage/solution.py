"""
Agglomerative Clustering with Linkage Comparison
=================================================
Comprehensive analysis of agglomerative hierarchical clustering with different
linkage methods (single, complete, average, ward) and dendrogram visualization.

Author: Data Science Team
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AgglomerativeAnalyzer:
    """
    Comprehensive agglomerative clustering analyzer with multiple linkage methods.
    """

    def __init__(self, n_clusters=None, distance_threshold=None):
        """
        Initialize agglomerative clustering analyzer.

        Parameters:
        -----------
        n_clusters : int or None
            Number of clusters (if None, use distance_threshold)
        distance_threshold : float or None
            Distance threshold for clustering
        """
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage_methods = ['ward', 'complete', 'average', 'single']
        self.results = {}

    def fit_all_linkages(self, X):
        """
        Fit agglomerative clustering with all linkage methods.
        """
        for linkage_method in self.linkage_methods:
            # Ward linkage requires euclidean affinity
            affinity = 'euclidean' if linkage_method == 'ward' else 'euclidean'

            model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                distance_threshold=self.distance_threshold,
                linkage=linkage_method,
                affinity=affinity
            )

            start_time = time.time()
            labels = model.fit_predict(X)
            elapsed = time.time() - start_time

            # Compute metrics
            n_clusters_found = len(np.unique(labels))

            if n_clusters_found > 1:
                sil_score = silhouette_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
            else:
                sil_score = db_score = ch_score = 0.0

            self.results[linkage_method] = {
                'model': model,
                'labels': labels,
                'n_clusters': n_clusters_found,
                'time': elapsed,
                'silhouette': sil_score,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score
            }

        return self.results


def generate_hierarchical_datasets():
    """
    Generate datasets suitable for hierarchical clustering.
    """
    datasets = {}

    # Dataset 1: Well-separated blobs
    X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    datasets['blobs'] = (X1, y1, 4)

    # Dataset 2: Chain-like structure (good for single linkage)
    np.random.seed(42)
    X2 = []
    y2 = []
    for i in range(3):
        chain = np.column_stack([
            np.linspace(i*5, i*5+3, 100) + np.random.randn(100)*0.2,
            np.random.randn(100)*0.5
        ])
        X2.append(chain)
        y2.extend([i]*100)
    X2 = np.vstack(X2)
    y2 = np.array(y2)
    datasets['chain'] = (X2, y2, 3)

    # Dataset 3: Compact clusters (good for complete linkage)
    X3, y3 = make_blobs(n_samples=400, centers=5, cluster_std=0.5, random_state=43)
    datasets['compact'] = (X3, y3, 5)

    # Dataset 4: Mixed densities
    centers = [[0, 0], [5, 5], [-5, 3]]
    stds = [0.5, 1.5, 1.0]
    X4_parts = []
    y4_parts = []
    for i, (center, std) in enumerate(zip(centers, stds)):
        X_temp, _ = make_blobs(n_samples=150, centers=[center],
                              cluster_std=std, random_state=44+i)
        X4_parts.append(X_temp)
        y4_parts.extend([i]*150)
    X4 = np.vstack(X4_parts)
    y4 = np.array(y4_parts)
    datasets['mixed_density'] = (X4, y4, 3)

    # Dataset 5: Nested clusters
    X5_outer, _ = make_blobs(n_samples=200, centers=1, cluster_std=3.0,
                            center_box=(0, 0), random_state=45)
    X5_inner, _ = make_blobs(n_samples=100, centers=1, cluster_std=0.5,
                            center_box=(0, 0), random_state=46)
    X5 = np.vstack([X5_outer, X5_inner])
    y5 = np.hstack([np.zeros(200), np.ones(100)])
    datasets['nested'] = (X5, y5, 2)

    return datasets


def plot_dendrograms(X, linkage_methods=['ward', 'complete', 'average', 'single'],
                     max_d=None):
    """
    Plot dendrograms for different linkage methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, method in enumerate(linkage_methods):
        ax = axes[idx]

        # Compute linkage
        Z = linkage(X, method=method)

        # Plot dendrogram
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=max_d)

        if max_d:
            ax.axhline(y=max_d, c='red', linestyle='--', linewidth=2,
                      label=f'Threshold={max_d:.2f}')
            ax.legend()

        ax.set_title(f'{method.capitalize()} Linkage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_clustering_comparison(X, results, title="Linkage Method Comparison"):
    """
    Visualize clustering results for all linkage methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, (method, result) in enumerate(results.items()):
        ax = axes[idx]

        scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                           cmap='viridis', alpha=0.7, s=50)

        ax.set_title(f'{method.capitalize()} Linkage\n'
                    f'k={result["n_clusters"]}, '
                    f'Silhouette: {result["silhouette"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_performance_metrics(results):
    """
    Plot performance metrics comparison across linkage methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = list(results.keys())
    silhouettes = [results[m]['silhouette'] for m in methods]
    db_scores = [results[m]['davies_bouldin'] for m in methods]
    ch_scores = [results[m]['calinski_harabasz'] for m in methods]
    times = [results[m]['time'] for m in methods]

    # Plot 1: Silhouette scores
    ax = axes[0, 0]
    bars = ax.bar(methods, silhouettes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Silhouette Score (Higher is Better)', fontweight='bold')
    ax.set_ylabel('Silhouette Score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')

    # Plot 2: Davies-Bouldin Index
    ax = axes[0, 1]
    bars = ax.bar(methods, db_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
    ax.set_ylabel('DB Index')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')

    # Plot 3: Calinski-Harabasz Score
    ax = axes[1, 0]
    bars = ax.bar(methods, ch_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontweight='bold')
    ax.set_ylabel('CH Score')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom')

    # Plot 4: Execution time
    ax = axes[1, 1]
    bars = ax.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_title('Execution Time (Lower is Better)', fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig


def analyze_distance_threshold(X, linkage_method='ward', threshold_range=None):
    """
    Analyze effect of distance threshold on number of clusters.
    """
    if threshold_range is None:
        # Compute linkage to determine reasonable threshold range
        Z = linkage(X, method=linkage_method)
        max_dist = Z[:, 2].max()
        threshold_range = np.linspace(0.1, max_dist, 20)

    results = []

    for threshold in threshold_range:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage=linkage_method
        )
        labels = model.fit_predict(X)
        n_clusters = len(np.unique(labels))

        if n_clusters > 1:
            sil = silhouette_score(X, labels)
        else:
            sil = 0.0

        results.append({
            'threshold': threshold,
            'n_clusters': n_clusters,
            'silhouette': sil
        })

    return pd.DataFrame(results)


def plot_threshold_analysis(threshold_df):
    """
    Plot threshold analysis results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Number of clusters vs threshold
    ax = axes[0]
    ax.plot(threshold_df['threshold'], threshold_df['n_clusters'],
           marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Distance Threshold', fontsize=11)
    ax.set_ylabel('Number of Clusters', fontsize=11)
    ax.set_title('Clusters vs Distance Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Silhouette score vs threshold
    ax = axes[1]
    ax.plot(threshold_df['threshold'], threshold_df['silhouette'],
           marker='s', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Distance Threshold', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Score vs Distance Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Mark best threshold
    best_idx = threshold_df['silhouette'].idxmax()
    best_threshold = threshold_df.loc[best_idx, 'threshold']
    best_sil = threshold_df.loc[best_idx, 'silhouette']
    ax.plot(best_threshold, best_sil, 'r*', markersize=20,
           label=f'Best (t={best_threshold:.2f})')
    ax.legend()

    plt.tight_layout()
    return fig


def cophenetic_correlation(X, linkage_method='ward'):
    """
    Compute cophenetic correlation coefficient to measure how well
    the dendrogram preserves pairwise distances.
    """
    from scipy.cluster.hierarchy import cophenet

    Z = linkage(X, method=linkage_method)
    orig_dists = pdist(X)
    cophen_dists = cophenet(Z)[0]

    correlation = np.corrcoef(orig_dists, cophenet(Z)[1])[0, 1]

    return correlation, Z


def plot_cophenetic_comparison(X):
    """
    Compare cophenetic correlations across linkage methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['ward', 'complete', 'average', 'single']
    correlations = []

    for method in methods:
        corr, _ = cophenetic_correlation(X, linkage_method=method)
        correlations.append(corr)

    bars = ax.bar(methods, correlations, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Cophenetic Correlation Coefficient', fontsize=11)
    ax.set_title('Cophenetic Correlation by Linkage Method\n(Higher = Better Preservation of Distances)',
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("AGGLOMERATIVE CLUSTERING WITH LINKAGE COMPARISON")
    print("="*80)

    # Generate datasets
    print("\n1. Generating synthetic datasets...")
    datasets = generate_hierarchical_datasets()
    print(f"   Generated {len(datasets)} datasets")

    # Standardize
    scaler = StandardScaler()
    for name in datasets:
        X, y, k = datasets[name]
        datasets[name] = (scaler.fit_transform(X), y, k)

    # Analyze each dataset
    for ds_name, (X, y_true, true_k) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name.upper()} (k={true_k})")
        print(f"{'='*80}")

        # Fit all linkage methods
        print(f"\n2. Fitting agglomerative clustering with all linkage methods...")
        analyzer = AgglomerativeAnalyzer(n_clusters=true_k)
        results = analyzer.fit_all_linkages(X)

        print(f"\n   Results:")
        for method, result in results.items():
            print(f"\n   {method.upper()}:")
            print(f"   - Silhouette: {result['silhouette']:.4f}")
            print(f"   - Davies-Bouldin: {result['davies_bouldin']:.4f}")
            print(f"   - Calinski-Harabasz: {result['calinski_harabasz']:.2f}")
            print(f"   - Time: {result['time']:.4f}s")

        # Cophenetic correlation
        print(f"\n3. Computing cophenetic correlations...")
        for method in ['ward', 'complete', 'average', 'single']:
            corr, _ = cophenetic_correlation(X, linkage_method=method)
            print(f"   {method.capitalize()}: {corr:.4f}")

        # Visualizations
        print(f"\n4. Creating visualizations...")

        # Dendrograms
        fig1 = plot_dendrograms(X)
        plt.savefig(f'/tmp/agglomerative_{ds_name}_dendrograms.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Clustering comparison
        fig2 = plot_clustering_comparison(X, results, title=f"{ds_name.capitalize()} Dataset")
        plt.savefig(f'/tmp/agglomerative_{ds_name}_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Performance metrics
        fig3 = plot_performance_metrics(results)
        plt.savefig(f'/tmp/agglomerative_{ds_name}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   - Saved visualization plots")

    # Additional analyses on blobs dataset
    print(f"\n{'='*80}")
    print("ADDITIONAL ANALYSES (Blobs Dataset)")
    print(f"{'='*80}")

    X, _, _ = datasets['blobs']

    # Distance threshold analysis
    print("\n5. Analyzing distance threshold effects...")
    threshold_df = analyze_distance_threshold(X, linkage_method='ward')
    print(f"\n   Best threshold: {threshold_df.loc[threshold_df['silhouette'].idxmax(), 'threshold']:.3f}")
    print(f"   Resulting clusters: {threshold_df.loc[threshold_df['silhouette'].idxmax(), 'n_clusters']}")

    fig4 = plot_threshold_analysis(threshold_df)
    plt.savefig('/tmp/agglomerative_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cophenetic comparison
    print("\n6. Creating cophenetic correlation comparison...")
    fig5 = plot_cophenetic_comparison(X)
    plt.savefig('/tmp/agglomerative_cophenetic.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Ward linkage minimizes within-cluster variance (good for compact clusters)")
    print("2. Single linkage can find elongated/chain-like clusters")
    print("3. Complete linkage is sensitive to outliers but finds compact clusters")
    print("4. Average linkage provides a compromise between single and complete")
    print("5. Cophenetic correlation measures dendrogram quality")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
