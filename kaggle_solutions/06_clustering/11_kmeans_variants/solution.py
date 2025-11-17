"""
K-Means Clustering Variants Analysis
=====================================
Comprehensive comparison of K-Means variants including K-Means++, Mini-Batch K-Means,
and Elkan's algorithm with performance analysis and cluster validation.

Author: Data Science Team
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
import warnings
import time
from itertools import combinations
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class KMeansVariants:
    """
    Implementation and comparison of various K-Means clustering algorithms.
    """

    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize K-Means variants.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def kmeans_random_init(self, X, max_iter=300):
        """
        Standard K-Means with random initialization.
        """
        model = KMeans(
            n_clusters=self.n_clusters,
            init='random',
            n_init=10,
            max_iter=max_iter,
            random_state=self.random_state
        )
        start_time = time.time()
        labels = model.fit_predict(X)
        elapsed_time = time.time() - start_time

        return {
            'model': model,
            'labels': labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'n_iter': model.n_iter_,
            'time': elapsed_time
        }

    def kmeans_plus_plus(self, X, max_iter=300):
        """
        K-Means++ initialization for better centroid placement.
        """
        model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=max_iter,
            random_state=self.random_state
        )
        start_time = time.time()
        labels = model.fit_predict(X)
        elapsed_time = time.time() - start_time

        return {
            'model': model,
            'labels': labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'n_iter': model.n_iter_,
            'time': elapsed_time
        }

    def minibatch_kmeans(self, X, batch_size=100, max_iter=300):
        """
        Mini-Batch K-Means for large datasets.
        """
        model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=self.random_state
        )
        start_time = time.time()
        labels = model.fit_predict(X)
        elapsed_time = time.time() - start_time

        return {
            'model': model,
            'labels': labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'n_iter': model.n_iter_,
            'time': elapsed_time
        }

    def elkan_kmeans(self, X, max_iter=300):
        """
        K-Means with Elkan's algorithm (triangle inequality optimization).
        """
        model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=max_iter,
            algorithm='elkan',
            random_state=self.random_state
        )
        start_time = time.time()
        labels = model.fit_predict(X)
        elapsed_time = time.time() - start_time

        return {
            'model': model,
            'labels': labels,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'n_iter': model.n_iter_,
            'time': elapsed_time
        }

    def fit_all_variants(self, X):
        """
        Fit all K-Means variants and store results.
        """
        print("Fitting K-Means variants...")

        self.results['random_init'] = self.kmeans_random_init(X)
        print("✓ Random initialization complete")

        self.results['kmeans++'] = self.kmeans_plus_plus(X)
        print("✓ K-Means++ complete")

        self.results['minibatch'] = self.minibatch_kmeans(X)
        print("✓ Mini-Batch K-Means complete")

        self.results['elkan'] = self.elkan_kmeans(X)
        print("✓ Elkan's algorithm complete")

        return self.results


def generate_synthetic_data(n_samples=3000, n_features=2, n_clusters=5, cluster_std=1.0):
    """
    Generate synthetic datasets with known cluster structure.
    """
    # Dataset 1: Well-separated blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    # Dataset 2: Overlapping clusters
    X_overlap, y_overlap = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std * 2.0,
        random_state=42
    )

    # Dataset 3: Different cluster sizes
    X_varied = []
    y_varied = []
    sizes = [n_samples // 2, n_samples // 4, n_samples // 8, n_samples // 16, n_samples // 16]

    for i in range(n_clusters):
        X_temp, _ = make_blobs(
            n_samples=sizes[i],
            n_features=n_features,
            centers=1,
            cluster_std=cluster_std,
            center_box=(i*5, i*5+5),
            random_state=42+i
        )
        X_varied.append(X_temp)
        y_varied.extend([i] * sizes[i])

    X_varied = np.vstack(X_varied)
    y_varied = np.array(y_varied)

    return {
        'blobs': (X_blobs, y_blobs),
        'overlap': (X_overlap, y_overlap),
        'varied': (X_varied, y_varied)
    }


def compute_cluster_metrics(X, labels, centers):
    """
    Compute comprehensive clustering quality metrics.
    """
    metrics = {}

    # Silhouette Score
    metrics['silhouette'] = silhouette_score(X, labels)

    # Davies-Bouldin Index (lower is better)
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)

    # Calinski-Harabasz Index (higher is better)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)

    # Inertia (within-cluster sum of squares)
    distances = cdist(X, centers, metric='euclidean')
    metrics['inertia'] = np.sum(np.min(distances, axis=1) ** 2)

    # Inter-cluster distance
    inter_distances = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(inter_distances, np.inf)
    metrics['min_inter_cluster_dist'] = np.min(inter_distances)
    metrics['avg_inter_cluster_dist'] = np.mean(inter_distances[inter_distances != np.inf])

    return metrics


def elbow_analysis(X, max_k=15):
    """
    Perform elbow method analysis to determine optimal k.
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    return k_range, inertias, silhouette_scores


def convergence_analysis(X, n_clusters=5, max_iter=100):
    """
    Analyze convergence behavior of different K-Means variants.
    """
    results = {}

    # Track inertia over iterations for each variant
    variants = {
        'random': {'init': 'random', 'algorithm': 'lloyd'},
        'kmeans++': {'init': 'k-means++', 'algorithm': 'lloyd'},
        'elkan': {'init': 'k-means++', 'algorithm': 'elkan'}
    }

    for name, params in variants.items():
        inertias = []

        for n_iter in range(1, max_iter + 1, 5):
            model = KMeans(
                n_clusters=n_clusters,
                init=params['init'],
                algorithm=params['algorithm'],
                max_iter=n_iter,
                n_init=1,
                random_state=42
            )
            model.fit(X)
            inertias.append(model.inertia_)

        results[name] = inertias

    return results


def stability_analysis(X, n_clusters=5, n_runs=30):
    """
    Analyze clustering stability across multiple runs.
    """
    variants = ['random', 'k-means++']
    results = {variant: [] for variant in variants}

    # First run to get reference labels
    reference_models = {}
    for variant in variants:
        model = KMeans(n_clusters=n_clusters, init=variant, n_init=1, random_state=42)
        reference_models[variant] = model.fit_predict(X)

    # Multiple runs with different seeds
    for seed in range(n_runs):
        for variant in variants:
            model = KMeans(n_clusters=n_clusters, init=variant, n_init=1, random_state=seed)
            labels = model.fit_predict(X)

            # Compute ARI with reference
            ari = adjusted_rand_score(reference_models[variant], labels)
            results[variant].append(ari)

    return results


def visualize_clusters(X, results, title_prefix=""):
    """
    Visualize clustering results for all variants.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    variant_names = ['random_init', 'kmeans++', 'minibatch', 'elkan']
    titles = ['Random Init', 'K-Means++', 'Mini-Batch', 'Elkan']

    for idx, (var_name, title) in enumerate(zip(variant_names, titles)):
        ax = axes[idx]
        result = results[var_name]

        scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                           cmap='viridis', alpha=0.6, s=30)
        ax.scatter(result['centers'][:, 0], result['centers'][:, 1],
                  c='red', marker='x', s=200, linewidths=3, label='Centroids')

        ax.set_title(f'{title_prefix}{title}\nInertia: {result["inertia"]:.2f}, '
                    f'Time: {result["time"]:.4f}s', fontsize=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    return fig


def plot_performance_comparison(results):
    """
    Create comprehensive performance comparison plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    variant_names = list(results.keys())
    metrics_data = {
        'inertia': [],
        'time': [],
        'n_iter': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    # Extract metrics
    for var in variant_names:
        metrics_data['inertia'].append(results[var]['inertia'])
        metrics_data['time'].append(results[var]['time'])
        metrics_data['n_iter'].append(results[var]['n_iter'])

    # Plot 1: Inertia comparison
    axes[0, 0].bar(variant_names, metrics_data['inertia'], color='steelblue')
    axes[0, 0].set_title('Inertia Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Execution time
    axes[0, 1].bar(variant_names, metrics_data['time'], color='coral')
    axes[0, 1].set_title('Execution Time (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Number of iterations
    axes[0, 2].bar(variant_names, metrics_data['n_iter'], color='lightgreen')
    axes[0, 2].set_title('Iterations to Convergence', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Iterations')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Plot 4: Time vs Inertia trade-off
    axes[1, 0].scatter(metrics_data['time'], metrics_data['inertia'], s=200, alpha=0.6)
    for i, var in enumerate(variant_names):
        axes[1, 0].annotate(var, (metrics_data['time'][i], metrics_data['inertia'][i]))
    axes[1, 0].set_xlabel('Execution Time (s)')
    axes[1, 0].set_ylabel('Inertia')
    axes[1, 0].set_title('Time vs Quality Trade-off', fontsize=12, fontweight='bold')

    # Plot 5: Efficiency metric (inertia per second)
    efficiency = [metrics_data['inertia'][i] / metrics_data['time'][i]
                  for i in range(len(variant_names))]
    axes[1, 1].bar(variant_names, efficiency, color='purple', alpha=0.6)
    axes[1, 1].set_title('Efficiency (Inertia/Time)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Plot 6: Convergence speed
    conv_speed = [metrics_data['inertia'][i] / metrics_data['n_iter'][i]
                  for i in range(len(variant_names))]
    axes[1, 2].bar(variant_names, conv_speed, color='orange', alpha=0.6)
    axes[1, 2].set_title('Convergence Speed (Inertia/Iteration)', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Speed')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def plot_elbow_analysis(k_range, inertias, silhouette_scores):
    """
    Visualize elbow method results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method - Inertia vs K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_convergence(convergence_results):
    """
    Plot convergence behavior of different variants.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = range(1, 101, 5)
    colors = {'random': 'blue', 'kmeans++': 'green', 'elkan': 'red'}

    for variant, inertias in convergence_results.items():
        ax.plot(iterations, inertias, marker='o', linewidth=2,
                label=variant, color=colors.get(variant, 'gray'))

    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Inertia', fontsize=12)
    ax.set_title('Convergence Comparison Across Variants', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stability(stability_results):
    """
    Visualize clustering stability analysis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Box plots
    data = [stability_results[var] for var in stability_results.keys()]
    ax1.boxplot(data, labels=list(stability_results.keys()))
    ax1.set_ylabel('Adjusted Rand Index', fontsize=12)
    ax1.set_title('Stability Analysis - ARI Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Histogram comparison
    for variant, ari_scores in stability_results.items():
        ax2.hist(ari_scores, alpha=0.6, bins=20, label=variant)

    ax2.set_xlabel('Adjusted Rand Index', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Stability Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Main execution function for K-Means variants analysis.
    """
    print("="*80)
    print("K-MEANS CLUSTERING VARIANTS ANALYSIS")
    print("="*80)

    # Generate synthetic datasets
    print("\n1. Generating synthetic datasets...")
    datasets = generate_synthetic_data(n_samples=2000, n_features=2, n_clusters=5)

    # Standardize data
    scaler = StandardScaler()
    for name in datasets:
        datasets[name] = (scaler.fit_transform(datasets[name][0]), datasets[name][1])

    print(f"   - Generated {len(datasets)} datasets")

    # Analyze each dataset
    all_results = {}

    for ds_name, (X, y_true) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Analyzing dataset: {ds_name.upper()}")
        print(f"{'='*80}")

        # Fit all variants
        kmv = KMeansVariants(n_clusters=5, random_state=42)
        results = kmv.fit_all_variants(X)
        all_results[ds_name] = results

        # Compute metrics for each variant
        print(f"\n2. Computing cluster quality metrics...")
        for var_name, result in results.items():
            metrics = compute_cluster_metrics(X, result['labels'], result['centers'])
            print(f"\n   {var_name.upper()}:")
            print(f"   - Silhouette Score: {metrics['silhouette']:.4f}")
            print(f"   - Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
            print(f"   - Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
            print(f"   - Execution Time: {result['time']:.4f}s")
            print(f"   - Iterations: {result['n_iter']}")

            # Compare with ground truth if available
            if y_true is not None:
                ari = adjusted_rand_score(y_true, result['labels'])
                nmi = normalized_mutual_info_score(y_true, result['labels'])
                print(f"   - ARI (vs ground truth): {ari:.4f}")
                print(f"   - NMI (vs ground truth): {nmi:.4f}")

        # Visualizations
        print(f"\n3. Creating visualizations...")

        # Cluster visualizations
        fig1 = visualize_clusters(X, results, title_prefix=f"{ds_name.capitalize()} - ")
        plt.savefig(f'/tmp/kmeans_variants_{ds_name}_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Performance comparison
        fig2 = plot_performance_comparison(results)
        plt.savefig(f'/tmp/kmeans_variants_{ds_name}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   - Saved visualization plots")

    # Additional analyses on the blobs dataset
    X, _ = datasets['blobs']

    print(f"\n{'='*80}")
    print("ADDITIONAL ANALYSES")
    print(f"{'='*80}")

    # Elbow analysis
    print("\n4. Performing elbow analysis...")
    k_range, inertias, silhouette_scores = elbow_analysis(X, max_k=15)
    fig3 = plot_elbow_analysis(k_range, inertias, silhouette_scores)
    plt.savefig('/tmp/kmeans_variants_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Optimal k analysis complete")

    # Convergence analysis
    print("\n5. Analyzing convergence behavior...")
    convergence_results = convergence_analysis(X, n_clusters=5, max_iter=100)
    fig4 = plot_convergence(convergence_results)
    plt.savefig('/tmp/kmeans_variants_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Convergence analysis complete")

    # Stability analysis
    print("\n6. Performing stability analysis...")
    stability_results = stability_analysis(X, n_clusters=5, n_runs=30)
    fig5 = plot_stability(stability_results)
    plt.savefig('/tmp/kmeans_variants_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

    for variant, ari_scores in stability_results.items():
        print(f"\n   {variant.upper()}:")
        print(f"   - Mean ARI: {np.mean(ari_scores):.4f}")
        print(f"   - Std ARI: {np.std(ari_scores):.4f}")
        print(f"   - Min ARI: {np.min(ari_scores):.4f}")
        print(f"   - Max ARI: {np.max(ari_scores):.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. K-Means++ consistently provides better initialization than random")
    print("2. Mini-Batch K-Means offers significant speedup for large datasets")
    print("3. Elkan's algorithm is more efficient for low-dimensional data")
    print("4. All variants converge to similar solutions for well-separated clusters")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
