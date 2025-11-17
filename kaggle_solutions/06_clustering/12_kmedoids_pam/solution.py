"""
K-Medoids and PAM Clustering Analysis
======================================
Comprehensive implementation of K-Medoids (Partitioning Around Medoids) clustering
with comparison to K-Means and various distance metrics analysis.

Author: Data Science Team
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class KMedoids:
    """
    K-Medoids (PAM - Partitioning Around Medoids) clustering implementation.
    More robust to outliers than K-Means as it uses actual data points as centers.
    """

    def __init__(self, n_clusters=3, max_iter=300, metric='euclidean', random_state=42):
        """
        Initialize K-Medoids clustering.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
        metric : str
            Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _initialize_medoids(self, X):
        """
        Initialize medoids using k-means++ style initialization.
        """
        n_samples = X.shape[0]
        np.random.seed(self.random_state)

        # First medoid: random point
        medoid_indices = [np.random.randint(n_samples)]

        # Remaining medoids: select points far from existing medoids
        for _ in range(1, self.n_clusters):
            distances = pairwise_distances(X, X[medoid_indices], metric=self.metric)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / min_distances.sum()
            next_medoid = np.random.choice(n_samples, p=probabilities)
            medoid_indices.append(next_medoid)

        return np.array(medoid_indices)

    def _assign_clusters(self, X, medoid_indices):
        """
        Assign each point to the nearest medoid.
        """
        distances = pairwise_distances(X, X[medoid_indices], metric=self.metric)
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_cost(self, X, medoid_indices, labels):
        """
        Calculate total cost (sum of distances to medoids).
        """
        cost = 0
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                distances = pairwise_distances(
                    X[cluster_mask],
                    X[medoid_indices[i]].reshape(1, -1),
                    metric=self.metric
                )
                cost += np.sum(distances)
        return cost

    def _update_medoids(self, X, labels):
        """
        Update medoids by finding the point that minimizes total distance within each cluster.
        """
        new_medoid_indices = np.zeros(self.n_clusters, dtype=int)

        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_points = X[cluster_mask]

            if len(cluster_points) == 0:
                # Empty cluster: keep old medoid or select random point
                new_medoid_indices[i] = self.medoid_indices_[i] if self.medoid_indices_ is not None else np.random.randint(len(X))
                continue

            # Find point that minimizes sum of distances within cluster
            distances = pairwise_distances(cluster_points, cluster_points, metric=self.metric)
            costs = np.sum(distances, axis=1)
            best_idx = np.argmin(costs)

            # Get the global index
            global_indices = np.where(cluster_mask)[0]
            new_medoid_indices[i] = global_indices[best_idx]

        return new_medoid_indices

    def fit(self, X):
        """
        Fit K-Medoids clustering.
        """
        # Initialize medoids
        self.medoid_indices_ = self._initialize_medoids(X)

        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, self.medoid_indices_)

            # Update medoids
            new_medoid_indices = self._update_medoids(X, labels)

            # Check convergence
            if np.array_equal(new_medoid_indices, self.medoid_indices_):
                self.n_iter_ = iteration + 1
                break

            self.medoid_indices_ = new_medoid_indices

        # Final assignment
        self.labels_ = self._assign_clusters(X, self.medoid_indices_)
        self.inertia_ = self._calculate_cost(X, self.medoid_indices_, self.labels_)

        return self

    def fit_predict(self, X):
        """
        Fit and return cluster labels.
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data.
        """
        if self.medoid_indices_ is None:
            raise ValueError("Model has not been fitted yet.")

        return self._assign_clusters(X, self.medoid_indices_)


def generate_datasets():
    """
    Generate various synthetic datasets for testing.
    """
    datasets = {}

    # Dataset 1: Well-separated blobs
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, n_features=2,
                                   cluster_std=1.0, random_state=42)
    datasets['blobs'] = (X_blobs, y_blobs)

    # Dataset 2: Blobs with outliers
    X_outliers, y_outliers = make_blobs(n_samples=450, centers=4, n_features=2,
                                         cluster_std=1.0, random_state=42)
    # Add outliers
    outliers = np.random.uniform(low=X_outliers.min()-3, high=X_outliers.max()+3, size=(50, 2))
    X_outliers = np.vstack([X_outliers, outliers])
    y_outliers = np.hstack([y_outliers, np.full(50, -1)])
    datasets['outliers'] = (X_outliers, y_outliers)

    # Dataset 3: Non-convex shapes (moons)
    X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)
    datasets['moons'] = (X_moons, y_moons)

    # Dataset 4: Concentric circles
    X_circles, y_circles = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
    datasets['circles'] = (X_circles, y_circles)

    # Dataset 5: Varied density
    X1, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, center_box=(0, 0), random_state=42)
    X2, _ = make_blobs(n_samples=100, centers=1, cluster_std=1.5, center_box=(5, 5), random_state=43)
    X3, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.8, center_box=(-4, 4), random_state=44)
    X_density = np.vstack([X1, X2, X3])
    y_density = np.hstack([np.zeros(200), np.ones(100), np.full(300, 2)])
    datasets['density'] = (X_density, y_density)

    return datasets


def compare_metrics(X, n_clusters=4):
    """
    Compare K-Medoids with different distance metrics.
    """
    metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
    results = {}

    for metric in metrics:
        kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=42)
        start_time = time.time()
        labels = kmedoids.fit_predict(X)
        elapsed = time.time() - start_time

        results[metric] = {
            'labels': labels,
            'medoids': kmedoids.medoid_indices_,
            'inertia': kmedoids.inertia_,
            'time': elapsed,
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)
        }

    return results


def compare_kmedoids_kmeans(X, n_clusters=4):
    """
    Compare K-Medoids with K-Means.
    """
    results = {}

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    start_time = time.time()
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_time = time.time() - start_time

    results['kmeans'] = {
        'labels': kmeans_labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'time': kmeans_time,
        'silhouette': silhouette_score(X, kmeans_labels),
        'davies_bouldin': davies_bouldin_score(X, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels)
    }

    # K-Medoids
    kmedoids = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
    start_time = time.time()
    kmedoids_labels = kmedoids.fit_predict(X)
    kmedoids_time = time.time() - start_time

    results['kmedoids'] = {
        'labels': kmedoids_labels,
        'medoids': kmedoids.medoid_indices_,
        'medoid_points': X[kmedoids.medoid_indices_],
        'inertia': kmedoids.inertia_,
        'time': kmedoids_time,
        'silhouette': silhouette_score(X, kmedoids_labels),
        'davies_bouldin': davies_bouldin_score(X, kmedoids_labels),
        'calinski_harabasz': calinski_harabasz_score(X, kmedoids_labels)
    }

    return results


def plot_comparison(X, results, title="K-Medoids vs K-Means"):
    """
    Visualize K-Medoids vs K-Means comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # K-Means
    ax = axes[0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=results['kmeans']['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(results['kmeans']['centers'][:, 0],
              results['kmeans']['centers'][:, 1],
              c='red', marker='x', s=300, linewidths=3, label='Centroids')
    ax.set_title(f'K-Means\nSilhouette: {results["kmeans"]["silhouette"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

    # K-Medoids
    ax = axes[1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=results['kmedoids']['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(results['kmedoids']['medoid_points'][:, 0],
              results['kmedoids']['medoid_points'][:, 1],
              c='red', marker='*', s=500, edgecolors='black',
              linewidths=2, label='Medoids')
    ax.set_title(f'K-Medoids\nSilhouette: {results["kmedoids"]["silhouette"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_metric_comparison(results):
    """
    Plot comparison of different distance metrics for K-Medoids.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    metrics = list(results.keys())
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.scatter(X[:, 0], X[:, 1], c=results[metric]['labels'],
                  cmap='viridis', alpha=0.6, s=50)
        medoids = X[results[metric]['medoids']]
        ax.scatter(medoids[:, 0], medoids[:, 1],
                  c='red', marker='*', s=500, edgecolors='black',
                  linewidths=2, label='Medoids')
        ax.set_title(f'{metric.capitalize()} Distance\n'
                    f'Silhouette: {results[metric]["silhouette"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()

    plt.tight_layout()
    return fig


def plot_performance_metrics(comparison_results, metric_results):
    """
    Create comprehensive performance comparison plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Extract data
    methods = ['K-Means', 'K-Medoids']
    silhouette_scores = [comparison_results['kmeans']['silhouette'],
                        comparison_results['kmedoids']['silhouette']]
    db_scores = [comparison_results['kmeans']['davies_bouldin'],
                comparison_results['kmedoids']['davies_bouldin']]
    ch_scores = [comparison_results['kmeans']['calinski_harabasz'],
                comparison_results['kmedoids']['calinski_harabasz']]
    times = [comparison_results['kmeans']['time'],
            comparison_results['kmedoids']['time']]

    # Plot 1: Silhouette Score
    axes[0, 0].bar(methods, silhouette_scores, color=['steelblue', 'coral'])
    axes[0, 0].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Davies-Bouldin Index
    axes[0, 1].bar(methods, db_scores, color=['steelblue', 'coral'])
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Calinski-Harabasz Score
    axes[0, 2].bar(methods, ch_scores, color=['steelblue', 'coral'])
    axes[0, 2].set_title('Calinski-Harabasz Score (Higher is Better)', fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Execution Time
    axes[1, 0].bar(methods, times, color=['steelblue', 'coral'])
    axes[1, 0].set_title('Execution Time (Lower is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Metric comparison for K-Medoids
    metric_names = list(metric_results.keys())
    metric_silhouettes = [metric_results[m]['silhouette'] for m in metric_names]
    axes[1, 1].bar(metric_names, metric_silhouettes, color='purple', alpha=0.7)
    axes[1, 1].set_title('K-Medoids: Distance Metric Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Silhouette Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Time comparison across metrics
    metric_times = [metric_results[m]['time'] for m in metric_names]
    axes[1, 2].bar(metric_names, metric_times, color='green', alpha=0.7)
    axes[1, 2].set_title('Execution Time by Distance Metric', fontweight='bold')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def robustness_analysis(base_data, n_outliers_range=[0, 10, 20, 50, 100]):
    """
    Analyze robustness to outliers.
    """
    results = {'kmeans': [], 'kmedoids': []}

    for n_outliers in n_outliers_range:
        # Add outliers
        if n_outliers > 0:
            outliers = np.random.uniform(
                low=base_data.min()-5,
                high=base_data.max()+5,
                size=(n_outliers, 2)
            )
            X = np.vstack([base_data, outliers])
        else:
            X = base_data.copy()

        # K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_sil = silhouette_score(X, kmeans_labels)
        results['kmeans'].append(kmeans_sil)

        # K-Medoids
        kmedoids = KMedoids(n_clusters=4, random_state=42)
        kmedoids_labels = kmedoids.fit_predict(X)
        kmedoids_sil = silhouette_score(X, kmedoids_labels)
        results['kmedoids'].append(kmedoids_sil)

    return n_outliers_range, results


def plot_robustness(n_outliers_range, robustness_results):
    """
    Plot robustness analysis results.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(n_outliers_range, robustness_results['kmeans'],
           marker='o', linewidth=2, markersize=8, label='K-Means')
    ax.plot(n_outliers_range, robustness_results['kmedoids'],
           marker='s', linewidth=2, markersize=8, label='K-Medoids')

    ax.set_xlabel('Number of Outliers', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Robustness to Outliers Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("K-MEDOIDS AND PAM CLUSTERING ANALYSIS")
    print("="*80)

    # Generate datasets
    print("\n1. Generating synthetic datasets...")
    datasets = generate_datasets()
    print(f"   Generated {len(datasets)} datasets")

    # Standardize datasets
    scaler = StandardScaler()
    for name in datasets:
        datasets[name] = (scaler.fit_transform(datasets[name][0]), datasets[name][1])

    # Analyze each dataset
    all_comparisons = {}

    for ds_name, (X, y_true) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*80}")

        # Determine number of clusters
        n_clusters = len(np.unique(y_true[y_true >= 0]))

        # Compare K-Medoids vs K-Means
        print(f"\n2. Comparing K-Medoids vs K-Means...")
        comparison = compare_kmedoids_kmeans(X, n_clusters=n_clusters)

        print(f"\n   K-MEANS:")
        print(f"   - Silhouette: {comparison['kmeans']['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {comparison['kmeans']['davies_bouldin']:.4f}")
        print(f"   - Time: {comparison['kmeans']['time']:.4f}s")

        print(f"\n   K-MEDOIDS:")
        print(f"   - Silhouette: {comparison['kmedoids']['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {comparison['kmedoids']['davies_bouldin']:.4f}")
        print(f"   - Time: {comparison['kmedoids']['time']:.4f}s")

        # Compare with ground truth
        if y_true is not None and len(np.unique(y_true)) > 1:
            y_filtered = y_true[y_true >= 0]
            if len(y_filtered) > 0:
                kmeans_ari = adjusted_rand_score(
                    y_true[y_true >= 0],
                    comparison['kmeans']['labels'][y_true >= 0]
                )
                kmedoids_ari = adjusted_rand_score(
                    y_true[y_true >= 0],
                    comparison['kmedoids']['labels'][y_true >= 0]
                )
                print(f"\n   Ground Truth Comparison:")
                print(f"   - K-Means ARI: {kmeans_ari:.4f}")
                print(f"   - K-Medoids ARI: {kmedoids_ari:.4f}")

        # Visualize
        fig1 = plot_comparison(X, comparison, title=f"{ds_name.capitalize()} Dataset")
        plt.savefig(f'/tmp/kmedoids_{ds_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        all_comparisons[ds_name] = comparison

    # Additional analyses on blobs dataset
    print(f"\n{'='*80}")
    print("ADDITIONAL ANALYSES (Blobs Dataset)")
    print(f"{'='*80}")

    X, _ = datasets['blobs']

    # Distance metric comparison
    print("\n3. Comparing distance metrics...")
    metric_results = compare_metrics(X, n_clusters=4)

    for metric, result in metric_results.items():
        print(f"\n   {metric.upper()}:")
        print(f"   - Silhouette: {result['silhouette']:.4f}")
        print(f"   - Time: {result['time']:.4f}s")

    # Visualize metric comparison
    fig2 = plot_metric_comparison(metric_results)
    plt.savefig('/tmp/kmedoids_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Performance metrics
    fig3 = plot_performance_metrics(all_comparisons['blobs'], metric_results)
    plt.savefig('/tmp/kmedoids_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Robustness analysis
    print("\n4. Performing robustness analysis...")
    X_clean, _ = make_blobs(n_samples=400, centers=4, random_state=42)
    X_clean = scaler.fit_transform(X_clean)

    n_outliers_range, robustness_results = robustness_analysis(X_clean)

    print("\n   Outlier Robustness (Silhouette Scores):")
    for n_out, km_sil, kmed_sil in zip(n_outliers_range,
                                        robustness_results['kmeans'],
                                        robustness_results['kmedoids']):
        print(f"   {n_out:3d} outliers - K-Means: {km_sil:.4f}, K-Medoids: {kmed_sil:.4f}")

    fig4 = plot_robustness(n_outliers_range, robustness_results)
    plt.savefig('/tmp/kmedoids_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. K-Medoids is more robust to outliers than K-Means")
    print("2. K-Medoids uses actual data points as cluster centers (medoids)")
    print("3. Manhattan distance often performs well for high-dimensional data")
    print("4. K-Medoids has higher computational cost but better robustness")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
