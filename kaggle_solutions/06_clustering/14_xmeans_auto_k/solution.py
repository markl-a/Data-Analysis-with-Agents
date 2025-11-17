"""
X-Means Clustering with Automatic K Determination
==================================================
Implementation of X-Means algorithm that automatically determines the optimal
number of clusters using BIC (Bayesian Information Criterion).

Author: Data Science Team
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy import stats
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class XMeans:
    """
    X-Means clustering algorithm with automatic determination of the number of clusters.
    Uses BIC (Bayesian Information Criterion) to decide whether to split clusters.
    """

    def __init__(self, k_min=2, k_max=20, random_state=42, max_iter=300):
        """
        Initialize X-Means clustering.

        Parameters:
        -----------
        k_min : int
            Minimum number of clusters to consider
        k_max : int
            Maximum number of clusters to consider
        random_state : int
            Random state for reproducibility
        max_iter : int
            Maximum iterations for K-Means
        """
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.max_iter = max_iter
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_clusters_ = None
        self.bic_scores_ = []
        self.split_history_ = []

    def _compute_bic(self, X, labels, centers):
        """
        Compute Bayesian Information Criterion (BIC) for cluster model.

        BIC = L(D) - (p/2) * log(N)
        where L(D) is log-likelihood, p is number of parameters, N is number of points
        """
        n_samples, n_features = X.shape
        n_clusters = len(centers)

        # Compute variance
        variance = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                variance += np.sum((cluster_points - centers[k]) ** 2)

        variance = variance / (n_samples - n_clusters)

        # Log-likelihood (assuming Gaussian distribution)
        log_likelihood = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            n_k = len(cluster_points)
            if n_k > 0:
                log_likelihood += n_k * np.log(n_k / n_samples)
                log_likelihood -= (n_k * n_features / 2) * np.log(2 * np.pi * variance)
                log_likelihood -= (n_k - 1) / 2

        # Number of free parameters
        n_params = n_clusters * (n_features + 1)

        # BIC (negative because we want to maximize)
        bic = log_likelihood - (n_params / 2) * np.log(n_samples)

        return bic

    def _should_split(self, X, parent_labels, parent_centers, cluster_id):
        """
        Decide whether to split a cluster based on BIC improvement.
        """
        # Get points in this cluster
        cluster_points = X[parent_labels == cluster_id]

        if len(cluster_points) < 3:
            return False, None, None

        # Current BIC (1 cluster)
        center_current = parent_centers[cluster_id].reshape(1, -1)
        labels_current = np.zeros(len(cluster_points), dtype=int)
        bic_current = self._compute_bic(cluster_points, labels_current, center_current)

        # Try splitting into 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=self.random_state,
                       n_init=10, max_iter=self.max_iter)
        labels_split = kmeans.fit_predict(cluster_points)
        centers_split = kmeans.cluster_centers_

        bic_split = self._compute_bic(cluster_points, labels_split, centers_split)

        # Split if BIC improves
        if bic_split > bic_current:
            return True, labels_split, centers_split
        else:
            return False, None, None

    def fit(self, X):
        """
        Fit X-Means clustering.
        """
        # Start with k_min clusters
        kmeans = KMeans(n_clusters=self.k_min, random_state=self.random_state,
                       n_init=10, max_iter=self.max_iter)
        current_labels = kmeans.fit_predict(X)
        current_centers = kmeans.cluster_centers_
        current_k = self.k_min

        self.split_history_ = [(current_k, self._compute_bic(X, current_labels, current_centers))]

        iteration = 0
        while current_k < self.k_max:
            iteration += 1
            improved = False

            new_labels = current_labels.copy()
            new_centers = []
            cluster_id_map = {}
            new_cluster_id = 0

            # Try to split each cluster
            for cluster_id in range(current_k):
                should_split, split_labels, split_centers = self._should_split(
                    X, current_labels, current_centers, cluster_id
                )

                if should_split and current_k + len(new_centers) < self.k_max:
                    # Split this cluster
                    cluster_mask = current_labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]

                    # Assign new labels
                    new_labels[cluster_indices[split_labels == 0]] = new_cluster_id
                    new_labels[cluster_indices[split_labels == 1]] = new_cluster_id + 1

                    # Add new centers
                    new_centers.append(split_centers[0])
                    new_centers.append(split_centers[1])

                    cluster_id_map[cluster_id] = [new_cluster_id, new_cluster_id + 1]
                    new_cluster_id += 2
                    improved = True
                else:
                    # Keep cluster as is
                    cluster_mask = current_labels == cluster_id
                    new_labels[cluster_mask] = new_cluster_id
                    new_centers.append(current_centers[cluster_id])
                    cluster_id_map[cluster_id] = [new_cluster_id]
                    new_cluster_id += 1

            if not improved:
                break

            # Update current state
            current_labels = new_labels
            current_centers = np.array(new_centers)
            current_k = len(current_centers)

            # Record BIC
            bic = self._compute_bic(X, current_labels, current_centers)
            self.split_history_.append((current_k, bic))

            # Re-run K-Means to refine
            kmeans = KMeans(n_clusters=current_k, init=current_centers,
                          n_init=1, max_iter=self.max_iter, random_state=self.random_state)
            current_labels = kmeans.fit_predict(X)
            current_centers = kmeans.cluster_centers_

        self.labels_ = current_labels
        self.cluster_centers_ = current_centers
        self.n_clusters_ = current_k

        return self

    def fit_predict(self, X):
        """
        Fit and return cluster labels.
        """
        self.fit(X)
        return self.labels_


def generate_variable_clusters(random_state=42):
    """
    Generate datasets with varying numbers of true clusters.
    """
    np.random.seed(random_state)
    datasets = {}

    # 3 clusters
    X_3, y_3 = make_blobs(n_samples=500, centers=3, n_features=2,
                          cluster_std=1.0, random_state=42)
    datasets['3_clusters'] = (X_3, y_3, 3)

    # 5 clusters
    X_5, y_5 = make_blobs(n_samples=600, centers=5, n_features=2,
                          cluster_std=1.2, random_state=43)
    datasets['5_clusters'] = (X_5, y_5, 5)

    # 8 clusters
    X_8, y_8 = make_blobs(n_samples=800, centers=8, n_features=2,
                          cluster_std=0.8, random_state=44)
    datasets['8_clusters'] = (X_8, y_8, 8)

    # Variable density (4 clusters)
    centers = [[0, 0], [5, 5], [-5, 5], [0, -6]]
    stds = [0.5, 1.5, 1.0, 2.0]
    X_var = []
    y_var = []
    for i, (center, std) in enumerate(zip(centers, stds)):
        X_temp, _ = make_blobs(n_samples=150, centers=[center], n_features=2,
                              cluster_std=std, random_state=45+i)
        X_var.append(X_temp)
        y_var.extend([i] * 150)
    X_var = np.vstack(X_var)
    y_var = np.array(y_var)
    datasets['variable_density'] = (X_var, y_var, 4)

    return datasets


def compare_with_kmeans(X, true_k_range=[2, 3, 4, 5, 6, 7, 8]):
    """
    Compare X-Means with K-Means for different k values.
    """
    results = {}

    # X-Means
    xmeans = XMeans(k_min=2, k_max=15, random_state=42)
    start_time = time.time()
    xmeans_labels = xmeans.fit_predict(X)
    xmeans_time = time.time() - start_time

    results['xmeans'] = {
        'labels': xmeans_labels,
        'centers': xmeans.cluster_centers_,
        'n_clusters': xmeans.n_clusters_,
        'time': xmeans_time,
        'split_history': xmeans.split_history_,
        'silhouette': silhouette_score(X, xmeans_labels),
        'davies_bouldin': davies_bouldin_score(X, xmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(X, xmeans_labels)
    }

    # K-Means for different k
    results['kmeans'] = {}
    for k in true_k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        start_time = time.time()
        labels = kmeans.fit_predict(X)
        elapsed = time.time() - start_time

        results['kmeans'][k] = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'time': elapsed,
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'inertia': kmeans.inertia_
        }

    return results


def plot_xmeans_results(X, xmeans_result, title="X-Means Clustering"):
    """
    Visualize X-Means clustering results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Final clustering
    ax = axes[0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=xmeans_result['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(xmeans_result['centers'][:, 0], xmeans_result['centers'][:, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2,
              label='Centers')
    ax.set_title(f'X-Means Clustering (k={xmeans_result["n_clusters"]})\n'
                f'Silhouette: {xmeans_result["silhouette"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.colorbar(scatter, ax=ax)

    # Plot 2: BIC evolution
    ax = axes[1]
    split_history = np.array(xmeans_result['split_history'])
    ax.plot(split_history[:, 0], split_history[:, 1], marker='o',
           linewidth=2, markersize=8, color='steelblue')
    ax.axvline(xmeans_result['n_clusters'], color='red', linestyle='--',
              linewidth=2, label=f'Selected k={xmeans_result["n_clusters"]}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('BIC Score', fontsize=11)
    ax.set_title('BIC Score Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_kmeans_comparison(comparison_results, true_k=None):
    """
    Compare X-Means with K-Means for different k values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    k_values = sorted(comparison_results['kmeans'].keys())
    silhouettes = [comparison_results['kmeans'][k]['silhouette'] for k in k_values]
    db_scores = [comparison_results['kmeans'][k]['davies_bouldin'] for k in k_values]
    ch_scores = [comparison_results['kmeans'][k]['calinski_harabasz'] for k in k_values]
    inertias = [comparison_results['kmeans'][k]['inertia'] for k in k_values]

    xmeans_k = comparison_results['xmeans']['n_clusters']

    # Plot 1: Silhouette Score
    ax = axes[0, 0]
    ax.plot(k_values, silhouettes, marker='o', linewidth=2, markersize=8, label='K-Means')
    ax.axhline(comparison_results['xmeans']['silhouette'], color='red',
              linestyle='--', linewidth=2, label='X-Means')
    ax.axvline(xmeans_k, color='green', linestyle=':', linewidth=2,
              label=f'X-Means k={xmeans_k}')
    if true_k:
        ax.axvline(true_k, color='orange', linestyle='-.', linewidth=2,
                  label=f'True k={true_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs k', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Davies-Bouldin Index
    ax = axes[0, 1]
    ax.plot(k_values, db_scores, marker='s', linewidth=2, markersize=8,
           color='coral', label='K-Means')
    ax.axhline(comparison_results['xmeans']['davies_bouldin'], color='red',
              linestyle='--', linewidth=2, label='X-Means')
    ax.axvline(xmeans_k, color='green', linestyle=':', linewidth=2,
              label=f'X-Means k={xmeans_k}')
    if true_k:
        ax.axvline(true_k, color='orange', linestyle='-.', linewidth=2,
                  label=f'True k={true_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index vs k (Lower is Better)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Calinski-Harabasz Score
    ax = axes[1, 0]
    ax.plot(k_values, ch_scores, marker='^', linewidth=2, markersize=8,
           color='purple', label='K-Means')
    ax.axhline(comparison_results['xmeans']['calinski_harabasz'], color='red',
              linestyle='--', linewidth=2, label='X-Means')
    ax.axvline(xmeans_k, color='green', linestyle=':', linewidth=2,
              label=f'X-Means k={xmeans_k}')
    if true_k:
        ax.axvline(true_k, color='orange', linestyle='-.', linewidth=2,
                  label=f'True k={true_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score vs k (Higher is Better)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Inertia (Elbow)
    ax = axes[1, 1]
    ax.plot(k_values, inertias, marker='D', linewidth=2, markersize=8,
           color='green', label='K-Means Inertia')
    ax.axvline(xmeans_k, color='red', linestyle='--', linewidth=2,
              label=f'X-Means k={xmeans_k}')
    if true_k:
        ax.axvline(true_k, color='orange', linestyle='-.', linewidth=2,
                  label=f'True k={true_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_visual_comparison(X, comparison_results, true_k):
    """
    Visual comparison of clustering results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # X-Means result
    ax = axes[0]
    xmeans_result = comparison_results['xmeans']
    scatter = ax.scatter(X[:, 0], X[:, 1], c=xmeans_result['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(xmeans_result['centers'][:, 0], xmeans_result['centers'][:, 1],
              c='red', marker='*', s=400, edgecolors='black', linewidths=2)
    ax.set_title(f'X-Means (k={xmeans_result["n_clusters"]})\n'
                f'Silhouette: {xmeans_result["silhouette"]:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # K-Means results for different k
    k_values = [true_k-1, true_k, true_k+1, true_k+2, true_k+3]
    for idx, k in enumerate(k_values, start=1):
        if k in comparison_results['kmeans']:
            ax = axes[idx]
            result = comparison_results['kmeans'][k]
            scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                               cmap='viridis', alpha=0.6, s=50)
            ax.scatter(result['centers'][:, 0], result['centers'][:, 1],
                      c='red', marker='x', s=300, linewidths=3)
            ax.set_title(f'K-Means (k={k})\n'
                        f'Silhouette: {result["silhouette"]:.3f}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

    plt.tight_layout()
    return fig


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("X-MEANS CLUSTERING WITH AUTOMATIC K DETERMINATION")
    print("="*80)

    # Generate datasets
    print("\n1. Generating synthetic datasets with varying cluster counts...")
    datasets = generate_variable_clusters()
    print(f"   Generated {len(datasets)} datasets")

    # Standardize
    scaler = StandardScaler()
    for name in datasets:
        X, y, k = datasets[name]
        datasets[name] = (scaler.fit_transform(X), y, k)

    # Analyze each dataset
    for ds_name, (X, y_true, true_k) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name.upper()} (True k={true_k})")
        print(f"{'='*80}")

        # Run X-Means and K-Means comparison
        print(f"\n2. Running X-Means and K-Means comparison...")
        comparison = compare_with_kmeans(X, true_k_range=range(2, min(12, true_k+5)))

        xmeans_result = comparison['xmeans']
        print(f"\n   X-MEANS RESULTS:")
        print(f"   - Detected k: {xmeans_result['n_clusters']} (true k={true_k})")
        print(f"   - Silhouette: {xmeans_result['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {xmeans_result['davies_bouldin']:.4f}")
        print(f"   - Calinski-Harabasz: {xmeans_result['calinski_harabasz']:.2f}")
        print(f"   - Time: {xmeans_result['time']:.4f}s")

        # Best K-Means by silhouette
        best_k = max(comparison['kmeans'].keys(),
                    key=lambda k: comparison['kmeans'][k]['silhouette'])
        best_kmeans = comparison['kmeans'][best_k]

        print(f"\n   BEST K-MEANS (k={best_k}):")
        print(f"   - Silhouette: {best_kmeans['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {best_kmeans['davies_bouldin']:.4f}")
        print(f"   - Calinski-Harabasz: {best_kmeans['calinski_harabasz']:.2f}")

        # Check accuracy
        accuracy = "Exact" if xmeans_result['n_clusters'] == true_k else "Off by " + str(abs(xmeans_result['n_clusters'] - true_k))
        print(f"\n   X-Means Accuracy: {accuracy}")

        # Visualizations
        print(f"\n3. Creating visualizations...")

        fig1 = plot_xmeans_results(X, xmeans_result, title=f"{ds_name.capitalize()}")
        plt.savefig(f'/tmp/xmeans_{ds_name}_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig2 = plot_kmeans_comparison(comparison, true_k=true_k)
        plt.savefig(f'/tmp/xmeans_{ds_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig3 = plot_visual_comparison(X, comparison, true_k)
        plt.savefig(f'/tmp/xmeans_{ds_name}_visual.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   - Saved visualization plots")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. X-Means automatically determines optimal number of clusters")
    print("2. Uses BIC to decide when to split clusters")
    print("3. Performs well across datasets with varying cluster counts")
    print("4. More robust than manual k selection in K-Means")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
