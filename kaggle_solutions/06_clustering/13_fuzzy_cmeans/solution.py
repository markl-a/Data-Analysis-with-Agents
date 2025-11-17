"""
Fuzzy C-Means Clustering Analysis
==================================
Comprehensive implementation of Fuzzy C-Means (FCM) clustering with soft cluster
assignments, membership degree analysis, and comparison with hard clustering.

Author: Data Science Team
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering implementation.
    Allows soft cluster assignments where each point has a membership degree to each cluster.
    """

    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5, random_state=42):
        """
        Initialize Fuzzy C-Means clustering.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        m : float
            Fuzziness parameter (m > 1). Higher values create fuzzier clusters.
            Typical range: 1.5 to 3.0
        max_iter : int
            Maximum number of iterations
        error : float
            Convergence threshold
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.m = m  # Fuzziness coefficient
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.centers_ = None
        self.u_ = None  # Membership matrix
        self.labels_ = None
        self.n_iter_ = 0

    def _initialize_membership(self, n_samples):
        """
        Initialize membership matrix randomly.
        Each row sums to 1 (total membership = 1 for each point).
        """
        np.random.seed(self.random_state)
        u = np.random.rand(n_samples, self.n_clusters)
        # Normalize so each row sums to 1
        u = u / np.sum(u, axis=1, keepdims=True)
        return u

    def _update_centers(self, X, u):
        """
        Update cluster centers based on membership matrix.
        """
        um = u ** self.m  # Raise membership to power m
        centers = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return centers

    def _update_membership(self, X, centers):
        """
        Update membership matrix based on distances to centers.
        """
        n_samples = X.shape[0]
        u = np.zeros((n_samples, self.n_clusters))

        # Calculate distances from each point to each center
        distances = cdist(X, centers, metric='euclidean')

        # Avoid division by zero
        distances = np.fmax(distances, np.finfo(float).eps)

        for i in range(n_samples):
            for j in range(self.n_clusters):
                # Fuzzy membership formula
                denominator = np.sum(
                    (distances[i, j] / distances[i, :]) ** (2 / (self.m - 1))
                )
                u[i, j] = 1 / denominator

        return u

    def _compute_cost(self, X, centers, u):
        """
        Compute the objective function (within-cluster variance).
        """
        distances = cdist(X, centers, metric='euclidean')
        um = u ** self.m
        cost = np.sum(um * (distances ** 2))
        return cost

    def fit(self, X):
        """
        Fit Fuzzy C-Means clustering.
        """
        n_samples = X.shape[0]

        # Initialize membership matrix
        u = self._initialize_membership(n_samples)

        for iteration in range(self.max_iter):
            u_old = u.copy()

            # Update centers
            self.centers_ = self._update_centers(X, u)

            # Update membership
            u = self._update_membership(X, self.centers_)

            # Check convergence
            if np.linalg.norm(u - u_old) < self.error:
                self.n_iter_ = iteration + 1
                break

        self.u_ = u
        # Hard clustering: assign to cluster with highest membership
        self.labels_ = np.argmax(u, axis=1)

        return self

    def fit_predict(self, X):
        """
        Fit and return hard cluster labels.
        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predict cluster labels for new data.
        """
        if self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        u = self._update_membership(X, self.centers_)
        return np.argmax(u, axis=1)

    def predict_proba(self, X):
        """
        Predict membership probabilities for new data.
        """
        if self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        return self._update_membership(X, self.centers_)


def generate_overlapping_clusters(n_samples=800, n_clusters=3, cluster_std=1.5):
    """
    Generate datasets with overlapping clusters (ideal for fuzzy clustering).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=2,
        cluster_std=cluster_std,
        random_state=42
    )
    return X, y


def compare_fuzziness_parameters(X, n_clusters=3, m_values=[1.5, 2.0, 2.5, 3.0, 4.0]):
    """
    Compare different fuzziness parameters.
    """
    results = {}

    for m in m_values:
        fcm = FuzzyCMeans(n_clusters=n_clusters, m=m, random_state=42)
        start_time = time.time()
        labels = fcm.fit_predict(X)
        elapsed = time.time() - start_time

        results[f'm={m}'] = {
            'labels': labels,
            'centers': fcm.centers_,
            'membership': fcm.u_,
            'n_iter': fcm.n_iter_,
            'time': elapsed,
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
        }

    return results


def compare_fcm_kmeans(X, n_clusters=3):
    """
    Compare Fuzzy C-Means with K-Means.
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
        'time': kmeans_time,
        'silhouette': silhouette_score(X, kmeans_labels),
        'davies_bouldin': davies_bouldin_score(X, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels)
    }

    # Fuzzy C-Means
    fcm = FuzzyCMeans(n_clusters=n_clusters, m=2.0, random_state=42)
    start_time = time.time()
    fcm_labels = fcm.fit_predict(X)
    fcm_time = time.time() - start_time

    results['fcm'] = {
        'labels': fcm_labels,
        'centers': fcm.centers_,
        'membership': fcm.u_,
        'time': fcm_time,
        'n_iter': fcm.n_iter_,
        'silhouette': silhouette_score(X, fcm_labels),
        'davies_bouldin': davies_bouldin_score(X, fcm_labels),
        'calinski_harabasz': calinski_harabasz_score(X, fcm_labels)
    }

    return results


def analyze_membership_uncertainty(membership_matrix):
    """
    Analyze uncertainty in cluster assignments based on membership values.
    """
    # Calculate entropy for each point
    # Higher entropy = more uncertainty
    epsilon = 1e-10  # Avoid log(0)
    entropy = -np.sum(membership_matrix * np.log(membership_matrix + epsilon), axis=1)

    # Maximum membership for each point
    max_membership = np.max(membership_matrix, axis=1)

    # Membership difference (between top 2 clusters)
    sorted_membership = np.sort(membership_matrix, axis=1)
    membership_diff = sorted_membership[:, -1] - sorted_membership[:, -2]

    return {
        'entropy': entropy,
        'max_membership': max_membership,
        'membership_diff': membership_diff
    }


def plot_fuzzy_clusters(X, fcm_result, title="Fuzzy C-Means Clustering"):
    """
    Visualize fuzzy clustering with membership-based coloring.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Hard clustering (standard visualization)
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=fcm_result['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(fcm_result['centers'][:, 0], fcm_result['centers'][:, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2)
    ax.set_title('Hard Cluster Assignment', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)

    # Plot 2: Membership to Cluster 0
    ax = axes[0, 1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=fcm_result['membership'][:, 0],
                        cmap='YlOrRd', alpha=0.7, s=50, vmin=0, vmax=1)
    ax.scatter(fcm_result['centers'][0, 0], fcm_result['centers'][0, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2)
    ax.set_title('Membership to Cluster 0', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Membership Degree')

    # Plot 3: Maximum membership value
    ax = axes[1, 0]
    max_membership = np.max(fcm_result['membership'], axis=1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=max_membership,
                        cmap='RdYlGn', alpha=0.7, s=50, vmin=0, vmax=1)
    ax.scatter(fcm_result['centers'][:, 0], fcm_result['centers'][:, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2)
    ax.set_title('Maximum Membership (Certainty)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Max Membership')

    # Plot 4: Uncertainty (entropy)
    ax = axes[1, 1]
    uncertainty = analyze_membership_uncertainty(fcm_result['membership'])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=uncertainty['entropy'],
                        cmap='coolwarm', alpha=0.7, s=50)
    ax.scatter(fcm_result['centers'][:, 0], fcm_result['centers'][:, 1],
              c='red', marker='*', s=500, edgecolors='black', linewidths=2)
    ax.set_title('Cluster Assignment Uncertainty (Entropy)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Entropy')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_fuzziness_comparison(X, fuzziness_results):
    """
    Compare different fuzziness parameters.
    """
    n_params = len(fuzziness_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, (param_name, result) in enumerate(fuzziness_results.items()):
        if idx >= 6:
            break

        ax = axes[idx]
        max_membership = np.max(result['membership'], axis=1)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=max_membership,
                           cmap='RdYlGn', alpha=0.7, s=50, vmin=0, vmax=1)
        ax.scatter(result['centers'][:, 0], result['centers'][:, 1],
                  c='red', marker='*', s=400, edgecolors='black', linewidths=2)
        ax.set_title(f'{param_name}\nSilhouette: {result["silhouette"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax, label='Max Membership')

    # Hide unused subplots
    for idx in range(n_params, 6):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_comparison_fcm_kmeans(X, comparison_results):
    """
    Visualize FCM vs K-Means comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # K-Means
    ax = axes[0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=comparison_results['kmeans']['labels'],
                        cmap='viridis', alpha=0.6, s=50)
    ax.scatter(comparison_results['kmeans']['centers'][:, 0],
              comparison_results['kmeans']['centers'][:, 1],
              c='red', marker='x', s=300, linewidths=3, label='Centroids')
    ax.set_title(f'K-Means (Hard Clustering)\n'
                f'Silhouette: {comparison_results["kmeans"]["silhouette"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

    # Fuzzy C-Means
    ax = axes[1]
    max_membership = np.max(comparison_results['fcm']['membership'], axis=1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=max_membership,
                        cmap='RdYlGn', alpha=0.7, s=50, vmin=0, vmax=1)
    ax.scatter(comparison_results['fcm']['centers'][:, 0],
              comparison_results['fcm']['centers'][:, 1],
              c='red', marker='*', s=500, edgecolors='black',
              linewidths=2, label='Centers')
    ax.set_title(f'Fuzzy C-Means (Soft Clustering)\n'
                f'Silhouette: {comparison_results["fcm"]["silhouette"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Max Membership')

    plt.tight_layout()
    return fig


def plot_membership_analysis(membership_matrix, labels):
    """
    Analyze and visualize membership distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    uncertainty = analyze_membership_uncertainty(membership_matrix)

    # Plot 1: Membership distribution
    ax = axes[0, 0]
    for i in range(membership_matrix.shape[1]):
        cluster_memberships = membership_matrix[labels == i, i]
        ax.hist(cluster_memberships, bins=30, alpha=0.6, label=f'Cluster {i}')
    ax.set_xlabel('Membership Degree', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Membership Distribution per Cluster', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Maximum membership distribution
    ax = axes[0, 1]
    ax.hist(uncertainty['max_membership'], bins=30, color='steelblue', alpha=0.7)
    ax.axvline(np.mean(uncertainty['max_membership']), color='red',
              linestyle='--', linewidth=2, label=f'Mean: {np.mean(uncertainty["max_membership"]):.3f}')
    ax.set_xlabel('Maximum Membership', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Maximum Membership Values', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Entropy distribution
    ax = axes[1, 0]
    ax.hist(uncertainty['entropy'], bins=30, color='coral', alpha=0.7)
    ax.axvline(np.mean(uncertainty['entropy']), color='red',
              linestyle='--', linewidth=2, label=f'Mean: {np.mean(uncertainty["entropy"]):.3f}')
    ax.set_xlabel('Entropy', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Uncertainty Distribution (Entropy)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Membership difference
    ax = axes[1, 1]
    ax.hist(uncertainty['membership_diff'], bins=30, color='lightgreen', alpha=0.7)
    ax.axvline(np.mean(uncertainty['membership_diff']), color='red',
              linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(uncertainty["membership_diff"]):.3f}')
    ax.set_xlabel('Membership Difference (Top 2)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Separation Between Top 2 Memberships', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_performance_metrics(comparison_results, fuzziness_results):
    """
    Plot performance metrics comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Silhouette Score Comparison
    ax = axes[0, 0]
    methods = ['K-Means', 'FCM (m=2.0)']
    silhouettes = [
        comparison_results['kmeans']['silhouette'],
        comparison_results['fcm']['silhouette']
    ]
    ax.bar(methods, silhouettes, color=['steelblue', 'coral'])
    ax.set_title('Silhouette Score Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score')
    ax.grid(True, alpha=0.3)

    # Plot 2: Time Comparison
    ax = axes[0, 1]
    times = [
        comparison_results['kmeans']['time'],
        comparison_results['fcm']['time']
    ]
    ax.bar(methods, times, color=['steelblue', 'coral'])
    ax.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Fuzziness Parameter Impact
    ax = axes[1, 0]
    m_values = [key for key in fuzziness_results.keys()]
    silhouettes_m = [fuzziness_results[m]['silhouette'] for m in m_values]
    ax.plot(range(len(m_values)), silhouettes_m, marker='o',
           linewidth=2, markersize=8, color='purple')
    ax.set_xticks(range(len(m_values)))
    ax.set_xticklabels(m_values)
    ax.set_xlabel('Fuzziness Parameter')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Impact of Fuzziness Parameter (m)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Iterations vs Fuzziness
    ax = axes[1, 1]
    iterations_m = [fuzziness_results[m]['n_iter'] for m in m_values]
    ax.plot(range(len(m_values)), iterations_m, marker='s',
           linewidth=2, markersize=8, color='green')
    ax.set_xticks(range(len(m_values)))
    ax.set_xticklabels(m_values)
    ax.set_xlabel('Fuzziness Parameter')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Convergence Speed vs Fuzziness', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """
    Main execution function.
    """
    print("="*80)
    print("FUZZY C-MEANS CLUSTERING ANALYSIS")
    print("="*80)

    # Generate datasets
    print("\n1. Generating synthetic datasets...")

    # Dataset 1: Overlapping clusters
    X_overlap, y_overlap = generate_overlapping_clusters(n_samples=800, n_clusters=3, cluster_std=1.5)

    # Dataset 2: Well-separated
    X_separated, y_separated = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=42)

    # Dataset 3: Complex shapes
    X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_overlap = scaler.fit_transform(X_overlap)
    X_separated = scaler.fit_transform(X_separated)
    X_moons = scaler.fit_transform(X_moons)

    datasets = {
        'overlapping': (X_overlap, y_overlap, 3),
        'separated': (X_separated, y_separated, 4),
        'moons': (X_moons, y_moons, 2)
    }

    print(f"   Generated {len(datasets)} datasets")

    # Analyze each dataset
    for ds_name, (X, y_true, n_clusters) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name.upper()}")
        print(f"{'='*80}")

        # Compare FCM vs K-Means
        print(f"\n2. Comparing Fuzzy C-Means vs K-Means...")
        comparison = compare_fcm_kmeans(X, n_clusters=n_clusters)

        print(f"\n   K-MEANS:")
        print(f"   - Silhouette: {comparison['kmeans']['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {comparison['kmeans']['davies_bouldin']:.4f}")
        print(f"   - Time: {comparison['kmeans']['time']:.4f}s")

        print(f"\n   FUZZY C-MEANS:")
        print(f"   - Silhouette: {comparison['fcm']['silhouette']:.4f}")
        print(f"   - Davies-Bouldin: {comparison['fcm']['davies_bouldin']:.4f}")
        print(f"   - Iterations: {comparison['fcm']['n_iter']}")
        print(f"   - Time: {comparison['fcm']['time']:.4f}s")

        # Membership analysis
        uncertainty = analyze_membership_uncertainty(comparison['fcm']['membership'])
        print(f"\n   MEMBERSHIP ANALYSIS:")
        print(f"   - Avg Max Membership: {np.mean(uncertainty['max_membership']):.4f}")
        print(f"   - Avg Entropy: {np.mean(uncertainty['entropy']):.4f}")
        print(f"   - Avg Membership Diff: {np.mean(uncertainty['membership_diff']):.4f}")

        # Visualizations
        fig1 = plot_fuzzy_clusters(X, comparison['fcm'],
                                   title=f"Fuzzy C-Means - {ds_name.capitalize()}")
        plt.savefig(f'/tmp/fcm_{ds_name}_fuzzy.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig2 = plot_comparison_fcm_kmeans(X, comparison)
        plt.savefig(f'/tmp/fcm_{ds_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig3 = plot_membership_analysis(comparison['fcm']['membership'],
                                       comparison['fcm']['labels'])
        plt.savefig(f'/tmp/fcm_{ds_name}_membership.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Additional analysis on overlapping dataset
    print(f"\n{'='*80}")
    print("FUZZINESS PARAMETER ANALYSIS")
    print(f"{'='*80}")

    print("\n3. Comparing different fuzziness parameters...")
    fuzziness_results = compare_fuzziness_parameters(X_overlap, n_clusters=3)

    for param_name, result in fuzziness_results.items():
        print(f"\n   {param_name.upper()}:")
        print(f"   - Silhouette: {result['silhouette']:.4f}")
        print(f"   - Iterations: {result['n_iter']}")
        print(f"   - Time: {result['time']:.4f}s")

    fig4 = plot_fuzziness_comparison(X_overlap, fuzziness_results)
    plt.savefig('/tmp/fcm_fuzziness_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig5 = plot_performance_metrics(comparison, fuzziness_results)
    plt.savefig('/tmp/fcm_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Fuzzy C-Means provides soft cluster assignments with membership degrees")
    print("2. Higher fuzziness (m) creates smoother cluster boundaries")
    print("3. FCM is better for overlapping clusters than hard clustering")
    print("4. Membership uncertainty can identify boundary/ambiguous points")
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
