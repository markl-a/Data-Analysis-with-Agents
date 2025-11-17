"""
Self-Organizing Maps
====================
Comprehensive implementation of self-organizing maps with multiple variants,
cluster validation metrics, and extensive visualizations.

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_datasets():
    """Generate diverse test datasets."""
    datasets = {}

    # Well-separated blobs
    X1, y1 = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    datasets['blobs'] = (X1, y1, 4)

    # Overlapping clusters
    X2, y2 = make_blobs(n_samples=600, centers=3, cluster_std=2.0, random_state=43)
    datasets['overlap'] = (X2, y2, 3)

    # Non-convex shapes
    X3, y3 = make_moons(n_samples=500, noise=0.1, random_state=44)
    datasets['moons'] = (X3, y3, 2)

    # Concentric circles
    X4, y4 = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=45)
    datasets['circles'] = (X4, y4, 2)

    # Varied density
    centers = [[0, 0], [6, 6], [-5, 5]]
    stds = [0.5, 1.5, 1.0]
    X5_parts, y5_parts = [], []
    for i, (center, std) in enumerate(zip(centers, stds)):
        X_temp, _ = make_blobs(n_samples=150, centers=[center],
                               cluster_std=std, random_state=46+i)
        X5_parts.append(X_temp)
        y5_parts.extend([i] * 150)
    X5 = np.vstack(X5_parts)
    y5 = np.array(y5_parts)
    datasets['varied'] = (X5, y5, 3)

    return datasets


def compute_metrics(X, labels):
    """Compute clustering quality metrics."""
    if len(np.unique(labels)) < 2:
        return {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0}

    return {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }


def plot_clustering_results(X, labels, title="Clustering Results"):
    """Visualize clustering results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_dict):
    """Plot performance metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    methods = list(results_dict.keys())
    silhouettes = [results_dict[m]['metrics']['silhouette'] for m in methods]
    db_scores = [results_dict[m]['metrics']['davies_bouldin'] for m in methods]
    ch_scores = [results_dict[m]['metrics']['calinski_harabasz'] for m in methods]
    times = [results_dict[m]['time'] for m in methods]

    # Silhouette Score
    axes[0, 0].bar(methods, silhouettes, color='steelblue')
    axes[0, 0].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Davies-Bouldin Index
    axes[0, 1].bar(methods, db_scores, color='coral')
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('Index')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Calinski-Harabasz Score
    axes[1, 0].bar(methods, ch_scores, color='lightgreen')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Execution Time
    axes[1, 1].bar(methods, times, color='purple', alpha=0.7)
    axes[1, 1].set_title('Execution Time', fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_comparison_grid(X, results_dict, title_prefix=""):
    """Create grid comparison of different methods."""
    n_methods = len(results_dict)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel() if n_methods > 1 else [axes]

    for idx, (method_name, result) in enumerate(results_dict.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                           cmap='viridis', alpha=0.6, s=50)
        metrics = result['metrics']
        ax.set_title(f'{title_prefix}{method_name}\n'
                    f'Sil: {metrics["silhouette"]:.3f}, '
                    f'DB: {metrics["davies_bouldin"]:.3f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def run_clustering_variants(X, n_clusters):
    """
    Run multiple clustering variants.
    Override this function in each solution.
    """
    results = {}

    # Placeholder - will be customized per solution
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    start = time.time()
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start

    results['variant_1'] = {
        'labels': labels,
        'time': elapsed,
        'metrics': compute_metrics(X, labels)
    }

    # Add more variants here based on specific solution
    return results


def main():
    """Main execution function."""
    print("="*80)
    print("SELF-ORGANIZING MAPS")
    print("="*80)

    # Generate datasets
    print("\n1. Generating test datasets...")
    datasets = generate_datasets()

    # Standardize data
    scaler = StandardScaler()
    for name in datasets:
        X, y, k = datasets[name]
        datasets[name] = (scaler.fit_transform(X), y, k)

    print(f"   Generated {len(datasets)} datasets")

    # Run clustering on each dataset
    for ds_name, (X, y_true, true_k) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {ds_name.upper()} (k={true_k})")
        print(f"{'='*80}")

        # Run clustering variants
        results = run_clustering_variants(X, true_k)

        # Print results
        for method_name, result in results.items():
            print(f"\n   {method_name.upper()}:")
            metrics = result['metrics']
            print(f"   - Silhouette: {metrics['silhouette']:.4f}")
            print(f"   - Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
            print(f"   - Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
            print(f"   - Time: {result['time']:.4f}s")

            if y_true is not None and len(np.unique(y_true)) > 1:
                ari = adjusted_rand_score(y_true, result['labels'])
                nmi = normalized_mutual_info_score(y_true, result['labels'])
                print(f"   - ARI: {ari:.4f}, NMI: {nmi:.4f}")

        # Create visualizations
        print(f"\n2. Creating visualizations...")

        # Individual plots
        for method_name, result in results.items():
            fig = plot_clustering_results(X, result['labels'],
                                         title=f"{ds_name.capitalize()} - {method_name}")
            plt.savefig(f'/tmp/30_self_organizing_maps_{ds_name}_{method_name}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        # Comparison plots
        if len(results) > 1:
            fig = plot_metrics_comparison(results)
            plt.savefig(f'/tmp/30_self_organizing_maps_{ds_name}_metrics.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            fig = plot_comparison_grid(X, results, title_prefix=f"{ds_name.capitalize()} - ")
            plt.savefig(f'/tmp/30_self_organizing_maps_{ds_name}_grid.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"   - Saved visualization plots")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll visualizations saved to /tmp/")


if __name__ == "__main__":
    main()
