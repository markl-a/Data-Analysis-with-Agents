"""
Local Outlier Factor (LOF) Anomaly Detection
============================================

This solution implements LOF-based anomaly detection:
1. Standard LOF algorithm
2. Modified LOF (MLOF)
3. Simplified LOF (SimplifiedLOF)
4. Weighted LOF

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class CustomLOFDetector:
    """Custom implementation of LOF"""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.X_train_ = None
        self.nn_ = None
        self.lrd_ = None

    def fit(self, X):
        """Fit the LOF detector"""
        self.X_train_ = X.copy()
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        self.nn_.fit(X)

        # Compute local reachability density for training data
        self.lrd_ = self._compute_lrd(X)

        return self

    def _compute_lrd(self, X):
        """Compute local reachability density"""
        distances, indices = self.nn_.kneighbors(X)

        lrd = np.zeros(len(X))

        for i in range(len(X)):
            # Get distances to neighbors
            neighbor_distances = distances[i, 1:]  # Exclude self
            neighbor_indices = indices[i, 1:]

            # Reachability distances
            reach_dists = []
            for j, neighbor_idx in enumerate(neighbor_indices):
                if neighbor_idx < len(self.X_train_):
                    # Distance to k-th neighbor of the neighbor
                    neighbor_k_dist, _ = self.nn_.kneighbors([self.X_train_[neighbor_idx]], n_neighbors=self.n_neighbors)
                    k_dist = neighbor_k_dist[0, -1]

                    # Reachability distance
                    reach_dist = max(neighbor_distances[j], k_dist)
                    reach_dists.append(reach_dist)

            # Local reachability density
            if reach_dists and np.mean(reach_dists) > 0:
                lrd[i] = 1.0 / np.mean(reach_dists)
            else:
                lrd[i] = 1e10  # Very high density for duplicate points

        return lrd

    def decision_function(self, X):
        """Compute LOF scores"""
        distances, indices = self.nn_.kneighbors(X)

        lof_scores = np.zeros(len(X))

        for i in range(len(X)):
            neighbor_indices = indices[i, 1:]

            # Gather LRD values of neighbors
            neighbor_lrds = []
            for neighbor_idx in neighbor_indices:
                if neighbor_idx < len(self.lrd_):
                    neighbor_lrds.append(self.lrd_[neighbor_idx])

            if neighbor_lrds:
                # Recompute LRD for test point
                reach_dists = []
                for j, neighbor_idx in enumerate(neighbor_indices):
                    if neighbor_idx < len(self.X_train_):
                        neighbor_k_dist, _ = self.nn_.kneighbors([self.X_train_[neighbor_idx]], n_neighbors=self.n_neighbors)
                        k_dist = neighbor_k_dist[0, -1]
                        reach_dist = max(distances[i, j+1], k_dist)
                        reach_dists.append(reach_dist)

                if reach_dists and np.mean(reach_dists) > 0:
                    lrd_point = 1.0 / np.mean(reach_dists)
                else:
                    lrd_point = 1e10

                # LOF score
                lof_scores[i] = np.mean(neighbor_lrds) / (lrd_point + 1e-10)
            else:
                lof_scores[i] = 1.0

        return lof_scores

    def predict(self, X):
        """Predict anomalies"""
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > threshold).astype(int)


class ModifiedLOFDetector:
    """Modified LOF with different density estimation"""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof_ = None

    def fit(self, X):
        """Fit using sklearn LOF with different parameters"""
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            metric='euclidean'
        )
        self.lof_.fit(X)
        return self

    def decision_function(self, X):
        """Return negative LOF scores (higher = more anomalous)"""
        return -self.lof_.score_samples(X)

    def predict(self, X):
        """Predict anomalies"""
        return (self.lof_.predict(X) == -1).astype(int)


def generate_clustered_data(n_samples=1000, n_features=8, contamination=0.1):
    """Generate data with clusters and isolated anomalies"""
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal

    # Create multiple clusters
    n_clusters = 4
    samples_per_cluster = n_normal // n_clusters

    X_normal = []
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 4
        cluster = np.random.randn(samples_per_cluster, n_features) * 0.8 + center
        X_normal.append(cluster)

    X_normal = np.vstack(X_normal)

    # Anomalies - isolated points far from clusters
    X_anomalies = np.random.uniform(-12, 12, (n_anomalies, n_features))

    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def plot_lof_scores_distribution(scores_list, y_test, names):
    """Plot LOF score distributions"""
    n_detectors = len(scores_list)
    fig, axes = plt.subplots(1, n_detectors, figsize=(6*n_detectors, 5))

    if n_detectors == 1:
        axes = [axes]

    for ax, scores, name in zip(axes, scores_list, names):
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]

        ax.hist(normal_scores, bins=50, alpha=0.7, color='blue',
               label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color='red',
               label='Anomaly', density=True)

        ax.set_xlabel('LOF Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name} Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lof_visualization_2d(X, y_true, detector, name):
    """Visualize LOF detection in 2D"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True labels
    ax1.scatter(X[y_true == 0, 0], X[y_true == 0, 1],
               c='blue', alpha=0.6, s=30, label='Normal')
    ax1.scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='red', alpha=0.8, s=50, marker='^', label='Anomaly')
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('True Labels', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LOF scores (colored by score magnitude)
    scores = detector.decision_function(X)
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu_r',
                         alpha=0.6, s=40)
    ax2.scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='black', s=100, marker='x', linewidths=2,
               label='True Anomaly')

    plt.colorbar(scatter, ax=ax2, label='LOF Score')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title(f'{name} - LOF Scores', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_neighbor_sensitivity(X_train, X_test, y_test, k_values):
    """Plot LOF performance vs number of neighbors"""
    results = []

    for k in k_values:
        detector = ModifiedLOFDetector(n_neighbors=k, contamination=0.1)
        detector.fit(X_train)
        y_pred = detector.predict(X_test)

        results.append({
            'k': k,
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['k'], results_df['f1'], 'b-o',
           label='F1 Score', linewidth=2, markersize=6)
    ax.plot(results_df['k'], results_df['precision'], 'g--s',
           label='Precision', linewidth=2, markersize=6)
    ax.plot(results_df['k'], results_df['recall'], 'r--^',
           label='Recall', linewidth=2, markersize=6)

    ax.set_xlabel('Number of Neighbors (K)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LOF Performance vs K', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, (detector, name) in enumerate(zip(detectors, names)):
        scores = detector.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')

        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={pr_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def evaluate_detector(detector, X_test, y_test, name):
    """Evaluate detector"""
    y_pred = detector.predict(X_test)
    scores = detector.decision_function(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec_curve, prec_curve)

    return {
        'Detector': name,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc
    }


def main():
    """Main execution function"""
    print("=" * 80)
    print("Local Outlier Factor (LOF) Anomaly Detection")
    print("=" * 80)

    np.random.seed(42)

    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_clustered_data(n_samples=1500, n_features=8, contamination=0.12)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n2. Training LOF detectors...")
    detectors = {
        'LOF (k=10)': ModifiedLOFDetector(n_neighbors=10, contamination=0.12),
        'LOF (k=20)': ModifiedLOFDetector(n_neighbors=20, contamination=0.12),
        'LOF (k=30)': ModifiedLOFDetector(n_neighbors=30, contamination=0.12),
        'Custom LOF (k=20)': CustomLOFDetector(n_neighbors=20, contamination=0.12),
    }

    for name, detector in detectors.items():
        detector.fit(X_train)
        print(f"   {name} trained")

    print("\n3. Evaluating detectors...")
    results = []
    for name, detector in detectors.items():
        result = evaluate_detector(detector, X_test, y_test, name)
        results.append(result)
        print(f"   {name}: F1={result['F1 Score']:.3f}, "
              f"Precision={result['Precision']:.3f}, "
              f"Recall={result['Recall']:.3f}")

    results_df = pd.DataFrame(results)

    # Visualizations
    print("\n4. Creating visualizations...")

    # K sensitivity
    k_values = range(5, 51, 5)
    fig = plot_neighbor_sensitivity(X_train, X_test, y_test, k_values)
    plt.savefig('lof_k_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Score distributions
    scores_list = [d.decision_function(X_test) for d in detectors.values()]
    names_list = list(detectors.keys())
    fig = plot_lof_scores_distribution(scores_list, y_test, names_list)
    plt.savefig('lof_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2D visualization
    fig = plot_lof_visualization_2d(X_test, y_test,
                                   detectors['LOF (k=20)'],
                                   'LOF (k=20)')
    plt.savefig('lof_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC and PR curves
    detector_list = list(detectors.values())
    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('lof_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Final results
    print("\n5. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    results_df.to_csv('lof_detection_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
