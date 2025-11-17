"""
Z-Score and Modified Z-Score Anomaly Detection
==============================================

This solution implements statistical anomaly detection using:
1. Standard Z-score method
2. Modified Z-score (using median absolute deviation)
3. Threshold tuning and optimization
4. Comprehensive performance evaluation with ROC and PR curves

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ZScoreDetector:
    """Standard Z-Score anomaly detector"""

    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Fit the detector by computing mean and std"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def predict(self, X):
        """Predict anomalies using Z-score threshold"""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        # Maximum z-score across features
        max_z = np.max(z_scores, axis=1)
        return (max_z > self.threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores (max z-score)"""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        return np.max(z_scores, axis=1)


class ModifiedZScoreDetector:
    """Modified Z-Score detector using Median Absolute Deviation (MAD)"""

    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.median_ = None
        self.mad_ = None

    def fit(self, X):
        """Fit the detector using median and MAD"""
        self.median_ = np.median(X, axis=0)
        # Median Absolute Deviation
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)
        return self

    def predict(self, X):
        """Predict anomalies using modified Z-score threshold"""
        # Modified Z-score = 0.6745 * (x - median) / MAD
        modified_z = 0.6745 * np.abs(X - self.median_) / (self.mad_ + 1e-10)
        max_modified_z = np.max(modified_z, axis=1)
        return (max_modified_z > self.threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores (max modified z-score)"""
        modified_z = 0.6745 * np.abs(X - self.median_) / (self.mad_ + 1e-10)
        return np.max(modified_z, axis=1)


class MultiFeatureZScoreDetector:
    """Z-Score detector that considers all features jointly"""

    def __init__(self, threshold=3.0, aggregation='max'):
        self.threshold = threshold
        self.aggregation = aggregation
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Fit the detector"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def predict(self, X):
        """Predict anomalies"""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def decision_function(self, X):
        """Return aggregated anomaly scores"""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))

        if self.aggregation == 'max':
            return np.max(z_scores, axis=1)
        elif self.aggregation == 'mean':
            return np.mean(z_scores, axis=1)
        elif self.aggregation == 'sum':
            return np.sum(z_scores, axis=1)
        else:
            return np.linalg.norm(z_scores, axis=1)


def generate_anomaly_data(n_samples=1000, n_features=10, contamination=0.1):
    """Generate synthetic data with anomalies"""
    # Normal data
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal

    # Generate normal samples from multivariate Gaussian
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    X_normal = np.random.multivariate_normal(mean, cov, n_normal)

    # Generate anomalies - samples from different distributions
    X_anomalies = []

    # Type 1: High magnitude anomalies
    n_type1 = n_anomalies // 3
    X_type1 = np.random.multivariate_normal(mean, cov * 5, n_type1)
    X_anomalies.append(X_type1)

    # Type 2: Shifted anomalies
    n_type2 = n_anomalies // 3
    shifted_mean = np.ones(n_features) * 4
    X_type2 = np.random.multivariate_normal(shifted_mean, cov, n_type2)
    X_anomalies.append(X_type2)

    # Type 3: Mixed anomalies
    n_type3 = n_anomalies - n_type1 - n_type2
    X_type3 = np.random.uniform(-5, 5, (n_type3, n_features))
    X_anomalies.append(X_type3)

    X_anomalies = np.vstack(X_anomalies)

    # Combine and create labels
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def inject_point_anomalies(X, contamination=0.05):
    """Inject point anomalies into clean data"""
    n_anomalies = int(len(X) * contamination)
    anomaly_indices = np.random.choice(len(X), n_anomalies, replace=False)

    X_modified = X.copy()
    y = np.zeros(len(X))

    for idx in anomaly_indices:
        # Randomly select anomaly type
        anomaly_type = np.random.choice(['scale', 'shift', 'noise'])

        if anomaly_type == 'scale':
            X_modified[idx] *= np.random.uniform(3, 5)
        elif anomaly_type == 'shift':
            X_modified[idx] += np.random.uniform(4, 6) * np.random.choice([-1, 1])
        else:
            X_modified[idx] = np.random.uniform(-5, 5, X.shape[1])

        y[idx] = 1

    return X_modified, y


def tune_threshold(detector, X_val, y_val, thresholds):
    """Tune threshold to maximize F1 score"""
    best_threshold = thresholds[0]
    best_f1 = 0

    scores = detector.decision_function(X_val)

    threshold_results = []
    for threshold in thresholds:
        y_pred = (scores > threshold).astype(int)
        f1 = f1_score(y_val, y_pred)

        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, pd.DataFrame(threshold_results)


def plot_threshold_tuning(results_df, title):
    """Plot threshold tuning results"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['threshold'], results_df['f1'], 'b-', label='F1 Score', linewidth=2)
    ax.plot(results_df['threshold'], results_df['precision'], 'g--', label='Precision')
    ax.plot(results_df['threshold'], results_df['recall'], 'r--', label='Recall')

    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']

    ax.axvline(best_threshold, color='orange', linestyle=':',
               label=f'Best Threshold: {best_threshold:.2f}')
    ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and Precision-Recall curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, (detector, name) in enumerate(zip(detectors, names)):
        scores = detector.decision_function(X_test)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')

    # ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # PR plot
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_anomaly_detection_2d(X, y_true, y_pred, title):
    """Visualize anomaly detection in 2D (using first 2 features)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True labels
    ax1.scatter(X[y_true == 0, 0], X[y_true == 0, 1],
               c='blue', alpha=0.6, s=30, label='Normal')
    ax1.scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='red', alpha=0.8, s=50, marker='^', label='True Anomaly')
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('True Labels', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Predicted labels
    correct_normal = (y_true == 0) & (y_pred == 0)
    correct_anomaly = (y_true == 1) & (y_pred == 1)
    false_positive = (y_true == 0) & (y_pred == 1)
    false_negative = (y_true == 1) & (y_pred == 0)

    ax2.scatter(X[correct_normal, 0], X[correct_normal, 1],
               c='blue', alpha=0.6, s=30, label='True Negative')
    ax2.scatter(X[correct_anomaly, 0], X[correct_anomaly, 1],
               c='green', alpha=0.8, s=50, marker='^', label='True Positive')
    ax2.scatter(X[false_positive, 0], X[false_positive, 1],
               c='orange', alpha=0.8, s=50, marker='x', label='False Positive')
    ax2.scatter(X[false_negative, 0], X[false_negative, 1],
               c='red', alpha=0.8, s=50, marker='s', label='False Negative')

    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title(f'{title} - Predictions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_score_distributions(scores_list, y_test, names):
    """Plot anomaly score distributions"""
    n_detectors = len(scores_list)
    fig, axes = plt.subplots(1, n_detectors, figsize=(6*n_detectors, 5))

    if n_detectors == 1:
        axes = [axes]

    for ax, scores, name in zip(axes, scores_list, names):
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]

        ax.hist(normal_scores, bins=50, alpha=0.7, color='blue', label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color='red', label='Anomaly', density=True)

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name} Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def evaluate_detector(detector, X_test, y_test, name):
    """Comprehensive evaluation of a detector"""
    y_pred = detector.predict(X_test)
    scores = detector.decision_function(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    # PR AUC
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
    print("Z-Score and Modified Z-Score Anomaly Detection")
    print("=" * 80)

    # Set random seed
    np.random.seed(42)

    # Generate synthetic data
    print("\n1. Generating synthetic anomaly data...")
    X, y = generate_anomaly_data(n_samples=2000, n_features=10, contamination=0.15)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"   Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Initialize detectors
    print("\n2. Training anomaly detectors...")
    detectors = {
        'Z-Score': ZScoreDetector(threshold=3.0),
        'Modified Z-Score': ModifiedZScoreDetector(threshold=3.5),
        'Z-Score (Mean Agg)': MultiFeatureZScoreDetector(threshold=2.5, aggregation='mean'),
        'Z-Score (Norm Agg)': MultiFeatureZScoreDetector(threshold=3.0, aggregation='norm')
    }

    # Train detectors
    for name, detector in detectors.items():
        detector.fit(X_train)
        print(f"   {name} trained")

    # Tune thresholds
    print("\n3. Tuning thresholds on validation set...")
    thresholds = np.linspace(1.0, 5.0, 50)

    tuned_detectors = {}
    threshold_results = {}

    for name, detector in detectors.items():
        best_threshold, results_df = tune_threshold(detector, X_val, y_val, thresholds)
        threshold_results[name] = results_df

        # Create new detector with best threshold
        if name == 'Z-Score':
            tuned_detectors[name] = ZScoreDetector(threshold=best_threshold)
        elif name == 'Modified Z-Score':
            tuned_detectors[name] = ModifiedZScoreDetector(threshold=best_threshold)
        elif 'Mean Agg' in name:
            tuned_detectors[name] = MultiFeatureZScoreDetector(
                threshold=best_threshold, aggregation='mean'
            )
        else:
            tuned_detectors[name] = MultiFeatureZScoreDetector(
                threshold=best_threshold, aggregation='norm'
            )

        tuned_detectors[name].fit(X_train)
        print(f"   {name}: Best threshold = {best_threshold:.3f}")

    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    results = []
    for name, detector in tuned_detectors.items():
        result = evaluate_detector(detector, X_test, y_test, name)
        results.append(result)
        print(f"   {name}: F1={result['F1 Score']:.3f}, "
              f"Precision={result['Precision']:.3f}, "
              f"Recall={result['Recall']:.3f}, "
              f"ROC-AUC={result['ROC AUC']:.3f}")

    results_df = pd.DataFrame(results)

    # Visualizations
    print("\n5. Creating visualizations...")

    # Threshold tuning plots
    for name, results_data in threshold_results.items():
        fig = plot_threshold_tuning(results_data, f'{name} - Threshold Tuning')
        plt.savefig(f'zscore_threshold_tuning_{name.replace(" ", "_").lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    # ROC and PR curves
    detector_list = list(tuned_detectors.values())
    names_list = list(tuned_detectors.keys())

    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('zscore_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Score distributions
    scores_list = [d.decision_function(X_test) for d in detector_list]
    fig = plot_score_distributions(scores_list, y_test, names_list)
    plt.savefig('zscore_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2D visualizations for each detector
    for name, detector in tuned_detectors.items():
        y_pred = detector.predict(X_test)
        fig = plot_anomaly_detection_2d(X_test, y_test, y_pred, name)
        plt.savefig(f'zscore_detection_2d_{name.replace(" ", "_").lower()}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Comparison metrics
    print("\n6. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    # Save results
    results_df.to_csv('zscore_detection_results.csv', index=False)
    print("\nResults saved to zscore_detection_results.csv")
    print("Visualizations saved to PNG files")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
