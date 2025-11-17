"""
Interquartile Range (IQR) Method for Anomaly Detection
======================================================

This solution implements IQR-based anomaly detection:
1. Standard IQR method (1.5 * IQR)
2. Tukey's fences (inner and outer)
3. Adjusted boxplot method
4. Multi-feature IQR detection

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class IQRDetector:
    """Standard IQR anomaly detector"""

    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X):
        """Fit the detector by computing quartiles"""
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_bound_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.multiplier * self.iqr_

        return self

    def predict(self, X):
        """Predict anomalies - samples outside bounds"""
        below_lower = X < self.lower_bound_
        above_upper = X > self.upper_bound_

        # Anomaly if any feature is outside bounds
        anomalies = np.any(below_lower | above_upper, axis=1)
        return anomalies.astype(int)

    def decision_function(self, X):
        """Return anomaly scores (distance from bounds)"""
        # Distance from lower bound
        lower_dist = np.maximum(0, self.lower_bound_ - X)
        # Distance from upper bound
        upper_dist = np.maximum(0, X - self.upper_bound_)

        # Maximum distance across all features
        total_dist = lower_dist + upper_dist
        return np.max(total_dist / (self.iqr_ + 1e-10), axis=1)


class TukeyFencesDetector:
    """Tukey's fences detector with inner and outer fences"""

    def __init__(self, fence_type='inner'):
        """
        fence_type: 'inner' (1.5*IQR) or 'outer' (3*IQR)
        """
        self.fence_type = fence_type
        self.multiplier = 1.5 if fence_type == 'inner' else 3.0
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_fence_ = None
        self.upper_fence_ = None

    def fit(self, X):
        """Fit the detector"""
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_fence_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_fence_ = self.q3_ + self.multiplier * self.iqr_

        return self

    def predict(self, X):
        """Predict anomalies"""
        below_lower = X < self.lower_fence_
        above_upper = X > self.upper_fence_

        anomalies = np.any(below_lower | above_upper, axis=1)
        return anomalies.astype(int)

    def decision_function(self, X):
        """Return anomaly scores"""
        lower_dist = np.maximum(0, self.lower_fence_ - X)
        upper_dist = np.maximum(0, X - self.upper_fence_)

        total_dist = lower_dist + upper_dist
        return np.max(total_dist / (self.iqr_ + 1e-10), axis=1)


class AdjustedBoxplotDetector:
    """Adjusted boxplot for skewed distributions"""

    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.median_ = None
        self.q1_ = None
        self.q3_ = None
        self.mc_ = None  # Medcouple (skewness measure)
        self.lower_bound_ = None
        self.upper_bound_ = None

    def _medcouple(self, x):
        """Calculate medcouple (robust skewness measure)"""
        median = np.median(x)
        left = x[x <= median]
        right = x[x >= median]

        if len(left) == 0 or len(right) == 0:
            return 0.0

        # Simplified medcouple calculation
        h_values = []
        for xi in left:
            for xj in right:
                if xi != xj:
                    h = ((xj - median) - (median - xi)) / (xj - xi)
                    h_values.append(h)

        return np.median(h_values) if h_values else 0.0

    def fit(self, X):
        """Fit the detector"""
        self.median_ = np.median(X, axis=0)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_

        # Calculate medcouple for each feature
        self.mc_ = np.array([self._medcouple(X[:, i]) for i in range(X.shape[1])])

        # Adjust bounds based on skewness
        lower_adj = np.where(self.mc_ >= 0,
                            -self.multiplier * np.exp(-4 * self.mc_),
                            -self.multiplier * np.exp(-3 * self.mc_))

        upper_adj = np.where(self.mc_ >= 0,
                            self.multiplier * np.exp(3 * self.mc_),
                            self.multiplier * np.exp(4 * self.mc_))

        self.lower_bound_ = self.q1_ + lower_adj * iqr
        self.upper_bound_ = self.q3_ + upper_adj * iqr

        return self

    def predict(self, X):
        """Predict anomalies"""
        below_lower = X < self.lower_bound_
        above_upper = X > self.upper_bound_

        anomalies = np.any(below_lower | above_upper, axis=1)
        return anomalies.astype(int)

    def decision_function(self, X):
        """Return anomaly scores"""
        iqr = self.q3_ - self.q1_
        lower_dist = np.maximum(0, self.lower_bound_ - X)
        upper_dist = np.maximum(0, X - self.upper_bound_)

        total_dist = lower_dist + upper_dist
        return np.max(total_dist / (iqr + 1e-10), axis=1)


class MultiFeatureIQRDetector:
    """IQR detector with various aggregation methods"""

    def __init__(self, multiplier=1.5, aggregation='any'):
        """
        aggregation: 'any', 'all', 'count', 'score'
        """
        self.multiplier = multiplier
        self.aggregation = aggregation
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X):
        """Fit the detector"""
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_bound_ = self.q1_ - self.multiplier * self.iqr_
        self.upper_bound_ = self.q3_ + self.multiplier * self.iqr_

        return self

    def predict(self, X):
        """Predict anomalies based on aggregation method"""
        below_lower = X < self.lower_bound_
        above_upper = X > self.upper_bound_
        outliers = below_lower | above_upper

        if self.aggregation == 'any':
            return np.any(outliers, axis=1).astype(int)
        elif self.aggregation == 'all':
            return np.all(outliers, axis=1).astype(int)
        elif self.aggregation == 'count':
            count = np.sum(outliers, axis=1)
            threshold = X.shape[1] // 3  # At least 1/3 of features
            return (count >= threshold).astype(int)
        else:  # 'score'
            scores = self.decision_function(X)
            threshold = np.percentile(scores, 90)
            return (scores > threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores"""
        lower_dist = np.maximum(0, self.lower_bound_ - X)
        upper_dist = np.maximum(0, X - self.upper_bound_)

        total_dist = lower_dist + upper_dist
        normalized_dist = total_dist / (self.iqr_ + 1e-10)

        # Sum of normalized distances
        return np.sum(normalized_dist, axis=1)


def generate_anomaly_data(n_samples=1000, n_features=8, contamination=0.1):
    """Generate synthetic data with various anomaly types"""
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal

    # Normal data with some skewness
    X_normal = np.random.randn(n_normal, n_features)
    # Add skewness to some features
    for i in range(0, n_features, 2):
        X_normal[:, i] = np.exp(X_normal[:, i] * 0.5)

    # Anomalies
    anomaly_types = []
    n_per_type = n_anomalies // 3

    # Type 1: Extreme values
    X_extreme = np.random.randn(n_per_type, n_features) * 5 + 5
    anomaly_types.append(X_extreme)

    # Type 2: Negative extreme
    X_negative = np.random.randn(n_per_type, n_features) * 5 - 5
    anomaly_types.append(X_negative)

    # Type 3: Mixed
    remaining = n_anomalies - 2 * n_per_type
    X_mixed = np.random.uniform(-10, 10, (remaining, n_features))
    anomaly_types.append(X_mixed)

    X_anomalies = np.vstack(anomaly_types)

    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def tune_multiplier(detector_class, X_val, y_val, multipliers, **kwargs):
    """Tune IQR multiplier"""
    best_multiplier = multipliers[0]
    best_f1 = 0

    results = []
    for mult in multipliers:
        detector = detector_class(multiplier=mult, **kwargs)
        detector.fit(X_val)
        y_pred = detector.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        results.append({
            'multiplier': mult,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

        if f1 > best_f1:
            best_f1 = f1
            best_multiplier = mult

    return best_multiplier, pd.DataFrame(results)


def plot_boxplots(X, y, feature_names=None):
    """Plot boxplots for each feature"""
    n_features = X.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i in range(n_features):
        ax = axes[i]

        # Prepare data for boxplot
        normal_data = X[y == 0, i]
        anomaly_data = X[y == 1, i]

        bp = ax.boxplot([normal_data, anomaly_data],
                        labels=['Normal', 'Anomaly'],
                        patch_artist=True)

        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        feature_name = feature_names[i] if feature_names else f'Feature {i+1}'
        ax.set_title(feature_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_iqr_bounds(X, detector, feature_idx=0):
    """Visualize IQR bounds for a specific feature"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x_data = X[:, feature_idx]

    # Plot histogram
    ax.hist(x_data, bins=50, alpha=0.6, color='skyblue', edgecolor='black')

    # Plot bounds
    if hasattr(detector, 'lower_bound_'):
        lower = detector.lower_bound_[feature_idx]
        upper = detector.upper_bound_[feature_idx]

        ax.axvline(lower, color='red', linestyle='--', linewidth=2, label=f'Lower Bound: {lower:.2f}')
        ax.axvline(upper, color='red', linestyle='--', linewidth=2, label=f'Upper Bound: {upper:.2f}')

    if hasattr(detector, 'q1_'):
        ax.axvline(detector.q1_[feature_idx], color='green', linestyle=':',
                  linewidth=2, label=f'Q1: {detector.q1_[feature_idx]:.2f}')
        ax.axvline(detector.q3_[feature_idx], color='green', linestyle=':',
                  linewidth=2, label=f'Q3: {detector.q3_[feature_idx]:.2f}')

    if hasattr(detector, 'median_'):
        ax.axvline(detector.median_[feature_idx], color='blue', linestyle='-',
                  linewidth=2, label=f'Median: {detector.median_[feature_idx]:.2f}')

    ax.set_xlabel(f'Feature {feature_idx + 1} Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('IQR Bounds Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for i, (detector, name) in enumerate(zip(detectors, names)):
        scores = detector.decision_function(X_test)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={pr_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def evaluate_detector(detector, X_test, y_test, name):
    """Evaluate detector performance"""
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
    print("IQR Method for Anomaly Detection")
    print("=" * 80)

    np.random.seed(42)

    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_anomaly_data(n_samples=2000, n_features=8, contamination=0.12)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Train detectors
    print("\n2. Training IQR-based detectors...")

    # Tune multipliers
    multipliers = np.linspace(0.5, 3.5, 30)

    # Standard IQR
    best_mult_iqr, _ = tune_multiplier(IQRDetector, X_val, y_val, multipliers)
    print(f"   Standard IQR: Best multiplier = {best_mult_iqr:.2f}")

    # Tukey's fences
    best_mult_tukey_inner, _ = tune_multiplier(
        TukeyFencesDetector, X_val, y_val, multipliers, fence_type='inner'
    )
    best_mult_tukey_outer, _ = tune_multiplier(
        TukeyFencesDetector, X_val, y_val, multipliers, fence_type='outer'
    )
    print(f"   Tukey Inner: Best multiplier = {best_mult_tukey_inner:.2f}")
    print(f"   Tukey Outer: Best multiplier = {best_mult_tukey_outer:.2f}")

    # Create tuned detectors
    detectors = {
        'IQR Standard': IQRDetector(multiplier=best_mult_iqr),
        'Tukey Inner': TukeyFencesDetector(multiplier=best_mult_tukey_inner, fence_type='inner'),
        'Tukey Outer': TukeyFencesDetector(multiplier=best_mult_tukey_outer, fence_type='outer'),
        'Adjusted Boxplot': AdjustedBoxplotDetector(multiplier=1.5),
        'Multi-Feature (Any)': MultiFeatureIQRDetector(multiplier=1.5, aggregation='any'),
        'Multi-Feature (Score)': MultiFeatureIQRDetector(multiplier=1.5, aggregation='score')
    }

    # Fit all detectors
    for name, detector in detectors.items():
        detector.fit(X_train)

    # Evaluate
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

    # Boxplots
    fig = plot_boxplots(X_test, y_test)
    plt.savefig('iqr_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # IQR bounds
    for i in [0, 1]:
        fig = plot_iqr_bounds(X_test, detectors['IQR Standard'], feature_idx=i)
        plt.savefig(f'iqr_bounds_feature_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ROC and PR curves
    detector_list = list(detectors.values())
    names_list = list(detectors.keys())
    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('iqr_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Results
    print("\n5. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    results_df.to_csv('iqr_detection_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
