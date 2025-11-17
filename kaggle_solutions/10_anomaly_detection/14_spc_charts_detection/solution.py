"""
Statistical Process Control (SPC) Charts for Anomaly Detection
==============================================================

This solution implements SPC charts for anomaly detection:
1. Shewhart control charts (X-bar, R, S charts)
2. CUSUM (Cumulative Sum) charts
3. EWMA (Exponentially Weighted Moving Average) charts
4. Multivariate control charts (Hotelling T²)

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ShewhartControlChart:
    """Shewhart Control Chart (X-bar and R chart)"""

    def __init__(self, n_sigma=3.0):
        self.n_sigma = n_sigma
        self.mean_ = None
        self.std_ = None
        self.ucl_ = None  # Upper Control Limit
        self.lcl_ = None  # Lower Control Limit

    def fit(self, X):
        """Fit control limits"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.ucl_ = self.mean_ + self.n_sigma * self.std_
        self.lcl_ = self.mean_ - self.n_sigma * self.std_

        return self

    def predict(self, X):
        """Detect out-of-control points"""
        above_ucl = X > self.ucl_
        below_lcl = X < self.lcl_

        # Anomaly if any feature is out of control
        return np.any(above_ucl | below_lcl, axis=1).astype(int)

    def decision_function(self, X):
        """Return distance from control limits"""
        # Distance to nearest control limit
        dist_to_ucl = np.maximum(0, X - self.ucl_)
        dist_to_lcl = np.maximum(0, self.lcl_ - X)

        total_dist = dist_to_ucl + dist_to_lcl
        return np.max(total_dist / (self.std_ + 1e-10), axis=1)


class CUSUMChart:
    """CUSUM (Cumulative Sum) Control Chart"""

    def __init__(self, k=0.5, h=4.0):
        """
        k: slack parameter (typically 0.5 * sigma)
        h: decision interval (typically 4-5 * sigma)
        """
        self.k = k
        self.h = h
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Fit parameters"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def predict(self, X):
        """Detect anomalies using CUSUM"""
        scores = self._compute_cusum(X)
        threshold = self.h * self.std_

        return np.any(scores > threshold, axis=1).astype(int)

    def decision_function(self, X):
        """Return CUSUM scores"""
        scores = self._compute_cusum(X)
        return np.max(scores / (self.std_ + 1e-10), axis=1)

    def _compute_cusum(self, X):
        """Compute CUSUM statistics"""
        n_samples, n_features = X.shape
        cusum_pos = np.zeros((n_samples, n_features))
        cusum_neg = np.zeros((n_samples, n_features))

        k_threshold = self.k * self.std_

        for i in range(n_samples):
            if i == 0:
                cusum_pos[i] = np.maximum(0, X[i] - self.mean_ - k_threshold)
                cusum_neg[i] = np.maximum(0, self.mean_ - X[i] - k_threshold)
            else:
                cusum_pos[i] = np.maximum(0, cusum_pos[i-1] + X[i] - self.mean_ - k_threshold)
                cusum_neg[i] = np.maximum(0, cusum_neg[i-1] + self.mean_ - X[i] - k_threshold)

        return np.maximum(cusum_pos, cusum_neg)


class EWMAChart:
    """EWMA (Exponentially Weighted Moving Average) Chart"""

    def __init__(self, lambda_=0.2, L=3.0):
        """
        lambda_: smoothing parameter (0 < lambda <= 1)
        L: width of control limits (typically 2.6-3.0)
        """
        self.lambda_ = lambda_
        self.L = L
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """Fit parameters"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def predict(self, X):
        """Detect anomalies using EWMA"""
        ewma = self._compute_ewma(X)

        # Control limits for EWMA
        n = X.shape[0]
        sigma_ewma = self.std_ * np.sqrt(
            (self.lambda_ / (2 - self.lambda_)) *
            (1 - (1 - self.lambda_)**(2 * np.arange(1, n+1)))
        )[:, np.newaxis]

        ucl = self.mean_ + self.L * sigma_ewma
        lcl = self.mean_ - self.L * sigma_ewma

        above_ucl = ewma > ucl
        below_lcl = ewma < lcl

        return np.any(above_ucl | below_lcl, axis=1).astype(int)

    def decision_function(self, X):
        """Return EWMA deviations"""
        ewma = self._compute_ewma(X)

        # Normalized deviation from mean
        deviation = np.abs(ewma - self.mean_) / (self.std_ + 1e-10)
        return np.max(deviation, axis=1)

    def _compute_ewma(self, X):
        """Compute EWMA values"""
        n_samples, n_features = X.shape
        ewma = np.zeros_like(X)

        ewma[0] = X[0]
        for i in range(1, n_samples):
            ewma[i] = self.lambda_ * X[i] + (1 - self.lambda_) * ewma[i-1]

        return ewma


class HotellingT2Chart:
    """Hotelling T² multivariate control chart"""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.ucl_ = None

    def fit(self, X):
        """Fit multivariate parameters"""
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)

        # Add small regularization for numerical stability
        self.cov_ += np.eye(X.shape[1]) * 1e-6

        self.inv_cov_ = np.linalg.inv(self.cov_)

        # Control limit based on F-distribution
        n, p = X.shape
        self.ucl_ = ((p * (n - 1) * (n + 1)) / (n * (n - p))) * stats.f.ppf(
            1 - self.alpha, p, n - p
        )

        return self

    def predict(self, X):
        """Detect anomalies"""
        t2_scores = self._compute_t2(X)
        return (t2_scores > self.ucl_).astype(int)

    def decision_function(self, X):
        """Return T² scores"""
        return self._compute_t2(X)

    def _compute_t2(self, X):
        """Compute Hotelling T² statistic"""
        diff = X - self.mean_
        t2 = np.sum((diff @ self.inv_cov_) * diff, axis=1)
        return t2


def generate_process_data(n_samples=1000, n_features=6, shift_points=None, contamination=0.1):
    """Generate process data with potential shifts and anomalies"""
    # Normal process
    mean = np.zeros(n_features)
    cov = np.eye(n_features)

    X = np.random.multivariate_normal(mean, cov, n_samples)
    y = np.zeros(n_samples)

    # Add process shifts
    if shift_points is None:
        shift_points = [n_samples // 3, 2 * n_samples // 3]

    for shift in shift_points:
        if shift < n_samples:
            # Mean shift
            shift_magnitude = np.random.uniform(0.5, 1.5, n_features)
            X[shift:shift+50] += shift_magnitude
            y[shift:shift+50] = 1

    # Add random anomalies
    n_anomalies = int(n_samples * contamination)
    anomaly_indices = np.random.choice(
        [i for i in range(n_samples) if y[i] == 0],
        min(n_anomalies, np.sum(y == 0)),
        replace=False
    )

    for idx in anomaly_indices:
        X[idx] += np.random.uniform(3, 5, n_features) * np.random.choice([-1, 1])
        y[idx] = 1

    return X, y


def plot_control_chart(X, y_true, y_pred, detector, feature_idx=0, title="Control Chart"):
    """Plot control chart for a specific feature"""
    fig, ax = plt.subplots(figsize=(14, 6))

    indices = np.arange(len(X))
    feature_data = X[:, feature_idx]

    # Plot data
    ax.plot(indices, feature_data, 'b-', alpha=0.5, linewidth=1, label='Process Data')

    # Mark true anomalies
    ax.scatter(indices[y_true == 1], feature_data[y_true == 1],
              c='red', s=50, marker='x', linewidth=2, label='True Anomalies', zorder=5)

    # Mark detected anomalies
    false_positives = (y_true == 0) & (y_pred == 1)
    if np.any(false_positives):
        ax.scatter(indices[false_positives], feature_data[false_positives],
                  c='orange', s=30, marker='o', alpha=0.6,
                  label='False Positives', zorder=4)

    # Control limits
    if hasattr(detector, 'mean_'):
        ax.axhline(detector.mean_[feature_idx], color='green',
                  linestyle='-', linewidth=2, label='Center Line')

    if hasattr(detector, 'ucl_'):
        ax.axhline(detector.ucl_[feature_idx], color='red',
                  linestyle='--', linewidth=2, label='UCL')
        ax.axhline(detector.lcl_[feature_idx], color='red',
                  linestyle='--', linewidth=2, label='LCL')

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel(f'Feature {feature_idx + 1}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cusum_chart(X, detector, feature_idx=0):
    """Plot CUSUM chart"""
    cusum_scores = detector._compute_cusum(X)

    fig, ax = plt.subplots(figsize=(14, 6))

    indices = np.arange(len(X))
    ax.plot(indices, cusum_scores[:, feature_idx], 'b-', linewidth=2, label='CUSUM')

    # Decision interval
    h_line = detector.h * detector.std_[feature_idx]
    ax.axhline(h_line, color='red', linestyle='--', linewidth=2, label=f'H = {detector.h}σ')
    ax.axhline(-h_line, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('CUSUM Value', fontsize=12)
    ax.set_title(f'CUSUM Chart - Feature {feature_idx + 1}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ewma_chart(X, detector, feature_idx=0):
    """Plot EWMA chart"""
    ewma = detector._compute_ewma(X)

    fig, ax = plt.subplots(figsize=(14, 6))

    indices = np.arange(len(X))
    ax.plot(indices, X[:, feature_idx], 'lightblue', alpha=0.5, label='Original Data')
    ax.plot(indices, ewma[:, feature_idx], 'b-', linewidth=2, label=f'EWMA (λ={detector.lambda_})')

    # Center line
    ax.axhline(detector.mean_[feature_idx], color='green',
              linestyle='-', linewidth=2, label='Center Line')

    # Control limits (simplified)
    sigma_ewma = detector.std_[feature_idx] * np.sqrt(
        detector.lambda_ / (2 - detector.lambda_)
    )
    ucl = detector.mean_[feature_idx] + detector.L * sigma_ewma
    lcl = detector.mean_[feature_idx] - detector.L * sigma_ewma

    ax.axhline(ucl, color='red', linestyle='--', linewidth=2, label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', linewidth=2, label='LCL')

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel(f'Feature {feature_idx + 1}', fontsize=12)
    ax.set_title('EWMA Chart', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_hotelling_t2(X, y_true, detector):
    """Plot Hotelling T² chart"""
    t2_scores = detector.decision_function(X)

    fig, ax = plt.subplots(figsize=(14, 6))

    indices = np.arange(len(X))
    ax.plot(indices, t2_scores, 'b-', linewidth=1, label='T² Statistic')

    # Mark anomalies
    ax.scatter(indices[y_true == 1], t2_scores[y_true == 1],
              c='red', s=50, marker='x', linewidth=2, label='True Anomalies')

    # UCL
    ax.axhline(detector.ucl_, color='red', linestyle='--',
              linewidth=2, label=f'UCL = {detector.ucl_:.2f}')

    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('T² Statistic', fontsize=12)
    ax.set_title("Hotelling T² Chart", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

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
    print("Statistical Process Control (SPC) Charts for Anomaly Detection")
    print("=" * 80)

    np.random.seed(42)

    # Generate process data
    print("\n1. Generating process data...")
    X, y = generate_process_data(n_samples=1000, n_features=6, contamination=0.08)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n2. Training SPC detectors...")
    detectors = {
        'Shewhart (3σ)': ShewhartControlChart(n_sigma=3.0),
        'Shewhart (2σ)': ShewhartControlChart(n_sigma=2.0),
        'CUSUM': CUSUMChart(k=0.5, h=4.0),
        'EWMA (λ=0.2)': EWMAChart(lambda_=0.2, L=3.0),
        'EWMA (λ=0.3)': EWMAChart(lambda_=0.3, L=3.0),
        'Hotelling T²': HotellingT2Chart(alpha=0.05)
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

    # Shewhart chart
    y_pred = detectors['Shewhart (3σ)'].predict(X_test)
    fig = plot_control_chart(X_test, y_test, y_pred,
                            detectors['Shewhart (3σ)'],
                            feature_idx=0,
                            title="Shewhart Control Chart (3σ)")
    plt.savefig('spc_shewhart_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # CUSUM chart
    fig = plot_cusum_chart(X_test, detectors['CUSUM'], feature_idx=0)
    plt.savefig('spc_cusum_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # EWMA chart
    fig = plot_ewma_chart(X_test, detectors['EWMA (λ=0.2)'], feature_idx=0)
    plt.savefig('spc_ewma_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Hotelling T² chart
    fig = plot_hotelling_t2(X_test, y_test, detectors['Hotelling T²'])
    plt.savefig('spc_hotelling_t2_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC and PR curves
    detector_list = list(detectors.values())
    names_list = list(detectors.keys())
    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('spc_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Final results
    print("\n5. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    results_df.to_csv('spc_detection_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
