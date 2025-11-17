"""
GESD (Generalized Extreme Studentized Deviate) Anomaly Detection
=================================================================

This solution implements GESD test for outlier detection:
1. Standard GESD test
2. Seasonal GESD for time series
3. Multi-feature GESD
4. Comparison with other methods

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


class GESDDetector:
    """Generalized Extreme Studentized Deviate (GESD) detector"""

    def __init__(self, max_outliers=10, alpha=0.05):
        """
        Parameters:
        - max_outliers: Maximum number of outliers to detect
        - alpha: Significance level
        """
        self.max_outliers = max_outliers
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None
        self.outlier_indices_ = []
        self.critical_values_ = []

    def fit(self, X):
        """Fit the detector"""
        # For univariate, flatten
        if X.ndim > 1:
            X_flat = X.flatten()
        else:
            X_flat = X.copy()

        self.mean_ = np.mean(X_flat)
        self.std_ = np.std(X_flat)

        # Perform GESD test
        self.outlier_indices_ = []
        self.critical_values_ = []

        data = X_flat.copy()
        n = len(data)

        for i in range(min(self.max_outliers, n - 2)):
            # Calculate test statistic
            mean = np.mean(data)
            std = np.std(data)

            # Find maximum deviation
            deviations = np.abs(data - mean)
            max_idx = np.argmax(deviations)
            max_dev = deviations[max_idx]

            # Test statistic
            R = max_dev / std if std > 0 else 0

            # Critical value using t-distribution
            n_current = len(data)
            t_crit = stats.t.ppf(1 - self.alpha / (2 * n_current), n_current - 2)
            lambda_crit = ((n_current - 1) * t_crit) / np.sqrt(
                (n_current - 2 + t_crit**2) * n_current
            )

            self.critical_values_.append(lambda_crit)

            # Check if outlier
            if R > lambda_crit:
                self.outlier_indices_.append(max_idx)
                # Remove the outlier and continue
                data = np.delete(data, max_idx)
            else:
                break

        return self

    def predict(self, X):
        """Predict outliers"""
        if X.ndim > 1:
            X_flat = X.flatten()
        else:
            X_flat = X.copy()

        # Calculate deviations
        deviations = np.abs(X_flat - self.mean_) / (self.std_ + 1e-10)

        # Use the last critical value as threshold
        threshold = self.critical_values_[-1] if self.critical_values_ else 3.0

        return (deviations > threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores"""
        if X.ndim > 1:
            X_flat = X.flatten()
        else:
            X_flat = X.copy()

        return np.abs(X_flat - self.mean_) / (self.std_ + 1e-10)


class SeasonalGESDDetector:
    """GESD for seasonal time series data"""

    def __init__(self, max_outliers=10, alpha=0.05, period=12):
        self.max_outliers = max_outliers
        self.alpha = alpha
        self.period = period
        self.trend_ = None
        self.seasonal_ = None
        self.residual_ = None
        self.outlier_indices_ = []

    def _decompose(self, X):
        """Simple seasonal decomposition"""
        n = len(X)

        # Trend (moving average)
        window = self.period
        trend = np.zeros(n)
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            trend[i] = np.mean(X[start:end])

        # Detrended
        detrended = X - trend

        # Seasonal component
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            seasonal[indices] = np.mean(detrended[indices])

        # Residual
        residual = X - trend - seasonal

        return trend, seasonal, residual

    def fit(self, X):
        """Fit the detector"""
        # Decompose
        self.trend_, self.seasonal_, self.residual_ = self._decompose(X)

        # Apply GESD to residuals
        gesd = GESDDetector(max_outliers=self.max_outliers, alpha=self.alpha)
        gesd.fit(self.residual_)

        self.outlier_indices_ = gesd.outlier_indices_

        return self

    def predict(self, X):
        """Predict outliers"""
        _, _, residual = self._decompose(X)

        # Use std of training residuals
        std_residual = np.std(self.residual_)
        threshold = 3.0 * std_residual

        return (np.abs(residual) > threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores"""
        _, _, residual = self._decompose(X)
        std_residual = np.std(self.residual_)
        return np.abs(residual) / (std_residual + 1e-10)


class MultiFeatureGESDDetector:
    """GESD for multivariate data"""

    def __init__(self, max_outliers=10, alpha=0.05):
        self.max_outliers = max_outliers
        self.alpha = alpha
        self.detectors_ = []

    def fit(self, X):
        """Fit detector for each feature"""
        n_features = X.shape[1]
        self.detectors_ = []

        for i in range(n_features):
            detector = GESDDetector(max_outliers=self.max_outliers, alpha=self.alpha)
            detector.fit(X[:, i])
            self.detectors_.append(detector)

        return self

    def predict(self, X):
        """Predict outliers - anomaly if any feature is outlier"""
        predictions = np.zeros((X.shape[0], X.shape[1]))

        for i, detector in enumerate(self.detectors_):
            predictions[:, i] = detector.predict(X[:, i])

        # Any feature is outlier
        return np.any(predictions, axis=1).astype(int)

    def decision_function(self, X):
        """Return max anomaly score across features"""
        scores = np.zeros((X.shape[0], X.shape[1]))

        for i, detector in enumerate(self.detectors_):
            scores[:, i] = detector.decision_function(X[:, i])

        return np.max(scores, axis=1)


def generate_time_series_anomalies(n_samples=500, n_anomalies=20):
    """Generate time series with anomalies"""
    t = np.arange(n_samples)

    # Trend
    trend = 0.05 * t

    # Seasonal component
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)

    # Noise
    noise = np.random.randn(n_samples) * 2

    # Clean signal
    signal = trend + seasonal + noise

    # Inject anomalies
    y = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'dip', 'shift'])

        if anomaly_type == 'spike':
            signal[idx] += np.random.uniform(20, 40)
        elif anomaly_type == 'dip':
            signal[idx] -= np.random.uniform(20, 40)
        else:  # shift
            shift_len = min(10, n_samples - idx)
            signal[idx:idx+shift_len] += np.random.uniform(15, 25)

        y[idx] = 1

    return signal, y


def generate_multivariate_anomalies(n_samples=1000, n_features=8, contamination=0.1):
    """Generate multivariate data with anomalies"""
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal

    # Normal data
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    X_normal = np.random.multivariate_normal(mean, cov, n_normal)

    # Anomalies
    X_anomalies = []

    # Type 1: Extreme values
    X_extreme = np.random.multivariate_normal(mean, cov * 10, n_anomalies // 2)
    X_anomalies.append(X_extreme)

    # Type 2: Shifted values
    shifted_mean = np.ones(n_features) * 5
    X_shifted = np.random.multivariate_normal(shifted_mean, cov, n_anomalies - n_anomalies // 2)
    X_anomalies.append(X_shifted)

    X_anomalies = np.vstack(X_anomalies)

    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def plot_gesd_process(data, outlier_indices, critical_values):
    """Visualize GESD detection process"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Data with outliers marked
    ax1.plot(data, 'b-', alpha=0.6, label='Data')
    ax1.scatter(outlier_indices, data[outlier_indices],
               c='red', s=100, marker='x', linewidth=3,
               label=f'Detected Outliers ({len(outlier_indices)})')

    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('GESD Outlier Detection Results', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Critical values
    iterations = np.arange(1, len(critical_values) + 1)
    ax2.plot(iterations, critical_values, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Critical Value (λ)', fontsize=12)
    ax2.set_title('GESD Critical Values', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_seasonal_decomposition(t, original, trend, seasonal, residual, anomalies):
    """Plot seasonal decomposition with anomalies"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Original
    axes[0].plot(t, original, 'b-', alpha=0.7)
    axes[0].scatter(t[anomalies == 1], original[anomalies == 1],
                   c='red', s=50, marker='x', linewidth=2)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].set_title('Seasonal Decomposition with Anomalies',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(t, trend, 'g-', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(t, seasonal, 'orange', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(t, residual, 'purple', alpha=0.7)
    axes[3].scatter(t[anomalies == 1], residual[anomalies == 1],
                   c='red', s=50, marker='x', linewidth=2)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

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
    print("GESD (Generalized Extreme Studentized Deviate) Anomaly Detection")
    print("=" * 80)

    np.random.seed(42)

    # Part 1: Time series analysis
    print("\n" + "="*80)
    print("PART 1: Time Series Anomaly Detection")
    print("="*80)

    print("\n1. Generating time series data...")
    ts_data, ts_labels = generate_time_series_anomalies(n_samples=500, n_anomalies=25)
    print(f"   Time series length: {len(ts_data)}")
    print(f"   Number of anomalies: {ts_labels.sum()}")

    print("\n2. Applying GESD to time series...")
    gesd = GESDDetector(max_outliers=30, alpha=0.05)
    gesd.fit(ts_data)
    print(f"   Detected {len(gesd.outlier_indices_)} outliers")

    # Plot GESD process
    print("\n3. Visualizing GESD detection...")
    fig = plot_gesd_process(ts_data, gesd.outlier_indices_, gesd.critical_values_)
    plt.savefig('gesd_time_series_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Seasonal GESD
    print("\n4. Applying Seasonal GESD...")
    seasonal_gesd = SeasonalGESDDetector(max_outliers=30, alpha=0.05, period=50)
    seasonal_gesd.fit(ts_data)

    t = np.arange(len(ts_data))
    y_pred = seasonal_gesd.predict(ts_data)

    fig = plot_seasonal_decomposition(
        t, ts_data, seasonal_gesd.trend_,
        seasonal_gesd.seasonal_, seasonal_gesd.residual_,
        y_pred
    )
    plt.savefig('gesd_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate time series detection
    y_pred_gesd = gesd.predict(ts_data)
    y_pred_seasonal = seasonal_gesd.predict(ts_data)

    print("\n5. Time Series Results:")
    print(f"   GESD - F1: {f1_score(ts_labels, y_pred_gesd):.3f}, "
          f"Precision: {precision_score(ts_labels, y_pred_gesd):.3f}, "
          f"Recall: {recall_score(ts_labels, y_pred_gesd):.3f}")
    print(f"   Seasonal GESD - F1: {f1_score(ts_labels, y_pred_seasonal):.3f}, "
          f"Precision: {precision_score(ts_labels, y_pred_seasonal):.3f}, "
          f"Recall: {recall_score(ts_labels, y_pred_seasonal):.3f}")

    # Part 2: Multivariate analysis
    print("\n" + "="*80)
    print("PART 2: Multivariate Anomaly Detection")
    print("="*80)

    print("\n1. Generating multivariate data...")
    X, y = generate_multivariate_anomalies(n_samples=1500, n_features=8, contamination=0.12)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n2. Training multivariate detectors...")
    detectors = {
        'Multi-Feature GESD': MultiFeatureGESDDetector(max_outliers=50, alpha=0.05),
        'GESD (alpha=0.01)': MultiFeatureGESDDetector(max_outliers=50, alpha=0.01),
        'GESD (alpha=0.10)': MultiFeatureGESDDetector(max_outliers=50, alpha=0.10)
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

    # ROC and PR curves
    print("\n4. Creating visualizations...")
    detector_list = list(detectors.values())
    names_list = list(detectors.keys())

    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('gesd_multivariate_roc_pr.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Final results
    print("\n5. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    results_df.to_csv('gesd_detection_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
