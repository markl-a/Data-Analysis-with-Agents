"""
Time Series Outlier Detection
Detects anomalies in time series data using statistical and decomposition methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_timeseries_data(n_samples=2000, anomaly_ratio=0.05):
    """Generate synthetic time series with various anomaly types"""
    n_anomalies = int(n_samples * anomaly_ratio)

    # Time index
    time = np.arange(n_samples)

    # Base signal: trend + seasonality + noise
    trend = 0.05 * time  # Linear trend
    seasonality = 10 * np.sin(2 * np.pi * time / 100) + 5 * np.sin(2 * np.pi * time / 30)
    noise = np.random.normal(0, 2, n_samples)

    signal = 50 + trend + seasonality + noise

    # Add anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    anomaly_types = np.random.choice(['spike', 'dip', 'level_shift', 'trend_change'], n_anomalies)

    labels = np.zeros(n_samples)

    for idx, anom_type in zip(anomaly_indices, anomaly_types):
        if anom_type == 'spike':
            # Sudden spike
            signal[idx] += np.random.uniform(20, 40)
            labels[idx] = 1
        elif anom_type == 'dip':
            # Sudden dip
            signal[idx] -= np.random.uniform(20, 40)
            labels[idx] = 1
        elif anom_type == 'level_shift':
            # Level shift for a period
            shift_length = min(20, n_samples - idx)
            signal[idx:idx+shift_length] += np.random.uniform(-15, 15)
            labels[idx:idx+shift_length] = 1
        else:  # trend_change
            # Temporary trend change
            change_length = min(30, n_samples - idx)
            trend_change = np.linspace(0, np.random.uniform(-20, 20), change_length)
            signal[idx:idx+change_length] += trend_change
            labels[idx:idx+change_length] = 1

    data = pd.DataFrame({
        'time': time,
        'value': signal,
        'label': labels
    })

    return data

def simple_seasonal_decompose(series, period=100):
    """Simple seasonal decomposition"""
    # Trend (moving average)
    trend = series.rolling(window=period, center=True).mean()

    # Detrend
    detrended = series - trend

    # Seasonal (average for each period position)
    seasonal = np.zeros(len(series))
    for i in range(period):
        indices = np.arange(i, len(series), period)
        seasonal[indices] = np.nanmean(detrended.iloc[indices])

    # Residual
    residual = series - trend - seasonal

    return trend, seasonal, residual

def plot_timeseries(data):
    """Visualize time series with anomalies"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle('Time Series with Anomalies', fontsize=16)

    # Full series
    ax = axes[0]
    normal_mask = data['label'] == 0
    anomaly_mask = data['label'] == 1

    ax.plot(data[normal_mask]['time'], data[normal_mask]['value'],
           'b-', alpha=0.6, label='Normal', linewidth=1)
    ax.scatter(data[anomaly_mask]['time'], data[anomaly_mask]['value'],
              c='red', marker='x', s=50, label='Anomaly', zorder=5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Original Time Series')

    # Zoomed section
    ax = axes[1]
    zoom_start, zoom_end = 500, 700
    zoom_data = data[(data['time'] >= zoom_start) & (data['time'] < zoom_end)]
    zoom_normal = zoom_data[zoom_data['label'] == 0]
    zoom_anomaly = zoom_data[zoom_data['label'] == 1]

    ax.plot(zoom_normal['time'], zoom_normal['value'],
           'b-', alpha=0.6, label='Normal', linewidth=1)
    ax.scatter(zoom_anomaly['time'], zoom_anomaly['value'],
              c='red', marker='x', s=100, label='Anomaly', zorder=5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f'Zoomed View (Time {zoom_start}-{zoom_end})')

    plt.tight_layout()
    plt.savefig('timeseries_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: timeseries_overview.png")

def plot_decomposition(data, trend, seasonal, residual):
    """Plot time series decomposition"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('Time Series Decomposition', fontsize=16)

    # Original
    axes[0].plot(data['time'], data['value'], 'b-', alpha=0.6, linewidth=1)
    axes[0].set_ylabel('Original')
    axes[0].grid(alpha=0.3)

    # Trend
    axes[1].plot(data['time'], trend, 'g-', alpha=0.8, linewidth=2)
    axes[1].set_ylabel('Trend')
    axes[1].grid(alpha=0.3)

    # Seasonal
    axes[2].plot(data['time'], seasonal, 'orange', alpha=0.8, linewidth=1)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(alpha=0.3)

    # Residual
    axes[3].plot(data['time'], residual, 'r-', alpha=0.6, linewidth=1)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('timeseries_decomposition.png', dpi=300, bbox_inches='tight')
    print("Saved: timeseries_decomposition.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def main():
    print("Time Series Outlier Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic time series data...")
    data = generate_timeseries_data(n_samples=2000, anomaly_ratio=0.05)
    print(f"Total samples: {len(data)}")
    print(f"Anomalies: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing time series...")
    plot_timeseries(data)

    y_true = data['label'].values
    metrics_dict = {}

    # Method 1: Statistical (Z-score on residuals)
    print("\n" + "="*60)
    print("Applying Statistical Method (Residual Z-Score)...")

    # Decompose
    trend, seasonal, residual = simple_seasonal_decompose(data['value'], period=100)

    # Z-score on residuals
    residual_filled = pd.Series(residual).fillna(0)
    z_scores = np.abs(stats.zscore(residual_filled, nan_policy='omit'))
    threshold_z = 3.0  # 3 standard deviations
    y_pred_stat = (z_scores > threshold_z).astype(int)

    metrics_dict['Statistical'] = evaluate_detector(y_true, y_pred_stat, "Statistical (Z-Score)")

    # Visualize decomposition
    plot_decomposition(data, trend, seasonal, residual_filled)

    # Method 2: Isolation Forest on features
    print("\n" + "="*60)
    print("Training Isolation Forest...")

    # Create features
    data['rolling_mean'] = data['value'].rolling(window=10, center=True).mean().fillna(data['value'])
    data['rolling_std'] = data['value'].rolling(window=10, center=True).std().fillna(0)
    data['diff'] = data['value'].diff().fillna(0)
    data['diff2'] = data['diff'].diff().fillna(0)

    feature_cols = ['value', 'rolling_mean', 'rolling_std', 'diff', 'diff2']
    X = data[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)

    metrics_dict['Isolation Forest'] = evaluate_detector(y_true, y_pred_if, "Isolation Forest")

    # Method 3: Moving Average + Threshold
    print("\n" + "="*60)
    print("Applying Moving Average Method...")

    window = 20
    ma = data['value'].rolling(window=window, center=True).mean()
    ma_std = data['value'].rolling(window=window, center=True).std()

    # Fill NaN values
    ma = ma.fillna(data['value'])
    ma_std = ma_std.fillna(data['value'].std())

    # Detect points outside 3*std from moving average
    deviation = np.abs(data['value'] - ma)
    threshold_ma = 3 * ma_std

    y_pred_ma = (deviation > threshold_ma).astype(int)

    metrics_dict['Moving Average'] = evaluate_detector(y_true, y_pred_ma, "Moving Average")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection Results Comparison', fontsize=16)

    methods = [
        ('Statistical', y_pred_stat),
        ('Isolation Forest', y_pred_if),
        ('Moving Average', y_pred_ma)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Performance comparison
    ax = axes[1, 1]
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: detection_comparison.png")

    # Visualize detections on time series
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('Detections on Time Series', fontsize=16)

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx]
        ax.plot(data['time'], data['value'], 'b-', alpha=0.3, linewidth=1, label='Signal')
        detected = data[y_pred == 1]
        ax.scatter(detected['time'], detected['value'],
                  c='red', marker='x', s=50, label='Detected Anomaly', zorder=5)
        true_anomalies = data[data['label'] == 1]
        ax.scatter(true_anomalies['time'], true_anomalies['value'],
                  c='orange', marker='o', s=30, alpha=0.5, label='True Anomaly', zorder=4)
        ax.set_ylabel('Value')
        ax.set_title(f'{name}')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('detection_overlay.png', dpi=300, bbox_inches='tight')
    print("Saved: detection_overlay.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- Statistical methods work well for simple seasonal patterns")
    print("- Isolation Forest captures complex multivariate patterns")
    print("- Moving average good for detecting level shifts and spikes")
    print("- Decomposition helps separate trend, seasonality, and anomalies")
    print("- Combine methods for robust detection")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
