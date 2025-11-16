"""
Server Log Anomaly Detection
Detects anomalous server behavior from log data using clustering and statistical methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_server_logs(n_samples=25000, anomaly_ratio=0.04):
    """Generate synthetic server log data with anomalies"""
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal server behavior
    base_time = datetime(2024, 1, 1)
    timestamps_normal = [base_time + timedelta(seconds=i*10) for i in range(n_normal)]

    normal_response_time = np.random.lognormal(mean=np.log(100), sigma=0.5, size=n_normal)
    normal_cpu_usage = np.random.beta(2, 5, n_normal) * 100  # Typical usage 20-40%
    normal_memory_usage = np.random.beta(3, 4, n_normal) * 100  # Typical usage 40-60%
    normal_requests_per_min = np.random.poisson(lam=50, size=n_normal)
    normal_error_rate = np.random.beta(1, 50, n_normal)  # Very low error rate
    normal_connections = np.random.poisson(lam=100, size=n_normal)
    normal_disk_io = np.random.exponential(scale=50, size=n_normal)
    normal_network_io = np.random.exponential(scale=1000, size=n_normal)

    # HTTP status codes (normal)
    normal_status_200 = np.random.binomial(1, 0.95, n_normal)  # 95% success
    normal_status_404 = np.random.binomial(1, 0.03, n_normal)
    normal_status_500 = np.random.binomial(1, 0.02, n_normal)

    # Anomalous server behavior (different types)
    timestamps_anomaly = [base_time + timedelta(seconds=np.random.randint(0, n_normal*10)) for _ in range(n_anomalies)]

    anomaly_types = np.random.choice(['ddos', 'server_crash', 'memory_leak', 'slow_query'], n_anomalies)

    anomaly_response_time = np.zeros(n_anomalies)
    anomaly_cpu_usage = np.zeros(n_anomalies)
    anomaly_memory_usage = np.zeros(n_anomalies)
    anomaly_requests_per_min = np.zeros(n_anomalies)
    anomaly_error_rate = np.zeros(n_anomalies)
    anomaly_connections = np.zeros(n_anomalies)
    anomaly_disk_io = np.zeros(n_anomalies)
    anomaly_network_io = np.zeros(n_anomalies)
    anomaly_status_200 = np.zeros(n_anomalies)
    anomaly_status_404 = np.zeros(n_anomalies)
    anomaly_status_500 = np.zeros(n_anomalies)

    for i, anom_type in enumerate(anomaly_types):
        if anom_type == 'ddos':
            anomaly_response_time[i] = np.random.uniform(500, 5000)  # Very slow
            anomaly_cpu_usage[i] = np.random.uniform(80, 100)  # Maxed out
            anomaly_memory_usage[i] = np.random.uniform(70, 95)
            anomaly_requests_per_min[i] = np.random.poisson(lam=500)  # High volume
            anomaly_error_rate[i] = np.random.uniform(0.2, 0.5)  # High errors
            anomaly_connections[i] = np.random.poisson(lam=1000)  # Many connections
            anomaly_disk_io[i] = np.random.exponential(scale=100)
            anomaly_network_io[i] = np.random.exponential(scale=10000)  # High network
            anomaly_status_200[i] = np.random.binomial(1, 0.5)
            anomaly_status_404[i] = np.random.binomial(1, 0.1)
            anomaly_status_500[i] = np.random.binomial(1, 0.4)
        elif anom_type == 'server_crash':
            anomaly_response_time[i] = np.random.uniform(1000, 10000)  # Timeout
            anomaly_cpu_usage[i] = np.random.choice([5, 100])  # Either idle or stuck
            anomaly_memory_usage[i] = np.random.uniform(90, 100)  # Memory full
            anomaly_requests_per_min[i] = np.random.poisson(lam=10)  # Few requests
            anomaly_error_rate[i] = np.random.uniform(0.5, 1.0)  # Mostly errors
            anomaly_connections[i] = np.random.poisson(lam=10)
            anomaly_disk_io[i] = np.random.exponential(scale=30)
            anomaly_network_io[i] = np.random.exponential(scale=500)
            anomaly_status_200[i] = 0
            anomaly_status_404[i] = 0
            anomaly_status_500[i] = 1  # All server errors
        elif anom_type == 'memory_leak':
            anomaly_response_time[i] = np.random.lognormal(mean=np.log(300), sigma=0.8)
            anomaly_cpu_usage[i] = np.random.uniform(50, 80)
            anomaly_memory_usage[i] = np.random.uniform(85, 99)  # Memory leak
            anomaly_requests_per_min[i] = np.random.poisson(lam=40)
            anomaly_error_rate[i] = np.random.uniform(0.1, 0.3)
            anomaly_connections[i] = np.random.poisson(lam=80)
            anomaly_disk_io[i] = np.random.exponential(scale=70)
            anomaly_network_io[i] = np.random.exponential(scale=1200)
            anomaly_status_200[i] = np.random.binomial(1, 0.7)
            anomaly_status_404[i] = np.random.binomial(1, 0.05)
            anomaly_status_500[i] = np.random.binomial(1, 0.25)
        else:  # slow_query
            anomaly_response_time[i] = np.random.uniform(1000, 3000)  # Slow responses
            anomaly_cpu_usage[i] = np.random.uniform(60, 90)  # High CPU
            anomaly_memory_usage[i] = np.random.uniform(50, 70)
            anomaly_requests_per_min[i] = np.random.poisson(lam=30)  # Normal-ish
            anomaly_error_rate[i] = np.random.uniform(0.05, 0.15)
            anomaly_connections[i] = np.random.poisson(lam=120)
            anomaly_disk_io[i] = np.random.exponential(scale=200)  # High disk I/O
            anomaly_network_io[i] = np.random.exponential(scale=800)
            anomaly_status_200[i] = np.random.binomial(1, 0.85)
            anomaly_status_404[i] = np.random.binomial(1, 0.05)
            anomaly_status_500[i] = np.random.binomial(1, 0.1)

    # Combine data
    data = pd.DataFrame({
        'timestamp': timestamps_normal + timestamps_anomaly,
        'response_time_ms': np.concatenate([normal_response_time, anomaly_response_time]),
        'cpu_usage_pct': np.concatenate([normal_cpu_usage, anomaly_cpu_usage]),
        'memory_usage_pct': np.concatenate([normal_memory_usage, anomaly_memory_usage]),
        'requests_per_minute': np.concatenate([normal_requests_per_min, anomaly_requests_per_min]),
        'error_rate': np.concatenate([normal_error_rate, anomaly_error_rate]),
        'active_connections': np.concatenate([normal_connections, anomaly_connections]),
        'disk_io_mbps': np.concatenate([normal_disk_io, anomaly_disk_io]),
        'network_io_mbps': np.concatenate([normal_network_io, anomaly_network_io]),
        'http_200': np.concatenate([normal_status_200, anomaly_status_200]),
        'http_404': np.concatenate([normal_status_404, anomaly_status_404]),
        'http_500': np.concatenate([normal_status_500, anomaly_status_500]),
        'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    })

    # Add derived features
    data['cpu_memory_product'] = data['cpu_usage_pct'] * data['memory_usage_pct']
    data['requests_per_connection'] = data['requests_per_minute'] / (data['active_connections'] + 1)
    data['response_time_per_request'] = data['response_time_ms'] / (data['requests_per_minute'] + 1)

    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    return data

def plot_metrics_over_time(data):
    """Plot server metrics over time"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Server Metrics Over Time', fontsize=16)

    metrics = [
        ('response_time_ms', 'Response Time (ms)'),
        ('cpu_usage_pct', 'CPU Usage (%)'),
        ('memory_usage_pct', 'Memory Usage (%)'),
        ('requests_per_minute', 'Requests/Minute'),
        ('error_rate', 'Error Rate'),
        ('active_connections', 'Active Connections')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Plot normal and anomaly separately
        normal_mask = data['label'] == 0
        anomaly_mask = data['label'] == 1

        ax.scatter(range(len(data[normal_mask])), data[normal_mask][metric],
                  alpha=0.3, s=5, c='blue', label='Normal')
        ax.scatter(np.where(anomaly_mask)[0], data[anomaly_mask][metric],
                  alpha=0.8, s=20, c='red', label='Anomaly', marker='x')

        ax.set_xlabel('Time Index')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('server_metrics_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved: server_metrics_timeline.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

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
    print("Server Log Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic server log data...")
    data = generate_server_logs(n_samples=25000, anomaly_ratio=0.04)
    print(f"Total log entries: {len(data)}")
    print(f"Anomalous entries: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Show sample
    print("\nSample log entries:")
    print(data.head(10))

    # Visualize
    print("\nVisualizing metrics over time...")
    plot_metrics_over_time(data)

    # Prepare features
    feature_cols = ['response_time_ms', 'cpu_usage_pct', 'memory_usage_pct',
                   'requests_per_minute', 'error_rate', 'active_connections',
                   'disk_io_mbps', 'network_io_mbps', 'http_200', 'http_404', 'http_500',
                   'cpu_memory_product', 'requests_per_connection', 'response_time_per_request']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Method 1: DBSCAN (Density-based clustering)
    print("\n" + "="*60)
    print("Applying DBSCAN Clustering...")
    dbscan = DBSCAN(eps=3, min_samples=50)
    clusters = dbscan.fit_predict(X_scaled)
    y_pred_dbscan = (clusters == -1).astype(int)  # -1 are outliers
    metrics_dict['DBSCAN'] = evaluate_detector(y_true, y_pred_dbscan, "DBSCAN")

    # Method 2: Isolation Forest
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.04, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)
    metrics_dict['Isolation Forest'] = evaluate_detector(y_true, y_pred_if, "Isolation Forest")

    # Method 3: Statistical (Multiple metrics threshold)
    print("\n" + "="*60)
    print("Applying Statistical Method...")

    # Define thresholds for key metrics
    response_threshold = np.percentile(data['response_time_ms'], 95)
    cpu_threshold = np.percentile(data['cpu_usage_pct'], 90)
    error_threshold = np.percentile(data['error_rate'], 95)

    y_pred_stat = (
        (data['response_time_ms'] > response_threshold) |
        (data['cpu_usage_pct'] > cpu_threshold) |
        (data['error_rate'] > error_threshold)
    ).astype(int)

    metrics_dict['Statistical'] = evaluate_detector(y_true, y_pred_stat, "Statistical Threshold")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Results Comparison', fontsize=16)

    # Confusion matrices
    methods = [('DBSCAN', y_pred_dbscan), ('Isolation Forest', y_pred_if), ('Statistical', y_pred_stat)]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Performance comparison
    ax = axes[1, 1]
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xlabel('Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: detection_results.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- DBSCAN identifies density-based outliers effectively")
    print("- Isolation Forest is fast and robust")
    print("- Statistical methods provide interpretable thresholds")
    print("- Use ensemble voting for production systems")
    print("- Set up real-time alerting for detected anomalies")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
