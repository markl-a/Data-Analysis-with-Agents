"""
Network Intrusion Detection System
Detects anomalous network traffic using multiple anomaly detection methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_network_traffic_data(n_samples=15000, attack_ratio=0.05):
    """Generate synthetic network traffic data with various attack types"""
    n_attacks = int(n_samples * attack_ratio)
    n_normal = n_samples - n_attacks

    # Normal traffic patterns
    normal_packet_size = np.random.normal(500, 150, n_normal)
    normal_duration = np.random.exponential(scale=5, size=n_normal)
    normal_src_bytes = np.random.lognormal(mean=8, sigma=1.5, size=n_normal)
    normal_dst_bytes = np.random.lognormal(mean=8, sigma=1.5, size=n_normal)
    normal_packets_per_sec = np.random.poisson(lam=10, size=n_normal)
    normal_failed_logins = np.zeros(n_normal)
    normal_num_compromised = np.zeros(n_normal)
    normal_root_shell = np.zeros(n_normal)
    normal_port = np.random.choice([80, 443, 22, 21, 25], n_normal, p=[0.5, 0.3, 0.1, 0.05, 0.05])

    # Attack patterns (different types)
    # DoS attacks - high packet rate, small packets
    n_dos = n_attacks // 3
    dos_packet_size = np.random.normal(100, 30, n_dos)
    dos_duration = np.random.uniform(0.1, 2, n_dos)
    dos_src_bytes = np.random.lognormal(mean=5, sigma=0.5, size=n_dos)
    dos_dst_bytes = np.random.lognormal(mean=5, sigma=0.5, size=n_dos)
    dos_packets_per_sec = np.random.poisson(lam=100, size=n_dos)

    # Port scan attacks - many connections, minimal data
    n_probe = n_attacks // 3
    probe_packet_size = np.random.normal(64, 10, n_probe)
    probe_duration = np.random.uniform(0.01, 0.5, n_probe)
    probe_src_bytes = np.random.uniform(0, 100, n_probe)
    probe_dst_bytes = np.random.uniform(0, 100, n_probe)
    probe_packets_per_sec = np.random.poisson(lam=50, size=n_probe)

    # Intrusion attempts - unusual ports, failed logins
    n_intrusion = n_attacks - n_dos - n_probe
    intrusion_packet_size = np.random.normal(400, 100, n_intrusion)
    intrusion_duration = np.random.exponential(scale=10, size=n_intrusion)
    intrusion_src_bytes = np.random.lognormal(mean=9, sigma=2, size=n_intrusion)
    intrusion_dst_bytes = np.random.lognormal(mean=7, sigma=1.5, size=n_intrusion)
    intrusion_packets_per_sec = np.random.poisson(lam=20, size=n_intrusion)

    # Combine attack data
    attack_packet_size = np.concatenate([dos_packet_size, probe_packet_size, intrusion_packet_size])
    attack_duration = np.concatenate([dos_duration, probe_duration, intrusion_duration])
    attack_src_bytes = np.concatenate([dos_src_bytes, probe_src_bytes, intrusion_src_bytes])
    attack_dst_bytes = np.concatenate([dos_dst_bytes, probe_dst_bytes, intrusion_dst_bytes])
    attack_packets_per_sec = np.concatenate([dos_packets_per_sec, probe_packets_per_sec, intrusion_packets_per_sec])

    attack_failed_logins = np.concatenate([
        np.zeros(n_dos),
        np.zeros(n_probe),
        np.random.poisson(lam=5, size=n_intrusion)
    ])
    attack_num_compromised = np.concatenate([
        np.zeros(n_dos),
        np.zeros(n_probe),
        np.random.binomial(1, 0.3, n_intrusion)
    ])
    attack_root_shell = np.concatenate([
        np.zeros(n_dos),
        np.zeros(n_probe),
        np.random.binomial(1, 0.2, n_intrusion)
    ])
    attack_port = np.concatenate([
        np.random.choice([80, 443], n_dos),
        np.random.randint(1024, 65535, n_probe),  # Random high ports
        np.random.choice([23, 3389, 445, 139], n_intrusion)  # Telnet, RDP, SMB
    ])

    # Create DataFrame
    data = pd.DataFrame({
        'duration': np.concatenate([normal_duration, attack_duration]),
        'src_bytes': np.concatenate([normal_src_bytes, attack_src_bytes]),
        'dst_bytes': np.concatenate([normal_dst_bytes, attack_dst_bytes]),
        'packet_size': np.concatenate([normal_packet_size, attack_packet_size]),
        'packets_per_sec': np.concatenate([normal_packets_per_sec, attack_packets_per_sec]),
        'failed_login_attempts': np.concatenate([normal_failed_logins, attack_failed_logins]),
        'num_compromised': np.concatenate([normal_num_compromised, attack_num_compromised]),
        'root_shell': np.concatenate([normal_root_shell, attack_root_shell]),
        'dst_port': np.concatenate([normal_port, attack_port]),
        'label': np.concatenate([np.zeros(n_normal), np.ones(n_attacks)])
    })

    # Add derived features
    data['byte_ratio'] = data['src_bytes'] / (data['dst_bytes'] + 1)
    data['avg_packet_size'] = data['packet_size']
    data['connection_rate'] = data['packets_per_sec'] / (data['duration'] + 1)

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def plot_feature_distributions(data):
    """Visualize feature distributions"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Network Traffic Feature Distributions', fontsize=16)

    features = ['duration', 'src_bytes', 'dst_bytes', 'packet_size', 'packets_per_sec',
               'failed_login_attempts', 'byte_ratio', 'connection_rate', 'dst_port']

    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]

        normal_data = data[data['label'] == 0][feature]
        attack_data = data[data['label'] == 1][feature]

        # Use log scale for highly skewed features
        if feature in ['src_bytes', 'dst_bytes', 'packets_per_sec']:
            bins = np.logspace(np.log10(max(data[feature].min(), 0.1)),
                             np.log10(data[feature].max()), 50)
            ax.set_xscale('log')
        else:
            bins = 50

        ax.hist(normal_data, bins=bins, alpha=0.6, label='Normal', density=True)
        ax.hist(attack_data, bins=bins, alpha=0.6, label='Attack', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature}')

    plt.tight_layout()
    plt.savefig('network_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: network_feature_distributions.png")

def evaluate_anomaly_detector(y_true, y_pred, scores, model_name):
    """Evaluate and visualize anomaly detection performance"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Detection Rate): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    # ROC curve if scores available
    if scores is not None:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr_curve, tpr_curve)
        print(f"ROC AUC: {roc_auc:.4f}")
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
               'f1': f1, 'fpr': fpr, 'roc_auc': roc_auc,
               'fpr_curve': fpr_curve, 'tpr_curve': tpr_curve}

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
           'f1': f1, 'fpr': fpr}

def plot_results(metrics_dict, data):
    """Plot comprehensive results"""
    # Performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(metrics_dict.keys())
    metrics = ['precision', 'recall', 'f1', 'fpr']
    x = np.arange(len(models))
    width = 0.2

    for idx, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        axes[0].bar(x + idx * width, values, width, label=metric.upper())

    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Performance Metrics')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # ROC curves
    for model_name, metrics in metrics_dict.items():
        if 'fpr_curve' in metrics:
            axes[1].plot(metrics['fpr_curve'], metrics['tpr_curve'],
                        label=f"{model_name} (AUC={metrics['roc_auc']:.3f})")

    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('network_intrusion_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: network_intrusion_results.png")

def main():
    print("Network Intrusion Detection System")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic network traffic data...")
    data = generate_network_traffic_data(n_samples=15000, attack_ratio=0.05)
    print(f"Total connections: {len(data)}")
    print(f"Attack connections: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Feature engineering
    print("\nFeature statistics:")
    print(data.describe())

    # Visualize
    print("\nVisualizing feature distributions...")
    plot_feature_distributions(data)

    # Prepare features
    feature_cols = ['duration', 'src_bytes', 'dst_bytes', 'packet_size', 'packets_per_sec',
                   'failed_login_attempts', 'num_compromised', 'root_shell',
                   'byte_ratio', 'connection_rate']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Model 1: Isolation Forest
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)
    scores_if = -iso_forest.score_samples(X_scaled)  # Negative for anomaly score
    metrics_dict['Isolation Forest'] = evaluate_anomaly_detector(
        y_true, y_pred_if, scores_if, "Isolation Forest"
    )

    # Model 2: One-Class SVM
    print("\n" + "="*60)
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(gamma='auto', nu=0.05)
    y_pred_svm = ocsvm.fit_predict(X_scaled)
    y_pred_svm = (y_pred_svm == -1).astype(int)
    scores_svm = -ocsvm.score_samples(X_scaled)
    metrics_dict['One-Class SVM'] = evaluate_anomaly_detector(
        y_true, y_pred_svm, scores_svm, "One-Class SVM"
    )

    # Model 3: Statistical Z-Score Method
    print("\n" + "="*60)
    print("Applying Statistical Z-Score Method...")
    z_scores = np.abs(stats.zscore(X_scaled, axis=0))
    max_z_scores = np.max(z_scores, axis=1)
    threshold = np.percentile(max_z_scores, 95)  # Top 5% as anomalies
    y_pred_stat = (max_z_scores > threshold).astype(int)
    metrics_dict['Statistical'] = evaluate_anomaly_detector(
        y_true, y_pred_stat, max_z_scores, "Statistical Z-Score"
    )

    # Visualize results
    print("\n" + "="*60)
    print("Generating result visualizations...")
    plot_results(metrics_dict, data)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_f1_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    best_recall_model = max(metrics_dict.items(), key=lambda x: x[1]['recall'])
    print(f"\nBest F1-Score: {best_f1_model[0]} ({best_f1_model[1]['f1']:.4f})")
    print(f"Best Detection Rate: {best_recall_model[0]} ({best_recall_model[1]['recall']:.4f})")

    print("\nRecommendation:")
    print("- Use Isolation Forest for real-time detection (fast, good balance)")
    print("- Combine multiple methods for higher confidence alerts")
    print("- Monitor false positive rate for operational feasibility")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
