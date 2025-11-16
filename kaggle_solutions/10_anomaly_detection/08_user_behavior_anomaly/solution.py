"""
User Behavior Anomaly Detection
Detects compromised accounts and insider threats through behavioral analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_user_behavior_data(n_users=200, sessions_per_user=150, anomaly_ratio=0.05):
    """Generate synthetic user activity data with anomalous behavior"""
    all_data = []

    for user_id in range(n_users):
        n_sessions = sessions_per_user
        n_anomalies = int(n_sessions * anomaly_ratio)
        n_normal = n_sessions - n_anomalies

        # User's normal behavior profile
        typical_login_hour = np.random.choice([9, 10, 14, 15, 16])  # Work hours
        typical_session_duration = np.random.lognormal(mean=np.log(30), sigma=0.5)  # Minutes
        typical_pages_visited = np.random.poisson(lam=15)
        typical_clicks = np.random.poisson(lam=50)

        # Normal behavior
        normal_login_hour = (typical_login_hour + np.random.normal(0, 2, n_normal)) % 24
        normal_session_duration = np.random.lognormal(mean=np.log(typical_session_duration), sigma=0.4, size=n_normal)
        normal_pages_visited = np.random.poisson(lam=typical_pages_visited, size=n_normal)
        normal_clicks = np.random.poisson(lam=typical_clicks, size=n_normal)
        normal_downloads = np.random.poisson(lam=2, size=n_normal)
        normal_failed_logins = np.random.binomial(1, 0.05, n_normal)  # 5% have failed login
        normal_location_changes = np.zeros(n_normal)  # Same location
        normal_device_changes = np.random.binomial(1, 0.1, n_normal)  # Occasionally different device

        # Anomalous behavior (account compromise or insider threat)
        anomaly_types = np.random.choice(['compromised_account', 'data_exfiltration',
                                         'unusual_access', 'suspicious_login'], n_anomalies)

        anomaly_login_hour = np.zeros(n_anomalies)
        anomaly_session_duration = np.zeros(n_anomalies)
        anomaly_pages_visited = np.zeros(n_anomalies)
        anomaly_clicks = np.zeros(n_anomalies)
        anomaly_downloads = np.zeros(n_anomalies)
        anomaly_failed_logins = np.zeros(n_anomalies)
        anomaly_location_changes = np.zeros(n_anomalies)
        anomaly_device_changes = np.zeros(n_anomalies)

        for i, anom_type in enumerate(anomaly_types):
            if anom_type == 'compromised_account':
                # Account accessed from unusual location/time after breach
                anomaly_login_hour[i] = np.random.choice([2, 3, 4, 23])  # Late night
                anomaly_session_duration[i] = np.random.uniform(5, 60)  # Varied
                anomaly_pages_visited[i] = np.random.poisson(lam=25)  # More exploration
                anomaly_clicks[i] = np.random.poisson(lam=70)
                anomaly_downloads[i] = np.random.poisson(lam=5)
                anomaly_failed_logins[i] = np.random.binomial(1, 0.8)  # Multiple attempts
                anomaly_location_changes[i] = 1  # Different location
                anomaly_device_changes[i] = 1  # Different device
            elif anom_type == 'data_exfiltration':
                # Downloading large amounts of data
                anomaly_login_hour[i] = typical_login_hour + np.random.normal(0, 2)
                anomaly_session_duration[i] = np.random.uniform(60, 180)  # Long session
                anomaly_pages_visited[i] = np.random.poisson(lam=50)  # Many pages
                anomaly_clicks[i] = np.random.poisson(lam=100)
                anomaly_downloads[i] = np.random.poisson(lam=30)  # LOTS of downloads
                anomaly_failed_logins[i] = 0
                anomaly_location_changes[i] = 0
                anomaly_device_changes[i] = np.random.binomial(1, 0.2)
            elif anom_type == 'unusual_access':
                # Accessing resources not typically used
                anomaly_login_hour[i] = np.random.choice(range(24))
                anomaly_session_duration[i] = np.random.uniform(10, 120)
                anomaly_pages_visited[i] = np.random.poisson(lam=60)  # Lots of exploration
                anomaly_clicks[i] = np.random.poisson(lam=120)
                anomaly_downloads[i] = np.random.poisson(lam=10)
                anomaly_failed_logins[i] = np.random.binomial(1, 0.3)
                anomaly_location_changes[i] = np.random.binomial(1, 0.3)
                anomaly_device_changes[i] = np.random.binomial(1, 0.5)
            else:  # suspicious_login
                # Multiple failed logins then success
                anomaly_login_hour[i] = np.random.choice(range(24))
                anomaly_session_duration[i] = np.random.uniform(1, 30)  # Short
                anomaly_pages_visited[i] = np.random.poisson(lam=5)
                anomaly_clicks[i] = np.random.poisson(lam=10)
                anomaly_downloads[i] = 0
                anomaly_failed_logins[i] = 1  # Failed attempts
                anomaly_location_changes[i] = np.random.binomial(1, 0.7)
                anomaly_device_changes[i] = np.random.binomial(1, 0.8)

        # Combine
        user_data = pd.DataFrame({
            'user_id': user_id,
            'login_hour': np.concatenate([normal_login_hour, anomaly_login_hour]),
            'session_duration_min': np.concatenate([normal_session_duration, anomaly_session_duration]),
            'pages_visited': np.concatenate([normal_pages_visited, anomaly_pages_visited]),
            'clicks': np.concatenate([normal_clicks, anomaly_clicks]),
            'downloads': np.concatenate([normal_downloads, anomaly_downloads]),
            'failed_login_attempt': np.concatenate([normal_failed_logins, anomaly_failed_logins]),
            'location_changed': np.concatenate([normal_location_changes, anomaly_location_changes]),
            'device_changed': np.concatenate([normal_device_changes, anomaly_device_changes]),
            'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        })

        all_data.append(user_data)

    # Combine all users
    data = pd.concat(all_data, ignore_index=True)

    # Add derived features
    data['clicks_per_page'] = data['clicks'] / (data['pages_visited'] + 1)
    data['downloads_per_page'] = data['downloads'] / (data['pages_visited'] + 1)
    data['session_intensity'] = (data['clicks'] + data['pages_visited']) / (data['session_duration_min'] + 1)
    data['is_night'] = ((data['login_hour'] <= 6) | (data['login_hour'] >= 22)).astype(int)

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def plot_behavior_patterns(data):
    """Visualize user behavior patterns"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('User Behavior Patterns', fontsize=16)

    features = ['login_hour', 'session_duration_min', 'pages_visited', 'clicks',
               'downloads', 'clicks_per_page', 'downloads_per_page', 'session_intensity',
               'failed_login_attempt']

    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]

        normal_data = data[data['label'] == 0][feature]
        anomaly_data = data[data['label'] == 1][feature]

        if feature in ['failed_login_attempt', 'location_changed', 'device_changed']:
            # Bar plot for binary features
            normal_counts = normal_data.value_counts()
            anomaly_counts = anomaly_data.value_counts()
            x = [0, 1]
            width = 0.35
            ax.bar([i - width/2 for i in x], [normal_counts.get(i, 0) for i in x],
                  width, alpha=0.6, label='Normal')
            ax.bar([i + width/2 for i in x], [anomaly_counts.get(i, 0) for i in x],
                  width, alpha=0.6, label='Anomaly')
        else:
            ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', density=True)
            ax.hist(anomaly_data, bins=50, alpha=0.6, label='Anomaly', density=True)

        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density' if feature not in ['failed_login_attempt'] else 'Count')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('behavior_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: behavior_patterns.png")

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
    print("User Behavior Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic user behavior data...")
    data = generate_user_behavior_data(n_users=200, sessions_per_user=150, anomaly_ratio=0.05)
    print(f"Total sessions: {len(data)}")
    print(f"Number of users: {data['user_id'].nunique()}")
    print(f"Anomalous sessions: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing behavior patterns...")
    plot_behavior_patterns(data)

    # Prepare features
    feature_cols = ['login_hour', 'session_duration_min', 'pages_visited', 'clicks',
                   'downloads', 'failed_login_attempt', 'location_changed', 'device_changed',
                   'clicks_per_page', 'downloads_per_page', 'session_intensity', 'is_night']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Method 1: Isolation Forest
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)
    metrics_dict['Isolation Forest'] = evaluate_detector(y_true, y_pred_if, "Isolation Forest")

    # Method 2: User-specific baseline
    print("\n" + "="*60)
    print("Applying User-Specific Baseline...")

    # For each user, calculate z-score from their normal behavior
    z_scores = []
    for user_id in data['user_id'].unique():
        user_mask = data['user_id'] == user_id
        user_features = X[user_mask]
        user_labels = y_true[user_mask]

        # Use normal data for baseline
        normal_mask = user_labels == 0
        if normal_mask.sum() > 5:
            mean = np.mean(user_features[normal_mask], axis=0)
            std = np.std(user_features[normal_mask], axis=0) + 1e-6
            user_z = np.abs((user_features - mean) / std)
            max_z = np.max(user_z, axis=1)
            z_scores.extend(max_z)
        else:
            z_scores.extend([0] * user_mask.sum())

    z_scores = np.array(z_scores)
    threshold = np.percentile(z_scores[y_true == 0], 95)
    y_pred_baseline = (z_scores > threshold).astype(int)

    metrics_dict['User Baseline'] = evaluate_detector(y_true, y_pred_baseline, "User-Specific Baseline")

    # Method 3: Rule-based
    print("\n" + "="*60)
    print("Applying Behavior Rules...")

    y_pred_rules = (
        (data['is_night'] == 1) &
        ((data['failed_login_attempt'] == 1) |
         (data['location_changed'] == 1) |
         (data['device_changed'] == 1) |
         (data['downloads'] > 20) |
         (data['session_duration_min'] > 120))
    ).astype(int)

    metrics_dict['Rule-Based'] = evaluate_detector(y_true, y_pred_rules, "Rule-Based System")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection Results', fontsize=16)

    methods = [
        ('Isolation Forest', y_pred_if),
        ('User Baseline', y_pred_baseline),
        ('Rule-Based', y_pred_rules)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
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
    print("- User-specific baselines capture individual behavior patterns")
    print("- Rule-based systems provide interpretable alerts")
    print("- Isolation Forest detects complex multivariate anomalies")
    print("- Combine methods for robust insider threat detection")
    print("- Monitor for gradual behavior changes over time")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
