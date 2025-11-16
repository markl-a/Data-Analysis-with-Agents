"""
Industrial Equipment Anomaly Detection
Detects equipment failures and anomalies using Autoencoder and statistical methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Try to import keras/tensorflow
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. Using only statistical methods.")

np.random.seed(42)
if KERAS_AVAILABLE:
    keras.utils.set_random_seed(42)

def generate_equipment_data(n_samples=20000, anomaly_ratio=0.03):
    """Generate synthetic industrial equipment sensor data"""
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal operating conditions
    # Temperature follows sinusoidal pattern with noise
    time = np.linspace(0, 100, n_normal)
    normal_temp = 70 + 10 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 2, n_normal)
    normal_pressure = 100 + 5 * np.sin(2 * np.pi * time / 24 + np.pi/4) + np.random.normal(0, 1.5, n_normal)
    normal_vibration = 0.5 + 0.2 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 0.1, n_normal)
    normal_rpm = 1500 + 50 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 20, n_normal)
    normal_current = 10 + 2 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 0.5, n_normal)
    normal_power = normal_current * 220 + np.random.normal(0, 50, n_normal)

    # Anomalous conditions (equipment failures)
    anomaly_time = np.random.uniform(0, 100, n_anomalies)

    # Different failure modes
    failure_types = np.random.choice(['overheat', 'pressure_spike', 'vibration', 'electrical'], n_anomalies)

    anomaly_temp = np.zeros(n_anomalies)
    anomaly_pressure = np.zeros(n_anomalies)
    anomaly_vibration = np.zeros(n_anomalies)
    anomaly_rpm = np.zeros(n_anomalies)
    anomaly_current = np.zeros(n_anomalies)

    for i, failure in enumerate(failure_types):
        if failure == 'overheat':
            anomaly_temp[i] = np.random.uniform(90, 110)  # High temperature
            anomaly_pressure[i] = np.random.normal(105, 3)
            anomaly_vibration[i] = np.random.normal(0.8, 0.2)
            anomaly_rpm[i] = np.random.normal(1400, 50)  # Reduced RPM
            anomaly_current[i] = np.random.normal(12, 1)  # High current
        elif failure == 'pressure_spike':
            anomaly_temp[i] = np.random.normal(75, 5)
            anomaly_pressure[i] = np.random.uniform(120, 150)  # Very high pressure
            anomaly_vibration[i] = np.random.normal(0.7, 0.15)
            anomaly_rpm[i] = np.random.normal(1550, 30)
            anomaly_current[i] = np.random.normal(11, 0.8)
        elif failure == 'vibration':
            anomaly_temp[i] = np.random.normal(72, 3)
            anomaly_pressure[i] = np.random.normal(102, 2)
            anomaly_vibration[i] = np.random.uniform(1.5, 3.0)  # Excessive vibration
            anomaly_rpm[i] = np.random.normal(1500, 100)  # Unstable RPM
            anomaly_current[i] = np.random.normal(10.5, 1.2)
        else:  # electrical
            anomaly_temp[i] = np.random.normal(73, 4)
            anomaly_pressure[i] = np.random.normal(100, 2)
            anomaly_vibration[i] = np.random.normal(0.6, 0.15)
            anomaly_rpm[i] = np.random.normal(1450, 80)
            anomaly_current[i] = np.random.uniform(15, 25)  # Current spike

    anomaly_power = anomaly_current * 220 + np.random.normal(0, 100, n_anomalies)

    # Combine data
    data = pd.DataFrame({
        'temperature': np.concatenate([normal_temp, anomaly_temp]),
        'pressure': np.concatenate([normal_pressure, anomaly_pressure]),
        'vibration': np.concatenate([normal_vibration, anomaly_vibration]),
        'rpm': np.concatenate([normal_rpm, anomaly_rpm]),
        'current': np.concatenate([normal_current, anomaly_current]),
        'power': np.concatenate([normal_power, anomaly_power]),
        'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    })

    # Add derived features
    data['temp_pressure_ratio'] = data['temperature'] / data['pressure']
    data['power_efficiency'] = data['power'] / (data['rpm'] + 1)
    data['vibration_rpm_ratio'] = data['vibration'] * 1000 / data['rpm']

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def build_autoencoder(input_dim, encoding_dim=8):
    """Build autoencoder model for anomaly detection"""
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(16, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    # Autoencoder model
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def plot_sensor_data(data):
    """Visualize sensor readings over time"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Equipment Sensor Readings', fontsize=16)

    sensors = ['temperature', 'pressure', 'vibration', 'rpm', 'current', 'power']

    for idx, sensor in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]

        normal = data[data['label'] == 0][sensor]
        anomaly = data[data['label'] == 1][sensor]

        ax.hist(normal, bins=50, alpha=0.6, label='Normal', density=True, color='blue')
        ax.hist(anomaly, bins=50, alpha=0.6, label='Anomaly', density=True, color='red')
        ax.set_xlabel(sensor.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('equipment_sensor_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: equipment_sensor_distributions.png")

def plot_correlation_matrix(data):
    """Plot correlation matrix of features"""
    plt.figure(figsize=(10, 8))
    corr = data.drop('label', axis=1).corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_correlation.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity}

def plot_reconstruction_error(reconstruction_errors, y_true, threshold):
    """Plot reconstruction error distribution"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    normal_errors = reconstruction_errors[y_true == 0]
    anomaly_errors = reconstruction_errors[y_true == 1]

    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.6, label='Anomaly', density=True)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Reconstruction Error Distribution')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(reconstruction_errors)), reconstruction_errors,
               c=y_true, cmap='coolwarm', alpha=0.5, s=10)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Over Samples')
    plt.colorbar(label='True Label')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('autoencoder_reconstruction_error.png', dpi=300, bbox_inches='tight')
    print("Saved: autoencoder_reconstruction_error.png")

def main():
    print("Industrial Equipment Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic equipment sensor data...")
    data = generate_equipment_data(n_samples=20000, anomaly_ratio=0.03)
    print(f"Total samples: {len(data)}")
    print(f"Anomalies: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing sensor data...")
    plot_sensor_data(data)
    plot_correlation_matrix(data)

    # Prepare features
    feature_cols = ['temperature', 'pressure', 'vibration', 'rpm', 'current', 'power',
                   'temp_pressure_ratio', 'power_efficiency', 'vibration_rpm_ratio']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Split for autoencoder training (use only normal data)
    normal_indices = y_true == 0
    X_train_normal = X[normal_indices]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_scaled = scaler.transform(X)

    metrics_dict = {}

    # Method 1: Autoencoder (if available)
    if KERAS_AVAILABLE:
        print("\n" + "="*60)
        print("Training Autoencoder...")
        autoencoder = build_autoencoder(input_dim=X_scaled.shape[1])

        history = autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        print("Training complete.")

        # Predict
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

        # Set threshold (95th percentile of normal data errors)
        normal_errors = reconstruction_errors[y_true == 0]
        threshold = np.percentile(normal_errors, 95)

        y_pred_ae = (reconstruction_errors > threshold).astype(int)
        metrics_dict['Autoencoder'] = evaluate_detector(y_true, y_pred_ae, "Autoencoder")

        # Plot reconstruction errors
        plot_reconstruction_error(reconstruction_errors, y_true, threshold)

    # Method 2: Isolation Forest
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.03, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)
    metrics_dict['Isolation Forest'] = evaluate_detector(y_true, y_pred_if, "Isolation Forest")

    # Method 3: Statistical (Mahalanobis distance)
    print("\n" + "="*60)
    print("Applying Statistical Method...")

    mean = np.mean(X_train_scaled, axis=0)
    cov = np.cov(X_train_scaled.T)
    cov_inv = np.linalg.pinv(cov)

    # Mahalanobis distance
    diff = X_scaled - mean
    mahal_dist = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))

    threshold_stat = np.percentile(mahal_dist[y_true == 0], 95)
    y_pred_stat = (mahal_dist > threshold_stat).astype(int)
    metrics_dict['Statistical'] = evaluate_detector(y_true, y_pred_stat, "Statistical Mahalanobis")

    # Compare methods
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)

    comparison_df = pd.DataFrame(metrics_dict).T
    print("\n", comparison_df.to_string())

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Anomaly Detection Method Comparison')
    ax.set_xlabel('Method')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: method_comparison.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest overall method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- Deploy ensemble of multiple methods for robust detection")
    print("- Set thresholds based on maintenance cost vs. downtime risk")
    print("- Monitor model performance and retrain with new data")
    print("- Autoencoder captures complex multivariate patterns")
    print("- Isolation Forest is fast and interpretable")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
