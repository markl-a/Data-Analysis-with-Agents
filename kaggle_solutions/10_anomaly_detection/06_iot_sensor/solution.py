"""
IoT Sensor Data Anomaly Detection
Detects sensor failures and environmental anomalies in IoT sensor networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_iot_sensor_data(n_sensors=50, samples_per_sensor=400, anomaly_ratio=0.04):
    """Generate synthetic IoT sensor network data"""
    all_data = []

    for sensor_id in range(n_sensors):
        n_samples = samples_per_sensor
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Sensor location affects readings
        location_temp_offset = np.random.uniform(-5, 5)  # Different locations
        location_humidity_offset = np.random.uniform(-10, 10)

        # Normal sensor readings (environmental monitoring)
        time = np.linspace(0, 24, n_normal)  # 24 hour cycle

        # Temperature with daily cycle
        normal_temp = 20 + location_temp_offset + 8 * np.sin(2 * np.pi * time / 24 - np.pi/2) + \
                     np.random.normal(0, 1, n_normal)

        # Humidity (inverse relationship with temperature)
        normal_humidity = 60 + location_humidity_offset - 10 * np.sin(2 * np.pi * time / 24 - np.pi/2) + \
                         np.random.normal(0, 3, n_normal)

        # Light (follows sun cycle)
        normal_light = 500 * np.maximum(0, np.sin(2 * np.pi * time / 24 - np.pi/2)) + \
                      np.random.normal(0, 50, n_normal)

        # Sound (random with occasional spikes)
        normal_sound = np.random.exponential(scale=40, size=n_normal)

        # Battery voltage (slowly decreasing)
        normal_battery = 3.7 - 0.5 * (time / 24) + np.random.normal(0, 0.05, n_normal)

        # Signal strength
        normal_rssi = -70 + np.random.normal(0, 5, n_normal)

        # Anomalous readings (sensor failures and environmental events)
        anomaly_types = np.random.choice(['sensor_failure', 'battery_failure', 'environmental_spike',
                                         'connectivity_issue'], n_anomalies)

        anomaly_time = np.random.uniform(0, 24, n_anomalies)
        anomaly_temp = np.zeros(n_anomalies)
        anomaly_humidity = np.zeros(n_anomalies)
        anomaly_light = np.zeros(n_anomalies)
        anomaly_sound = np.zeros(n_anomalies)
        anomaly_battery = np.zeros(n_anomalies)
        anomaly_rssi = np.zeros(n_anomalies)

        for i, anom_type in enumerate(anomaly_types):
            if anom_type == 'sensor_failure':
                # Sensor stuck or giving erratic readings
                anomaly_temp[i] = np.random.choice([0, 100, -40])  # Unrealistic values
                anomaly_humidity[i] = np.random.choice([0, 100])
                anomaly_light[i] = np.random.choice([0, 10000])
                anomaly_sound[i] = np.random.uniform(0, 200)
                anomaly_battery[i] = np.random.uniform(3.0, 3.8)
                anomaly_rssi[i] = -70 + np.random.normal(0, 5)
            elif anom_type == 'battery_failure':
                # Low battery affecting readings
                anomaly_temp[i] = 20 + location_temp_offset + np.random.normal(0, 5)
                anomaly_humidity[i] = 60 + location_humidity_offset + np.random.normal(0, 10)
                anomaly_light[i] = 500 + np.random.normal(0, 200)
                anomaly_sound[i] = np.random.exponential(scale=40)
                anomaly_battery[i] = np.random.uniform(2.8, 3.1)  # Low battery
                anomaly_rssi[i] = np.random.uniform(-100, -85)  # Weak signal
            elif anom_type == 'environmental_spike':
                # Actual environmental anomaly (fire, etc.)
                anomaly_temp[i] = np.random.uniform(35, 60)  # High temperature
                anomaly_humidity[i] = np.random.uniform(10, 30)  # Low humidity
                anomaly_light[i] = np.random.uniform(2000, 8000)  # Bright light
                anomaly_sound[i] = np.random.uniform(200, 500)  # Loud noise
                anomaly_battery[i] = 3.7 + np.random.normal(0, 0.1)
                anomaly_rssi[i] = -70 + np.random.normal(0, 5)
            else:  # connectivity_issue
                # Communication problems
                anomaly_temp[i] = 20 + location_temp_offset + np.random.normal(0, 2)
                anomaly_humidity[i] = 60 + location_humidity_offset + np.random.normal(0, 5)
                anomaly_light[i] = 500 + np.random.normal(0, 100)
                anomaly_sound[i] = np.random.exponential(scale=40)
                anomaly_battery[i] = 3.5 + np.random.normal(0, 0.1)
                anomaly_rssi[i] = np.random.uniform(-100, -90)  # Very weak signal

        # Combine
        sensor_data = pd.DataFrame({
            'sensor_id': sensor_id,
            'timestamp': list(time) + list(anomaly_time),
            'temperature': np.concatenate([normal_temp, anomaly_temp]),
            'humidity': np.concatenate([normal_humidity, anomaly_humidity]),
            'light': np.concatenate([normal_light, anomaly_light]),
            'sound': np.concatenate([normal_sound, anomaly_sound]),
            'battery_voltage': np.concatenate([normal_battery, anomaly_battery]),
            'rssi': np.concatenate([normal_rssi, anomaly_rssi]),
            'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        })

        # Sort by time
        sensor_data = sensor_data.sort_values('timestamp').reset_index(drop=True)
        all_data.append(sensor_data)

    # Combine all sensors
    data = pd.concat(all_data, ignore_index=True)

    # Add derived features
    data['temp_humidity_ratio'] = data['temperature'] / (data['humidity'] + 1)
    data['battery_health'] = (data['battery_voltage'] - 2.8) / (3.7 - 2.8)  # 0-1 scale
    data['signal_quality'] = (data['rssi'] + 100) / 30  # Normalized signal

    return data

def plot_sensor_network(data):
    """Visualize sensor network data"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('IoT Sensor Network Readings', fontsize=16)

    sensors = ['temperature', 'humidity', 'light', 'sound', 'battery_voltage', 'rssi']

    for idx, sensor in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]

        normal_data = data[data['label'] == 0][sensor]
        anomaly_data = data[data['label'] == 1][sensor]

        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', density=True)
        ax.hist(anomaly_data, bins=50, alpha=0.6, label='Anomaly', density=True)
        ax.set_xlabel(sensor.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('iot_sensor_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: iot_sensor_distributions.png")

def plot_sensor_timeline(data, sensor_id=0):
    """Plot readings for a single sensor over time"""
    sensor_data = data[data['sensor_id'] == sensor_id]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Sensor {sensor_id} Readings Over Time', fontsize=16)

    sensors = ['temperature', 'humidity', 'light', 'sound', 'battery_voltage', 'rssi']

    for idx, sensor in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]

        normal_mask = sensor_data['label'] == 0
        anomaly_mask = sensor_data['label'] == 1

        ax.plot(sensor_data[normal_mask]['timestamp'],
               sensor_data[normal_mask][sensor],
               'b.-', alpha=0.5, markersize=3, label='Normal')
        ax.scatter(sensor_data[anomaly_mask]['timestamp'],
                  sensor_data[anomaly_mask][sensor],
                  c='red', marker='x', s=100, label='Anomaly', zorder=5)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(sensor.replace('_', ' ').title())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensor_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved: sensor_timeline.png")

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
    print("IoT Sensor Network Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic IoT sensor data...")
    data = generate_iot_sensor_data(n_sensors=50, samples_per_sensor=400, anomaly_ratio=0.04)
    print(f"Total readings: {len(data)}")
    print(f"Number of sensors: {data['sensor_id'].nunique()}")
    print(f"Anomalous readings: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing sensor data...")
    plot_sensor_network(data)
    plot_sensor_timeline(data, sensor_id=0)

    # Prepare features
    feature_cols = ['temperature', 'humidity', 'light', 'sound', 'battery_voltage', 'rssi',
                   'temp_humidity_ratio', 'battery_health', 'signal_quality']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Method 1: Isolation Forest
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.04, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)
    metrics_dict['Isolation Forest'] = evaluate_detector(y_true, y_pred_if, "Isolation Forest")

    # Method 2: Elliptic Envelope (Robust covariance)
    print("\n" + "="*60)
    print("Training Elliptic Envelope...")
    elliptic = EllipticEnvelope(contamination=0.04, random_state=42)
    y_pred_ee = elliptic.fit_predict(X_scaled)
    y_pred_ee = (y_pred_ee == -1).astype(int)
    metrics_dict['Elliptic Envelope'] = evaluate_detector(y_true, y_pred_ee, "Elliptic Envelope")

    # Method 3: Rule-based (Hardware thresholds)
    print("\n" + "="*60)
    print("Applying Hardware Threshold Rules...")

    y_pred_rules = (
        (data['temperature'] < -20) | (data['temperature'] > 50) |
        (data['humidity'] < 5) | (data['humidity'] > 95) |
        (data['light'] < 0) | (data['light'] > 5000) |
        (data['battery_voltage'] < 3.0) |
        (data['rssi'] < -90)
    ).astype(int)

    metrics_dict['Hardware Rules'] = evaluate_detector(y_true, y_pred_rules, "Hardware Threshold Rules")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Results', fontsize=16)

    methods = [
        ('Isolation Forest', y_pred_if),
        ('Elliptic Envelope', y_pred_ee),
        ('Hardware Rules', y_pred_rules)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
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
    plt.savefig('iot_detection_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: iot_detection_results.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- Isolation Forest handles diverse failure modes well")
    print("- Elliptic Envelope good for Gaussian-distributed normal data")
    print("- Hardware rules catch obvious sensor failures")
    print("- Combine methods for robust detection")
    print("- Monitor battery levels for preventive replacement")
    print("- Use edge computing for real-time detection")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
