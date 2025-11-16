"""
Healthcare Vital Signs Anomaly Detection
Detects critical patient condition changes using time-series anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_patient_vitals(n_patients=100, samples_per_patient=200, anomaly_ratio=0.05):
    """Generate synthetic patient vital signs with anomalies"""
    all_data = []

    for patient_id in range(n_patients):
        n_samples = samples_per_patient
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Normal vital signs (varies by patient baseline)
        baseline_hr = np.random.randint(60, 80)  # Heart rate
        baseline_bp_sys = np.random.randint(110, 130)  # Systolic BP
        baseline_bp_dia = np.random.randint(70, 85)  # Diastolic BP
        baseline_temp = np.random.uniform(36.5, 37.2)  # Temperature
        baseline_spo2 = np.random.uniform(96, 100)  # Oxygen saturation
        baseline_rr = np.random.randint(12, 18)  # Respiratory rate

        # Normal variations
        normal_hr = baseline_hr + np.random.normal(0, 5, n_normal)
        normal_bp_sys = baseline_bp_sys + np.random.normal(0, 8, n_normal)
        normal_bp_dia = baseline_bp_dia + np.random.normal(0, 5, n_normal)
        normal_temp = baseline_temp + np.random.normal(0, 0.3, n_normal)
        normal_spo2 = baseline_spo2 + np.random.normal(0, 1, n_normal)
        normal_rr = baseline_rr + np.random.normal(0, 2, n_normal)

        # Anomalous vital signs (critical conditions)
        anomaly_types = np.random.choice(['tachycardia', 'bradycardia', 'hypertension',
                                         'hypotension', 'fever', 'hypoxia'], n_anomalies)

        anomaly_hr = np.zeros(n_anomalies)
        anomaly_bp_sys = np.zeros(n_anomalies)
        anomaly_bp_dia = np.zeros(n_anomalies)
        anomaly_temp = np.zeros(n_anomalies)
        anomaly_spo2 = np.zeros(n_anomalies)
        anomaly_rr = np.zeros(n_anomalies)

        for i, anom_type in enumerate(anomaly_types):
            if anom_type == 'tachycardia':  # High heart rate
                anomaly_hr[i] = np.random.uniform(120, 180)
                anomaly_bp_sys[i] = baseline_bp_sys + np.random.normal(10, 8)
                anomaly_bp_dia[i] = baseline_bp_dia + np.random.normal(5, 5)
                anomaly_temp[i] = baseline_temp + np.random.normal(0, 0.5)
                anomaly_spo2[i] = baseline_spo2 + np.random.normal(-2, 1)
                anomaly_rr[i] = baseline_rr + np.random.normal(4, 2)
            elif anom_type == 'bradycardia':  # Low heart rate
                anomaly_hr[i] = np.random.uniform(40, 55)
                anomaly_bp_sys[i] = baseline_bp_sys + np.random.normal(-10, 8)
                anomaly_bp_dia[i] = baseline_bp_dia + np.random.normal(-5, 5)
                anomaly_temp[i] = baseline_temp + np.random.normal(0, 0.3)
                anomaly_spo2[i] = baseline_spo2 + np.random.normal(-1, 1)
                anomaly_rr[i] = baseline_rr + np.random.normal(-2, 2)
            elif anom_type == 'hypertension':  # High blood pressure
                anomaly_hr[i] = baseline_hr + np.random.normal(10, 5)
                anomaly_bp_sys[i] = np.random.uniform(160, 200)
                anomaly_bp_dia[i] = np.random.uniform(100, 120)
                anomaly_temp[i] = baseline_temp + np.random.normal(0, 0.3)
                anomaly_spo2[i] = baseline_spo2 + np.random.normal(0, 1)
                anomaly_rr[i] = baseline_rr + np.random.normal(2, 2)
            elif anom_type == 'hypotension':  # Low blood pressure
                anomaly_hr[i] = baseline_hr + np.random.normal(15, 8)  # Compensatory
                anomaly_bp_sys[i] = np.random.uniform(70, 90)
                anomaly_bp_dia[i] = np.random.uniform(40, 60)
                anomaly_temp[i] = baseline_temp + np.random.normal(-0.5, 0.3)
                anomaly_spo2[i] = baseline_spo2 + np.random.normal(-3, 2)
                anomaly_rr[i] = baseline_rr + np.random.normal(3, 2)
            elif anom_type == 'fever':  # High temperature
                anomaly_hr[i] = baseline_hr + np.random.normal(20, 8)
                anomaly_bp_sys[i] = baseline_bp_sys + np.random.normal(5, 8)
                anomaly_bp_dia[i] = baseline_bp_dia + np.random.normal(0, 5)
                anomaly_temp[i] = np.random.uniform(38.5, 40.5)
                anomaly_spo2[i] = baseline_spo2 + np.random.normal(-2, 1.5)
                anomaly_rr[i] = baseline_rr + np.random.normal(6, 3)
            else:  # hypoxia - Low oxygen
                anomaly_hr[i] = baseline_hr + np.random.normal(25, 10)
                anomaly_bp_sys[i] = baseline_bp_sys + np.random.normal(-5, 8)
                anomaly_bp_dia[i] = baseline_bp_dia + np.random.normal(-3, 5)
                anomaly_temp[i] = baseline_temp + np.random.normal(0, 0.4)
                anomaly_spo2[i] = np.random.uniform(75, 92)
                anomaly_rr[i] = baseline_rr + np.random.normal(8, 3)

        # Combine
        patient_data = pd.DataFrame({
            'patient_id': patient_id,
            'time_index': range(n_samples),
            'heart_rate': np.concatenate([normal_hr, anomaly_hr]),
            'bp_systolic': np.concatenate([normal_bp_sys, anomaly_bp_sys]),
            'bp_diastolic': np.concatenate([normal_bp_dia, anomaly_bp_dia]),
            'temperature': np.concatenate([normal_temp, anomaly_temp]),
            'spo2': np.concatenate([normal_spo2, anomaly_spo2]),
            'respiratory_rate': np.concatenate([normal_rr, anomaly_rr]),
            'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        })

        # Shuffle within patient
        patient_data = patient_data.sample(frac=1, random_state=patient_id).reset_index(drop=True)
        patient_data = patient_data.sort_values('time_index').reset_index(drop=True)

        all_data.append(patient_data)

    # Combine all patients
    data = pd.concat(all_data, ignore_index=True)

    # Add derived features
    data['pulse_pressure'] = data['bp_systolic'] - data['bp_diastolic']
    data['mean_arterial_pressure'] = data['bp_diastolic'] + data['pulse_pressure'] / 3
    data['shock_index'] = data['heart_rate'] / data['bp_systolic']

    return data

def plot_patient_vitals(data, patient_id=0):
    """Plot vitals for a single patient"""
    patient_data = data[data['patient_id'] == patient_id]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f'Patient {patient_id} Vital Signs Over Time', fontsize=16)

    vitals = [
        ('heart_rate', 'Heart Rate (bpm)'),
        ('bp_systolic', 'Systolic BP (mmHg)'),
        ('bp_diastolic', 'Diastolic BP (mmHg)'),
        ('temperature', 'Temperature (°C)'),
        ('spo2', 'SpO2 (%)'),
        ('respiratory_rate', 'Respiratory Rate')
    ]

    for idx, (vital, label) in enumerate(vitals):
        ax = axes[idx // 2, idx % 2]

        normal_mask = patient_data['label'] == 0
        anomaly_mask = patient_data['label'] == 1

        ax.plot(patient_data[normal_mask]['time_index'],
               patient_data[normal_mask][vital],
               'b.-', alpha=0.6, label='Normal', markersize=4)
        ax.scatter(patient_data[anomaly_mask]['time_index'],
                  patient_data[anomaly_mask][vital],
                  c='red', marker='x', s=100, label='Anomaly', zorder=5)

        ax.set_xlabel('Time Index')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('patient_vitals_timeline.png', dpi=300, bbox_inches='tight')
    print("Saved: patient_vitals_timeline.png")

def plot_vital_distributions(data):
    """Plot distribution of vital signs"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Vital Signs Distributions', fontsize=16)

    vitals = ['heart_rate', 'bp_systolic', 'bp_diastolic',
             'temperature', 'spo2', 'respiratory_rate']

    for idx, vital in enumerate(vitals):
        ax = axes[idx // 3, idx % 3]

        normal_data = data[data['label'] == 0][vital]
        anomaly_data = data[data['label'] == 1][vital]

        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', density=True)
        ax.hist(anomaly_data, bins=50, alpha=0.6, label='Anomaly', density=True)
        ax.set_xlabel(vital.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig('vital_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: vital_distributions.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Critical']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity}

def main():
    print("Healthcare Vital Signs Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic patient vital signs...")
    data = generate_patient_vitals(n_patients=100, samples_per_patient=200, anomaly_ratio=0.05)
    print(f"Total measurements: {len(data)}")
    print(f"Number of patients: {data['patient_id'].nunique()}")
    print(f"Critical conditions: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing patient data...")
    plot_patient_vitals(data, patient_id=0)
    plot_vital_distributions(data)

    # Prepare features
    feature_cols = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'temperature',
                   'spo2', 'respiratory_rate', 'pulse_pressure',
                   'mean_arterial_pressure', 'shock_index']

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

    # Method 2: Statistical (Clinical thresholds)
    print("\n" + "="*60)
    print("Applying Clinical Threshold Rules...")

    # Define clinical thresholds
    y_pred_clinical = (
        (data['heart_rate'] < 50) | (data['heart_rate'] > 120) |
        (data['bp_systolic'] < 90) | (data['bp_systolic'] > 160) |
        (data['bp_diastolic'] < 60) | (data['bp_diastolic'] > 100) |
        (data['temperature'] < 36.0) | (data['temperature'] > 38.3) |
        (data['spo2'] < 94) |
        (data['respiratory_rate'] < 10) | (data['respiratory_rate'] > 24)
    ).astype(int)

    metrics_dict['Clinical Rules'] = evaluate_detector(y_true, y_pred_clinical, "Clinical Threshold Rules")

    # Method 3: Z-score (Patient-specific baseline)
    print("\n" + "="*60)
    print("Applying Patient-Specific Z-Score Method...")

    # Calculate z-scores per patient
    z_scores = []
    for patient_id in data['patient_id'].unique():
        patient_mask = data['patient_id'] == patient_id
        patient_features = X[patient_mask]

        # Use normal data only for baseline
        patient_labels = y_true[patient_mask]
        normal_mask = patient_labels == 0

        if normal_mask.sum() > 5:  # Need enough normal samples
            mean = np.mean(patient_features[normal_mask], axis=0)
            std = np.std(patient_features[normal_mask], axis=0) + 1e-6

            patient_z = np.abs((patient_features - mean) / std)
            max_z = np.max(patient_z, axis=1)
            z_scores.extend(max_z)
        else:
            z_scores.extend([0] * patient_mask.sum())

    z_scores = np.array(z_scores)
    threshold_z = np.percentile(z_scores[y_true == 0], 95)
    y_pred_zscore = (z_scores > threshold_z).astype(int)

    metrics_dict['Patient Z-Score'] = evaluate_detector(y_true, y_pred_zscore, "Patient-Specific Z-Score")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Results', fontsize=16)

    methods = [
        ('Isolation Forest', y_pred_if),
        ('Clinical Rules', y_pred_clinical),
        ('Patient Z-Score', y_pred_zscore)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax,
                   xticklabels=['Normal', 'Critical'], yticklabels=['Normal', 'Critical'])
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
    ax.legend(title='Metric', loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: detection_comparison.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_recall = max(metrics_dict.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(metrics_dict.items(), key=lambda x: x[1]['f1'])

    print(f"\nBest Recall (Safety): {best_recall[0]} ({best_recall[1]['recall']:.4f})")
    print(f"Best F1-Score (Balance): {best_f1[0]} ({best_f1[1]['f1']:.4f})")

    print("\nClinical Recommendations:")
    print("- High recall is critical in healthcare (minimize false negatives)")
    print("- Clinical rules provide interpretable baseline")
    print("- Patient-specific baselines account for individual variation")
    print("- Combine methods for robust early warning system")
    print("- Adjust thresholds based on patient risk profile")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
