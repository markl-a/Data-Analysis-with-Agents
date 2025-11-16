"""
Manufacturing Quality Control - Defect Detection
=================================================

This solution predicts manufacturing defects using ensemble methods
with comprehensive process parameter and sensor data features.

Business Context:
- Manufacturing defects cost billions in recalls and rework
- Early detection prevents downstream costs
- Quality control critical for customer satisfaction
- Predictive maintenance reduces equipment failures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_manufacturing_data(n_samples=4500):
    """
    Generate realistic manufacturing quality control data

    Features include:
    - Process parameters (temperature, pressure, speed)
    - Material properties
    - Machine settings and conditions
    - Sensor measurements
    - Environmental factors
    - Operator and shift information
    """
    print("Generating manufacturing quality control data...")

    # Production line and machine
    production_line = np.random.choice(['Line_A', 'Line_B', 'Line_C', 'Line_D'],
                                      n_samples, p=[0.30, 0.25, 0.25, 0.20])

    machine_id = np.random.choice(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
                                  n_samples)

    # Machine age (years) - older machines more prone to issues
    machine_age = np.random.choice([1, 2, 3, 5, 7, 10, 15], n_samples,
                                   p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])

    # Shift information
    shift = np.random.choice(['Morning', 'Afternoon', 'Night'], n_samples,
                            p=[0.40, 0.35, 0.25])

    operator_id = np.random.choice(range(1, 21), n_samples)  # 20 operators

    # Batch and material information
    batch_id = np.random.choice(range(1, 101), n_samples)  # 100 batches
    material_grade = np.random.choice(['Grade_A', 'Grade_B', 'Grade_C'],
                                     n_samples, p=[0.60, 0.30, 0.10])

    # Process parameters - temperature
    target_temp = 180  # degrees Celsius
    temp_variation = np.random.normal(0, 5, n_samples)
    temperature = target_temp + temp_variation

    # Add anomalies for some samples
    temp_anomalies = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    temperature = np.where(temp_anomalies == 1,
                          temperature + np.random.uniform(-20, 20, n_samples),
                          temperature)

    # Process parameters - pressure
    target_pressure = 100  # PSI
    pressure_variation = np.random.normal(0, 3, n_samples)
    pressure = target_pressure + pressure_variation

    pressure_anomalies = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
    pressure = np.where(pressure_anomalies == 1,
                       pressure + np.random.uniform(-15, 15, n_samples),
                       pressure)

    # Process parameters - speed
    target_speed = 50  # units per minute
    speed_variation = np.random.normal(0, 2, n_samples)
    production_speed = target_speed + speed_variation

    # Humidity and ambient temperature
    ambient_temp = np.random.normal(22, 3, n_samples).clip(15, 30)
    humidity = np.random.normal(45, 10, n_samples).clip(20, 80)

    # Vibration level (from sensors)
    base_vibration = np.random.normal(2.0, 0.5, n_samples)
    vibration = base_vibration + (machine_age / 15) * 1.5  # Older machines vibrate more

    # Power consumption
    base_power = 75  # kW
    power_consumption = base_power + np.random.normal(0, 5, n_samples)

    # Material thickness (with tolerance)
    target_thickness = 2.5  # mm
    thickness = np.random.normal(target_thickness, 0.1, n_samples)

    # Measurement sensors
    sensor_1 = np.random.normal(100, 5, n_samples)  # Generic sensor reading
    sensor_2 = np.random.normal(50, 3, n_samples)
    sensor_3 = np.random.normal(75, 4, n_samples)

    # Surface roughness
    surface_roughness = np.random.normal(1.6, 0.3, n_samples).clip(0.5, 3.5)

    # Cycle time
    target_cycle_time = 120  # seconds
    cycle_time = np.random.normal(target_cycle_time, 8, n_samples).clip(90, 180)

    # Tool wear percentage
    tool_wear = np.random.uniform(0, 100, n_samples)

    # Maintenance hours since last service
    hours_since_maintenance = np.random.exponential(200, n_samples).clip(0, 1000)

    # Number of defects in previous batch
    prev_batch_defects = np.random.choice([0, 0, 0, 1, 2, 3, 5],
                                         n_samples, p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.007, 0.003])

    # Calculate defect probability based on features
    defect_score = (
        0.10 * (np.abs(temperature - target_temp) > 10) +
        0.10 * (np.abs(pressure - target_pressure) > 8) +
        0.08 * (vibration > 4.0) +
        0.08 * (machine_age > 8) +
        0.07 * (tool_wear > 70) +
        0.07 * (hours_since_maintenance > 500) +
        0.06 * (shift == 'Night') +
        0.06 * (material_grade == 'Grade_C') +
        0.05 * (np.abs(thickness - target_thickness) > 0.15) +
        0.05 * (cycle_time < 100) +  # Too fast
        0.05 * (surface_roughness > 2.5) +
        0.05 * (humidity > 65) +
        0.04 * (prev_batch_defects > 0) +
        0.04 * (ambient_temp > 27) +
        0.05 * np.random.random(n_samples)
    )

    # Generate defect labels
    defect = (defect_score > 0.25).astype(int)

    # Defect type (for defective items)
    defect_types = np.where(
        defect == 1,
        np.random.choice(['Surface', 'Dimensional', 'Material', 'Assembly'],
                        n_samples, p=[0.40, 0.30, 0.20, 0.10]),
        'None'
    )

    # Create DataFrame
    data = pd.DataFrame({
        'ProductionLine': production_line,
        'MachineID': machine_id,
        'MachineAge': machine_age,
        'Shift': shift,
        'OperatorID': operator_id,
        'BatchID': batch_id,
        'MaterialGrade': material_grade,
        'Temperature': temperature,
        'Pressure': pressure,
        'ProductionSpeed': production_speed,
        'AmbientTemp': ambient_temp,
        'Humidity': humidity,
        'Vibration': vibration,
        'PowerConsumption': power_consumption,
        'Thickness': thickness,
        'Sensor1': sensor_1,
        'Sensor2': sensor_2,
        'Sensor3': sensor_3,
        'SurfaceRoughness': surface_roughness,
        'CycleTime': cycle_time,
        'ToolWear': tool_wear,
        'HoursSinceMaintenance': hours_since_maintenance,
        'PrevBatchDefects': prev_batch_defects,
        'DefectType': defect_types,
        'Defect': defect
    })

    return data

def engineer_quality_features(df):
    """Create advanced quality control features"""
    print("Engineering quality control features...")

    df_eng = df.copy()

    # Process deviation metrics
    df_eng['TempDeviation'] = np.abs(df_eng['Temperature'] - 180)
    df_eng['PressureDeviation'] = np.abs(df_eng['Pressure'] - 100)
    df_eng['ThicknessDeviation'] = np.abs(df_eng['Thickness'] - 2.5)

    # Combined process deviation score
    df_eng['ProcessDeviationScore'] = (
        df_eng['TempDeviation'] / 20 +
        df_eng['PressureDeviation'] / 15 +
        df_eng['ThicknessDeviation'] / 0.2
    ) / 3

    # Machine health indicators
    df_eng['MachineHealthScore'] = (
        (1 - df_eng['MachineAge'] / 15) * 0.3 +
        (1 - df_eng['Vibration'] / 10) * 0.3 +
        (1 - df_eng['ToolWear'] / 100) * 0.2 +
        (1 - df_eng['HoursSinceMaintenance'] / 1000) * 0.2
    ).clip(0, 1)

    # Environmental stress index
    df_eng['EnvironmentalStress'] = (
        (df_eng['Humidity'] - 45) / 35 * 0.5 +
        (df_eng['AmbientTemp'] - 22) / 8 * 0.5
    )

    # Optimal operating range indicators
    df_eng['TempInRange'] = ((df_eng['Temperature'] >= 170) &
                             (df_eng['Temperature'] <= 190)).astype(int)

    df_eng['PressureInRange'] = ((df_eng['Pressure'] >= 92) &
                                  (df_eng['Pressure'] <= 108)).astype(int)

    df_eng['ThicknessInRange'] = ((df_eng['Thickness'] >= 2.35) &
                                   (df_eng['Thickness'] <= 2.65)).astype(int)

    # All critical parameters in range
    df_eng['AllInRange'] = (df_eng['TempInRange'] &
                            df_eng['PressureInRange'] &
                            df_eng['ThicknessInRange']).astype(int)

    # Speed-quality relationship
    df_eng['SpeedCategory'] = pd.cut(df_eng['ProductionSpeed'],
                                      bins=[0, 45, 55, 100],
                                      labels=['Slow', 'Optimal', 'Fast'])

    # Sensor correlation feature
    df_eng['SensorAvg'] = (df_eng['Sensor1'] + df_eng['Sensor2'] + df_eng['Sensor3']) / 3
    df_eng['SensorStd'] = df_eng[['Sensor1', 'Sensor2', 'Sensor3']].std(axis=1)

    # Maintenance urgency
    df_eng['MaintenanceUrgency'] = (
        (df_eng['HoursSinceMaintenance'] / 1000) * 0.5 +
        (df_eng['ToolWear'] / 100) * 0.5
    )

    # Historical quality indicator
    df_eng['HistoricalQualityIssue'] = (df_eng['PrevBatchDefects'] > 0).astype(int)

    # Power efficiency
    expected_power = 75
    df_eng['PowerEfficiency'] = df_eng['PowerConsumption'] / expected_power

    # Quality risk score
    df_eng['QualityRiskScore'] = (
        df_eng['ProcessDeviationScore'] * 0.3 +
        (1 - df_eng['MachineHealthScore']) * 0.3 +
        df_eng['MaintenanceUrgency'] * 0.2 +
        (df_eng['SurfaceRoughness'] / 3.5) * 0.1 +
        (df_eng['PrevBatchDefects'] / 5) * 0.1
    )

    # Shift risk (night shift typically higher defects)
    df_eng['ShiftRisk'] = df_eng['Shift'].map({
        'Morning': 0.0,
        'Afternoon': 0.3,
        'Night': 0.7
    })

    return df_eng

def create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names):
    """Create comprehensive quality control visualizations"""
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Defect rate by production line
    ax1 = plt.subplot(3, 4, 1)
    defect_by_line = df.groupby('ProductionLine')['Defect'].mean().sort_values(ascending=False)
    ax1.bar(defect_by_line.index, defect_by_line.values, color='steelblue', alpha=0.7)
    ax1.set_title('Defect Rate by Production Line', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Production Line')
    ax1.set_ylabel('Defect Rate')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Defect rate by machine age
    ax2 = plt.subplot(3, 4, 2)
    defect_by_age = df.groupby('MachineAge')['Defect'].mean()
    ax2.plot(defect_by_age.index, defect_by_age.values, marker='o', linewidth=2, color='darkred')
    ax2.set_title('Defect Rate by Machine Age', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Machine Age (years)')
    ax2.set_ylabel('Defect Rate')
    ax2.grid(True, alpha=0.3)

    # 3. Temperature distribution by defect
    ax3 = plt.subplot(3, 4, 3)
    df[df['Defect'] == 0]['Temperature'].hist(bins=40, alpha=0.6, label='No Defect',
                                                color='green', ax=ax3)
    df[df['Defect'] == 1]['Temperature'].hist(bins=40, alpha=0.6, label='Defect',
                                                color='red', ax=ax3)
    ax3.axvline(180, color='black', linestyle='--', linewidth=2, label='Target')
    ax3.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 4. Pressure distribution by defect
    ax4 = plt.subplot(3, 4, 4)
    df[df['Defect'] == 0]['Pressure'].hist(bins=40, alpha=0.6, label='No Defect',
                                            color='green', ax=ax4)
    df[df['Defect'] == 1]['Pressure'].hist(bins=40, alpha=0.6, label='Defect',
                                            color='red', ax=ax4)
    ax4.axvline(100, color='black', linestyle='--', linewidth=2, label='Target')
    ax4.set_title('Pressure Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Pressure (PSI)')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # 5. Vibration vs Machine Age
    ax5 = plt.subplot(3, 4, 5)
    for defect in [0, 1]:
        subset = df[df['Defect'] == defect].sample(min(300, len(df[df['Defect'] == defect])))
        ax5.scatter(subset['MachineAge'], subset['Vibration'], alpha=0.5, label=f"Defect={defect}")
    ax5.set_title('Vibration vs Machine Age', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Machine Age (years)')
    ax5.set_ylabel('Vibration Level')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Defect rate by shift
    ax6 = plt.subplot(3, 4, 6)
    defect_by_shift = df.groupby('Shift')['Defect'].mean()
    shift_order = ['Morning', 'Afternoon', 'Night']
    defect_by_shift = defect_by_shift.reindex(shift_order)
    ax6.bar(defect_by_shift.index, defect_by_shift.values, color='purple', alpha=0.7)
    ax6.set_title('Defect Rate by Shift', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Shift')
    ax6.set_ylabel('Defect Rate')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Defect type distribution
    ax7 = plt.subplot(3, 4, 7)
    defect_types = df[df['Defect'] == 1]['DefectType'].value_counts()
    ax7.pie(defect_types.values, labels=defect_types.index, autopct='%1.1f%%',
           colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'])
    ax7.set_title('Defect Type Distribution', fontsize=12, fontweight='bold')

    # 8. Material grade impact
    ax8 = plt.subplot(3, 4, 8)
    defect_by_material = df.groupby('MaterialGrade')['Defect'].mean().sort_values(ascending=False)
    ax8.bar(defect_by_material.index, defect_by_material.values, color='coral', alpha=0.7)
    ax8.set_title('Defect Rate by Material Grade', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Material Grade')
    ax8.set_ylabel('Defect Rate')
    ax8.grid(axis='y', alpha=0.3)

    # 9. Tool wear vs defects
    ax9 = plt.subplot(3, 4, 9)
    df.boxplot(column='ToolWear', by='Defect', ax=ax9)
    ax9.set_title('Tool Wear by Defect Status', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Defect (0=No, 1=Yes)')
    ax9.set_ylabel('Tool Wear (%)')
    plt.suptitle('')

    # 10. Feature importance
    ax10 = plt.subplot(3, 4, 10)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(15)

    ax10.barh(importance_df['feature'], importance_df['importance'], color='darkgreen')
    ax10.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Importance')

    # 11. Confusion matrix
    ax11 = plt.subplot(3, 4, 11)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax11, cbar=False)
    ax11.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Predicted')
    ax11.set_ylabel('Actual')

    # 12. ROC Curve
    ax12 = plt.subplot(3, 4, 12)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax12.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}', color='darkorange')
    ax12.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax12.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax12.set_xlabel('False Positive Rate')
    ax12.set_ylabel('True Positive Rate')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quality_control_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'quality_control_analysis.png'")
    plt.close()

def main():
    print("="*60)
    print("Manufacturing Quality Control - Defect Detection")
    print("="*60)

    # Generate data
    df = generate_manufacturing_data(n_samples=4500)
    print(f"\nDataset shape: {df.shape}")
    print(f"Defect rate: {df['Defect'].mean():.2%}")
    print(f"\nDefect types distribution:")
    print(df[df['Defect'] == 1]['DefectType'].value_counts())

    # Engineer features
    df_eng = engineer_quality_features(df)

    # Encode categorical variables
    categorical_cols = ['ProductionLine', 'MachineID', 'Shift', 'MaterialGrade',
                       'DefectType', 'SpeedCategory']

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col].astype(str))
        label_encoders[col] = le

    # Prepare features
    X = df_eng.drop('Defect', axis=1)
    y = df_eng['Defect']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train ensemble models
    print("\nTraining ensemble models...")

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1
    )

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    # Logistic Regression (for diversity)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    # Create voting ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[2, 2, 1]  # Give more weight to tree models
    )

    # Train on appropriate data
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    gb_model.fit(X_train, y_train)

    print("Training Logistic Regression...")
    lr_model.fit(X_train_scaled, y_train)

    print("Creating ensemble model...")
    # For ensemble, we need to retrain with scaled data for LR
    ensemble_model.fit(X_train, y_train)

    # Predictions
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("Model Performance (Ensemble)")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Individual model performance
    print("\nIndividual Model Performance:")
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    print(f"Random Forest AUC: {roc_auc_score(y_test, rf_pred_proba):.4f}")
    print(f"Gradient Boosting AUC: {roc_auc_score(y_test, gb_pred_proba):.4f}")
    print(f"Logistic Regression AUC: {roc_auc_score(y_test, lr_pred_proba):.4f}")

    # Create visualizations
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns.tolist()

    create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
