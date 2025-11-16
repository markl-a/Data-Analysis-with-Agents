# Industrial Equipment Anomaly Detection

## Overview
This example demonstrates predictive maintenance for industrial equipment using multiple anomaly detection techniques including deep learning autoencoders, Isolation Forest, and statistical methods to detect equipment failures before they occur.

## Problem Description
Industrial equipment failures can cause:
- **Costly downtime**: Production stops
- **Safety hazards**: Equipment malfunctions
- **Maintenance costs**: Emergency repairs

Early detection enables:
- Scheduled maintenance during planned downtime
- Prevention of catastrophic failures
- Optimization of spare parts inventory

## Dataset
Synthetic equipment sensor data with 20,000 readings:
- **Normal operation (97%)**: Standard operating conditions
- **Anomalies (3%)**: Various failure modes

### Sensor Features
- `temperature`: Operating temperature (°C)
- `pressure`: System pressure (PSI)
- `vibration`: Vibration level (g)
- `rpm`: Rotational speed (RPM)
- `current`: Electrical current (A)
- `power`: Power consumption (W)

### Derived Features
- `temp_pressure_ratio`: Temperature-to-pressure ratio
- `power_efficiency`: Power consumption efficiency
- `vibration_rpm_ratio`: Vibration intensity relative to speed

## Failure Modes Detected

### 1. Overheating
- High temperature (90-110°C)
- Elevated pressure
- Increased vibration
- Reduced RPM
- High current draw

### 2. Pressure Spikes
- Extreme pressure (120-150 PSI)
- Moderate vibration increase
- Normal to high RPM
- Moderate current

### 3. Excessive Vibration
- Very high vibration (1.5-3.0 g)
- Unstable RPM
- Normal temperature/pressure
- Variable current

### 4. Electrical Anomalies
- Current spikes (15-25 A)
- Reduced RPM
- Normal vibration
- Normal temperature/pressure

## Methods Used

### 1. Autoencoder (Deep Learning)
- **Architecture**: 9 → 16 → 8 → 16 → 9
- **Training**: Only on normal data
- **Detection**: Reconstruction error > threshold
- **Strengths**: Captures complex multivariate relationships
- **Threshold**: 95th percentile of normal reconstruction errors

### 2. Isolation Forest
- **Approach**: Tree-based isolation
- **Configuration**: 100 estimators, 3% contamination
- **Strengths**: Fast, interpretable, handles outliers well

### 3. Statistical Mahalanobis Distance
- **Approach**: Distance from normal distribution
- **Metric**: Mahalanobis distance using covariance
- **Strengths**: Statistical foundation, interpretable
- **Threshold**: 95th percentile of normal distances

## Evaluation Metrics
- **Precision**: Accuracy of anomaly predictions
- **Recall**: Percentage of failures detected
- **F1-Score**: Balance between precision and recall
- **Specificity**: Correct identification of normal operation

## Results Visualizations
1. **equipment_sensor_distributions.png**: Sensor reading patterns
2. **feature_correlation.png**: Feature relationships
3. **autoencoder_reconstruction_error.png**: Reconstruction error analysis
4. **method_comparison.png**: Performance comparison

## Key Insights
- Autoencoder excels at detecting subtle multivariate anomalies
- Isolation Forest provides fast, reliable detection
- Statistical methods offer interpretable baseline
- Multiple sensors together improve detection accuracy
- Different failure modes have distinct signatures

## Usage
```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow (optional, for autoencoder)

## Operational Deployment

### Real-Time Monitoring
1. Collect sensor data continuously
2. Apply standardization using fitted scaler
3. Run all three detectors in parallel
4. Alert if 2+ detectors agree (ensemble voting)

### Alert Levels
- **Low**: Single detector flags anomaly
- **Medium**: Two detectors agree
- **High**: All three detectors agree

### Threshold Tuning
- Adjust based on maintenance cost vs. downtime cost
- Lower threshold: More false alarms, fewer missed failures
- Higher threshold: Fewer false alarms, risk missing failures

## Extensions
1. Add time-series LSTM autoencoder for temporal patterns
2. Implement online learning for concept drift
3. Add specific failure mode classification
4. Integrate with maintenance scheduling system
5. Use transfer learning for similar equipment
6. Add explainability (SHAP values) for detected anomalies
7. Implement streaming pipeline with Apache Kafka

## Real-World Applications
- Manufacturing equipment
- HVAC systems
- Wind turbines
- Industrial pumps and compressors
- CNC machines
- Power generation equipment
