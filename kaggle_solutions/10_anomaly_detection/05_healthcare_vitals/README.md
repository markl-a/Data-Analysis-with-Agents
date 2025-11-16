# Healthcare Vital Signs Anomaly Detection

## Overview
This example demonstrates early warning system for critical patient conditions by detecting anomalous vital signs using machine learning, clinical rule-based, and patient-specific statistical methods.

## Problem Description
Patient vital sign monitoring is critical for:
- **Early intervention**: Detecting deterioration before crisis
- **ICU monitoring**: Continuous patient surveillance
- **Alert prioritization**: Reducing alarm fatigue
- **Resource allocation**: Identifying high-risk patients

## Dataset
Synthetic patient monitoring data:
- **100 patients** with individual baselines
- **200 measurements per patient** (20,000 total)
- **5% critical conditions** requiring immediate attention

### Vital Signs Monitored
- `heart_rate`: Heart rate (bpm) - Normal: 60-100
- `bp_systolic`: Systolic blood pressure (mmHg) - Normal: 90-140
- `bp_diastolic`: Diastolic blood pressure (mmHg) - Normal: 60-90
- `temperature`: Body temperature (°C) - Normal: 36.5-37.5
- `spo2`: Oxygen saturation (%) - Normal: 95-100
- `respiratory_rate`: Breaths per minute - Normal: 12-20

### Derived Clinical Indices
- `pulse_pressure`: Systolic - Diastolic
- `mean_arterial_pressure`: Diastolic + Pulse Pressure/3
- `shock_index`: Heart Rate / Systolic BP

## Critical Conditions Detected

### 1. Tachycardia
- Heart rate: 120-180 bpm
- Often indicates: Pain, stress, infection, cardiac issues
- Associated: Elevated respiratory rate

### 2. Bradycardia
- Heart rate: 40-55 bpm
- Indicates: Heart block, medication effects
- Associated: Reduced blood pressure

### 3. Hypertension
- Systolic BP: 160-200 mmHg
- Diastolic BP: 100-120 mmHg
- Risk: Stroke, cardiac events

### 4. Hypotension
- Systolic BP: 70-90 mmHg
- Indicates: Shock, bleeding, dehydration
- Associated: Compensatory tachycardia, low SpO2

### 5. Fever
- Temperature: 38.5-40.5°C
- Indicates: Infection, inflammation
- Associated: Tachycardia, tachypnea

### 6. Hypoxia
- SpO2: 75-92%
- Critical: Oxygen deficiency
- Associated: Tachycardia, tachypnea

## Methods Used

### 1. Isolation Forest
- **Application**: General anomaly detection
- **Strengths**: Handles multivariate patterns, no assumptions
- **Configuration**: 100 estimators, 5% contamination
- **Use case**: Automated screening

### 2. Clinical Threshold Rules
- **Based on**: Established medical guidelines
- **Thresholds**:
  - HR: <50 or >120 bpm
  - Systolic BP: <90 or >160 mmHg
  - Diastolic BP: <60 or >100 mmHg
  - Temperature: <36.0 or >38.3°C
  - SpO2: <94%
  - RR: <10 or >24 breaths/min
- **Strengths**: Interpretable, clinically validated
- **Use case**: Rule-based alerting

### 3. Patient-Specific Z-Score
- **Approach**: Deviation from individual baseline
- **Calculation**: Z-score using patient's normal data
- **Threshold**: 95th percentile of normal deviations
- **Strengths**: Accounts for individual variation
- **Use case**: Personalized monitoring

## Evaluation Metrics
- **Precision**: Accuracy of critical alerts
- **Recall (Sensitivity)**: Percentage of critical conditions detected
- **F1-Score**: Balance between precision and recall
- **Specificity**: Correct identification of normal conditions

**Note**: In healthcare, high recall is prioritized (minimize missed critical conditions)

## Results Visualizations
1. **patient_vitals_timeline.png**: Individual patient vital trends
2. **vital_distributions.png**: Normal vs. critical distributions
3. **detection_comparison.png**: Method performance comparison

## Key Insights
- Patient-specific baselines improve detection of subtle changes
- Clinical rules provide interpretable and validated thresholds
- Machine learning detects complex multivariate patterns
- Combination of methods reduces false alerts
- Recall is more important than precision in critical care

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
- scipy

## Clinical Deployment

### Early Warning Score System
1. Collect vitals every 5-15 minutes
2. Calculate derived indices
3. Run all three detection methods
4. Generate alert if 2+ methods agree

### Alert Levels
- **RED (Critical)**: All 3 methods detect + vital extremely abnormal
- **YELLOW (Warning)**: 2 methods detect anomaly
- **GREEN (Monitor)**: 1 method detects anomaly

### Integration with Hospital Systems
- Electronic Health Records (EHR)
- Nurse call systems
- ICU monitoring dashboards
- Mobile alerts for physicians
- Automated vital sign collection from bedside monitors

### Customization by Unit
- **ICU**: Lower thresholds, higher sensitivity
- **General ward**: Standard thresholds
- **Post-surgical**: Adjusted for expected variations
- **Pediatric**: Age-specific normal ranges
- **Geriatric**: Account for baseline differences

## Safety Considerations
1. **Never replace clinical judgment**: Algorithms support decisions
2. **Regular validation**: Monitor false positive/negative rates
3. **Threshold tuning**: Adjust based on patient population
4. **Alarm fatigue**: Balance sensitivity with specificity
5. **Documentation**: Log all alerts and responses
6. **Training**: Ensure staff understand system limitations

## Extensions
1. Add trend analysis (vital sign trajectories)
2. Implement LSTM for temporal pattern detection
3. Include lab values (lactate, troponin, etc.)
4. Add patient demographics and comorbidities
5. Develop condition-specific models (sepsis, cardiac)
6. Implement explainable AI for clinical transparency
7. Add medication interaction checks
8. Create automated escalation protocols

## Real-World Applications
- Intensive Care Units (ICU)
- Emergency Departments
- Post-operative recovery
- Remote patient monitoring
- Telemedicine
- Nursing homes
- Ambulance monitoring
