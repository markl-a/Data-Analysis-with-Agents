# Heartbeat Sound Classification

## Overview
This solution demonstrates classification of heartbeat sounds for medical diagnosis. The system classifies different cardiac conditions (normal, murmur, tachycardia, arrhythmia, bradycardia) from phonocardiogram (PCG) audio.

## Problem Statement
Automated heartbeat analysis is crucial for:
- Early detection of cardiac abnormalities
- Remote patient monitoring
- Screening in resource-limited settings
- Telemedicine applications
- Continuous cardiac monitoring

## Cardiac Conditions

### Normal
- **Heart Rate**: 60-100 BPM
- **Pattern**: Regular lub-dub (S1-S2)
- **S1 Sound**: 40 Hz, 120ms duration
- **S2 Sound**: 60 Hz, 80ms duration
- **Interval**: Regular intervals

### Murmur
- **Heart Rate**: 65-95 BPM
- **Pattern**: Lub-whoosh-dub
- **Additional Sound**: Whooshing between S1 and S2
- **Cause**: Turbulent blood flow
- **Frequency**: Low-pass filtered noise (< 200 Hz)

### Tachycardia
- **Heart Rate**: >100 BPM (110-150)
- **Pattern**: Rapid lub-dub
- **Characteristics**: Shorter intervals, faster sounds
- **S1/S2 Duration**: Reduced due to speed

### Arrhythmia
- **Heart Rate**: Variable
- **Pattern**: Irregular intervals
- **Characteristics**: Unpredictable beat timing
- **Variability**: High interval variance

### Bradycardia
- **Heart Rate**: <60 BPM (40-55)
- **Pattern**: Slow lub-dub
- **Characteristics**: Longer intervals between beats
- **S1/S2 Duration**: Slightly prolonged

## Dataset
200 synthetic heartbeat samples (40 per condition):
- Sample rate: 4 kHz (sufficient for heart sounds)
- Duration: 5 seconds per sample
- Synthesized with condition-specific characteristics

## Features Extracted (27 total)

### 1. Heart Rate Features (3)
- **Heart rate**: Beats per minute
- **Heart rate variability**: Interval standard deviation
- **Number of beats**: Total detected beats

### 2. Frequency Features (4)
- **Low-frequency power** (20-60 Hz): S1/S2 sounds
- **Mid-frequency power** (60-150 Hz): Murmurs
- **High-frequency power** (150+ Hz): Noise
- **Spectral centroid**: Frequency distribution

### 3. Temporal Features (2)
- **S1-S2 interval**: Systolic period
- **S1-S2 variability**: Interval variance

### 4. Energy Features (3)
- Total energy
- RMS energy
- Peak amplitude

### 5. MFCC Features (8)
- Mel-frequency cepstral coefficients

### 6. Statistical Features (4)
- Mean, std, 25th/75th percentiles

## Cardiac Cycle

### Heart Sounds
```
S1 (Lub) → Systole → S2 (Dub) → Diastole → S1 ...
   ↓                      ↓
Mitral/Tricuspid    Aortic/Pulmonic
valve closure        valve closure
```

### Normal Timing
- **S1**: Beginning of systole (~120ms)
- **S1-S2 interval**: Systolic period (~200-300ms)
- **S2**: Beginning of diastole (~80ms)
- **S2-S1 interval**: Diastolic period (~400-600ms)

### Frequency Characteristics
- **S1**: 20-50 Hz (lower pitch, longer)
- **S2**: 50-70 Hz (higher pitch, shorter)
- **Murmurs**: 60-300 Hz (turbulent flow)
- **S3/S4 (abnormal)**: <50 Hz (low frequency)

## Approach

### 1. Sound Generation
```python
- Generate S1 (lub) at appropriate frequency
- Generate S2 (dub) after systolic interval
- Add condition-specific characteristics
- Apply realistic timing and amplitude
```

### 2. Feature Extraction
```python
- Detect heartbeat peaks (envelope detection)
- Calculate heart rate and variability
- Extract frequency domain features
- Analyze temporal patterns
```

### 3. Classification
```python
- Random Forest (200 trees)
- Class balancing for medical data
- Feature standardization
```

## Requirements
```
numpy
scipy
scikit-learn
matplotlib
seaborn
```

## Usage
```bash
python solution.py
```

## Output
1. **heartbeat_patterns.png**: Waveforms and spectra for each condition
2. **heartbeat_results.png**: Confusion matrix, metrics, and ROC curves

## Results
Expected performance:
- Overall accuracy: 85-92%
- Normal: High accuracy (baseline pattern)
- Tachycardia/Bradycardia: Very high (heart rate discriminates)
- Murmur: Good (unique whooshing signature)
- Arrhythmia: Good (irregular intervals)

## Key Insights
1. **Heart rate** is primary discriminator for tachycardia/bradycardia
2. **Interval variability** identifies arrhythmia
3. **Mid-frequency power** detects murmurs
4. **Temporal patterns** distinguish normal from abnormal
5. **S1-S2 interval** provides diagnostic information

## Medical Significance

### S1 (First Heart Sound)
- **Cause**: Closure of mitral and tricuspid valves
- **Timing**: Beginning of ventricular systole
- **Frequency**: Lower (20-50 Hz)
- **Duration**: Longer (100-140ms)

### S2 (Second Heart Sound)
- **Cause**: Closure of aortic and pulmonic valves
- **Timing**: End of ventricular systole
- **Frequency**: Higher (50-70 Hz)
- **Duration**: Shorter (60-100ms)

### Murmurs
- **Cause**: Turbulent blood flow
- **Types**:
  - Systolic: Between S1 and S2
  - Diastolic: Between S2 and S1
  - Continuous: Throughout cycle
- **Grading**: I-VI scale
- **Significance**: May indicate valve disease

### Arrhythmias
- **Atrial fibrillation**: Irregularly irregular
- **Premature beats**: Extra beats
- **Heart block**: Missed beats
- **Significance**: Can be benign or life-threatening

## Extensions
- Add more conditions (S3, S4, split sounds)
- Multi-label classification (multiple abnormalities)
- Severity grading for murmurs
- Real PCG dataset (PhysioNet, PASCAL, CirCor)
- Deep learning models (CNN on spectrograms)
- Real-time monitoring
- Beat-by-beat analysis
- Heart rate variability (HRV) analysis
- Integration with ECG data
- Automated diagnosis assistance

## Technical Details
- **Sample Rate**: 4 kHz
- **Duration**: 5 seconds per sample
- **Classifier**: Random Forest (200 estimators)
- **Class Balancing**: Enabled for medical data

## Applications
- **Clinical**: Cardiac screening, diagnosis assistance
- **Telehealth**: Remote cardiac monitoring
- **Wearables**: Continuous health tracking
- **Mobile Health**: Smartphone-based screening
- **Resource-Limited**: Low-cost diagnosis tool
- **Emergency**: Rapid triage
- **Pediatric**: Non-invasive cardiac assessment

## Challenges
- Noise from breathing, movement
- Recording quality variations
- Individual anatomical differences
- Overlapping conditions
- Subtle abnormalities
- Need for medical validation
- Regulatory requirements for medical devices

## Feature Importance Ranking
1. Heart rate: Highest
2. Heart rate variability: Very high
3. Mid-frequency power: High (for murmurs)
4. S1-S2 interval: High
5. Temporal features: Medium

## Clinical Validation
- Sensitivity and specificity reporting
- ROC curve analysis
- Comparison with cardiologist diagnosis
- Multi-center validation
- False positive/negative analysis
- Clinical decision support integration

## Regulatory Considerations
- Medical device classification
- FDA/CE approval requirements
- Clinical trial evidence
- Risk management
- Quality management systems
- Post-market surveillance
