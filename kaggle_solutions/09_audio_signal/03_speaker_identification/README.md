# Speaker Identification System

## Overview
This solution demonstrates automatic speaker identification using voice biometrics. The system identifies speakers based on their unique vocal characteristics, including pitch, formant frequencies, and spectral patterns.

## Problem Statement
Speaker identification is crucial for security systems, personalized services, forensic analysis, and voice-controlled applications. This solution shows how to:
- Generate synthetic voices with speaker-specific characteristics
- Extract voice biometric features
- Build a robust speaker identification model
- Analyze vocal tract characteristics

## Dataset
Synthetic voice samples with speaker-specific parameters:
- **Speaker A**: Male (F0=120 Hz, formants=[700, 1220, 2600] Hz)
- **Speaker B**: Male (F0=100 Hz, formants=[650, 1080, 2500] Hz)
- **Speaker C**: Female (F0=220 Hz, formants=[800, 1400, 2800] Hz)
- **Speaker D**: Female (F0=200 Hz, formants=[850, 1500, 2900] Hz)
- **Speaker E**: Male (F0=140 Hz, formants=[720, 1300, 2650] Hz)

150 total samples (30 per speaker) at 16 kHz sample rate, 2 seconds each.

## Features

### Voice Biometric Features (28 total)

1. **Pitch Features** (2):
   - Fundamental frequency (F0) estimation
   - Pitch variation (jitter)

2. **Formant Frequencies** (5):
   - First 5 formant peaks (vocal tract resonances)

3. **Spectral Features** (3):
   - Spectral centroid
   - Spectral bandwidth
   - Spectral tilt

4. **MFCC Features** (13):
   - Mel-frequency cepstral coefficients

5. **Energy Features** (3):
   - Total energy
   - RMS energy
   - Peak amplitude

6. **Temporal Features** (2):
   - Zero crossing rate
   - Standard deviation

## Approach

### 1. Voice Synthesis
```python
- Generate glottal pulse train at speaker's F0
- Apply formant filtering (vocal tract model)
- Add harmonic richness
- Apply natural amplitude envelope
```

### 2. Feature Extraction
```python
- Pitch detection via autocorrelation
- Formant extraction from spectral peaks
- MFCC computation
- Energy and temporal statistics
```

### 3. Model Training
```python
- Support Vector Machine (RBF kernel)
- C=10, gamma='scale'
- StandardScaler for normalization
```

### 4. Evaluation
- Per-speaker accuracy analysis
- Confusion matrix
- Probability-based identification

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
The script generates:
1. **speaker_voice_characteristics.png**: Waveforms, spectra, and spectrograms
2. **speaker_identification_results.png**: Confusion matrix and accuracy analysis

## Results
Expected performance:
- Overall accuracy: 95-99%
- Male speakers: Very high (distinct F0 and formants)
- Female speakers: Very high (higher F0 range)
- Cross-gender confusion: Minimal
- Same-gender confusion: Rare (formant differences)

## Key Insights
1. **Fundamental frequency (F0)** is the primary gender discriminator
2. **Formant frequencies** uniquely identify vocal tract shape
3. **MFCC features** capture overall voice timbre
4. **Pitch variation** adds speaker-specific patterns
5. **Combination of features** provides robust identification

## Voice Characteristics

### Fundamental Frequency (F0)
- Male speakers: 80-180 Hz
- Female speakers: 180-300 Hz
- Child speakers: 300-500 Hz

### Formant Frequencies
- F1 (first formant): 200-1000 Hz (vocal tract length)
- F2 (second formant): 800-2500 Hz (tongue position)
- F3 (third formant): 2000-3500 Hz (lip rounding)

### Why Formants Matter
Formants are resonance frequencies of the vocal tract. They depend on:
- Physical dimensions (length, shape)
- Anatomical features (unique to each person)
- Articulation patterns

## Technical Details

### Glottal Pulse Train
Simulates vocal cord vibration:
```
- Square wave at fundamental frequency
- Natural pitch variation over time
- Speaker-specific F0 range
```

### Formant Filtering
Vocal tract resonance simulation:
```
- Bandpass filters at formant frequencies
- Bandwidth: ~50 Hz per formant
- Multiple formants create voice timbre
```

### Feature Importance
1. Formant frequencies: Highest
2. Fundamental frequency: High
3. MFCC coefficients: High
4. Spectral features: Medium
5. Energy features: Medium

## Extensions
- Add more speakers (scalability testing)
- Text-dependent vs text-independent identification
- Speaker verification (1:1 matching)
- Noise robustness testing
- Age and gender classification
- Emotion-invariant identification
- Real-time speaker identification
- Deep learning approaches (i-vectors, x-vectors)
- Multi-speaker diarization

## Applications
- Biometric authentication
- Forensic voice analysis
- Personalized voice assistants
- Call center automation
- Access control systems
- Smart home devices
- Video game character recognition
- Podcast speaker tracking

## Challenges
- Channel variability (microphone quality)
- Background noise
- Emotional state variations
- Health conditions affecting voice
- Aging effects on voice
- Mimicry and impersonation

## Technical Specifications
- **Sample Rate**: 16 kHz
- **Audio Duration**: 2 seconds per sample
- **Classifier**: SVM with RBF kernel
- **Feature Scaling**: StandardScaler
- **Train-Test Split**: 75-25 with stratification
