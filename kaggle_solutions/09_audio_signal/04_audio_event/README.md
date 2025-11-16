# Audio Event Detection

## Overview
This solution demonstrates automatic detection and classification of audio events in continuous audio streams. The system identifies various sound events (bell, alarm, door slam, glass break, applause) from audio data using machine learning.

## Problem Statement
Audio event detection is essential for:
- Security and surveillance systems
- Smart home automation
- Industrial monitoring
- Healthcare applications
- Environmental monitoring

This solution shows how to detect and classify discrete acoustic events in audio streams.

## Dataset
Synthetic audio events with distinctive characteristics:
- **Silence**: Low-amplitude background noise
- **Bell**: 880 Hz tone with harmonics and exponential decay
- **Alarm**: Oscillating square wave (1000-1200 Hz at 4 Hz)
- **Door Slam**: White noise burst with fast decay + low-frequency thump (60 Hz)
- **Glass Break**: High-frequency noise bursts (>2000 Hz)
- **Applause**: Multiple random claps (20-40 claps per sample)

300 total samples (50 per event type) at 16 kHz sample rate, 0.5 seconds each.

## Features

### Event Features Extracted (25 total)

1. **Energy Features** (3):
   - Total energy
   - RMS energy
   - Peak amplitude

2. **Temporal Features** (3):
   - Zero crossing rate
   - Temporal centroid (energy concentration in time)
   - Decay rate (energy decrease rate)

3. **Spectral Features** (4):
   - Spectral centroid
   - Spectral bandwidth
   - Spectral rolloff
   - Spectral flatness

4. **Frequency Band Energies** (6):
   - Low band (0-500 Hz) energy
   - Mid band (500-2000 Hz) energy
   - High band (2000+ Hz) energy
   - Energy ratios for each band

5. **Statistical Features** (6):
   - Mean, standard deviation
   - Min, max values
   - 25th and 75th percentiles

## Approach

### 1. Event Sound Generation
```python
- Bell: Harmonic series with exponential decay
- Alarm: Square wave with frequency oscillation
- Door Slam: Noise burst with low-frequency impact
- Glass Break: High-pass filtered noise bursts
- Applause: Random clap sequences
```

### 2. Feature Extraction
```python
- Temporal envelope analysis
- FFT for frequency content
- Band-specific energy computation
- Decay characteristics
```

### 3. Model Training
```python
- Random Forest Classifier (150 trees)
- Max depth: 20
- Feature standardization
```

### 4. Stream Processing
- Continuous audio stream generation
- Sliding window analysis (0.5s windows)
- Real-time event detection capability

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
1. **audio_event_types.png**: Waveforms and spectrograms for all event types
2. **audio_event_results.png**: Confusion matrix and per-event metrics

## Results
Expected performance:
- Overall accuracy: 95-98%
- Silence: Very high (unique low-energy signature)
- Bell: High (distinctive decay pattern and harmonics)
- Alarm: Very high (unique frequency oscillation)
- Door Slam: High (characteristic impact signature)
- Glass Break: High (high-frequency concentration)
- Applause: Good (random burst pattern)

## Key Insights
1. **Temporal envelope** (decay pattern) strongly distinguishes impact sounds
2. **Spectral centroid** separates low-frequency from high-frequency events
3. **Energy distribution** across frequency bands identifies event types
4. **Decay rate** distinguishes impulsive (slam, break) from sustained (alarm) events
5. **Spectral flatness** separates tonal (bell, alarm) from noise-like (applause, break) events

## Event Characteristics

### Bell
- **Frequency**: Harmonic series (880, 1760, 2640 Hz)
- **Envelope**: Exponential decay
- **Duration**: Sustained with long decay
- **Spectrum**: Strong harmonic peaks

### Alarm
- **Frequency**: Alternating 1000-1200 Hz
- **Envelope**: Constant amplitude
- **Pattern**: Regular oscillation (4 Hz)
- **Spectrum**: Strong fundamental with harmonics

### Door Slam
- **Frequency**: Broadband with low-frequency emphasis
- **Envelope**: Very fast decay (<0.1s)
- **Impact**: High initial amplitude
- **Spectrum**: Low-frequency peak at 60 Hz

### Glass Break
- **Frequency**: High-frequency dominant (>2000 Hz)
- **Envelope**: Multiple short bursts
- **Pattern**: Irregular fragmentation
- **Spectrum**: High-pass characteristic

### Applause
- **Frequency**: Broadband noise
- **Envelope**: Random impulse train
- **Pattern**: 20-40 claps per sample
- **Spectrum**: Flat across frequencies

## Extensions
- Add more event types (footsteps, car horn, gunshot, etc.)
- Multi-label detection (overlapping events)
- Real-time streaming detection
- Event localization (start/end time estimation)
- Noise-robust detection
- Deep learning models (CNNs on spectrograms)
- Online learning for new event types
- Event counting and statistics
- Acoustic scene classification
- Audio segmentation

## Technical Details
- **Sample Rate**: 16 kHz
- **Window Size**: 0.5 seconds
- **Overlap**: Can be configured for sliding window
- **Classifier**: Random Forest (150 estimators)
- **Feature Scaling**: StandardScaler

## Applications
- **Security**: Glass break detection, gunshot detection
- **Healthcare**: Fall detection, cough monitoring
- **Industrial**: Machine failure detection, quality control
- **Smart Home**: Activity recognition, appliance monitoring
- **Environmental**: Wildlife monitoring, traffic analysis
- **Emergency**: Fire alarm detection, emergency sound recognition

## Challenges
- Overlapping events
- Background noise interference
- Reverberant environments
- Event variations (different types of doors, glass, etc.)
- Real-time processing constraints
- False positive reduction

## Performance Optimization
- Feature selection (identify most discriminative features)
- Window size tuning
- Overlap for temporal continuity
- Ensemble methods
- Hard negative mining for false positives
