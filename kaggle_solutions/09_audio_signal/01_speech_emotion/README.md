# Speech Emotion Recognition

## Overview
This solution demonstrates speech emotion recognition using audio signal processing and machine learning. The system classifies emotions (happy, sad, angry, neutral, fear) from speech audio by analyzing acoustic features.

## Problem Statement
Emotion recognition from speech is crucial for human-computer interaction, mental health assessment, customer service analysis, and entertainment applications. This solution shows how to:
- Generate synthetic speech-like audio with emotional characteristics
- Extract acoustic features from audio signals
- Build a classification model for emotion recognition
- Visualize audio waveforms and spectrograms

## Dataset
Synthetic speech audio is generated with emotional characteristics:
- **Happy**: Higher pitch (200-280 Hz), faster tempo, more variation
- **Sad**: Lower pitch (100-150 Hz), slower tempo, less variation
- **Angry**: Variable pitch (150-220 Hz), fast tempo, high energy
- **Fear**: High pitch (220-300 Hz), trembling effect
- **Neutral**: Moderate values across all parameters

250 audio samples total (50 per emotion class) at 16kHz sample rate.

## Features

### Audio Features Extracted
1. **Spectral Features**:
   - Spectral centroid (brightness)
   - Spectral bandwidth (spread)
   - Spectral rolloff (frequency concentration)

2. **Time-domain Features**:
   - Zero crossing rate
   - Energy
   - RMS (Root Mean Square)

3. **MFCC-like Features**:
   - Power in 13 frequency bands

4. **Statistical Features**:
   - Mean, standard deviation
   - Min, max values
   - Percentiles (25th, 75th)

Total: 25 features per audio sample

## Approach

### 1. Audio Generation
```python
- Generate base frequency based on emotion
- Add harmonics to simulate speech formants
- Apply emotion-specific modulations
- Add amplitude envelope and noise
```

### 2. Feature Extraction
```python
- FFT for frequency analysis
- Spectral feature computation
- Time-domain statistics
- Frequency band power distribution
```

### 3. Model Training
```python
- Random Forest Classifier (100 trees)
- Feature standardization
- 80-20 train-test split
```

### 4. Evaluation
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Per-emotion accuracy analysis

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
1. **speech_emotion_audio_features.png**: Waveforms and spectrograms for each emotion
2. **speech_emotion_results.png**: Confusion matrix and per-emotion accuracy

## Results
Expected performance:
- Overall accuracy: 85-95%
- Happy: High accuracy (clear pitch and tempo patterns)
- Sad: High accuracy (distinctive low energy)
- Angry: Good accuracy (high energy signature)
- Neutral: Moderate accuracy (less distinctive features)
- Fear: Good accuracy (trembling patterns)

## Key Insights
1. **Pitch variations** are strong indicators of emotional states
2. **Energy levels** distinguish high-arousal (angry, happy) from low-arousal (sad) emotions
3. **Spectral features** capture voice quality differences
4. **Temporal patterns** (tempo, rhythm) provide additional discriminative power

## Extensions
- Add more emotions (surprise, disgust, etc.)
- Implement deep learning models (CNN, RNN)
- Use real speech datasets (RAVDESS, IEMOCAP)
- Add speaker-independent emotion recognition
- Include multilingual emotion recognition
- Real-time emotion detection from microphone input

## Technical Details
- **Sample Rate**: 16 kHz
- **Audio Duration**: 2 seconds per sample
- **Classifier**: Random Forest (100 estimators)
- **Feature Scaling**: StandardScaler
- **Train-Test Split**: 80-20 with stratification

## Applications
- Mental health monitoring
- Customer service quality analysis
- Interactive voice response systems
- Gaming and entertainment
- Educational tools
- Human-robot interaction
