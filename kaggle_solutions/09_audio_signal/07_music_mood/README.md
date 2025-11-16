# Music Mood Classification

## Overview
This solution demonstrates automatic mood classification from music audio. The system classifies music into different moods (happy, sad, energetic, calm, angry) based on musical and acoustic features.

## Problem Statement
Music mood classification is essential for:
- Music recommendation systems
- Playlist generation
- Music therapy applications
- Content tagging and organization
- Emotion-based music search

## Moods and Characteristics

### Happy
- **Musical**: Major key, upbeat tempo
- **Audio**: High brightness, moderate-high energy
- **Example**: C major chord progression
- **Arousal-Valence**: High arousal, positive valence

### Sad
- **Musical**: Minor key, slow tempo
- **Audio**: Low brightness, low energy, gentle dynamics
- **Example**: A minor chord progression
- **Arousal-Valence**: Low arousal, negative valence

### Energetic
- **Musical**: Fast tempo, driving rhythm
- **Audio**: High energy, complex rhythm, distortion
- **Example**: Power chords with beat
- **Arousal-Valence**: Very high arousal, positive valence

### Calm
- **Musical**: Slow tempo, simple harmony
- **Audio**: Soft dynamics, sustained tones, ambient
- **Example**: Sustained C major 7th chord
- **Arousal-Valence**: Low arousal, positive valence

### Angry
- **Musical**: Dissonant harmony, aggressive
- **Audio**: Harsh timbre, high energy, noise
- **Example**: Tritone intervals with distortion
- **Arousal-Valence**: High arousal, negative valence

## Dataset
175 synthetic music samples (35 per mood):
- Sample rate: 22.05 kHz
- Duration: 4 seconds per sample
- Synthesized with mood-specific musical characteristics

## Features Extracted (31 total)

### 1. Valence Features (2)
- **Spectral centroid**: Brightness indicator
- **High-frequency energy**: Mood positivity

### 2. Arousal Features (3)
- **Tempo estimate**: Beat rate
- **RMS energy**: Overall energy level
- **Dynamic range**: Volume variation

### 3. Timbre Features (3)
- **Spectral bandwidth**: Sound richness
- **Spectral rolloff**: Frequency concentration
- **Spectral flatness**: Noise vs tonal character

### 4. Harmony Features (2)
- **Harmonic-to-noise ratio**: Tonality
- **Dissonance**: Harmonic complexity

### 5. Rhythm Features (2)
- **Beat strength**: Rhythmic clarity
- **Zero crossing rate**: Transient density

### 6. MFCC Features (12)
- Mel-frequency cepstral coefficients

### 7. Temporal Features (1)
- **Attack time**: Sound onset speed

### 8. Statistical Features (4)
- Mean, std, 25th/75th percentiles

## Arousal-Valence Model

### Two-Dimensional Emotion Space
```
High Arousal
    ↑
    |  Angry    Energetic
    |            Happy
    |
----|-------|-------|---- Valence
    |           Calm
    |  Sad
    ↓
Low Arousal

Negative ← → Positive
```

### Mapping
- **Arousal**: Energy, tempo, dynamic range
- **Valence**: Mode (major/minor), brightness, harmony

## Approach

### 1. Music Generation
```python
- Define chord progressions for each mood
- Set tempo and rhythm patterns
- Apply timbre characteristics
- Add mood-specific effects
```

### 2. Feature Extraction
```python
- Compute arousal-related features
- Compute valence-related features
- Extract timbre and harmony features
- Calculate rhythm features
```

### 3. Classification
```python
- Gradient Boosting (150 estimators)
- Max depth: 7
- Feature standardization
- 5-fold cross-validation
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
1. **music_mood_audio.png**: Waveforms and spectrograms for each mood
2. **music_mood_results.png**: Confusion matrix and arousal-valence plot

## Results
Expected performance:
- Overall accuracy: 85-92%
- Energetic: Very high (distinctive tempo and energy)
- Calm: High (unique low-arousal signature)
- Happy vs Sad: Good separation (valence discrimination)
- Angry: High (dissonance and noise)

## Key Insights
1. **Tempo and energy** are primary arousal indicators
2. **Spectral brightness** correlates with positive valence
3. **Harmonic complexity** distinguishes angry from happy
4. **Dynamic envelope** separates calm from energetic
5. **Mode (major/minor)** is key for happy/sad discrimination

## Musical Theory

### Major vs Minor Keys
- **Major**: Perceived as happy, bright, uplifting
- **Minor**: Perceived as sad, dark, melancholic
- **Interval difference**: Third degree (major 3rd vs minor 3rd)

### Tempo Effects
- Fast (>120 BPM): Energetic, exciting
- Medium (80-120 BPM): Neutral, comfortable
- Slow (<80 BPM): Calm, relaxing, sad

### Harmonic Tension
- **Consonance**: Stable, pleasant (happy, calm)
- **Dissonance**: Unstable, tense (angry, energetic)

## Extensions
- Add more moods (melancholic, romantic, mysterious, etc.)
- Implement deep learning models (CNN on spectrograms)
- Multi-label classification (songs with mixed moods)
- Temporal modeling (mood changes over time)
- Use real music datasets (Million Song Dataset, MTG-Jamendo)
- Include lyrics analysis for text-audio fusion
- Real-time mood tracking
- Cultural mood variations
- Personalized mood perception

## Technical Details
- **Sample Rate**: 22.05 kHz
- **Duration**: 4 seconds per sample
- **Classifier**: Gradient Boosting (150 estimators)
- **Validation**: 5-fold cross-validation

## Applications
- Music streaming services (Spotify, Apple Music)
- DJ software and mixing tools
- Music therapy and wellness apps
- Video game dynamic soundtracks
- Film scoring and editing
- Retail and restaurant ambiance
- Workout playlist generation
- Sleep and meditation apps

## Challenges
- Subjectivity in mood perception
- Cultural differences in mood association
- Genre-specific mood expressions
- Temporal mood evolution in songs
- Multi-modal moods
- Individual listener differences

## Feature Importance
1. Tempo/beat strength: Highest
2. Spectral centroid: High
3. Energy features: High
4. MFCC: Medium
5. Harmony features: Medium

## Advanced Topics

### Circumplex Model
- Continuous arousal-valence space
- Regression instead of classification
- More nuanced mood representation

### Deep Learning Approaches
- CNN on Mel spectrograms
- Recurrent networks for temporal patterns
- Attention mechanisms
- Transfer learning from pretrained models

### Multi-Modal Learning
- Combine audio with lyrics
- Include album art visual features
- Use metadata (genre, artist, era)
