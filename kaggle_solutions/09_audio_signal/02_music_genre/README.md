# Music Genre Classification

## Overview
This solution demonstrates automatic music genre classification using audio signal analysis and machine learning. The system classifies music into different genres (rock, classical, jazz, electronic, hip-hop) based on spectral, rhythmic, and harmonic features.

## Problem Statement
Music genre classification is essential for music recommendation systems, library organization, and playlist generation. This solution shows how to:
- Generate synthetic music with genre-specific characteristics
- Extract comprehensive audio features
- Build a robust classification model
- Analyze genre-distinguishing patterns

## Dataset
Synthetic music samples are generated with genre-specific characteristics:
- **Rock**: Power chords (220, 277, 330 Hz), distortion, strong beat (120 BPM)
- **Classical**: Orchestra harmonics, vibrato, smooth dynamics
- **Jazz**: Complex 7th chords, swing rhythm, improvisation patterns
- **Electronic**: Sub-bass (55 Hz), synthesized leads, sawtooth beats
- **Hip-hop**: Heavy bass (50 Hz), kick-snare patterns, hi-hat rhythms

200 total samples (40 per genre) at 22.05 kHz sample rate, 3 seconds each.

## Features

### Audio Features Extracted (25 total)

1. **Spectral Features** (4):
   - Spectral centroid (brightness)
   - Spectral bandwidth
   - Spectral rolloff
   - Spectral flatness

2. **Rhythm Features** (2):
   - Tempo estimation
   - Beat strength

3. **Harmonic Features** (1):
   - Harmonic-to-noise ratio (HNR)

4. **Energy by Frequency Band** (4):
   - Sub-bass (20-60 Hz)
   - Bass (60-250 Hz)
   - Mid-range (250-2000 Hz)
   - High (2000+ Hz)

5. **MFCC-like Features** (10):
   - Mel-frequency cepstral coefficients

6. **Time-domain Features** (4):
   - Mean absolute value
   - Standard deviation
   - RMS energy
   - Zero crossing rate

## Approach

### 1. Music Generation
```python
- Genre-specific instrument simulation
- Harmonic structure creation
- Rhythm pattern generation
- Dynamic and timbral characteristics
```

### 2. Feature Extraction
```python
- FFT for frequency analysis
- Spectral feature computation
- Rhythm analysis via autocorrelation
- Energy distribution across bands
```

### 3. Model Training
```python
- Gradient Boosting Classifier
- 100 estimators, max depth 5
- Learning rate: 0.1
- 5-fold cross-validation
```

### 4. Evaluation
- Classification report with metrics
- Confusion matrix visualization
- Per-genre accuracy analysis
- Feature importance ranking

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
1. **music_genre_samples.png**: Waveforms and frequency spectra for each genre
2. **music_genre_results.png**: Confusion matrix, accuracies, and feature importance

## Results
Expected performance:
- Overall accuracy: 90-95%
- Rock: Very high (distinctive distortion and power chords)
- Classical: High (unique harmonic structure)
- Electronic: Very high (characteristic bass and synthesis)
- Hip-hop: High (strong sub-bass signature)
- Jazz: Good (complex harmonies distinctive)

## Key Insights
1. **Sub-bass energy** is the strongest discriminator for electronic and hip-hop
2. **Spectral flatness** distinguishes harmonic (classical, jazz) from noise-like (rock, electronic) genres
3. **Beat strength** separates rhythmic genres from classical
4. **Harmonic complexity** captured by MFCC coefficients identifies jazz and classical

## Genre Characteristics

### Rock
- High distortion (harmonic saturation)
- Power chord frequencies
- Strong regular beat
- Mid-to-high frequency emphasis

### Classical
- Clean harmonic series
- Vibrato modulation
- Dynamic envelope variation
- Distributed frequency spectrum

### Jazz
- 7th chord voicings
- Swing rhythm patterns
- Complex harmonic structure
- Mid-frequency richness

### Electronic
- Strong sub-bass component
- Synthesized waveforms
- Sawtooth/square wave characteristics
- Filter sweep effects

### Hip-hop
- Dominant sub-bass
- Kick-snare rhythm pattern
- High-frequency percussion
- Sparse mid-range

## Extensions
- Add more genres (metal, country, reggae, etc.)
- Implement CNN for spectrogram classification
- Use real music datasets (GTZAN, Million Song Dataset)
- Add sub-genre classification
- Real-time genre detection
- Multi-label classification for fusion genres
- Temporal modeling with RNNs

## Technical Details
- **Sample Rate**: 22.05 kHz
- **Audio Duration**: 3 seconds per sample
- **Classifier**: Gradient Boosting (100 estimators)
- **Feature Scaling**: StandardScaler
- **Validation**: 5-fold cross-validation

## Applications
- Music streaming services
- DJ software and mixing tools
- Music library organization
- Playlist generation
- Music recommendation engines
- Copyright detection
- Music production tools
