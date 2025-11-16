# Environmental Sound Classification

## Overview
This solution demonstrates classification of environmental sounds from audio recordings. The system identifies different environmental sounds (rain, wind, birds, traffic, ocean, thunder) using acoustic features and machine learning.

## Problem Statement
Environmental sound recognition is important for:
- Smart home and IoT devices
- Wildlife monitoring
- Urban planning and noise pollution analysis
- Security and surveillance systems
- Ambient context awareness
- Assistive technology for hearing-impaired

## Sound Types and Characteristics

### Rain
- **Character**: Continuous random impacts
- **Frequency**: Mid-range (200-3000 Hz)
- **Pattern**: Steady with amplitude modulation
- **Spectral**: Broadband noise filtered

### Wind
- **Character**: Low-frequency turbulence
- **Frequency**: Low (<800 Hz)
- **Pattern**: Fluctuating gusts
- **Spectral**: Low-pass filtered noise

### Birds
- **Character**: Discrete high-frequency chirps
- **Frequency**: High (2000-5000 Hz)
- **Pattern**: Intermittent calls with frequency sweeps
- **Spectral**: Tonal with modulation

### Traffic
- **Character**: Low rumble with occasional peaks
- **Frequency**: Low-mid (80-500 Hz)
- **Pattern**: Continuous with Doppler-like variations
- **Spectral**: Low-frequency dominated

### Ocean
- **Character**: Periodic wave sounds
- **Frequency**: Low-mid (100-1500 Hz)
- **Pattern**: Regular swells (0.1-0.3 Hz periodicity)
- **Spectral**: Band-limited noise with periodicity

### Thunder
- **Character**: Low-frequency rumble
- **Frequency**: Very low (<300 Hz)
- **Pattern**: Sharp attack, slow decay
- **Spectral**: Low-frequency burst

## Dataset
210 synthetic environmental sound samples (35 per sound type):
- Sample rate: 22.05 kHz
- Duration: 3 seconds per sample
- Synthesized with sound-specific characteristics

## Features Extracted (48 total)

### 1. Spectral Features (10)
- **Spectral centroid**: Center of mass of spectrum
- **Spectral bandwidth**: Spread around centroid
- **Spectral rolloff**: 85% energy cutoff frequency
- **Spectral flatness**: Tonality vs noise
- **Spectral contrast** (6 bands): Peak-valley difference

### 2. Temporal Features (2)
- **Zero crossing rate**: Sign change frequency
- **Temporal centroid**: Energy concentration in time

### 3. Energy Distribution (6)
- Sub-bass (20-60 Hz)
- Bass (60-250 Hz)
- Low-mid (250-800 Hz)
- Mid (800-2000 Hz)
- High-mid (2000-5000 Hz)
- High (5000+ Hz)

### 4. MFCC Features (13)
- Mel-frequency cepstral coefficients

### 5. Statistical Features (7)
- Mean, std, min, max
- 25th percentile, 75th percentile
- RMS energy

## Approach

### 1. Sound Generation
```python
# Rain: Filtered noise with modulation
# Wind: Low-pass noise with gusts
# Birds: Frequency sweeps (chirps)
# Traffic: Rumble with Doppler
# Ocean: Periodic waves
# Thunder: Low-frequency bursts
```

### 2. Feature Extraction
```python
- FFT for spectral analysis
- Band-specific energy computation
- Temporal envelope analysis
- MFCC calculation
- Statistical descriptors
```

### 3. Classification
```python
- Random Forest (200 trees)
- Max depth: 20
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
1. **environmental_sounds.png**: Waveforms and spectrograms for each sound type
2. **environmental_results.png**: Confusion matrix and F1-scores

## Results
Expected performance:
- Overall accuracy: 90-95%
- Thunder: Very high (unique low-frequency burst)
- Birds: Very high (distinct high-frequency chirps)
- Rain: High (characteristic filtered noise)
- Ocean: High (periodic pattern)
- Wind vs Traffic: Good (frequency distribution differs)

## Key Insights
1. **Frequency distribution** is the primary discriminator
2. **Spectral centroid** separates low (wind, traffic, thunder) from high (birds) frequency sounds
3. **Spectral flatness** distinguishes tonal (birds) from noise-like (rain, wind) sounds
4. **Temporal patterns** identify periodic sounds (ocean) vs continuous (rain, wind)
5. **Energy in specific bands** provides strong classification cues

## Sound Characteristics Analysis

### Frequency Signatures
```
Thunder:    ▓▓▓▓░░░░░░  (Very low)
Traffic:    ▓▓▓▓▓░░░░░  (Low)
Wind:       ▓▓▓▓░░░░░░  (Low)
Ocean:      ░▓▓▓▓░░░░░  (Low-mid)
Rain:       ░░▓▓▓▓▓░░░  (Mid)
Birds:      ░░░░░▓▓▓▓▓  (High)
```

### Temporal Patterns
- **Continuous**: Rain, wind, traffic, ocean
- **Discrete**: Birds, thunder
- **Periodic**: Ocean waves
- **Random**: Rain, birds

### Spectral Character
- **Tonal**: Birds (harmonic structure)
- **Noise-like**: Rain, wind, traffic
- **Mixed**: Ocean (noise with periodicity)
- **Impulsive**: Thunder

## Extensions
- Add more environmental sounds (fire, water, footsteps, etc.)
- Acoustic scene classification (forest, city, beach)
- Sound event detection (onset/offset times)
- Multi-label classification (overlapping sounds)
- Use real datasets (ESC-50, UrbanSound8K, AudioSet)
- Deep learning models (CNN on spectrograms)
- Real-time environmental monitoring
- Sound source localization
- Noise-robust classification
- Transfer learning from pretrained models

## Technical Details
- **Sample Rate**: 22.05 kHz
- **Duration**: 3 seconds per sample
- **Classifier**: Random Forest (200 estimators)
- **Feature Count**: 48 acoustic features

## Applications

### Smart Home
- Context-aware automation
- Security alerts (glass break, unusual sounds)
- Ambient sound control

### Environmental Monitoring
- Wildlife presence detection
- Noise pollution measurement
- Weather condition monitoring

### Safety and Security
- Gunshot detection
- Emergency sound recognition
- Intrusion detection

### Assistive Technology
- Sound alerts for hearing-impaired
- Environmental awareness
- Navigation assistance

### Urban Planning
- Traffic pattern analysis
- Noise mapping
- Quality of life assessment

## Challenges
- Overlapping sounds in real environments
- Background noise interference
- Sound variation (e.g., different bird species)
- Recording quality differences
- Distance and direction effects
- Reverberant environments
- Weather effects on outdoor sounds

## Feature Importance
1. Spectral centroid: Highest
2. Energy distribution: Very high
3. MFCC coefficients: High
4. Spectral contrast: High
5. Temporal features: Medium
6. Statistical features: Low

## Real-World Datasets

### ESC-50
- 50 classes of environmental sounds
- 2000 audio samples
- 5-second clips
- Curated from Freesound.org

### UrbanSound8K
- 8732 labeled urban sounds
- 10 classes (air conditioner, car horn, etc.)
- Real-world recordings
- Standard benchmark

### AudioSet
- 2+ million audio clips
- 632 audio event classes
- YouTube videos
- Large-scale dataset

## Advanced Techniques

### Spectral Processing
- Mel-spectrogram
- Gammatone filtering
- Constant-Q transform
- Wavelet decomposition

### Deep Learning
- CNNs on spectrograms
- Recurrent networks for temporal
- Attention mechanisms
- Multi-scale processing

### Data Augmentation
- Time stretching
- Pitch shifting
- Adding background noise
- Volume variation
- Mixing sounds

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion matrix analysis
- ROC curves
- Class-specific performance
