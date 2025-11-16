# Audio Noise Reduction

## Overview
This solution demonstrates various techniques for audio noise reduction and enhancement. The system removes different types of noise from audio signals using signal processing algorithms.

## Problem Statement
Noise reduction is essential for:
- Voice communication systems
- Audio recording and production
- Hearing aids and assistive devices
- Speech recognition preprocessing
- Audio restoration

This solution shows how to implement and compare multiple denoising techniques.

## Noise Types

### 1. White Noise
- Flat power spectrum across all frequencies
- Equal energy at all frequencies
- Common in electronic systems

### 2. Pink Noise (1/f)
- Power decreases with frequency
- More realistic for environmental noise
- Common in nature

### 3. Hum Noise
- Power line interference (50/60 Hz)
- Harmonics at multiples of fundamental
- Common in electrical equipment

### 4. Click Noise
- Impulsive disturbances
- Vinyl record pops and crackles
- Digital transmission errors

## Denoising Methods

### 1. Spectral Subtraction
**Principle**: Estimate and subtract noise spectrum from noisy signal
```
Clean Magnitude = max(Noisy Magnitude - Noise Estimate, 0)
```
**Best for**: Stationary noise (white, pink)
**Pros**: Simple, fast
**Cons**: Can introduce musical noise artifacts

### 2. Wiener Filter
**Principle**: Optimal MMSE filter based on noise/signal power
```
Wiener Gain = max(1 - Noise Power / Signal Power, 0)
```
**Best for**: Stationary noise with known statistics
**Pros**: Optimal for Gaussian noise, smooth results
**Cons**: Requires noise estimation

### 3. Median Filter
**Principle**: Replace each sample with median of neighborhood
**Best for**: Impulsive noise (clicks, pops)
**Pros**: Excellent for outliers, simple
**Cons**: Can blur signal, not for continuous noise

### 4. Lowpass Filter
**Principle**: Remove high-frequency components
**Best for**: High-frequency noise
**Pros**: Very simple, fast
**Cons**: Removes high-frequency signal content

### 5. Notch Filter
**Principle**: Remove specific frequency and harmonics
**Best for**: Hum and tonal interference
**Pros**: Surgical removal of specific frequencies
**Cons**: Only works for known frequencies

## Features

### Clean Signal Generation
- Speech-like signals with formants
- Musical signals with chords
- Pure tones

### Noise Addition
- Controlled SNR (Signal-to-Noise Ratio)
- Multiple noise types
- Realistic noise characteristics

### Performance Metrics
- **SNR**: Signal-to-Noise Ratio (dB)
- **SNR Improvement**: Difference before/after denoising
- **Spectral comparison**: Visual frequency domain analysis

## Approach

### 1. Signal Generation
```python
- Generate clean speech-like audio
- Add harmonics and formants
- Apply natural envelope
```

### 2. Noise Addition
```python
- Calculate required noise power for target SNR
- Generate specific noise type
- Mix with clean signal
```

### 3. Denoising
```python
- Select appropriate method for noise type
- Apply denoising algorithm
- Measure improvement
```

### 4. Evaluation
- Calculate SNR before and after
- Compare frequency spectra
- Visual time-domain comparison

## Requirements
```
numpy
scipy
matplotlib
```

## Usage
```bash
python solution.py
```

## Output
For each noise type, generates:
1. **noise_reduction_[type].png**: Time and frequency domain comparison
2. **method_comparison_[type].png**: Bar chart comparing all methods

## Results
Expected SNR improvements:

### White Noise
- Wiener Filter: +6 to +10 dB
- Spectral Subtraction: +4 to +8 dB
- Lowpass Filter: +3 to +6 dB

### Hum Noise
- Notch Filter: +15 to +25 dB
- Wiener Filter: +8 to +12 dB
- Spectral Subtraction: +6 to +10 dB

### Click Noise
- Median Filter: +10 to +15 dB
- Spectral Subtraction: +3 to +7 dB
- Wiener Filter: +4 to +8 dB

## Key Insights
1. **No universal denoiser**: Different noise types require different approaches
2. **Trade-off**: Noise reduction vs signal distortion
3. **Noise estimation**: Critical for statistical methods
4. **Frequency selectivity**: Notch filters excel at tonal noise
5. **Time-domain methods**: Better for impulsive noise

## Algorithm Selection Guide

### For Stationary Broadband Noise (White, Pink)
1. First choice: Wiener Filter
2. Alternative: Spectral Subtraction
3. Simple option: Lowpass Filter (if noise is high-frequency)

### For Tonal Interference (Hum)
1. First choice: Notch Filter (if frequency known)
2. Alternative: Wiener Filter
3. Fallback: Spectral Subtraction

### For Impulsive Noise (Clicks, Pops)
1. First choice: Median Filter
2. Alternative: Morphological filters
3. Advanced: Sparse coding methods

### For Mixed Noise
1. Cascaded processing (multiple methods)
2. Adaptive filtering
3. Machine learning approaches

## Extensions
- Adaptive Wiener filtering
- Deep learning denoisers (WaveNet, U-Net)
- Kalman filtering
- Non-local means denoising
- Dictionary learning methods
- Real-time processing
- Perceptual noise reduction
- Voice activity detection integration
- Multi-band processing
- Psychoacoustic noise shaping

## Technical Details
- **Sample Rate**: 16 kHz
- **Signal Duration**: 3 seconds
- **Default SNR**: 10 dB
- **Notch Filter Q**: 30
- **Median Kernel**: 5 samples

## Applications
- Voice communication (telephony, VoIP)
- Audio production and mastering
- Hearing aids
- Speech recognition preprocessing
- Audio forensics
- Music restoration
- Podcast editing
- Video conferencing
- IoT voice assistants

## Advanced Techniques

### Spectral Gating
- More aggressive noise reduction
- Set threshold for spectral components
- Zero out components below threshold

### Multi-Band Processing
- Divide spectrum into bands
- Apply different processing per band
- Better preservation of signal

### Machine Learning Methods
- Train neural networks on clean/noisy pairs
- Learn optimal denoising function
- Can handle complex noise patterns

## Challenges
- Musical noise artifacts
- Signal distortion
- Real-time processing requirements
- Unknown noise characteristics
- Non-stationary noise
- Very low SNR scenarios
- Preservation of natural sound quality
