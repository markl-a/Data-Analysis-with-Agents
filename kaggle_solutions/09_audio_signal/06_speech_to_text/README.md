# Simple Speech-to-Text System

## Overview
This solution demonstrates a simplified speech-to-text system for spoken digit recognition (0-9). The system uses formant-based speech synthesis and pattern recognition to convert spoken digits to text.

## Problem Statement
Speech recognition is fundamental to:
- Voice assistants and smart speakers
- Automated phone systems
- Accessibility tools
- Voice-controlled applications
- Transcription services

This solution shows the basics of speech-to-text conversion using a limited vocabulary.

## Vocabulary
Spoken digits: zero, one, two, three, four, five, six, seven, eight, nine

## Dataset
400 synthetic speech samples (40 per digit):
- Sample rate: 8 kHz
- Duration: 0.8 seconds per sample
- Formant-based synthesis
- Phoneme-level modeling

## Approach

### 1. Speech Synthesis
```
- Define formant frequencies for each digit's phonemes
- Generate glottal pulse train (vocal cord vibration)
- Apply formant resonances
- Create phoneme sequences
- Add natural pitch variation
```

### 2. Formant Patterns

#### Digit-Specific Formants
- **zero**: /ɪ/ (400-800 Hz, 900-1300 Hz) + /oʊ/
- **one**: /w/ (400-900 Hz) + /ʌ/ (700-1100 Hz)
- **two**: /t/ (350-750 Hz) + /u/ (900-1400 Hz)
- **three**: /θ/ (500-1000 Hz) + /i/ (1400-1800 Hz)
- **four**: /f/ (600-1100 Hz) + /ɔ/ (1000-1400 Hz)
- **five**: /f/ (450-900 Hz) + /aɪ/ (700-1200 Hz)
- **six**: /s/ (500-1000 Hz) + /ɪ/ (700-1300 Hz)
- **seven**: /s/ (550-1100 Hz) + /ɛ/ (1200-1700 Hz)
- **eight**: /eɪ/ (550-1050 Hz, 1100-1600 Hz)
- **nine**: /n/ (450-950 Hz) + /aɪ/ (700-1300 Hz)

### 3. Feature Extraction (25 features)

1. **Formant Features** (3):
   - First three formant frequencies
   - Most discriminative for vowel sounds

2. **Spectral Features** (3):
   - Spectral centroid
   - Spectral bandwidth
   - Spectral rolloff

3. **MFCC Features** (12):
   - Mel-frequency cepstral coefficients
   - Capture overall spectral shape

4. **Temporal Features** (2):
   - Active speech duration
   - Zero crossing rate

5. **Energy Features** (3):
   - Total energy
   - RMS energy
   - Peak amplitude

6. **Pitch Features** (1):
   - Fundamental frequency (F0)

### 4. Classification
```
- Random Forest Classifier (200 trees)
- Max depth: 20
- Feature standardization
- 80-20 train-test split
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
The script generates:
1. **speech_digit_spectrograms.png**: Spectrograms for all 10 digits
2. **speech_recognition_results.png**: Confusion matrix and accuracy
3. **speech_recognition_demo.png**: Example recognition with probabilities

## Results
Expected performance:
- Overall accuracy: 90-95%
- High accuracy: "three", "seven", "eight" (distinctive formants)
- Good accuracy: Most other digits
- Potential confusion: "five"/"nine" (similar /aɪ/ sound)

## Key Insights
1. **Formant frequencies** are the primary discriminators for vowels
2. **Phoneme sequences** distinguish similar-sounding digits
3. **Pitch variation** adds natural speech characteristics
4. **MFCC features** capture overall voice timbre
5. **Temporal patterns** help with rhythm and duration

## Speech Production Model

### Glottal Pulse Train
- Represents vocal cord vibration
- Frequency = fundamental frequency (pitch)
- Typically 80-400 Hz
- Higher for females, lower for males

### Formants
- Resonance frequencies of vocal tract
- Determined by tongue position, mouth shape
- F1 (first formant): Vocal tract opening
- F2 (second formant): Tongue position
- F3 (third formant): Lip rounding

### Why Formants Matter
Each vowel sound has characteristic formant patterns:
- /i/ (see): High F1, very high F2
- /u/ (too): Low F1, low F2
- /a/ (father): High F1, low F2
- /e/ (say): Mid F1, high F2

## Phoneme-Level Modeling

### Digit Phoneme Breakdown
- **zero**: /z/ + /ɪ/ + /r/ + /oʊ/
- **one**: /w/ + /ʌ/ + /n/
- **two**: /t/ + /u/
- **three**: /θ/ + /r/ + /i/
- **four**: /f/ + /ɔ/ + /r/
- **five**: /f/ + /aɪ/ + /v/
- **six**: /s/ + /ɪ/ + /k/ + /s/
- **seven**: /s/ + /ɛ/ + /v/ + /ən/
- **eight**: /eɪ/ + /t/
- **nine**: /n/ + /aɪ/ + /n/

## Extensions
- Expand vocabulary (all words, continuous speech)
- Add speaker independence
- Language modeling (word sequences, grammar)
- Deep learning models (RNN, LSTM, Transformer)
- Attention mechanisms
- End-to-end systems (no hand-crafted features)
- Real-time recognition
- Multi-language support
- Noise robustness
- Accent adaptation
- Use real speech datasets (LibriSpeech, Common Voice)

## Comparison with Modern Systems

### This Solution (Traditional)
- Hand-crafted features (formants, MFCC)
- Limited vocabulary
- Phoneme-level modeling
- Random Forest classifier

### Modern Systems (Deep Learning)
- Raw waveform or spectrograms
- Large vocabulary (thousands of words)
- Character or subword units
- Neural networks (CNN, RNN, Transformer)
- Examples: DeepSpeech, Wav2Vec, Whisper

## Applications
- Voice dialing
- PIN entry systems
- Digit recognition in IVR
- Number dictation
- Calculator voice input
- Educational tools
- Accessibility features

## Challenges
- Speaker variability (accent, gender, age)
- Background noise
- Recording quality
- Speaking rate variations
- Co-articulation effects
- Similar-sounding digits
- Continuous digit strings

## Technical Details
- **Sample Rate**: 8 kHz (sufficient for speech)
- **Duration**: 0.8 seconds per digit
- **Classifier**: Random Forest (200 estimators)
- **Features**: 25 acoustic features
- **Accuracy**: 90-95% on clean synthetic speech

## Feature Importance Ranking
1. Formant frequencies: Highest
2. MFCC coefficients: High
3. Spectral centroid: Medium
4. Pitch: Medium
5. Energy features: Low

## Real-World Deployment Considerations
- Noise reduction preprocessing
- Voice activity detection (VAD)
- Endpoint detection (start/end of speech)
- Confidence thresholds
- Rejection of non-speech
- User feedback mechanisms
- Continuous learning
