# Handwritten Text Recognition

## Problem Overview
Recognize sequences of handwritten digits using a combination of CNN and RNN. Applications in check processing, form digitization, and historical document transcription.

## Dataset
- **Synthetic handwritten digit sequences**:
  - 3-digit sequences (000-999)
  - Simple digit drawings with variations
  - Added noise for realism
- **Image Size**: 32x96 pixels (3 digits side by side)
- **Samples**: 1,500 sequences

## Approach

### Model Architecture
**CRNN (CNN + RNN)**:
```
Input (32x96x1)
  ↓
CNN Feature Extractor
├─ Conv Block 1 → 16x48x32
├─ Conv Block 2 → 8x24x64
└─ Conv Block 3 → 4x12x128
  ↓
Reshape → (batch, 12 time steps, 128 features)
  ↓
RNN (Bi-directional LSTM concept)
  → (batch, 12, 64)
  ↓
Select time steps (3 positions)
  ↓
Fully Connected (per position)
  → 3 predictions (each 10 classes)
```

### Key Techniques
1. **CNN Backbone**: Extracts visual features from image
2. **RNN Layer**: Models sequential dependencies
3. **CTC Loss Concept**: Align predictions to sequence
4. **Multi-Position Classification**: One classifier per digit

### Challenges
- **Sequence Alignment**: Map image features to characters
- **Variable Spacing**: Digits may be close or far apart
- **Handwriting Variation**: Different writing styles
- **Segmentation-Free**: No need to segment individual digits

## Results

### Performance Metrics
- **Sequence Accuracy**: ~60-75% (all 3 digits correct)
- **Per-Digit Accuracy**: ~80-90% (individual digit)
- **Better for**: Clear, well-spaced digits
- **Challenges**: Overlapping or touching digits

### Key Insights
1. CNN features capture digit shapes
2. RNN helps with context and sequence
3. Sequence-level prediction harder than single digit
4. Spatial features more important than temporal for digits

## Files Generated
1. `handwritten_samples.png` - Sample digit sequences
2. `handwritten_training_history.png` - Training curves
3. `handwritten_predictions.png` - Predicted vs true sequences

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/18_handwritten_text
python solution.py
```

## Requirements
- numpy
- matplotlib
- seaborn
- scikit-learn

## Applications
- **Check Processing**: Read amounts and numbers
- **Form Digitization**: Extract handwritten fields
- **Postal Automation**: Address recognition
- **Historical Documents**: Transcribe old texts
- **Note Taking Apps**: Convert handwriting to text
- **License Plate Recognition**: Read plate numbers

## Future Improvements
1. CTC (Connectionist Temporal Classification) loss
2. Attention mechanisms
3. Variable-length sequences
4. Full alphabet recognition (not just digits)
5. Cursive handwriting support
6. Language model integration for context

## Related Datasets
- IAM Handwriting Database
- RIMES: French handwritten documents
- CVL Database: Historical documents
- MNIST: Single digit recognition
