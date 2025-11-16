# Image Quality Assessment

## Problem Overview
Automated assessment of image quality without reference images. Applications in photo editing, video streaming, and quality control.

## Dataset
- **Synthetic images** with controlled degradations:
  - Noise: Gaussian noise at varying levels
  - Blur: Averaging-based blur simulation
  - Compression: Block artifacts
  - Color distortion: Channel shifts
- **Quality Scores**: 0-100 (normalized to 0-1)
- **Samples**: 1,500 images (64x64x3)

## Approach

### Model Architecture
**Regression CNN**:
```
Input (64x64x3)
  ↓
Conv Block 1 → 32x32x32
  ↓
Conv Block 2 → 16x16x64
  ↓
Conv Block 3 → 8x8x128
  ↓
Conv Block 4 → 4x4x256
  ↓
Global Avg Pool → 256
  ↓
FC (128) → ReLU
  ↓
FC (64) → ReLU
  ↓
Output (1) → Sigmoid
```

### Quality Degradation Types
1. **Noise**: Random pixel variations
2. **Blur**: Loss of sharpness
3. **Compression Artifacts**: Blocking effects
4. **Color Distortion**: Channel imbalances

### Loss Function
- **MSE**: Mean Squared Error for regression
- **MAE**: Mean Absolute Error for interpretability

## Results

### Performance Metrics
- **MAE**: ~5-10 points (on 0-100 scale)
- **RMSE**: ~8-15 points
- **R² Score**: ~0.70-0.85

### Key Insights
1. Multiple degradation types make task challenging
2. Blur and noise are easier to detect than compression
3. Combined degradations complicate assessment
4. Deep features capture quality indicators

## Files Generated
1. `quality_samples.png` - Images at different quality levels
2. `quality_training_history.png` - Training curves
3. `quality_predictions.png` - Prediction scatter and residuals

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/16_image_quality
python solution.py
```

## Requirements
- numpy
- matplotlib
- seaborn
- scikit-learn

## Applications
- **Photo Editing**: Automatic quality enhancement
- **Video Streaming**: Adaptive bitrate selection
- **Camera Systems**: Auto-quality adjustment
- **Quality Control**: Manufacturing inspection
- **Social Media**: Content filtering
- **Compression**: Optimal parameter selection

## Future Improvements
1. Multi-scale quality assessment
2. Perceptual quality metrics (SSIM, PSNR)
3. Separate scores for different degradation types
4. Attention to important regions
5. Reference-based quality comparison
6. Video quality assessment (temporal)

## Related Approaches
- BRISQUE: No-reference image quality
- NIQE: Natural image quality evaluator
- Deep IQA: Learning-based approaches
