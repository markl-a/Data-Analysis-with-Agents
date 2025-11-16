# Image Anomaly Detection

## Overview
Detects defective or anomalous images using PCA reconstruction, Isolation Forest, and pixel-based statistical methods. Applications include manufacturing quality control and defect detection.

## Problem Description
Identifies visual defects:
- **Scratches**: Linear defects across surface
- **Spots**: Missing or extra material
- **Incomplete**: Missing sections
- **Distortions**: Shape deviations

## Dataset
- 500 normal images (circles)
- 30 anomalous images (defects)
- 28x28 grayscale synthetic images

### Normal Pattern
- Circular shapes with slight variations
- Consistent radius and centering
- Minimal noise

### Anomaly Types
1. **Scratch**: Linear defect across object
2. **Spot**: Multiple missing sections
3. **Incomplete**: Missing arc segment
4. **Distorted**: Elliptical instead of circular

## Methods

### 1. PCA Reconstruction
- Train PCA on normal images only
- Reconstruct all images
- Detect high reconstruction error
- Threshold: 95th percentile of normal errors

### 2. Isolation Forest on PCA Features
- Apply Isolation Forest to PCA components
- Detects outliers in compressed representation
- Robust to noise

### 3. Pixel Statistics
- Calculate mean/std from normal images
- Compute max z-score per image
- Simple baseline method

## Evaluation Metrics
- Precision, Recall, F1-Score
- Visual inspection of detections
- Reconstruction error distribution

## Usage
```bash
python solution.py
```

## Requirements
- numpy, pandas, matplotlib, seaborn, scikit-learn

## Applications
- Manufacturing quality control
- Medical imaging (tumor detection)
- Defect inspection
- Security screening
- Product inspection

## Production Considerations
- Use deep learning autoencoders for better performance
- Convolutional autoencoders for spatial features
- Real-time inference requirements
- Adjustable sensitivity thresholds
