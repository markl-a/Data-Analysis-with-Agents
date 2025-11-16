# Autoencoder for Image Denoising

Use convolutional autoencoders to remove noise from images.

## Problem Description

Image denoising is the task of removing noise from corrupted images. This implementation uses a convolutional autoencoder that learns to map noisy images back to their clean versions by learning a compressed representation of the image features.

## Approach

### Architecture

```
Noisy Image ──> Encoder ──> Latent Code ──> Decoder ──> Clean Image
   (28×28)      (Conv+Pool)    (32-dim)    (Conv+Upsample)  (28×28)
```

### Encoder Architecture

```
Input (28, 28, 1)
    ↓
Conv2D (28, 28, 32) + ReLU
    ↓
MaxPooling2D (14, 14, 32)
    ↓
Conv2D (14, 14, 64) + ReLU
    ↓
MaxPooling2D (7, 7, 64)
    ↓
Conv2D (7, 7, 128) + ReLU
    ↓
MaxPooling2D (4, 4, 128)
    ↓
Flatten (2048)
    ↓
Dense (32) + ReLU
    ↓
Latent Code (32)
```

### Decoder Architecture

```
Latent Code (32)
    ↓
Dense (2048) + ReLU
    ↓
Reshape (4, 4, 128)
    ↓
Conv2D (4, 4, 128) + ReLU
    ↓
UpSampling2D (8, 8, 128)
    ↓
Conv2D (8, 8, 64) + ReLU
    ↓
UpSampling2D (16, 16, 64)
    ↓
Conv2D (16, 16, 32) + ReLU
    ↓
UpSampling2D (32, 32, 32)
    ↓
Conv2D (32, 32, 1) + Sigmoid
    ↓
Cropping2D (28, 28, 1)
    ↓
Output (28, 28, 1)
```

### Training Process

1. **Generate Clean Images**: Create synthetic images with geometric shapes
2. **Add Noise**: Apply Gaussian noise to create noisy versions
3. **Train Autoencoder**:
   - Input: Noisy images
   - Target: Clean images
   - Loss: Mean Squared Error (MSE)
4. **Optimize**: Use Adam optimizer to minimize reconstruction loss
5. **Evaluate**: Measure denoising quality using MSE, MAE, and PSNR

### Loss Function

**Reconstruction Loss (MSE):**
```
L = (1/N) Σ ||x_clean - decoder(encoder(x_noisy))||²
```

Where:
- x_clean: Clean image
- x_noisy: Noisy image
- N: Number of pixels

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam (default learning rate)
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Encoding Dimension**: 32
- **Batch Size**: 128
- **Epochs**: 30

## Features

1. Convolutional encoder-decoder architecture
2. Symmetric network design
3. Synthetic data generation with geometric shapes
4. Gaussian noise injection
5. Visual comparison of clean, noisy, and denoised images
6. Performance metrics (MSE, MAE, PSNR)

## Usage

```bash
python solution.py
```

## Output

The script generates:
- Denoised images from noisy inputs
- Side-by-side comparison visualization
- Training and validation loss curves
- MAE curves
- Denoising performance metrics

## Results

Expected outputs:
- Significant noise reduction in denoised images
- PSNR improvement of 10-20 dB
- Smooth convergence of training loss
- Good generalization to validation set

## Evaluation Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/N) Σ (clean - denoised)²
```

**Mean Absolute Error (MAE):**
```
MAE = (1/N) Σ |clean - denoised|
```

**Peak Signal-to-Noise Ratio (PSNR):**
```
PSNR = 10 * log₁₀(MAX² / MSE)
```
Higher PSNR indicates better denoising quality.

## Parameters

Key hyperparameters you can tune:

```python
encoding_dim = 32        # Size of latent representation
epochs = 30              # Training epochs
batch_size = 128         # Batch size
noise_factor = 0.5       # Amount of noise to add
img_size = 28           # Image dimensions
```

## Technical Notes

1. **Symmetric Architecture**: Decoder mirrors encoder structure
2. **Bottleneck**: Forces learning of compressed representations
3. **Noise Level**: Higher noise requires larger encoding dimension
4. **Padding**: 'same' padding preserves spatial dimensions
5. **Activation**: Sigmoid output ensures values in [0, 1]

## Types of Noise

This implementation uses **Gaussian noise**. Other noise types:

1. **Salt-and-pepper noise**: Random black/white pixels
2. **Poisson noise**: Shot noise from photon counting
3. **Speckle noise**: Multiplicative noise in radar images
4. **Uniform noise**: Random values from uniform distribution

## Extensions

Potential improvements:

1. **Residual Connections**: Skip connections for better gradient flow
2. **U-Net Architecture**: Concatenate encoder features with decoder
3. **Variational Autoencoder**: Add probabilistic latent space
4. **Attention Mechanism**: Focus on important regions
5. **Multi-scale Denoising**: Process multiple resolutions
6. **Blind Denoising**: Handle unknown noise levels
7. **Color Images**: Extend to RGB channels

## Comparison with Other Methods

**Classical Methods:**
- Median Filter: Fast but loses details
- Gaussian Filter: Smooth but blurry
- Bilateral Filter: Edge-preserving but slow

**Deep Learning Methods:**
- Denoising Autoencoder: This implementation
- U-Net: Better for medical images
- DnCNN: Specialized for denoising
- Noise2Noise: No clean images needed

## Applications

1. Medical imaging enhancement
2. Astronomical image processing
3. Surveillance camera improvement
4. Old photograph restoration
5. Low-light photography
6. Compressed image quality improvement

## References

- Vincent et al. (2008): "Extracting and Composing Robust Features with Denoising Autoencoders"
- Zhang et al. (2017): "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
- Gondara (2016): "Medical Image Denoising Using Convolutional Denoising Autoencoders"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
