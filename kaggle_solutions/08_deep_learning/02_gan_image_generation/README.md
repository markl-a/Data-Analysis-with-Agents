# GAN for Image Generation

Generate synthetic images using Generative Adversarial Networks (GANs).

## Problem Description

Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks contest with each other in a zero-sum game. This implementation demonstrates training a GAN to generate synthetic images that resemble the training data.

## Approach

### Architecture

```
        ┌──────────────┐
Noise ─>│  Generator   │──> Fake Images ──┐
        └──────────────┘                   │
                                           ├──> ┌──────────────┐
                                           │    │Discriminator │──> Real/Fake
Real Images ───────────────────────────────┘    └──────────────┘
                                                       │
                                                   Backprop
                                                       │
                                           ┌───────────┴──────────┐
                                           │                      │
                                      Update D              Update G
```

### Generator Architecture

```
Input (Latent Vector: 100)
    ↓
Dense (128 * 7 * 7) + ReLU + BatchNorm
    ↓
Reshape (7, 7, 128)
    ↓
UpSampling2D (14, 14, 128)
    ↓
Conv2D (14, 14, 128) + ReLU + BatchNorm
    ↓
UpSampling2D (28, 28, 128)
    ↓
Conv2D (28, 28, 64) + ReLU + BatchNorm
    ↓
Conv2D (28, 28, 1) + Tanh
    ↓
Output (28, 28, 1)
```

### Discriminator Architecture

```
Input (28, 28, 1)
    ↓
Conv2D (14, 14, 32) + LeakyReLU + Dropout
    ↓
Conv2D (7, 7, 64) + LeakyReLU + Dropout + BatchNorm
    ↓
Conv2D (4, 4, 128) + LeakyReLU + Dropout + BatchNorm
    ↓
Conv2D (4, 4, 256) + LeakyReLU + Dropout
    ↓
Flatten + Dense (1) + Sigmoid
    ↓
Output (Real/Fake probability)
```

### Training Algorithm

1. **Initialize**: Random weights for Generator (G) and Discriminator (D)
2. **For each training iteration**:
   - Sample random noise z from N(0,1)
   - Generate fake images: G(z)
   - Sample real images from training data
   - **Train Discriminator**:
     - Forward pass real images, label as 1
     - Forward pass fake images, label as 0
     - Compute loss and update D weights
   - **Train Generator**:
     - Generate new fake images
     - Forward through D, label as 1 (fool D)
     - Compute loss and update G weights
3. **Repeat** until convergence

### Loss Functions

**Discriminator Loss:**
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```

**Generator Loss:**
```
L_G = -E[log D(G(z))]
```

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam (lr=0.0002, beta_1=0.5)
- **Latent Dimension**: 100
- **Batch Size**: 32
- **Training Epochs**: 3000
- **Image Size**: 28×28 grayscale

## Features

1. Fully connected GAN architecture
2. Batch normalization for stable training
3. LeakyReLU and Dropout for discriminator
4. Synthetic dataset generation
5. Real-time training monitoring
6. Generated image visualization

## Usage

```bash
python solution.py
```

## Output

The script generates:
- Synthetic images from trained generator
- Comparison with real training images
- Training curves for both networks
- Loss and accuracy metrics

## Results

Expected outputs:
- Generated images resembling training distribution
- Discriminator accuracy around 50-70%
- Stable convergence of both losses
- Diverse generated samples

## Training Dynamics

**Key observations:**

1. **Early Training**: Discriminator dominates, high accuracy
2. **Mid Training**: Generator improves, D accuracy drops
3. **Convergence**: Nash equilibrium, D accuracy ~50-70%
4. **Mode Collapse**: If G produces limited variety, restart with different initialization

## Parameters

Key hyperparameters you can tune:

```python
latent_dim = 100           # Size of noise vector
epochs = 3000              # Training iterations
batch_size = 32            # Batch size
learning_rate = 0.0002     # Adam learning rate
beta_1 = 0.5              # Adam beta_1 parameter
```

## Technical Notes

1. **Normalization**: Images scaled to [-1, 1] for tanh activation
2. **Stability**: Batch normalization and label smoothing improve stability
3. **Mode Collapse**: Monitor diversity of generated images
4. **Training Balance**: D and G should improve together

## Common Issues

**Mode Collapse:**
- Generator produces limited variety
- Solution: Reduce learning rate, add noise to labels

**Discriminator Too Strong:**
- Generator loss doesn't decrease
- Solution: Train D less frequently, reduce D capacity

**Training Instability:**
- Losses oscillate wildly
- Solution: Reduce learning rate, add gradient clipping

## Extensions

Potential improvements:
1. **DCGAN**: Deep Convolutional GAN with architecture improvements
2. **WGAN**: Wasserstein GAN for better convergence
3. **Conditional GAN**: Control generated image class
4. **Progressive GAN**: Generate high-resolution images
5. **StyleGAN**: State-of-the-art image generation

## References

- Goodfellow et al. (2014): "Generative Adversarial Networks"
- Radford et al. (2015): "Unsupervised Representation Learning with DCGANs"
- Arjovsky et al. (2017): "Wasserstein GAN"
- Salimans et al. (2016): "Improved Techniques for Training GANs"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
