# Variational Autoencoder (VAE)

Generate new images using probabilistic latent representations with Variational Autoencoders.

## Problem Description

Variational Autoencoders (VAEs) are generative models that learn a continuous latent representation of data. Unlike regular autoencoders, VAEs learn a probability distribution in the latent space, allowing them to generate new, realistic samples by sampling from this distribution.

## Approach

### Architecture

```
Input Image ──> Encoder ──> [μ, σ²] ──> Sampling ──> z ──> Decoder ──> Reconstructed Image
   (28×28)       (CNN)      (Latent)  (Reparameterization)  (CNN)         (28×28)
```

### Mathematical Foundation

**Encoder outputs:**
- μ (z_mean): Mean of latent distribution
- σ² (z_log_var): Log variance of latent distribution

**Reparameterization Trick:**
```
z = μ + σ * ε,  where ε ~ N(0,1)
```

**Loss Function:**
```
L = Reconstruction Loss + KL Divergence
  = E[log p(x|z)] - KL(q(z|x) || p(z))
```

Where:
- Reconstruction Loss: Binary cross-entropy between input and output
- KL Divergence: Regularizes latent space to be close to N(0,1)

### Encoder Architecture

```
Input (28, 28, 1)
    ↓
Conv2D (14, 14, 32) + ReLU, stride=2
    ↓
Conv2D (7, 7, 64) + ReLU, stride=2
    ↓
Flatten (3136)
    ↓
Dense (16) + ReLU
    ↓
    ├──> Dense (latent_dim) → z_mean
    └──> Dense (latent_dim) → z_log_var
            ↓
    Sampling Layer (Reparameterization)
            ↓
         z (latent_dim)
```

### Decoder Architecture

```
z (latent_dim)
    ↓
Dense (7 * 7 * 64) + ReLU
    ↓
Reshape (7, 7, 64)
    ↓
Conv2DTranspose (14, 14, 64) + ReLU, stride=2
    ↓
Conv2DTranspose (28, 28, 32) + ReLU, stride=2
    ↓
Conv2DTranspose (28, 28, 1) + Sigmoid
    ↓
Output (28, 28, 1)
```

### Training Process

1. **Forward Pass**:
   - Encode input to get μ and σ²
   - Sample z using reparameterization trick
   - Decode z to reconstruct input

2. **Loss Calculation**:
   - Reconstruction loss: How well can we reconstruct input?
   - KL loss: How close is latent distribution to N(0,1)?

3. **Backpropagation**:
   - Update encoder and decoder weights
   - Minimize total loss

4. **Generation**:
   - Sample z from N(0,1)
   - Decode to generate new images

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam
- **Latent Dimension**: 2 (for easy visualization)
- **Batch Size**: 128
- **Epochs**: 30
- **Reconstruction Loss**: Binary cross-entropy
- **KL Weight**: 1.0 (β-VAE when β ≠ 1)

## Features

1. Custom VAE training loop with two-part loss
2. Reparameterization trick for backpropagation
3. 2D latent space for visualization
4. Image generation from latent samples
5. Latent space exploration
6. Training metrics tracking

## Usage

```bash
python solution.py
```

## Output

The script generates:
1. Generated images from random latent samples
2. Latent space visualization (for 2D latent dim)
3. Training curves (total, reconstruction, KL losses)
4. Performance metrics

## Results

Expected outputs:
- Diverse generated images
- Smooth latent space interpolation
- Convergence of both loss components
- Clustered latent representations

## Key Concepts

### Reparameterization Trick

**Problem**: Cannot backpropagate through random sampling

**Solution**:
```python
# Instead of: z ~ N(μ, σ²)
# Use: z = μ + σ * ε, where ε ~ N(0,1)
```

This makes sampling differentiable!

### KL Divergence

Measures distance between two distributions:
```
KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

Forces latent distribution to be close to standard normal N(0,1).

### β-VAE

Control KL weight with β:
```
L = Reconstruction Loss + β * KL Divergence
```

- β > 1: More disentangled representations
- β < 1: Better reconstructions
- β = 1: Standard VAE

## Parameters

Key hyperparameters you can tune:

```python
latent_dim = 2             # Latent space dimensions
epochs = 30                # Training epochs
batch_size = 128           # Batch size
kl_weight = 1.0           # β parameter for β-VAE
```

## Technical Notes

1. **Latent Dimension**: 2D allows visualization, higher dims for complex data
2. **KL Annealing**: Gradually increase KL weight during training
3. **Posterior Collapse**: If KL → 0, decoder ignores latent code
4. **Architecture Balance**: Encoder/decoder capacity must match

## Comparison: VAE vs Regular Autoencoder

| Aspect | Regular AE | VAE |
|--------|-----------|-----|
| Latent Space | Deterministic | Probabilistic |
| Generation | Poor interpolation | Smooth generation |
| Regularization | Optional | Built-in (KL) |
| Training | Simple | Two-part loss |
| Applications | Compression | Generation |

## Applications

1. **Image Generation**: Create new realistic images
2. **Anomaly Detection**: Identify outliers in latent space
3. **Data Interpolation**: Smooth transitions between images
4. **Disentanglement**: Separate factors of variation
5. **Semi-supervised Learning**: Use latent representations
6. **Drug Discovery**: Generate new molecular structures

## Extensions

Potential improvements:

1. **β-VAE**: Control disentanglement with β parameter
2. **Conditional VAE**: Generate specific classes
3. **Hierarchical VAE**: Multiple latent levels
4. **VQ-VAE**: Vector quantized latent space
5. **WAE**: Wasserstein autoencoder
6. **AAE**: Adversarial autoencoder

## Common Issues

**Posterior Collapse:**
- KL divergence → 0
- Solution: KL annealing, increase decoder depth

**Blurry Reconstructions:**
- MSE/BCE reconstruction loss
- Solution: Use perceptual loss, adversarial training

**Poor Generations:**
- Latent space not well-structured
- Solution: Increase KL weight, more training

## References

- Kingma & Welling (2013): "Auto-Encoding Variational Bayes"
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- Burgess et al. (2018): "Understanding disentangling in β-VAE"
- Razavi et al. (2019): "Generating Diverse High-Fidelity Images with VQ-VAE-2"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
