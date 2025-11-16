# Neural Style Transfer

Transfer artistic style from one image to another using deep learning features from VGG19.

## Problem Description

Neural style transfer is a technique that combines the content of one image with the artistic style of another. This implementation uses the VGG19 convolutional neural network to extract features and optimize a generated image to match both the content and style representations.

## Approach

### Architecture

```
Content Image ──┐
                ├──> VGG19 Feature Extractor ──> Optimization ──> Stylized Image
Style Image ────┘
```

**VGG19 Layers Used:**
- Content: `block5_conv2` (deep semantic features)
- Style: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`

### Algorithm

1. **Feature Extraction**: Use pretrained VGG19 to extract features from content and style images
2. **Content Representation**: Use activations from deep layers (block5_conv2)
3. **Style Representation**: Use Gram matrices of activations from multiple layers
4. **Optimization**: Start with content image and iteratively update to minimize:
   - Content Loss: MSE between content features
   - Style Loss: MSE between Gram matrices
5. **Loss Function**: `Total Loss = α * Content Loss + β * Style Loss`

### Key Components

**Gram Matrix:**
```
G[i,j] = Σ(F[i,k] * F[j,k]) / (H * W)
```
Where F is the feature map, H and W are height and width.

**Loss Functions:**
- Content Loss: L2 distance between feature representations
- Style Loss: L2 distance between Gram matrices
- Total Variation Loss: Smoothness regularization

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Base Model**: VGG19 pretrained on ImageNet
- **Optimizer**: Adam with learning rate 5.0
- **Image Size**: 400x400 pixels
- **Training**: 5 epochs × 50 steps = 250 iterations

## Features

1. Custom VGG19 feature extractor
2. Gram matrix computation for style representation
3. Multi-layer style transfer
4. Real-time loss monitoring
5. Visualization of content, style, and generated images

## Usage

```bash
python solution.py
```

## Output

The script generates:
- Stylized image combining content and style
- Training loss curves (total, content, style)
- Visual comparison of all three images
- Performance metrics

## Results

Expected outputs:
- Successfully transferred artistic style to content image
- Smooth convergence of loss functions
- Visually appealing stylized images
- Content preservation with style application

## Parameters

Key hyperparameters you can tune:

```python
epochs = 5              # Number of epochs
steps_per_epoch = 50    # Steps per epoch
style_weight = 1e-2     # Weight for style loss (β)
content_weight = 1e4    # Weight for content loss (α)
learning_rate = 5.0     # Adam optimizer learning rate
```

## Technical Notes

1. **Preprocessing**: Images are preprocessed using VGG19's expected format (BGR, mean-centered)
2. **Memory**: Requires ~2-3GB GPU memory for 400x400 images
3. **Speed**: ~10-20 seconds per epoch on GPU, ~2-3 minutes on CPU
4. **Stability**: Gradient clipping prevents numerical instabilities

## Extensions

Potential improvements:
1. Add total variation loss for smoother results
2. Implement fast neural style transfer (feed-forward network)
3. Support for multiple style images (style blending)
4. Semantic-aware style transfer
5. Real-time video style transfer

## References

- Gatys et al. (2016): "A Neural Algorithm of Artistic Style"
- Johnson et al. (2016): "Perceptual Losses for Real-Time Style Transfer"
- VGG19: Simonyan & Zisserman (2014)

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
```
