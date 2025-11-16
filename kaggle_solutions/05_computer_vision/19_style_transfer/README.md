# Neural Style Transfer

## Problem Overview
Transfer artistic style from one image to another while preserving content structure. Applications in art generation, photo editing, and creative tools.

## Dataset
- **Synthetic images**:
  - Content: Simple geometric shapes (house)
  - Styles: Brushstrokes, waves, geometric patterns
- **Image Size**: 64x64x3
- **Styles**: 3 different artistic patterns

## Approach

### Model Architecture
**Style Transfer Network**:
```
Content Image → Feature Extractor → Content Features
Style Image → Feature Extractor → Style Features → Gram Matrix

Optimization Loop:
  Generated Image → Features
    ├─ Content Loss: ||Features - Content Features||²
    └─ Style Loss: ||Gram(Features) - Style Gram||²

  Total Loss = α * Content Loss + β * Style Loss
  Update Generated Image
```

### Key Concepts
1. **Content Representation**: High-level CNN features preserve structure
2. **Style Representation**: Gram matrices capture texture/patterns
3. **Loss Balancing**: α (content weight) vs β (style weight)
4. **Iterative Optimization**: Gradually blend content and style

### Gram Matrix
- Captures correlations between feature maps
- Represents texture independent of spatial layout
- Used to measure style similarity

## Results

### Style Types
1. **Brushstrokes**: Impressionist painting effect
2. **Waves**: Sinusoidal pattern overlay
3. **Geometric**: Abstract block patterns

### Key Insights
1. Content loss preserves structure
2. Style loss transfers texture
3. Balance between α and β crucial
4. More iterations = stronger style
5. Different layers capture different features

## Files Generated
1. `style_patterns.png` - Different style examples
2. `style_transfer_brushstrokes.png` - Brushstroke style result
3. `style_transfer_waves.png` - Wave pattern result
4. `style_transfer_geometric.png` - Geometric pattern result
5. `style_transfer_comparison.png` - All styles compared

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/19_style_transfer
python solution.py
```

## Requirements
- numpy
- matplotlib

## Applications
- **Art Generation**: Create artistic images
- **Photo Filters**: Instagram-style effects
- **Video Stylization**: Apply styles to video frames
- **Creative Tools**: Photoshop/GIMP plugins
- **Game Graphics**: Procedural texture generation
- **AR Filters**: Real-time style application

## Hyperparameters
- **α (Content Weight)**: Controls content preservation (typical: 1.0)
- **β (Style Weight)**: Controls style strength (typical: 1000.0)
- **Iterations**: Number of optimization steps (typical: 100-1000)
- **Learning Rate**: Step size for updates

## Future Improvements
1. Fast neural style transfer (feed-forward networks)
2. Multi-style transfer (blend multiple styles)
3. Perceptual loss functions
4. Instance normalization for better results
5. Video temporal consistency
6. Real-time mobile implementation

## Related Approaches
- Gatys et al. (2015): Original neural style transfer
- Johnson et al. (2016): Fast style transfer
- Adaptive Instance Normalization (AdaIN)
- StyleGAN: Generative approach
