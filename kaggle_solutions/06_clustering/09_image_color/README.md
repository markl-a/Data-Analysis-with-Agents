# Image Color Quantization using Clustering

## Overview
This solution demonstrates color quantization - reducing the number of colors in an image while preserving visual quality using K-Means clustering. This is a fundamental technique in image processing and compression.

## Problem Statement
Digital images often contain thousands or millions of unique colors, leading to large file sizes and slow processing. Color quantization reduces colors to a palette while maintaining visual appeal, useful for compression, web optimization, and artistic effects.

## Technical Approach
Color quantization treats each pixel as a 3D point in RGB color space. Clustering groups similar colors together, replacing them with their cluster center (representative color).

## Data Representation
- **Input**: Image pixels as (R, G, B) triplets
- **Color Space**: RGB (0-255 for each channel)
- **Clustering**: Each pixel is a sample, colors are features
- **Output**: Quantized image with reduced color palette

## Synthetic Image Components
The generated test image includes:
- **Sky**: Blue gradient (natural color transition)
- **Sun**: Yellow circle (distinct bright object)
- **Grass**: Green field with variation
- **Flowers**: Multi-colored dots (red, pink, yellow, orange, purple)
- **Trees**: Brown trunks with green foliage
- **Noise**: Realistic color variation

## Clustering Algorithms
1. **K-Means**: Standard algorithm for color quantization
   - Finds k representative colors (cluster centers)
   - Assigns each pixel to nearest color
   - Iteratively optimizes color palette

2. **MiniBatch K-Means**: Faster variant for large images
   - Processes random subsets of pixels
   - Significantly faster with minimal quality loss
   - Ideal for real-time applications

## Analysis Components

### 1. Color Count Comparison
Shows visual quality at different compression levels:
- 4 colors: Highly compressed, posterized effect
- 8 colors: Moderate compression, visible reduction
- 16 colors: Good balance of quality and compression
- 32 colors: High quality, subtle differences
- 64 colors: Near-original quality

### 2. Color Palette Extraction
- Visual palette showing extracted colors
- Distribution chart showing frequency of each color
- Sorted by dominance in image

### 3. Quality Analysis
- **MSE (Mean Squared Error)**: Reconstruction error
- **Silhouette Score**: Clustering quality
- Trade-off curves showing quality vs color count

### 4. Algorithm Comparison
- Speed comparison: K-Means vs MiniBatch
- Visual quality comparison
- Computational efficiency analysis

## Evaluation Metrics
- **Mean Squared Error (MSE)**: Pixel-level reconstruction error (lower better)
- **Silhouette Score**: Color cluster separation quality (higher better)
- **Compression Ratio**: Percentage of colors reduced
- **Processing Time**: Algorithm execution speed

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage
```bash
python solution.py
```

## Output
1. Original synthetic image
2. Comparison grid with different color counts
3. Extracted color palette with distribution
4. Quality analysis plots (MSE and Silhouette vs colors)
5. Algorithm comparison (K-Means vs MiniBatch)
6. Quantized images at various compression levels

## Applications

### Web & Mobile
- **Image Optimization**: Reduce file sizes for faster loading
- **Bandwidth Savings**: Smaller images use less data
- **Responsive Design**: Adaptive image quality

### Design & Art
- **Palette Extraction**: Identify dominant colors for themes
- **Color Schemes**: Generate matching color palettes
- **Artistic Effects**: Posterization and stylization
- **Brand Colors**: Extract brand palette from images

### Image Processing
- **Preprocessing**: Simplify images for analysis
- **Compression**: Lossy image compression
- **Segmentation**: Initial step for object detection
- **Video Processing**: Reduce processing requirements

### Data Science
- **Feature Reduction**: Simplify image features
- **Pattern Recognition**: Focus on color patterns
- **Visualization**: Create cleaner visualizations

## Key Insights
- 16-32 colors often provide good quality-compression balance
- Dominant colors can be extracted and analyzed
- MiniBatch K-Means offers 5-10x speedup with similar quality
- Color quantization is reversible but lossy

## Technical Details
- **Sampling**: Random sampling for large images (>10K pixels)
- **Color Space**: RGB (could extend to LAB for perceptual uniformity)
- **Initialization**: K-Means++ for better initial centroids
- **Convergence**: Maximum 100 iterations

## Real-World Extensions
- LAB color space for perceptual quality
- Dithering for smoother gradients
- Adaptive quantization based on image content
- GIF generation with optimal palettes
- Region-specific quantization
- Temporal consistency for video

## Comparison with Other Methods
- **Median Cut**: Divides color space by median
- **Octree**: Tree-based color reduction
- **K-Means**: Generally produces best visual quality
- **Neural Networks**: Learning-based approaches

## Performance Considerations
- Image size impacts processing time
- Sampling reduces computation for large images
- MiniBatch K-Means recommended for real-time use
- Parallelization possible for batch processing

## Author
Kaggle Competition Solution - Image Processing Clustering
