# Image + Text Multi-Modal Classification

## Overview
This example demonstrates multi-modal learning by combining image and text features for product categorization. It showcases different fusion strategies and their impact on classification performance.

## Problem Statement
Classify products into categories using both visual features (product images) and textual features (product descriptions). This simulates real-world e-commerce scenarios where both modalities provide complementary information.

## Dataset
- **Size**: 1,000 synthetic product samples
- **Categories**: 5 (Electronics, Clothing, Home & Garden, Sports, Books)
- **Modalities**:
  - **Image Features**: 512-dimensional embeddings (simulating CNN features)
  - **Text Features**: 300-dimensional embeddings (simulating word embeddings)

## Multi-Modal Fusion Strategies

### 1. Early Fusion
- **Approach**: Concatenate image and text features before classification
- **Advantages**:
  - Simple implementation
  - Allows model to learn cross-modal interactions
- **Disadvantages**:
  - High-dimensional feature space
  - May not capture modality-specific patterns well

### 2. Late Fusion
- **Approach**: Train separate models for each modality, combine predictions
- **Advantages**:
  - Preserves modality-specific information
  - Can weight different modalities
- **Disadvantages**:
  - No cross-modal feature learning
  - May miss complex interactions

### 3. Hybrid Fusion
- **Approach**: Train base models per modality, use meta-classifier on predictions
- **Advantages**:
  - Best of both worlds
  - Learns optimal combination strategy
- **Disadvantages**:
  - More complex
  - Requires more training time

## Feature Engineering

### Image Features
- Statistical aggregations (mean, std, max, min, median, quartiles)
- Dimensionality reduction (top 20 principal components)
- Spatial pattern encoding

### Text Features
- Statistical aggregations (mean, std, max, min, median, quartiles)
- Semantic component extraction (top 20 components)
- Linguistic pattern encoding

## Models Used
- **Random Forest Classifier**: For robust non-linear classification
- **Logistic Regression**: For meta-classification in hybrid fusion
- **Ensemble Methods**: Combining multiple modalities

## Ablation Study
The solution includes an ablation study comparing:
1. **Image-Only Model**: Using only visual features
2. **Text-Only Model**: Using only textual features
3. **Combined Model**: Using both modalities

This demonstrates the value added by multi-modal learning.

## Key Metrics
- **Accuracy**: Overall classification accuracy
- **Per-Category Performance**: F1-scores for each product category
- **Fusion Comparison**: Performance across different fusion strategies
- **Modality Contribution**: Individual vs. combined performance

## Visualizations
1. **Fusion Strategy Comparison**: Bar plot comparing all strategies
2. **Confusion Matrix**: For the best-performing model
3. **Per-Category Performance**: F1-scores across categories
4. **Ablation Study**: Individual vs. combined modality performance

## Usage

```python
# Run the complete analysis
python solution.py
```

## Expected Output
```
================================================================================
Image + Text Multi-Modal Classification
================================================================================

1. Generating synthetic product data...
   Generated 1000 samples across 5 categories
   Categories: Electronics, Clothing, Home & Garden, Sports, Books
   Train: 800, Test: 200

2. Comparing fusion strategies...
   Training early fusion model...
   Early Fusion Accuracy: 0.8550

   Training late fusion model...
   Late Fusion Accuracy: 0.8400

   Training hybrid fusion model...
   Hybrid Fusion Accuracy: 0.8700

3. Running ablation study...
   Ablation Study Results:
   Image Only: 0.7150
   Text Only: 0.7200
   Combined: 0.8550

================================================================================
RESULTS SUMMARY
================================================================================

Best Fusion Strategy: HYBRID
Best Accuracy: 0.8700

Improvement over single modality:
  vs Image Only: +15.50%
  vs Text Only:  +15.00%

4. Generating visualizations...
✓ Visualization saved: multimodal_fusion_comparison.png
```

## Key Findings
1. **Multi-modal beats uni-modal**: Combined models outperform single-modality models by ~15%
2. **Fusion strategy matters**: Hybrid fusion typically performs best
3. **Complementary information**: Image and text provide different but complementary signals
4. **Category-specific patterns**: Some categories benefit more from visual features, others from text

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Difficulty
⭐⭐⭐ Intermediate

## Learning Objectives
- Understand multi-modal data fusion
- Compare different fusion strategies
- Conduct ablation studies
- Extract features from different modalities
- Evaluate cross-modal learning benefits

## Real-World Applications
- E-commerce product categorization
- Social media content classification
- Medical diagnosis (images + patient records)
- Multimedia content recommendation
- Document classification with images

## Extensions
1. Add attention mechanisms to weight modalities differently
2. Implement deep learning fusion (e.g., with neural networks)
3. Add more modalities (e.g., product metadata, reviews)
4. Experiment with different feature extraction methods
5. Implement cross-modal retrieval (find images from text, vice versa)

## References
- Multimodal Machine Learning: A Survey and Taxonomy (Baltrusaitis et al., 2019)
- Early vs Late Fusion in Multimodal Convolutional Neural Networks (Neverova et al., 2015)
- Multimodal Deep Learning (Ngiam et al., 2011)
