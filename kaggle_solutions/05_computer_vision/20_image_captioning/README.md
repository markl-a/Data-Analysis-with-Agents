# Image Captioning

## Problem Overview
Generate natural language descriptions of images using a CNN-RNN architecture. Combines computer vision and natural language processing for visual understanding.

## Dataset
- **Synthetic images** with simple scenes:
  - Shapes: Circle, Square, Triangle
  - Colors: Red, Blue, Green, Yellow
  - Positions: Center, Corner
- **Caption Format**: "a <color> <shape> in the <position>"
- **Vocabulary**: 15 words (including special tokens)
- **Samples**: 1,000 image-caption pairs

## Approach

### Model Architecture
**CNN Encoder + RNN Decoder**:
```
Image (64x64x3)
  ↓
CNN Encoder
├─ Conv Block 1 → 32x32x32
├─ Conv Block 2 → 16x16x64
├─ Conv Block 3 → 8x8x128
└─ Global Avg Pool → 128
  ↓
FC → Hidden Features (64)
  ↓
RNN Decoder (LSTM-style)
├─ Input: <start> token + image features
├─ Embedding layer (32 dim)
├─ RNN hidden state (64 dim)
└─ Output: Next word probabilities
  ↓
Caption: "<start> a red circle in the center <end>"
```

### Key Components
1. **CNN Encoder**: Extract visual features from image
2. **Image Features**: Initialize RNN hidden state
3. **Word Embeddings**: Map words to continuous vectors
4. **RNN Decoder**: Generate caption word by word
5. **Teacher Forcing**: Use ground truth during training

### Caption Generation
- **Training**: Teacher forcing (use true previous word)
- **Inference**: Autoregressive (use predicted previous word)
- **Decoding**: Greedy (select highest probability word)

## Results

### Performance Metrics
- **Caption Accuracy**: ~60-75% (exact match)
- **Word Accuracy**: ~85-95% (per-word)
- **BLEU Score**: Would measure n-gram overlap

### Key Insights
1. Simple scenes enable learning
2. Visual features crucial for object/color
3. RNN captures word dependencies
4. Teacher forcing stabilizes training
5. Limited vocabulary simplifies task

## Files Generated
1. `captioning_samples.png` - Images with true/predicted captions
2. `captioning_training_history.png` - Training curves

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/20_image_captioning
python solution.py
```

## Requirements
- numpy
- matplotlib
- scikit-learn

## Applications
- **Accessibility**: Screen readers for visually impaired
- **Photo Organization**: Automatic image tagging
- **Social Media**: Auto-generated captions
- **Visual Search**: Image retrieval by description
- **Robotics**: Scene understanding
- **Medical Imaging**: Report generation

## Vocabulary
- **Special Tokens**: `<start>`, `<end>`, `<pad>`
- **Articles**: a, the
- **Colors**: red, blue, green, yellow
- **Objects**: circle, square, triangle
- **Prepositions**: in
- **Locations**: center, corner

## Future Improvements
1. Attention mechanism (focus on image regions)
2. Beam search decoding (multiple hypotheses)
3. Larger vocabulary and complex scenes
4. BLEU/METEOR/CIDEr evaluation metrics
5. Pre-trained CNN (ResNet, VGG)
6. Transformer-based models
7. Multi-object scenes
8. Relationship descriptions

## Related Datasets
- MS COCO Captions: 330K images, 5 captions each
- Flickr8k/Flickr30k: Diverse image captions
- Visual Genome: Dense captions and relationships
- Conceptual Captions: 3.3M image-text pairs

## Training Techniques
- **Teacher Forcing**: Use true previous words during training
- **Scheduled Sampling**: Gradually reduce teacher forcing
- **Beam Search**: Keep top-k caption hypotheses
- **Attention**: Weight image regions by relevance
