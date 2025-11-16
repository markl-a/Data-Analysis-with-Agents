# Transfer Learning with Pre-trained Models

Leverage pre-trained deep learning models for image classification tasks.

## Problem Description

Transfer learning involves taking a model trained on one task and adapting it to a new related task. This implementation demonstrates how to use pre-trained models (trained on ImageNet) for a custom classification task, achieving high accuracy with limited training data.

## Approach

### Architecture

```
Input Image ──> Pre-trained Base Model ──> Global Average Pooling ──> Custom Classifier ──> Predictions
  (224×224×3)      (Frozen Layers)              (Feature Vector)        (Dense Layers)       (3 classes)
```

### Transfer Learning Strategy

```
┌────────────────────────────────────────────────────────┐
│ Pre-trained Model (e.g., MobileNetV2)                  │
│ Trained on ImageNet (1000 classes, 1.2M images)        │
│                                                        │
│ ┌──────────────────────────────────────────┐          │
│ │ Convolutional Layers (Feature Extractor)  │          │
│ │ - Extract low-level features (edges)      │ FROZEN  │
│ │ - Extract mid-level features (textures)   │          │
│ │ - Extract high-level features (patterns)  │          │
│ └──────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ Custom Classifier Head                                 │
│                                                        │
│ Global Average Pooling                                 │
│         ↓                                              │
│ Dense (256) + ReLU + Dropout(0.5)                     │ TRAINABLE
│         ↓                                              │
│ Dense (num_classes) + Softmax                         │
└────────────────────────────────────────────────────────┘
```

### Two-Phase Training

**Phase 1: Feature Extraction**
```
1. Freeze all pre-trained layers
2. Train only custom classifier head
3. Use higher learning rate (1e-3)
4. Quick convergence (10-20 epochs)
```

**Phase 2: Fine-tuning (Optional)**
```
1. Unfreeze top layers of base model
2. Train with lower learning rate (1e-4)
3. Fine-tune features for specific task
4. Further accuracy improvement
```

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Base Models**: MobileNetV2, VGG16, ResNet50
- **Input Size**: 224×224×3
- **Pre-trained Weights**: ImageNet
- **Feature Extraction**: Global Average Pooling
- **Classifier**: Dense(256) + Dropout(0.5) + Dense(num_classes)
- **Optimizer**: Adam (lr=1e-3 for feature extraction, 1e-4 for fine-tuning)
- **Loss**: Sparse Categorical Cross-Entropy
- **Epochs**: 10 (feature extraction)
- **Batch Size**: 32

## Features

1. Multiple pre-trained model options
2. Automatic weight loading from ImageNet
3. Frozen base model for feature extraction
4. Optional fine-tuning capability
5. Custom classifier head
6. Dropout for regularization
7. Synthetic dataset generation
8. Comprehensive evaluation

## Usage

```bash
python solution.py
```

## Output

The script generates:
1. Sample predictions with true/predicted labels
2. Training and validation curves
3. Test set evaluation metrics
4. Model architecture summary

## Results

Expected outputs:
- High accuracy with limited data (90%+)
- Fast training convergence
- Good generalization to test set
- Minimal overfitting

## Pre-trained Models Comparison

### MobileNetV2
- **Parameters**: 3.5M
- **Size**: 14 MB
- **Speed**: Very Fast
- **Accuracy**: Good
- **Use Case**: Mobile/embedded devices

### VGG16
- **Parameters**: 138M
- **Size**: 528 MB
- **Speed**: Slow
- **Accuracy**: Very Good
- **Use Case**: Research, when size doesn't matter

### ResNet50
- **Parameters**: 25.6M
- **Size**: 98 MB
- **Speed**: Medium
- **Accuracy**: Excellent
- **Use Case**: General purpose, production

### InceptionV3
- **Parameters**: 23.8M
- **Size**: 92 MB
- **Speed**: Medium
- **Accuracy**: Excellent
- **Use Case**: Complex image classification

### EfficientNet
- **Parameters**: Varies (B0-B7)
- **Size**: Varies
- **Speed**: Good
- **Accuracy**: State-of-the-art
- **Use Case**: Best accuracy-efficiency tradeoff

## Why Transfer Learning?

**Advantages:**

1. **Less Data Required**: Pre-trained features work well
2. **Faster Training**: Only train classifier head
3. **Better Generalization**: Features learned from large dataset
4. **Lower Computational Cost**: No need to train from scratch
5. **Better Performance**: Often beats training from scratch

**When to Use:**

- Limited training data (< 10k samples)
- Similar domain (natural images)
- Limited computational resources
- Need quick results

**When NOT to Use:**

- Very different domain (medical images, satellite)
- Plenty of training data (> 100k samples)
- Domain-specific features needed
- Specialized architecture required

## Parameters

Key hyperparameters you can tune:

```python
base_model_name = 'MobileNetV2'  # Base model choice
num_classes = 3                   # Number of classes
input_shape = (224, 224, 3)       # Input image shape
fine_tune_layers = 0              # Layers to fine-tune
epochs = 10                       # Training epochs
batch_size = 32                   # Batch size
learning_rate = 0.001             # Learning rate
dropout_rate = 0.5                # Dropout rate
```

## Technical Notes

1. **Input Preprocessing**: Match pre-trained model's preprocessing
2. **Learning Rate**: Lower for fine-tuning than feature extraction
3. **Batch Normalization**: Keep frozen during fine-tuning
4. **Data Augmentation**: Helps prevent overfitting
5. **Class Imbalance**: Use class weights or resampling

## Feature Extraction vs Fine-Tuning

### Feature Extraction
```python
# Freeze all base model layers
base_model.trainable = False

# Train only classifier
model.compile(optimizer=Adam(lr=1e-3), ...)
model.fit(X_train, y_train, epochs=10)
```

### Fine-Tuning
```python
# Unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Train with lower learning rate
model.compile(optimizer=Adam(lr=1e-4), ...)
model.fit(X_train, y_train, epochs=10)
```

## Best Practices

1. **Start with Feature Extraction**: Train classifier first
2. **Then Fine-Tune**: Unfreeze top layers if needed
3. **Use Data Augmentation**: Rotation, flip, zoom, etc.
4. **Monitor Validation Loss**: Stop if overfitting
5. **Learning Rate Scheduling**: Reduce LR on plateau
6. **Regularization**: Dropout, L2, early stopping

## Data Augmentation

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```

Benefits:
- Increases effective dataset size
- Improves generalization
- Reduces overfitting
- Makes model robust to variations

## Common Issues

**Low Accuracy:**
- Ensure proper preprocessing
- Check learning rate
- Add more training data
- Try different base model

**Overfitting:**
- Add dropout
- Use data augmentation
- Reduce model complexity
- Add L2 regularization

**Slow Training:**
- Reduce batch size
- Use smaller base model (MobileNet)
- Use GPU acceleration
- Freeze more layers

**Poor Generalization:**
- More diverse training data
- Stronger regularization
- Cross-validation
- Ensemble methods

## Applications

1. **Medical Imaging**: Disease classification
2. **Agriculture**: Plant disease detection
3. **Manufacturing**: Defect detection
4. **Retail**: Product classification
5. **Wildlife**: Species identification
6. **Security**: Object/person recognition
7. **Autonomous Vehicles**: Object detection

## Extensions

Potential improvements:

1. **Multi-Model Ensemble**: Combine multiple base models
2. **Progressive Fine-Tuning**: Gradually unfreeze layers
3. **Learning Rate Scheduling**: Cosine annealing, warm restarts
4. **Advanced Augmentation**: AutoAugment, RandAugment
5. **Knowledge Distillation**: Compress to smaller model
6. **Multi-Task Learning**: Train on related tasks
7. **Self-Supervised Pre-training**: Train on unlabeled data

## Comparison: Scratch vs Transfer

| Aspect | From Scratch | Transfer Learning |
|--------|-------------|-------------------|
| Data Required | 10k-1M+ | 100-10k |
| Training Time | Days-Weeks | Minutes-Hours |
| Compute | High | Low |
| Accuracy | Variable | Consistently Good |
| Expertise | High | Medium |

## Real-World Example

**Scenario**: Classify 3 types of products with 300 images each

**Option 1 - Train from Scratch:**
- Requires 10k+ images
- 24-48 hours training
- 70-80% accuracy
- High GPU cost

**Option 2 - Transfer Learning:**
- Works with 300 images
- 10-30 minutes training
- 90-95% accuracy
- Low GPU cost

**Winner**: Transfer Learning!

## References

- Yosinski et al. (2014): "How transferable are features in deep neural networks?"
- Donahue et al. (2014): "DeCAF: A Deep Convolutional Activation Feature"
- Sharif Razavian et al. (2014): "CNN Features off-the-shelf"
- Pan & Yang (2010): "A Survey on Transfer Learning"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
