# Siamese Networks for Similarity Learning

Learn similarity metrics using Siamese neural networks with contrastive loss.

## Problem Description

Siamese networks are designed to learn similarity between inputs. Instead of classifying inputs into fixed categories, they learn an embedding space where similar inputs are close together and dissimilar inputs are far apart. This is useful for face recognition, signature verification, one-shot learning, and more.

## Approach

### Architecture

```
Image A ──> Embedding Network ──> Embedding A ──┐
                  ↓ (shared weights)             ├──> Distance ──> Loss
Image B ──> Embedding Network ──> Embedding B ──┘
```

### Siamese Network Structure

```
┌─────────────────────────────────────────────────────────┐
│ Input A                        Input B                  │
│   ↓                              ↓                      │
│ ┌───────────────────────────────────────────┐           │
│ │   Shared Embedding Network (CNN)          │           │
│ │                                           │           │
│ │   Conv2D(32) → Pool → Conv2D(64) → Pool  │           │
│ │   → Conv2D(128) → Flatten → Dense(256)   │           │
│ │   → Dense(128) → L2 Normalize            │           │
│ └───────────────────────────────────────────┘           │
│   ↓                              ↓                      │
│ Embedding A                   Embedding B               │
│   ↓                              ↓                      │
│   └──────────── Distance ────────┘                     │
│                   ↓                                     │
│           Contrastive Loss                              │
└─────────────────────────────────────────────────────────┘
```

### Contrastive Loss

The contrastive loss function:

```
L(y, d) = y * d² + (1-y) * max(margin - d, 0)²
```

Where:
- y: Label (1 for similar, 0 for dissimilar)
- d: Euclidean distance between embeddings
- margin: Margin for dissimilar pairs (typically 1.0)

**Interpretation:**
- **Similar pairs (y=1)**: Minimize distance d
- **Dissimilar pairs (y=0)**: Push apart up to margin

### Training Process

1. **Create Pairs**:
   - Similar pairs: Same class
   - Dissimilar pairs: Different classes

2. **Forward Pass**:
   - Pass both images through shared embedding network
   - Compute L2 distance between embeddings

3. **Loss Calculation**:
   - Apply contrastive loss
   - Penalize similar pairs with large distance
   - Penalize dissimilar pairs with small distance

4. **Backpropagation**:
   - Update embedding network weights
   - Same weights used for both branches

5. **Inference**:
   - Compute embeddings for query and reference
   - Calculate distance
   - Threshold for similarity decision

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Embedding Network**: CNN (Conv2D + Dense layers)
- **Embedding Dimension**: 128
- **Distance Metric**: L2 (Euclidean) distance
- **Loss Function**: Contrastive loss
- **Margin**: 1.0
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 20
- **Batch Size**: 32
- **Normalization**: L2 normalization of embeddings

## Features

1. Shared weight architecture
2. Contrastive loss implementation
3. L2 normalized embeddings
4. Custom accuracy metric
5. Pair generation from classes
6. Similarity prediction
7. Embedding extraction

## Usage

```bash
python solution.py
```

## Output

The script generates:
1. Similar and dissimilar pair predictions with distances
2. Training and validation loss curves
3. Accuracy curves
4. Visual comparison of pairs

## Results

Expected outputs:
- Similar pairs: Distance < 0.5
- Dissimilar pairs: Distance > 1.0
- Validation accuracy: 85-95%
- Clear separation in embedding space

## Key Concepts

### Shared Weights

Both branches use the **same** embedding network:
```python
embedding_a = embedding_network(input_a)
embedding_b = embedding_network(input_b)
```

Benefits:
- Consistent feature extraction
- Learns symmetric similarity
- Reduces parameters

### L2 Normalization

Normalize embeddings to unit sphere:
```python
embedding = tf.nn.l2_normalize(embedding, axis=1)
```

Benefits:
- Distance bounded [0, 2]
- Focuses on direction, not magnitude
- Improves training stability

### Distance Metrics

**L2 (Euclidean) Distance:**
```
d = sqrt(Σ(a_i - b_i)²)
```

**Cosine Similarity:**
```
sim = (a · b) / (||a|| ||b||)
```

**Manhattan Distance:**
```
d = Σ|a_i - b_i|
```

## Parameters

Key hyperparameters you can tune:

```python
embedding_dim = 128       # Embedding vector size
margin = 1.0             # Contrastive loss margin
learning_rate = 0.001    # Adam learning rate
epochs = 20              # Training epochs
batch_size = 32          # Batch size
threshold = 0.5          # Similarity threshold
```

## Technical Notes

1. **Weight Sharing**: Crucial for symmetric similarity
2. **Pair Generation**: Balance similar/dissimilar pairs
3. **Margin Selection**: Affects separation in embedding space
4. **Normalization**: L2 normalization improves performance
5. **Hard Negative Mining**: Select challenging dissimilar pairs

## Contrastive vs Triplet Loss

### Contrastive Loss (This Implementation)
```
Input: (Image A, Image B, Label)
Loss: L(y, d) = y*d² + (1-y)*max(m-d, 0)²
```

**Pros**: Simple, effective, well-studied
**Cons**: Requires pair generation

### Triplet Loss
```
Input: (Anchor, Positive, Negative)
Loss: L = max(d(A,P) - d(A,N) + margin, 0)
```

**Pros**: Relative distances, better for ranking
**Cons**: Harder to optimize, needs triplet mining

## Applications

1. **Face Recognition**: Verify if two faces are the same person
2. **Signature Verification**: Authenticate signatures
3. **One-Shot Learning**: Classify with few examples
4. **Product Matching**: Find similar products
5. **Duplicate Detection**: Find duplicate images/documents
6. **Image Retrieval**: Search by similarity
7. **Metric Learning**: Learn custom distance metrics

## One-Shot Learning

Siamese networks excel at one-shot learning:

**Problem**: Classify with only one example per class

**Solution**:
1. Train Siamese network on similarity
2. At test time:
   - Compute embedding for query
   - Compute embeddings for one example per class
   - Predict class with smallest distance

**Advantages**:
- No retraining needed for new classes
- Works with minimal examples
- Generalizes to unseen classes

## Extensions

Potential improvements:

1. **Triplet Loss**: Use anchor-positive-negative triplets
2. **Hard Negative Mining**: Select challenging negatives
3. **Online Pair Mining**: Generate pairs during training
4. **Multi-Task Learning**: Combine with classification
5. **Attention Mechanism**: Focus on discriminative regions
6. **Metric Learning**: Learn custom distance metrics
7. **Deep Metric Learning**: Deeper architectures
8. **Ensemble**: Combine multiple embedding networks

## Comparison: Different Architectures

| Method | Input | Loss | Use Case |
|--------|-------|------|----------|
| Siamese | Pair + Label | Contrastive | Binary similarity |
| Triplet | Anchor+Pos+Neg | Triplet | Ranking |
| Quadruplet | 4 images | Quadruplet | Better margins |
| N-pair | Anchor+N | N-pair | Multiple negatives |

## Common Issues

**Poor Separation:**
- Increase margin
- Use harder negative examples
- Train longer
- Larger embedding dimension

**Overfitting:**
- Add dropout
- Data augmentation
- Reduce model capacity
- Regularization

**Slow Convergence:**
- Adjust learning rate
- Better pair sampling
- Use batch normalization
- Pre-train embedding network

**Embeddings Collapse:**
- Check normalization
- Verify loss implementation
- Ensure balanced pairs
- Add regularization

## Best Practices

1. **Balanced Pairs**: Equal similar/dissimilar pairs
2. **Hard Mining**: Include challenging examples
3. **Normalization**: L2 normalize embeddings
4. **Augmentation**: Augment images for robustness
5. **Margin Tuning**: Adjust based on distance distribution
6. **Batch Size**: Larger batches for stable gradients

## Evaluation Metrics

**Distance Distribution:**
- Similar pairs: Mean distance
- Dissimilar pairs: Mean distance
- Separation: Should be clear

**ROC Curve:**
- Plot TPR vs FPR at various thresholds
- AUC should be high (>0.9)

**Precision-Recall:**
- Precision at different recall levels
- Good for imbalanced scenarios

## Real-World Example

**Face Verification System:**

1. **Training**:
   - Dataset: 10k people, 50 images each
   - Create 500k pairs (250k similar, 250k dissimilar)
   - Train Siamese network

2. **Deployment**:
   - Store embedding for registered face
   - For verification:
     - Compute embedding of query face
     - Calculate distance to stored embedding
     - Accept if distance < threshold (e.g., 0.4)

3. **Advantages**:
   - Add new people without retraining
   - Robust to variations
   - Fast inference

## Distance Threshold Selection

**Method 1: Validation Set**
- Compute distances on validation pairs
- Select threshold maximizing accuracy

**Method 2: ROC Analysis**
- Plot ROC curve
- Select threshold balancing TPR/FPR

**Method 3: Application-Specific**
- Security: Low threshold (few false accepts)
- User experience: High threshold (few false rejects)

## Comparison: Classification vs Similarity

| Aspect | Classification | Siamese Network |
|--------|---------------|-----------------|
| New Classes | Retrain | No retrain |
| Few Examples | Poor | Excellent |
| Fixed Classes | Yes | No |
| Flexibility | Low | High |
| Training Data | Moderate | High (pairs) |

## References

- Bromley et al. (1993): "Signature Verification using a Siamese Time Delay Neural Network"
- Koch et al. (2015): "Siamese Neural Networks for One-shot Image Recognition"
- Schroff et al. (2015): "FaceNet: A Unified Embedding for Face Recognition"
- Chopra et al. (2005): "Learning a Similarity Metric Discriminatively"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
