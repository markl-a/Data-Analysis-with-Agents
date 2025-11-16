# Capsule Networks (CapsNet)

## 🎯 Problem Overview

Capsule Networks introduce a new building block called "capsules" - groups of neurons that represent specific properties of entities. Unlike traditional CNNs that lose spatial information through pooling, CapsNets preserve hierarchical pose relationships.

### Objective
Implement capsule networks with dynamic routing to achieve better representation of spatial hierarchies in image classification.

## 🔬 Methodology

### Architecture Components

1. **Feature Extraction**: Initial convolutional-like layer
2. **Primary Capsules**: Convert features to capsule representations
3. **Digit Capsules**: High-level capsules with dynamic routing
4. **Decoder Network**: Reconstruct input for regularization

### Capsule Representation
- **Vector Output**: Each capsule outputs a vector (not scalar)
- **Length**: Represents probability of entity presence
- **Direction**: Represents pose/properties of entity

### Dynamic Routing Algorithm
```
for each routing iteration:
    1. Compute coupling coefficients (softmax)
    2. Weighted sum of predictions
    3. Squash to normalize
    4. Update routing based on agreement
```

## 💻 Implementation Details

### Key Components

1. **Squash Activation**
   ```python
   squash(v) = ||v||² / (1 + ||v||²) · v / ||v||
   ```
   - Preserves direction
   - Normalizes length to [0,1]

2. **Primary Capsules**
   - Transform features to capsule vectors
   - Each capsule: 8D vector
   - 16 capsules total

3. **Digit Capsules**
   - One capsule per class (10 digits)
   - Each capsule: 16D vector
   - Dynamic routing between layers

4. **Margin Loss**
   ```python
   L = T_c·max(0, m⁺ - ||v_c||)² + λ·(1-T_c)·max(0, ||v_c|| - m⁻)²
   ```
   - T_c = 1 if class c present
   - m⁺ = 0.9, m⁻ = 0.1
   - λ = 0.5

5. **Reconstruction Regularizer**
   - Decoder network reconstructs input
   - MSE reconstruction loss
   - Weighted by 0.0005

## 📊 Visualizations

1. **Training Curves**: Loss and accuracy over epochs
2. **Capsule Activations**: Heatmap of average activations per class
3. **Reconstructions**: Original vs reconstructed images

## 🚀 Usage

```bash
python solution.py
```

### Expected Output
```
CAPSULE NETWORKS - KAGGLE SOLUTION
================================================================

📊 Loading dataset...
Dataset shape: (1797, 64)

🏗️ Building Capsule Network...
  Primary Capsules: 16 × 8D
  Digit Capsules: 10 × 16D

Training CapsNet for 50 epochs...
Epoch 10/50 | Train Loss: 0.0234 | Train Acc: 0.9532 | Val Acc: 0.9444
...

✅ Test Accuracy: 0.9611
```

## 🎓 Key Concepts

### Why Capsules?

#### Problems with CNNs
1. **Pooling loses spatial info**: Max pooling discards precise positions
2. **No pose encoding**: Can't represent rotations, scales
3. **No part-whole relationships**: Poor hierarchical understanding

#### Capsule Advantages
1. **Equivariance**: Activities change systematically with pose
2. **Pose information**: Vector output encodes properties
3. **Routing by agreement**: Dynamic parsing of scene
4. **Sample efficiency**: Better generalization

### Dynamic Routing

#### Routing by Agreement
- Lower-level capsules vote for higher-level
- Agreement strengthens connections
- Disagreement weakens connections
- Iterative refinement (typically 3 iterations)

#### Coupling Coefficients
- Softmax ensures they sum to 1
- Determined by routing algorithm
- Not learned weights (computed dynamically)

### Capsule Properties

#### Length (Magnitude)
- Represents probability entity exists
- 0 = absent, 1 = definitely present
- Used for classification

#### Direction (Orientation)
- Represents instantiation parameters
- Pose, lighting, deformation, etc.
- Preserved through routing

## 📈 Results Interpretation

### Training Dynamics

1. **Initial Phase**: High loss, routing adjusting
2. **Learning Phase**: Rapid improvement
3. **Fine-tuning**: Slower convergence
4. **Reconstruction**: Improves regularization

### Capsule Activation Patterns

- **Class-specific**: Each digit capsule activates for its class
- **Dimension meanings**: Different dimensions encode different properties
- **Consistent**: Similar inputs produce similar capsule outputs

## 🔧 Customization

### Modify Architecture
```python
model = CapsuleNetwork(
    input_dim=64,
    n_classes=10,
    n_primary_capsules=32,      # More capsules
    primary_capsule_dim=16,     # Higher dimension
    digit_capsule_dim=32        # More expressive
)
```

### Adjust Routing
```python
digit_caps = DigitCapsule(
    ...,
    n_routing_iterations=5  # More routing iterations
)
```

### Loss Weights
```python
# In training
margin_loss = ...
recon_loss = ...
total_loss = margin_loss + 0.001 * recon_loss  # Adjust weight
```

## 🎯 Practical Applications

### When to Use CapsNets
- Need to handle viewpoint changes
- Spatial relationships critical
- Limited training data
- Hierarchical structure important

### Applications
1. **Object Recognition**: Better with rotations/scales
2. **Medical Imaging**: Precise spatial relationships
3. **Face Recognition**: Handle poses and expressions
4. **Segmentation**: Part-whole relationships
5. **3D Object Understanding**: Pose estimation

## 📚 Advanced Topics

### Improvements to Try

1. **EM Routing**
   - Use expectation-maximization
   - More principled than dynamic routing
   - Better convergence properties

2. **Matrix Capsules**
   - Use pose matrices instead of vectors
   - More expressive transformations
   - Better viewpoint handling

3. **Self-Attention Routing**
   - Replace dynamic routing
   - More efficient computation
   - Better scalability

4. **Deeper Architectures**
   - Multiple capsule layers
   - Hierarchical feature learning
   - Better abstraction

### Challenges

1. **Computational Cost**: Routing is expensive
2. **Large Images**: Doesn't scale well yet
3. **Implementation Complexity**: More intricate than CNNs
4. **Training Stability**: Requires careful tuning

## 🏆 Competition Tips

1. **Start Simple**: Use small capsule dimensions first
2. **Monitor Routing**: Check if coupling coefficients make sense
3. **Balance Losses**: Tune reconstruction weight carefully
4. **Use Regularization**: Reconstruction helps prevent overfitting
5. **Compare with CNN**: Ensure capsules provide actual benefit

## 📖 References

- Sabour et al. (2017): "Dynamic Routing Between Capsules"
- Hinton et al. (2018): "Matrix Capsules with EM Routing"
- Rajasegaran et al. (2019): "DeepCaps: Going Deeper with Capsule Networks"

## 🔗 Related Techniques

- **Attention Mechanisms**: Similar idea of dynamic weighting
- **Graph Neural Networks**: Message passing resembles routing
- **Transformers**: Self-attention as alternative routing
- **Mixture of Experts**: Dynamic combination of components

## 💡 Key Takeaways

1. ✅ Capsules encode both presence and pose
2. ✅ Dynamic routing replaces pooling
3. ✅ Vector outputs more expressive than scalars
4. ✅ Better equivariance to transformations
5. ✅ Reconstruction provides regularization
6. ✅ More sample efficient than CNNs
