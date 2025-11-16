# Meta-Learning (Few-Shot Learning)

## 🎯 Problem Overview

Meta-learning, or "learning to learn," trains models that can quickly adapt to new tasks with minimal data. Few-shot learning is a meta-learning approach where models learn from just a few examples.

### Objective
Build a model that can classify novel classes (never seen during training) using only a handful of examples per class.

## 🔬 Methodology

### Few-Shot Learning Setup

**N-way K-shot Classification**
- **N-way**: N different classes in each task
- **K-shot**: K examples per class (support set)
- **Query set**: Examples to classify

Example: 5-way 5-shot
- 5 classes
- 5 examples per class (25 total support examples)
- Classify new query examples

### Prototypical Networks

1. **Embedding**: Map examples to metric space
2. **Prototypes**: Compute class centers (means)
3. **Classification**: Assign to nearest prototype

```
Prototype_c = mean(embed(x) for x in class c)
Predict(query) = argmin_c distance(embed(query), Prototype_c)
```

## 💻 Implementation Details

### Architecture

1. **Embedding Network**
   ```
   Input → FC(128) → ReLU → FC(128) → ReLU → FC(64) → L2-Norm
   ```
   - Learns metric space
   - L2 normalization for stable distances

2. **Prototype Computation**
   - Average embeddings per class
   - Forms class representatives

3. **Distance Metric**
   - Euclidean distance
   - Alternative: Cosine similarity

### Meta-Training Procedure

```python
for episode in meta_training:
    1. Sample N classes
    2. Sample K examples per class (support)
    3. Sample Q examples per class (query)
    4. Compute prototypes from support set
    5. Predict query examples
    6. Update embedding network
```

### Meta-Testing

- Use completely novel classes
- Never seen during meta-training
- Tests generalization ability

## 📊 Visualizations

1. **Training Curves**: Meta-training progress
2. **Few-Shot Example**: Support, query, prototypes in 2D
3. **K-shot Analysis**: Performance vs number of shots
4. **N-way Analysis**: Performance vs number of classes
5. **Embedding Space**: Learned metric space
6. **Distance Distribution**: Same vs different class
7. **Confusion Matrix**: Sample task errors
8. **Summary Statistics**: Overall performance

## 🚀 Usage

```bash
python solution.py
```

### Expected Output
```
META-LEARNING (FEW-SHOT LEARNING) - KAGGLE SOLUTION
================================================================

📊 Loading dataset...
Meta-train: 1232 samples, classes [0, 1, 2, 3, 4, 5, 6]
Meta-test: 565 samples, classes [7, 8, 9] (novel!)

Meta-Training Phase...
Episode 100/1000 | Loss: 0.4523 | Accuracy: 0.7867
Episode 200/1000 | Loss: 0.3145 | Accuracy: 0.8734
...

Meta-Testing Phase (novel classes)...
✅ Meta-Test Accuracy (3-way 5-shot): 0.8945 ± 0.0234

1-shot learning: Accuracy: 0.6534 ± 0.0445
10-shot learning: Accuracy: 0.9456 ± 0.0156
```

## 🎓 Key Concepts

### Why Meta-Learning?

#### Traditional ML Problems
- Need lots of labeled data
- Separate model per task
- Doesn't leverage related tasks
- Poor generalization to new tasks

#### Meta-Learning Solutions
- Learn from few examples
- Rapid adaptation
- Transfer across tasks
- Better sample efficiency

### Episode-Based Training

#### Traditional Training
- Fixed dataset
- Many epochs
- Learn specific task

#### Meta-Training
- Many episodes (tasks)
- Each episode is different
- Learn how to learn

### N-way K-shot Notation

#### Examples
- **5-way 1-shot**: 5 classes, 1 example each
- **5-way 5-shot**: 5 classes, 5 examples each
- **10-way 10-shot**: 10 classes, 10 examples each

#### Difficulty
- More ways (N) → Harder
- Fewer shots (K) → Harder
- 1-shot is extreme few-shot

### Metric Learning

#### Goal
Learn embedding space where:
- Same class examples are close
- Different class examples are far

#### Prototypical Approach
- Represent each class by prototype
- Prototype = mean of class embeddings
- Classify by nearest prototype

## 📈 Results Interpretation

### Good Performance Indicators

1. **Low intra-class distance**: Same class close together
2. **High inter-class distance**: Different classes far apart
3. **Increasing accuracy**: Meta-training progresses
4. **Generalizes to novel classes**: Non-zero test accuracy

### Common Patterns

- **1-shot**: ~60-70% accuracy (challenging)
- **5-shot**: ~80-90% accuracy
- **10-shot**: ~90-95% accuracy
- **Performance plateaus**: After ~10 shots

## 🔧 Customization

### Modify Embedding Network
```python
model = PrototypicalNetwork(
    input_dim=64,
    embedding_dim=128  # Larger embedding
)

# In EmbeddingNetwork
hidden_dims=[256, 256, 128]  # Deeper network
```

### Different Distance Metrics
```python
# Cosine distance
def cosine_distance(x, y):
    dot_product = np.dot(x, y.T)
    return 1 - dot_product  # Assumes L2-normalized

# Learned distance (relation networks)
distances = relation_network(concat(query, prototype))
```

### Task Configuration
```python
# Harder tasks
meta_train(model, X, y,
          n_way=10,    # More classes
          k_shot=1,    # Fewer examples
          n_query=30)  # More queries
```

## 🎯 Practical Applications

### Common Applications

1. **Computer Vision**
   - Face recognition from few images
   - Object detection with few examples
   - Medical image diagnosis (rare diseases)

2. **NLP**
   - Language adaptation
   - Intent classification
   - Named entity recognition

3. **Robotics**
   - Quick task adaptation
   - New object manipulation
   - Environment adaptation

4. **Drug Discovery**
   - Predict properties with few molecules
   - Rare disease treatment
   - Personalized medicine

### When to Use

- Limited labeled data
- Many related tasks
- Need rapid adaptation
- Cost of labeling is high

## 📚 Advanced Topics

### Meta-Learning Approaches

1. **Model-Agnostic Meta-Learning (MAML)**
   - Learn initialization
   - Fast fine-tuning
   - Finn et al., 2017

2. **Prototypical Networks** (This Implementation)
   - Metric learning
   - Simple and effective
   - Snell et al., 2017

3. **Matching Networks**
   - Attention over support set
   - Non-parametric
   - Vinyals et al., 2016

4. **Relation Networks**
   - Learn distance metric
   - More flexible
   - Sung et al., 2018

### Improvements

1. **Task Augmentation**
   - More diverse tasks
   - Better generalization

2. **Transductive Methods**
   - Use query set structure
   - Semi-supervised approach

3. **Multi-Modal**
   - Leverage text descriptions
   - Cross-modal learning

4. **Hierarchical**
   - Learn at multiple levels
   - Coarse-to-fine

## 🏆 Competition Tips

1. **Task Design**: Create diverse meta-training tasks
2. **Embedding Quality**: Monitor embedding space
3. **Augmentation**: Use data augmentation heavily
4. **Ensemble**: Combine multiple meta-learners
5. **Fine-tuning**: Can fine-tune on support set
6. **Cross-Validation**: Use episode-based CV

## 📖 References

- Snell et al. (2017): "Prototypical Networks for Few-shot Learning"
- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"
- Vinyals et al. (2016): "Matching Networks for One Shot Learning"
- Sung et al. (2018): "Learning to Compare: Relation Network for Few-Shot Learning"

## 🔗 Related Techniques

- **Transfer Learning**: Pre-train and fine-tune
- **Domain Adaptation**: Adapt to new domain
- **Zero-Shot Learning**: No examples (only descriptions)
- **Online Learning**: Continual adaptation

## 💡 Key Takeaways

1. ✅ Meta-learning enables learning from few examples
2. ✅ Prototypes represent class centers in embedding space
3. ✅ Episode-based training learns to learn
4. ✅ Works on completely novel classes
5. ✅ Sample efficiency is dramatically improved
6. ✅ Metric learning is key to success
