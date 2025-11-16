# Graph Neural Networks (GNN)

## 🎯 Problem Overview

Graph Neural Networks extend deep learning to graph-structured data, enabling learning on networks like social graphs, molecules, citation networks, and knowledge graphs.

### Objective
Implement a Graph Convolutional Network (GCN) for semi-supervised node classification.

## 🔬 Methodology

### Graph Representation
- **Nodes**: Entities with features
- **Edges**: Relationships between nodes
- **Adjacency Matrix**: A[i,j] = 1 if edge exists
- **Feature Matrix**: X ∈ ℝ^(N×F) where N=nodes, F=features

### Graph Convolution
```
H^(l+1) = σ(Â · H^(l) · W^(l))
```
Where:
- Â = Normalized adjacency matrix
- H^(l) = Node features at layer l
- W^(l) = Learnable weights
- σ = Activation function

### Message Passing
1. **Aggregate**: Collect information from neighbors
2. **Transform**: Apply learnable transformation
3. **Activate**: Non-linear activation
4. **Repeat**: Stack multiple layers

## 💻 Implementation Details

### Adjacency Normalization
```python
Â = D^(-1/2) · (A + I) · D^(-1/2)
```
- Add self-loops (A + I)
- Symmetric normalization
- Prevents vanishing/exploding gradients

### Architecture
```
Input (16) → GCN(32) → ReLU → GCN(16) → ReLU → GCN(4) → Softmax
```

### Semi-Supervised Learning
- Train on labeled nodes only
- Propagate information through graph
- Predict unlabeled nodes
- Leverages graph structure

## 📊 Visualizations

1. **Training Curves**: Loss and accuracy
2. **Node Embeddings**: t-SNE visualization
3. **Graph Structure**: Network topology
4. **Degree Distribution**: Connectivity patterns
5. **Class Distribution**: Label balance
6. **Predictions**: True vs predicted labels
7. **Per-Class Performance**: Class-wise accuracy
8. **Summary Statistics**: Overall metrics

## 🚀 Usage

```bash
python solution.py
```

### Expected Output
```
GRAPH NEURAL NETWORKS - KAGGLE SOLUTION
================================================================

📊 Creating synthetic graph dataset...
Nodes: 200
Features per node: 16
Edges: 650
Classes: 4

🏗️ Building Graph Neural Network...
Architecture: 16 → 32 → 16 → 4

Training GNN for 200 epochs...
Epoch 20/200 | Train Loss: 0.8234 | Train Acc: 0.7083 | Val Acc: 0.6750
...

✅ Test Accuracy: 0.8500
✅ Test F1 Score: 0.8421
```

## 🎓 Key Concepts

### Why GNNs?

#### Traditional ML Limitations
- **Assumes i.i.d. data**: Ignores relationships
- **Fixed structure**: Can't handle varying graphs
- **No relational info**: Loses connection patterns

#### GNN Advantages
- **Relational inductive bias**: Exploits graph structure
- **Permutation invariance**: Order doesn't matter
- **Local + Global**: Combines neighborhood and distant info
- **Flexible**: Works with any graph

### Message Passing Framework

#### Aggregation Functions
- **Mean**: `h_i = mean(h_j for j in N(i))`
- **Sum**: `h_i = sum(h_j for j in N(i))`
- **Max**: `h_i = max(h_j for j in N(i))`

#### Update Functions
- **Linear**: `h'_i = W · h_i`
- **GRU**: Recurrent update
- **Attention**: Weighted aggregation

### Over-Smoothing Problem
- Stacking too many layers
- Node features become similar
- Loss of discriminative power
- Solution: Residual connections, jumping knowledge

## 📈 Results Interpretation

### Good Signs
1. **Embeddings cluster by class**: t-SNE shows separation
2. **Val accuracy increases**: Model generalizing
3. **Train/val gap small**: Not overfitting
4. **Per-class balanced**: All classes learned

### Common Issues
1. **Over-smoothing**: Add fewer layers
2. **Disconnected components**: Check graph connectivity
3. **Class imbalance**: Use weighted loss
4. **High degree variance**: Use attention mechanisms

## 🔧 Customization

### Modify Architecture
```python
model = GraphNeuralNetwork(
    n_features=16,
    hidden_dims=[64, 32, 16],  # Deeper network
    n_classes=4
)
```

### Different Aggregation
```python
# In GraphConvolutionalLayer.forward()
# Mean aggregation
aggregated = (A_norm @ X) / degree

# Max aggregation
aggregated = max_pool_neighbors(X, A)
```

### Add Attention
```python
# Compute attention weights
alpha = softmax(attention_scores)
aggregated = alpha * neighbor_features
```

## 🎯 Practical Applications

### Common Applications
1. **Social Networks**: User recommendation, influence prediction
2. **Molecules**: Property prediction, drug discovery
3. **Citation Networks**: Paper classification, link prediction
4. **Knowledge Graphs**: Entity classification, relation extraction
5. **Traffic Networks**: Flow prediction, route optimization

### Problem Types
- **Node Classification**: Predict node labels
- **Link Prediction**: Predict missing edges
- **Graph Classification**: Classify entire graphs
- **Community Detection**: Find clusters

## 📚 Advanced Topics

### GNN Variants

1. **GraphSAGE**
   - Sample fixed-size neighborhoods
   - More scalable
   - Inductive learning

2. **Graph Attention Networks (GAT)**
   - Attention-based aggregation
   - Different weights for neighbors
   - Better expressiveness

3. **Graph Isomorphism Network (GIN)**
   - Maximally expressive
   - Injective aggregation
   - Theoretical guarantees

4. **Temporal GNNs**
   - Handle dynamic graphs
   - Time-aware aggregation
   - Evolving relationships

### Advanced Techniques

1. **Sampling**: For large graphs (GraphSAGE)
2. **Batching**: Mini-batch training on subgraphs
3. **Virtual Nodes**: Connect all nodes
4. **Edge Features**: Include edge attributes
5. **Hierarchical Pooling**: Coarsen graphs

## 🏆 Competition Tips

1. **Feature Engineering**: Create informative node features
2. **Graph Construction**: Design meaningful edges
3. **Normalization**: Critical for stable training
4. **Layer Count**: Usually 2-3 layers sufficient
5. **Regularization**: Dropout on features and adjacency
6. **Ensemble**: Combine multiple GNN variants

## 📖 References

- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"
- Veličković et al. (2018): "Graph Attention Networks"
- Xu et al. (2019): "How Powerful are Graph Neural Networks?"

## 🔗 Related Techniques

- **Knowledge Graph Embeddings**: TransE, DistMult
- **Network Embeddings**: DeepWalk, Node2Vec
- **Spectral Methods**: Laplacian eigenmaps
- **Message Passing Neural Networks**: General framework

## 💡 Key Takeaways

1. ✅ GNNs leverage graph structure
2. ✅ Message passing aggregates neighbor info
3. ✅ Normalization is crucial
4. ✅ Semi-supervised learning effective
5. ✅ 2-3 layers usually optimal
6. ✅ Many variants for different needs
