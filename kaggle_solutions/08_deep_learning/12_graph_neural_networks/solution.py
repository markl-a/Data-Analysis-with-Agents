"""
Graph Neural Networks (GNN) - Kaggle Solution Example
======================================================

This example demonstrates Graph Neural Networks for learning on graph-structured data.
GNNs aggregate information from node neighborhoods to learn node representations.

Problem: Node classification in citation network / social network

Approach:
1. Create synthetic graph dataset
2. Implement Graph Convolutional Network (GCN)
3. Message passing and aggregation
4. Train on semi-supervised node classification
5. Visualize learned embeddings and graph structure

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


def create_graph_dataset(n_nodes=200, n_features=16, n_classes=4):
    """
    Create synthetic graph dataset (social network / citation network).

    Args:
        n_nodes: Number of nodes in graph
        n_features: Feature dimension per node
        n_classes: Number of node classes

    Returns:
        features, adjacency matrix, labels
    """
    # Assign nodes to classes
    labels = np.random.randint(0, n_classes, n_nodes)

    # Generate features based on class (with some noise)
    features = np.random.randn(n_nodes, n_features) * 0.5

    for i in range(n_nodes):
        class_signal = np.random.randn(n_features)
        features[i] += class_signal * labels[i]

    # Create adjacency matrix (preferential attachment to same class)
    adjacency = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        # Connect to nodes with same class more often
        n_connections = np.random.randint(3, 8)

        for _ in range(n_connections):
            # 70% chance to connect to same class
            if np.random.random() < 0.7:
                same_class = np.where(labels == labels[i])[0]
                j = np.random.choice(same_class)
            else:
                j = np.random.randint(0, n_nodes)

            if i != j:
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # Undirected graph

    return features, adjacency, labels


def normalize_adjacency(adjacency):
    """
    Normalize adjacency matrix with self-loops.

    A_norm = D^(-1/2) * (A + I) * D^(-1/2)

    Args:
        adjacency: Adjacency matrix

    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops
    A = adjacency + np.eye(adjacency.shape[0])

    # Degree matrix
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))

    # Symmetric normalization
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


class GraphConvolutionalLayer:
    """
    Single Graph Convolutional Layer.

    Performs: H' = σ(A_norm * H * W)
    """

    def __init__(self, in_features, out_features):
        """
        Initialize GCN layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        self.in_features = in_features
        self.out_features = out_features

        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

    def forward(self, X, A_norm):
        """
        Forward pass.

        Args:
            X: Node features [n_nodes, in_features]
            A_norm: Normalized adjacency matrix [n_nodes, n_nodes]

        Returns:
            Updated node features [n_nodes, out_features]
        """
        # Aggregate from neighbors and transform
        self.input = X
        self.A_norm = A_norm

        # Message passing: aggregate neighbor features
        aggregated = A_norm @ X

        # Transform
        self.output = aggregated @ self.W + self.b

        return self.output

    def backward(self, grad_output, learning_rate=0.01):
        """
        Backward pass.

        Args:
            grad_output: Gradient from next layer
            learning_rate: Learning rate

        Returns:
            Gradient for previous layer
        """
        # Gradient w.r.t. weights
        aggregated = self.A_norm @ self.input
        grad_W = aggregated.T @ grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True)

        # Update weights
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        # Gradient w.r.t. input
        grad_input = grad_output @ self.W.T
        grad_input = self.A_norm @ grad_input

        return grad_input


class GraphNeuralNetwork:
    """
    Graph Neural Network with multiple GCN layers.
    """

    def __init__(self, n_features, hidden_dims, n_classes):
        """
        Initialize GNN.

        Args:
            n_features: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            n_classes: Number of output classes
        """
        self.layers = []

        # Build layers
        layer_sizes = [n_features] + hidden_dims + [n_classes]

        for i in range(len(layer_sizes) - 1):
            layer = GraphConvolutionalLayer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, A_norm):
        """
        Forward pass through GNN.

        Args:
            X: Node features [n_nodes, n_features]
            A_norm: Normalized adjacency matrix

        Returns:
            Class probabilities [n_nodes, n_classes]
        """
        self.activations = [X]
        self.z_values = []

        h = X

        # Hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward(h, A_norm)
            self.z_values.append(z)
            h = self.relu(z)
            self.activations.append(h)

        # Output layer with softmax
        z = self.layers[-1].forward(h, A_norm)
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)

        return output

    def cross_entropy_loss(self, y_true, y_pred, mask):
        """
        Cross-entropy loss for masked nodes.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            mask: Boolean mask for which nodes to include

        Returns:
            Loss value
        """
        n = np.sum(mask)
        if n == 0:
            return 0

        # One-hot encode
        y_onehot = np.zeros_like(y_pred)
        y_onehot[np.arange(len(y_true)), y_true] = 1

        # Cross-entropy
        loss = -np.sum(y_onehot[mask] * np.log(y_pred[mask] + 1e-10)) / n

        return loss

    def backward(self, X, A_norm, y_true, mask, learning_rate=0.01):
        """
        Backward pass with masked loss.

        Args:
            X: Input features
            A_norm: Normalized adjacency
            y_true: True labels
            mask: Training mask
            learning_rate: Learning rate
        """
        n_nodes = X.shape[0]
        n_classes = self.activations[-1].shape[1]

        # One-hot encode
        y_onehot = np.zeros((n_nodes, n_classes))
        y_onehot[np.arange(n_nodes), y_true] = 1

        # Gradient of loss w.r.t. output
        delta = self.activations[-1] - y_onehot

        # Apply mask
        delta[~mask] = 0
        delta = delta / (np.sum(mask) + 1e-10)

        # Backpropagate through output layer
        delta = self.layers[-1].backward(delta, learning_rate)

        # Backpropagate through hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            delta = delta * self.relu_derivative(self.z_values[i])
            delta = self.layers[i].backward(delta, learning_rate)

    def train_step(self, X, A_norm, y, train_mask, learning_rate=0.01):
        """Single training step."""
        # Forward
        y_pred = self.forward(X, A_norm)

        # Loss
        loss = self.cross_entropy_loss(y, y_pred, train_mask)

        # Backward
        self.backward(X, A_norm, y, train_mask, learning_rate)

        return loss

    def predict(self, X, A_norm):
        """Make predictions."""
        y_pred = self.forward(X, A_norm)
        return np.argmax(y_pred, axis=1)


def train_gnn(model, features, adjacency, labels, train_mask, val_mask,
              epochs=200, learning_rate=0.01):
    """
    Train Graph Neural Network.

    Args:
        model: GNN model
        features: Node features
        adjacency: Adjacency matrix
        labels: Node labels
        train_mask: Training nodes mask
        val_mask: Validation nodes mask
        epochs: Number of epochs
        learning_rate: Learning rate
    """
    # Normalize adjacency
    A_norm = normalize_adjacency(adjacency)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print(f"Training GNN for {epochs} epochs...")

    for epoch in range(epochs):
        # Training step
        loss = model.train_step(features, A_norm, labels, train_mask, learning_rate)

        # Evaluate
        train_pred = model.predict(features, A_norm)
        train_acc = accuracy_score(labels[train_mask], train_pred[train_mask])

        val_pred = model.predict(features, A_norm)
        val_acc = accuracy_score(labels[val_mask], val_pred[val_mask])

        # Validation loss
        y_pred_val = model.forward(features, A_norm)
        val_loss = model.cross_entropy_loss(labels, y_pred_val, val_mask)

        history['train_loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history, A_norm


def visualize_results(model, features, adjacency, labels, train_mask, val_mask, test_mask, history):
    """Create comprehensive visualizations."""
    A_norm = normalize_adjacency(adjacency)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Graph Neural Network Results', fontsize=16, fontweight='bold')

    # 1. Training curves
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy curves
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Node embeddings (t-SNE)
    ax = fig.add_subplot(gs[0, 2])

    # Get embeddings from second-to-last layer
    model.forward(features, A_norm)
    embeddings = model.activations[-2]  # Before final layer

    # t-SNE
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=labels, cmap='tab10', s=50, alpha=0.7)
    ax.set_title('Node Embeddings (t-SNE)', fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Class')

    # 4. Graph structure visualization
    ax = fig.add_subplot(gs[1, 0])

    # Compute layout (simple circular)
    n_nodes = features.shape[0]
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)

    # Draw edges (sample for visibility)
    edge_sample_rate = 0.1
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency[i, j] > 0 and np.random.random() < edge_sample_rate:
                ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                       'gray', alpha=0.2, linewidth=0.5)

    # Draw nodes
    ax.scatter(pos_x, pos_y, c=labels, cmap='tab10', s=30, alpha=0.8)
    ax.set_title('Graph Structure (Sample)', fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # 5. Degree distribution
    ax = fig.add_subplot(gs[1, 1])
    degrees = np.sum(adjacency, axis=1)
    ax.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Class distribution
    ax = fig.add_subplot(gs[1, 2])
    unique_labels, counts = np.unique(labels, return_counts=True)
    ax.bar(unique_labels, counts, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 7. Predictions visualization
    ax = fig.add_subplot(gs[2, 0])
    predictions = model.predict(features, A_norm)

    # Show first 50 nodes
    n_show = min(50, len(labels))
    x = np.arange(n_show)

    ax.scatter(x[train_mask[:n_show]], labels[:n_show][train_mask[:n_show]],
              label='Train (True)', marker='o', s=50, alpha=0.7)
    ax.scatter(x[val_mask[:n_show]], labels[:n_show][val_mask[:n_show]],
              label='Val (True)', marker='s', s=50, alpha=0.7)
    ax.scatter(x, predictions[:n_show], label='Predicted',
              marker='x', s=50, alpha=0.7, color='red')

    ax.set_xlabel('Node Index')
    ax.set_ylabel('Class')
    ax.set_title('Predictions vs True Labels (First 50 Nodes)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Per-class performance
    ax = fig.add_subplot(gs[2, 1])

    test_pred = model.predict(features, A_norm)
    per_class_acc = []

    for c in unique_labels:
        mask = (labels == c) & test_mask
        if np.sum(mask) > 0:
            acc = accuracy_score(labels[mask], test_pred[mask])
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)

    ax.bar(unique_labels, per_class_acc, color='lightgreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Per-Class Performance', fontweight='bold')
    ax.axhline(np.mean(per_class_acc), color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 9. Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    test_acc = accuracy_score(labels[test_mask], test_pred[test_mask])
    test_f1 = f1_score(labels[test_mask], test_pred[test_mask], average='macro')

    summary = f"""
    TRAINING SUMMARY
    ══════════════════════

    Dataset:
    • Nodes: {len(labels)}
    • Edges: {int(np.sum(adjacency)/2)}
    • Classes: {len(unique_labels)}

    Split:
    • Train: {np.sum(train_mask)}
    • Val: {np.sum(val_mask)}
    • Test: {np.sum(test_mask)}

    Performance:
    • Test Accuracy: {test_acc:.4f}
    • Test F1: {test_f1:.4f}

    Final:
    • Train Acc: {history['train_acc'][-1]:.4f}
    • Val Acc: {history['val_acc'][-1]:.4f}
    """

    ax.text(0.1, 0.5, summary, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.savefig('/tmp/gnn_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/gnn_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("GRAPH NEURAL NETWORKS - KAGGLE SOLUTION")
    print("=" * 70)

    # Create dataset
    print("\n📊 Creating synthetic graph dataset...")
    features, adjacency, labels = create_graph_dataset(
        n_nodes=200, n_features=16, n_classes=4
    )

    print(f"Nodes: {features.shape[0]}")
    print(f"Features per node: {features.shape[1]}")
    print(f"Edges: {int(np.sum(adjacency)/2)}")
    print(f"Classes: {len(np.unique(labels))}")

    # Create train/val/test split
    n_nodes = features.shape[0]
    indices = np.random.permutation(n_nodes)

    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train+n_val]] = True
    test_mask[indices[n_train+n_val:]] = True

    print(f"\nSplit: Train={np.sum(train_mask)}, Val={np.sum(val_mask)}, Test={np.sum(test_mask)}")

    # Create model
    print("\n🏗️ Building Graph Neural Network...")
    model = GraphNeuralNetwork(
        n_features=features.shape[1],
        hidden_dims=[32, 16],
        n_classes=len(np.unique(labels))
    )

    print(f"Architecture: {features.shape[1]} → 32 → 16 → {len(np.unique(labels))}")

    # Train model
    print("\n" + "=" * 70)
    history, A_norm = train_gnn(
        model, features, adjacency, labels,
        train_mask, val_mask,
        epochs=200, learning_rate=0.01
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("📊 Evaluating on test set...")

    test_pred = model.predict(features, A_norm)
    test_acc = accuracy_score(labels[test_mask], test_pred[test_mask])
    test_f1 = f1_score(labels[test_mask], test_pred[test_mask], average='macro')

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(model, features, adjacency, labels,
                     train_mask, val_mask, test_mask, history)

    print("\n" + "=" * 70)
    print("✅ GRAPH NEURAL NETWORK TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
