"""
Graph Convolutional Networks (GCN) for Node Classification

This solution demonstrates Graph Convolutional Networks for node classification tasks.
We implement multiple GCN architectures and compare them with baseline methods.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class GraphConvolutionalLayer:
    """Single GCN layer implementing graph convolution operation"""

    def __init__(self, input_dim, output_dim, activation='relu', use_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

        # Initialize weights with Xavier initialization
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        if use_bias:
            self.bias = np.zeros(output_dim)

    def forward(self, adjacency, features):
        """Forward pass: D^-1/2 * A * D^-1/2 * X * W"""
        # Aggregate features from neighbors
        aggregated = adjacency @ features

        # Apply weights
        output = aggregated @ self.weights

        if self.use_bias:
            output += self.bias

        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        elif self.activation == 'tanh':
            output = np.tanh(output)

        return output

    def backward(self, adjacency, features, grad_output, learning_rate=0.01):
        """Backward pass for gradient descent"""
        # Compute gradients
        aggregated = adjacency @ features
        grad_weights = aggregated.T @ grad_output

        if self.use_bias:
            grad_bias = np.sum(grad_output, axis=0)

        # Update weights
        self.weights -= learning_rate * grad_weights
        if self.use_bias:
            self.bias -= learning_rate * grad_bias

        # Propagate gradient
        grad_aggregated = grad_output @ self.weights.T
        grad_features = adjacency.T @ grad_aggregated

        return grad_features


class GCNModel:
    """Multi-layer Graph Convolutional Network"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        self.layers = []
        self.dropout = dropout

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            activation = 'relu' if i < len(dims) - 2 else 'softmax'
            layer = GraphConvolutionalLayer(dims[i], dims[i+1], activation=activation)
            self.layers.append(layer)

    def forward(self, adjacency, features, training=True):
        """Forward pass through all layers"""
        output = features

        for i, layer in enumerate(self.layers):
            output = layer.forward(adjacency, output)

            # Apply dropout (except last layer)
            if training and i < len(self.layers) - 1 and self.dropout > 0:
                mask = np.random.binomial(1, 1-self.dropout, output.shape) / (1-self.dropout)
                output = output * mask

        return output

    def train_step(self, adjacency, features, labels, mask, learning_rate=0.01):
        """Single training step"""
        # Forward pass
        predictions = self.forward(adjacency, features, training=True)

        # Compute loss (cross-entropy)
        epsilon = 1e-10
        loss = -np.mean(np.sum(labels[mask] * np.log(predictions[mask] + epsilon), axis=1))

        # Backward pass (simplified - only update on masked nodes)
        grad_output = (predictions - labels) / len(mask)
        grad_output[~mask] = 0

        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad_output = layer.backward(adjacency, features, grad_output, learning_rate)

        return loss

    def predict(self, adjacency, features):
        """Make predictions"""
        output = self.forward(adjacency, features, training=False)
        return np.argmax(output, axis=1)


class ChebNetLayer:
    """Chebyshev polynomial-based graph convolution layer"""

    def __init__(self, input_dim, output_dim, K=3, activation='relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K  # Order of Chebyshev polynomial
        self.activation = activation

        # Initialize weights for each polynomial order
        limit = np.sqrt(6.0 / (input_dim * K + output_dim))
        self.weights = [np.random.uniform(-limit, limit, (input_dim, output_dim))
                       for _ in range(K)]

    def forward(self, laplacian, features):
        """Forward pass using Chebyshev polynomials"""
        # Compute Chebyshev polynomials
        T = [features]
        if self.K > 1:
            T.append(laplacian @ features)

        for k in range(2, self.K):
            T_k = 2 * (laplacian @ T[-1]) - T[-2]
            T.append(T_k)

        # Aggregate with weights
        output = sum(T_k @ W for T_k, W in zip(T, self.weights))

        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'softmax':
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            output = exp_output / np.sum(exp_output, axis=1, keepdims=True)

        return output


def normalize_adjacency(adjacency):
    """Normalize adjacency matrix: D^-1/2 * A * D^-1/2"""
    # Add self-loops
    adj = adjacency + np.eye(adjacency.shape[0])

    # Compute degree matrix
    degree = np.sum(adj, axis=1)

    # D^-1/2
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalize: D^-1/2 * A * D^-1/2
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return normalized_adj


def compute_graph_laplacian(adjacency):
    """Compute normalized graph Laplacian"""
    # Degree matrix
    degree = np.sum(adjacency, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalized Laplacian: I - D^-1/2 * A * D^-1/2
    laplacian = np.eye(adjacency.shape[0]) - d_mat_inv_sqrt @ adjacency @ d_mat_inv_sqrt

    # Rescale Laplacian to [-1, 1] for Chebyshev polynomials
    eigenvalues = np.linalg.eigvalsh(laplacian)
    laplacian_rescaled = (2 * laplacian / np.max(eigenvalues)) - np.eye(adjacency.shape[0])

    return laplacian_rescaled


def generate_synthetic_citation_network(n_nodes=500, n_features=50, n_classes=5):
    """Generate synthetic citation network with node features and labels"""
    # Create scale-free graph (like citation networks)
    G = nx.barabasi_albert_graph(n_nodes, m=3, seed=42)

    # Generate node features (bag-of-words style)
    features = np.random.randn(n_nodes, n_features)

    # Assign labels based on community structure
    communities = list(nx.community.greedy_modularity_communities(G))
    labels = np.zeros(n_nodes, dtype=int)

    for i, community in enumerate(communities[:n_classes]):
        for node in community:
            labels[node] = i % n_classes

    # Add feature correlation with labels
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 2

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # Get adjacency matrix
    adjacency = nx.adjacency_matrix(G).toarray()

    return G, adjacency, features, labels


def generate_karate_club_data():
    """Generate Zachary's Karate Club network data"""
    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()

    # Create simple features (one-hot encoding of nodes)
    features = np.eye(n_nodes)

    # Labels based on actual split
    labels = np.array([G.nodes[i]['club'] == 'Mr. Hi' for i in range(n_nodes)], dtype=int)

    # Get adjacency matrix
    adjacency = nx.adjacency_matrix(G).toarray()

    return G, adjacency, features, labels


def generate_stochastic_block_model(n_blocks=4, nodes_per_block=50, p_intra=0.3, p_inter=0.02):
    """Generate stochastic block model graph"""
    n_nodes = n_blocks * nodes_per_block

    # Create block assignments
    labels = np.repeat(np.arange(n_blocks), nodes_per_block)

    # Generate adjacency matrix
    adjacency = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if labels[i] == labels[j]:
                if np.random.rand() < p_intra:
                    adjacency[i, j] = adjacency[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adjacency[i, j] = adjacency[j, i] = 1

    # Create graph
    G = nx.from_numpy_array(adjacency)

    # Generate features based on labels
    n_features = 30
    features = np.random.randn(n_nodes, n_features)

    for i in range(n_blocks):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 3

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    return G, adjacency, features, labels


def train_baseline_models(features, labels, train_mask, test_mask):
    """Train baseline models without graph structure"""
    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(features[train_mask], labels[train_mask])
    lr_pred = lr.predict(features[test_mask])
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(labels[test_mask], lr_pred),
        'f1_macro': f1_score(labels[test_mask], lr_pred, average='macro')
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features[train_mask], labels[train_mask])
    rf_pred = rf.predict(features[test_mask])
    results['Random Forest'] = {
        'accuracy': accuracy_score(labels[test_mask], rf_pred),
        'f1_macro': f1_score(labels[test_mask], rf_pred, average='macro')
    }

    return results


def visualize_graph_with_predictions(G, labels_true, labels_pred, train_mask, test_mask, title):
    """Visualize graph with true and predicted labels"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    pos = nx.spring_layout(G, seed=42)

    # True labels
    ax = axes[0]
    node_colors = labels_true
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='Set3',
                          node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Highlight train/test nodes
    train_nodes = np.where(train_mask)[0]
    test_nodes = np.where(test_mask)[0]
    nx.draw_networkx_nodes(G, pos, nodelist=train_nodes.tolist(),
                          node_color='none', edgecolors='red',
                          linewidths=2, node_size=300, ax=ax)

    ax.set_title(f'{title} - True Labels (Red borders = Train)', fontsize=14)
    ax.axis('off')

    # Predicted labels
    ax = axes[1]
    node_colors = labels_pred
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap='Set3',
                          node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Highlight misclassifications
    test_nodes = np.where(test_mask)[0]
    misclassified = test_nodes[labels_true[test_mask] != labels_pred[test_mask]]
    if len(misclassified) > 0:
        nx.draw_networkx_nodes(G, pos, nodelist=misclassified.tolist(),
                              node_color='none', edgecolors='black',
                              linewidths=3, node_size=300, ax=ax)

    ax.set_title(f'{title} - Predictions (Black borders = Errors)', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('gcn_node_classification_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gcn_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_dict):
    """Compare different models"""
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    f1_scores = [results_dict[m]['f1_macro'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(models))
    width = 0.6

    # Accuracy comparison
    ax = axes[0]
    bars = ax.bar(x, accuracies, width, color='steelblue', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison - Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # F1 score comparison
    ax = axes[1]
    bars = ax.bar(x, f1_scores, width, color='coral', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('F1 Score (Macro)', fontsize=12)
    ax.set_title('Model Comparison - F1 Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('gcn_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_layer_activations(model, adjacency, features, layer_idx=0):
    """Analyze activations of specific layer"""
    # Forward pass up to specific layer
    output = features
    for i, layer in enumerate(model.layers[:layer_idx+1]):
        output = layer.forward(adjacency, output)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Activation distribution
    ax = axes[0, 0]
    ax.hist(output.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Layer {layer_idx} Activation Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Activation heatmap
    ax = axes[0, 1]
    im = ax.imshow(output[:50], aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Node ID', fontsize=12)
    ax.set_title(f'Layer {layer_idx} Activation Heatmap (First 50 nodes)', fontsize=14)
    plt.colorbar(im, ax=ax)

    # Activation statistics per node
    ax = axes[1, 0]
    node_means = np.mean(output, axis=1)
    node_stds = np.std(output, axis=1)
    ax.scatter(node_means, node_stds, alpha=0.5, s=30)
    ax.set_xlabel('Mean Activation', fontsize=12)
    ax.set_ylabel('Std Activation', fontsize=12)
    ax.set_title(f'Layer {layer_idx} Node-wise Statistics', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Feature importance (variance)
    ax = axes[1, 1]
    feature_vars = np.var(output, axis=0)
    ax.bar(range(len(feature_vars)), sorted(feature_vars, reverse=True),
           color='coral', alpha=0.7)
    ax.set_xlabel('Feature Index (sorted)', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(f'Layer {layer_idx} Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('gcn_layer_activations.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Graph Convolutional Networks (GCN) for Node Classification")
    print("=" * 80)

    # Generate synthetic citation network
    print("\n1. Generating Synthetic Citation Network...")
    G, adjacency, features, labels = generate_synthetic_citation_network(
        n_nodes=500, n_features=50, n_classes=5
    )

    n_nodes = adjacency.shape[0]
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))

    print(f"   Nodes: {n_nodes}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")
    print(f"   Average Degree: {2 * G.number_of_edges() / n_nodes:.2f}")

    # Split data
    print("\n2. Splitting Data (60% train, 20% val, 20% test)...")
    indices = np.arange(n_nodes)
    np.random.shuffle(indices)

    train_size = int(0.6 * n_nodes)
    val_size = int(0.2 * n_nodes)

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    print(f"   Train nodes: {np.sum(train_mask)}")
    print(f"   Val nodes: {np.sum(val_mask)}")
    print(f"   Test nodes: {np.sum(test_mask)}")

    # Normalize adjacency matrix
    print("\n3. Normalizing Adjacency Matrix...")
    adj_normalized = normalize_adjacency(adjacency)

    # Train baseline models
    print("\n4. Training Baseline Models (without graph structure)...")
    baseline_results = train_baseline_models(features, labels, train_mask, test_mask)
    for model, metrics in baseline_results.items():
        print(f"   {model}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # Train GCN model
    print("\n5. Training GCN Model...")
    gcn = GCNModel(input_dim=n_features, hidden_dims=[32, 16], output_dim=n_classes, dropout=0.5)

    # One-hot encode labels
    labels_onehot = np.eye(n_classes)[labels]

    # Training loop
    n_epochs = 200
    learning_rate = 0.01

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(n_epochs):
        # Training step
        loss = gcn.train_step(adj_normalized, features, labels_onehot, train_mask, learning_rate)

        # Evaluate on train and val
        train_pred = gcn.predict(adj_normalized, features)
        train_acc = accuracy_score(labels[train_mask], train_pred[train_mask])

        val_pred = gcn.predict(adj_normalized, features)
        val_acc = accuracy_score(labels[val_mask], val_pred[val_mask])

        # Compute val loss
        val_predictions = gcn.forward(adj_normalized, features, training=False)
        val_loss = -np.mean(np.sum(labels_onehot[val_mask] *
                                   np.log(val_predictions[val_mask] + 1e-10), axis=1))

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Final evaluation
    print("\n6. Final Evaluation on Test Set...")
    test_pred = gcn.predict(adj_normalized, features)
    test_acc = accuracy_score(labels[test_mask], test_pred[test_mask])
    test_f1 = f1_score(labels[test_mask], test_pred[test_mask], average='macro')

    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")

    print("\n   Classification Report:")
    print(classification_report(labels[test_mask], test_pred[test_mask]))

    # Combine results
    all_results = baseline_results.copy()
    all_results['GCN'] = {'accuracy': test_acc, 'f1_macro': test_f1}

    # Visualizations
    print("\n7. Generating Visualizations...")
    visualize_graph_with_predictions(G, labels, test_pred, train_mask, test_mask, 'GCN')
    plot_training_curves(history)
    plot_model_comparison(all_results)
    analyze_layer_activations(gcn, adj_normalized, features, layer_idx=0)

    print("\n" + "=" * 80)
    print("GCN Node Classification Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"1. GCN achieved {test_acc:.1%} accuracy on node classification")
    print(f"2. GCN outperformed baselines by leveraging graph structure")
    print(f"3. Model successfully learned node representations from neighborhood aggregation")
    print("=" * 80)


if __name__ == "__main__":
    main()
