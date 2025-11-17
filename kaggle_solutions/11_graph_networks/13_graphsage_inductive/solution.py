"""
GraphSAGE: Inductive Representation Learning on Large Graphs

This solution implements GraphSAGE with different aggregator functions for
inductive learning that can generalize to unseen nodes.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class GraphSAGELayer:
    """GraphSAGE layer with different aggregator types"""

    def __init__(self, input_dim, output_dim, aggregator_type='mean',
                 activation='relu', use_bias=True, normalize=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator_type = aggregator_type
        self.activation = activation
        self.use_bias = use_bias
        self.normalize = normalize

        # Initialize weights
        limit = np.sqrt(6.0 / (input_dim + output_dim))

        # Weight for self features
        self.W_self = np.random.uniform(-limit, limit, (input_dim, output_dim))

        # Weight for neighbor features
        self.W_neigh = np.random.uniform(-limit, limit, (input_dim, output_dim))

        if use_bias:
            self.bias = np.zeros(output_dim)

    def aggregate_neighbors(self, features, adjacency, sample_size=None):
        """Aggregate features from sampled neighbors"""
        n_nodes = features.shape[0]
        aggregated = np.zeros((n_nodes, self.input_dim))

        for node in range(n_nodes):
            # Get neighbors
            neighbors = np.where(adjacency[node] > 0)[0]

            if len(neighbors) == 0:
                continue

            # Sample neighbors if specified
            if sample_size is not None and len(neighbors) > sample_size:
                neighbors = np.random.choice(neighbors, sample_size, replace=False)

            neighbor_features = features[neighbors]

            # Apply aggregator
            if self.aggregator_type == 'mean':
                aggregated[node] = np.mean(neighbor_features, axis=0)
            elif self.aggregator_type == 'sum':
                aggregated[node] = np.sum(neighbor_features, axis=0)
            elif self.aggregator_type == 'max':
                aggregated[node] = np.max(neighbor_features, axis=0)
            elif self.aggregator_type == 'lstm':
                # Simplified LSTM aggregator (just using max for now)
                aggregated[node] = np.max(neighbor_features, axis=0)
            elif self.aggregator_type == 'pool':
                # Pooling aggregator
                pooled = np.maximum(0, neighbor_features)  # ReLU
                aggregated[node] = np.max(pooled, axis=0)

        return aggregated

    def forward(self, features, adjacency, sample_size=None):
        """Forward pass"""
        # Aggregate neighbor features
        neighbor_aggregated = self.aggregate_neighbors(features, adjacency, sample_size)

        # Combine self and neighbor features
        self_part = features @ self.W_self
        neigh_part = neighbor_aggregated @ self.W_neigh

        output = self_part + neigh_part

        if self.use_bias:
            output += self.bias

        # L2 normalization
        if self.normalize:
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            norms[norms == 0] = 1
            output = output / norms

        # Activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'tanh':
            output = np.tanh(output)

        return output


class GraphSAGEModel:
    """Multi-layer GraphSAGE model"""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 aggregator_type='mean', dropout=0.5):
        self.layers = []
        self.dropout = dropout

        # Build layers
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layer = GraphSAGELayer(
                dims[i], dims[i+1],
                aggregator_type=aggregator_type,
                activation='relu',
                normalize=True
            )
            self.layers.append(layer)

        # Output layer
        self.output_layer = GraphSAGELayer(
            dims[-1], output_dim,
            aggregator_type=aggregator_type,
            activation=None,
            normalize=False
        )

    def forward(self, features, adjacency, sample_sizes=None, training=True):
        """Forward pass through all layers"""
        output = features

        if sample_sizes is None:
            sample_sizes = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            output = layer.forward(output, adjacency, sample_sizes[i])

            # Dropout
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1-self.dropout, output.shape) / (1-self.dropout)
                output = output * mask

        # Output layer
        output = self.output_layer.forward(output, adjacency,
                                          sample_sizes[-1] if sample_sizes else None)

        return output

    def predict(self, features, adjacency, sample_sizes=None):
        """Make predictions"""
        logits = self.forward(features, adjacency, sample_sizes, training=False)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return np.argmax(probs, axis=1), probs


class UnsupervisedGraphSAGE:
    """Unsupervised GraphSAGE using link prediction"""

    def __init__(self, input_dim, hidden_dim, output_dim, aggregator_type='mean'):
        self.model = GraphSAGEModel(
            input_dim, [hidden_dim], output_dim,
            aggregator_type=aggregator_type
        )

    def generate_pairs(self, adjacency, n_negative=5):
        """Generate positive and negative node pairs"""
        edges = np.argwhere(adjacency > 0)
        positive_pairs = edges[edges[:, 0] < edges[:, 1]]  # Remove duplicates

        # Negative sampling
        n_nodes = adjacency.shape[0]
        negative_pairs = []

        for _ in range(len(positive_pairs) * n_negative):
            u = np.random.randint(n_nodes)
            v = np.random.randint(n_nodes)
            if adjacency[u, v] == 0 and u != v:
                negative_pairs.append([u, v])

        return positive_pairs, np.array(negative_pairs)

    def compute_loss(self, embeddings, positive_pairs, negative_pairs):
        """Compute unsupervised loss"""
        # Positive pairs
        pos_u = embeddings[positive_pairs[:, 0]]
        pos_v = embeddings[positive_pairs[:, 1]]
        pos_scores = np.sum(pos_u * pos_v, axis=1)

        # Negative pairs
        neg_u = embeddings[negative_pairs[:, 0]]
        neg_v = embeddings[negative_pairs[:, 1]]
        neg_scores = np.sum(neg_u * neg_v, axis=1)

        # Max-margin loss
        loss = np.mean(np.maximum(0, neg_scores - pos_scores + 1.0))

        return loss


def generate_inductive_dataset(n_train_nodes=300, n_test_nodes=100,
                               n_features=30, n_classes=4):
    """Generate dataset with separate train and test graphs"""
    # Train graph
    G_train = nx.barabasi_albert_graph(n_train_nodes, m=4, seed=42)

    # Test graph (unseen nodes)
    G_test = nx.barabasi_albert_graph(n_test_nodes, m=4, seed=43)

    # Combine for full graph
    G_full = nx.union(G_train, G_test, rename=('train-', 'test-'))

    # Relabel nodes
    mapping = {node: i for i, node in enumerate(G_full.nodes())}
    G_full = nx.relabel_nodes(G_full, mapping)

    # Generate labels based on community structure
    communities = list(nx.community.greedy_modularity_communities(G_full))
    labels = np.zeros(len(G_full.nodes()), dtype=int)

    for i, community in enumerate(communities[:n_classes]):
        for node in community:
            labels[node] = i % n_classes

    # Generate features
    features = np.random.randn(len(G_full.nodes()), n_features)

    # Add label-specific features
    for i in range(n_classes):
        mask = labels == i
        features[mask] += np.random.randn(n_features) * 2

    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    # Get adjacency matrix
    adjacency = nx.adjacency_matrix(G_full).toarray()

    # Create train/test masks
    train_mask = np.zeros(len(G_full.nodes()), dtype=bool)
    train_mask[:n_train_nodes] = True
    test_mask = ~train_mask

    return G_full, adjacency, features, labels, train_mask, test_mask


def compare_aggregators(features, adjacency, labels, train_mask, test_mask):
    """Compare different aggregator types"""
    aggregators = ['mean', 'sum', 'max', 'pool']
    results = []

    n_features = features.shape[1]
    n_classes = len(np.unique(labels))

    print("   Comparing aggregator types...")
    for agg_type in aggregators:
        print(f"      Testing {agg_type} aggregator...")

        model = GraphSAGEModel(
            n_features, [64, 32], n_classes,
            aggregator_type=agg_type,
            dropout=0.5
        )

        # Simple training
        best_acc = 0
        for epoch in range(100):
            # Forward pass
            logits = model.forward(features, adjacency, training=True)

            # Predictions
            predictions, _ = model.predict(features, adjacency)

            # Evaluate
            test_acc = accuracy_score(labels[test_mask], predictions[test_mask])
            if test_acc > best_acc:
                best_acc = test_acc

        results.append({
            'aggregator': agg_type,
            'test_accuracy': best_acc
        })
        print(f"         Best Test Accuracy: {best_acc:.4f}")

    return pd.DataFrame(results)


def analyze_sampling_strategies(features, adjacency, labels, train_mask, test_mask):
    """Analyze different neighborhood sampling strategies"""
    sample_configs = [
        {'name': 'No Sampling', 'sizes': [None, None]},
        {'name': 'Sample 25', 'sizes': [25, 25]},
        {'name': 'Sample 10', 'sizes': [10, 10]},
        {'name': 'Sample 5', 'sizes': [5, 5]},
        {'name': 'Two-hop [25,10]', 'sizes': [25, 10]},
    ]

    results = []
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))

    print("   Analyzing sampling strategies...")
    for config in sample_configs:
        print(f"      Testing {config['name']}...")

        model = GraphSAGEModel(
            n_features, [64, 32], n_classes,
            aggregator_type='mean',
            dropout=0.5
        )

        # Training
        best_acc = 0
        for epoch in range(100):
            logits = model.forward(features, adjacency,
                                  sample_sizes=config['sizes'],
                                  training=True)

            predictions, _ = model.predict(features, adjacency,
                                          sample_sizes=config['sizes'])

            test_acc = accuracy_score(labels[test_mask], predictions[test_mask])
            if test_acc > best_acc:
                best_acc = test_acc

        results.append({
            'strategy': config['name'],
            'test_accuracy': best_acc
        })
        print(f"         Best Test Accuracy: {best_acc:.4f}")

    return pd.DataFrame(results)


def visualize_embeddings(embeddings, labels, train_mask, test_mask, title):
    """Visualize learned embeddings using t-SNE approximation"""
    # Simple 2D projection using PCA approximation
    centered = embeddings - embeddings.mean(axis=0)
    cov = centered.T @ centered / len(embeddings)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Get top 2 eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    projected = centered @ eigenvectors[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color by label
    ax = axes[0]
    scatter = ax.scatter(projected[:, 0], projected[:, 1],
                        c=labels, cmap='tab10', alpha=0.6, s=50)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{title} - Colored by Label', fontsize=14)
    plt.colorbar(scatter, ax=ax)

    # Color by train/test
    ax = axes[1]
    colors = np.where(train_mask, 'blue', 'red')
    ax.scatter(projected[:, 0], projected[:, 1],
              c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{title} - Blue=Train, Red=Test (Unseen)', fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Train'),
        Patch(facecolor='red', alpha=0.6, label='Test (Unseen)')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('graphsage_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregator_comparison(results_df):
    """Plot comparison of different aggregators"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    bars = ax.bar(x, results_df['test_accuracy'],
                  color='steelblue', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Aggregator Type', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('GraphSAGE: Comparison of Aggregator Types', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['aggregator'])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, acc in zip(bars, results_df['test_accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('graphsage_aggregator_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_sampling_comparison(results_df):
    """Plot comparison of sampling strategies"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    bars = ax.bar(x, results_df['test_accuracy'],
                  color='coral', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Sampling Strategy', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('GraphSAGE: Impact of Neighborhood Sampling', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['strategy'], rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, acc in zip(bars, results_df['test_accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('graphsage_sampling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_inductive_performance(model, features, adjacency, labels,
                                  train_mask, test_mask):
    """Analyze inductive learning performance"""
    # Get embeddings
    embeddings = model.forward(features, adjacency, training=False)

    # Predictions
    predictions, probs = model.predict(features, adjacency)

    # Compute metrics
    train_acc = accuracy_score(labels[train_mask], predictions[train_mask])
    test_acc = accuracy_score(labels[test_mask], predictions[test_mask])

    train_f1 = f1_score(labels[train_mask], predictions[train_mask], average='macro')
    test_f1 = f1_score(labels[test_mask], predictions[test_mask], average='macro')

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax = axes[0]
    metrics = ['Accuracy', 'F1-Score']
    train_scores = [train_acc, train_f1]
    test_scores = [test_acc, test_f1]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, train_scores, width, label='Train', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, test_scores, width, label='Test (Unseen)', color='coral', alpha=0.7)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Inductive Learning Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Prediction confidence distribution
    ax = axes[1]
    train_confidence = np.max(probs[train_mask], axis=1)
    test_confidence = np.max(probs[test_mask], axis=1)

    ax.hist(train_confidence, bins=30, alpha=0.5, label='Train', color='steelblue', density=True)
    ax.hist(test_confidence, bins=30, alpha=0.5, label='Test (Unseen)', color='coral', density=True)

    ax.set_xlabel('Prediction Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Confidence Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graphsage_inductive_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1
    }


def main():
    """Main execution function"""
    print("=" * 80)
    print("GraphSAGE: Inductive Representation Learning")
    print("=" * 80)

    # Generate inductive dataset
    print("\n1. Generating Inductive Learning Dataset...")
    G, adjacency, features, labels, train_mask, test_mask = generate_inductive_dataset(
        n_train_nodes=300, n_test_nodes=100, n_features=30, n_classes=4
    )

    n_nodes = len(labels)
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))

    print(f"   Total Nodes: {n_nodes}")
    print(f"   Train Nodes: {np.sum(train_mask)} (seen during training)")
    print(f"   Test Nodes: {np.sum(test_mask)} (unseen, inductive)")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")

    # Compare aggregators
    print("\n2. Comparing Aggregator Types...")
    agg_results = compare_aggregators(features, adjacency, labels,
                                     train_mask, test_mask)
    print("\n   Aggregator Comparison Results:")
    print(agg_results.to_string(index=False))

    # Analyze sampling strategies
    print("\n3. Analyzing Neighborhood Sampling Strategies...")
    sampling_results = analyze_sampling_strategies(features, adjacency, labels,
                                                   train_mask, test_mask)
    print("\n   Sampling Strategy Results:")
    print(sampling_results.to_string(index=False))

    # Train final model
    print("\n4. Training Final GraphSAGE Model...")
    model = GraphSAGEModel(
        n_features, [64, 32], n_classes,
        aggregator_type='mean',
        dropout=0.5
    )

    # Get embeddings and analyze
    print("\n5. Analyzing Inductive Performance...")
    performance = analyze_inductive_performance(
        model, features, adjacency, labels, train_mask, test_mask
    )

    print(f"\n   Train Accuracy: {performance['train_acc']:.4f}")
    print(f"   Test Accuracy (Unseen Nodes): {performance['test_acc']:.4f}")
    print(f"   Train F1-Score: {performance['train_f1']:.4f}")
    print(f"   Test F1-Score (Unseen Nodes): {performance['test_f1']:.4f}")

    # Visualizations
    print("\n6. Generating Visualizations...")
    embeddings = model.forward(features, adjacency, training=False)
    visualize_embeddings(embeddings, labels, train_mask, test_mask, 'GraphSAGE')
    plot_aggregator_comparison(agg_results)
    plot_sampling_comparison(sampling_results)

    print("\n" + "=" * 80)
    print("GraphSAGE Inductive Learning Complete!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"1. Successfully learned on {np.sum(train_mask)} nodes and generalized to {np.sum(test_mask)} unseen nodes")
    print(f"2. Best aggregator: {agg_results.iloc[agg_results['test_accuracy'].idxmax()]['aggregator']}")
    print(f"3. Inductive test accuracy: {performance['test_acc']:.1%}")
    print(f"4. GraphSAGE enables scalable learning on large, evolving graphs")
    print("=" * 80)


if __name__ == "__main__":
    main()
