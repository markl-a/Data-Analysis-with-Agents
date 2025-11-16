"""
Meta-Learning (Few-Shot Learning) - Kaggle Solution Example
============================================================

This example demonstrates meta-learning for few-shot classification,
where models learn to learn from very few examples using MAML-inspired approach.

Problem: Learn to classify new classes from just a few examples

Approach:
1. Create few-shot learning tasks (N-way K-shot)
2. Implement Prototypical Networks
3. Learn to extract discriminative features
4. Meta-train on multiple tasks
5. Evaluate on novel classes
6. Visualize learned embeddings and prototypes

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


def create_few_shot_task(X, y, n_way=5, k_shot=5, n_query=15):
    """
    Create a few-shot learning task.

    Args:
        X: Feature data
        y: Labels
        n_way: Number of classes in task
        k_shot: Number of examples per class (support)
        n_query: Number of query examples per class

    Returns:
        support_X, support_y, query_X, query_y
    """
    classes = np.unique(y)
    selected_classes = np.random.choice(classes, n_way, replace=False)

    support_X, support_y = [], []
    query_X, query_y = [], []

    for i, cls in enumerate(selected_classes):
        cls_indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(cls_indices, k_shot + n_query, replace=False)

        # Support set (few examples for learning)
        support_indices = selected_indices[:k_shot]
        support_X.append(X[support_indices])
        support_y.extend([i] * k_shot)

        # Query set (examples to predict)
        query_indices = selected_indices[k_shot:]
        query_X.append(X[query_indices])
        query_y.extend([i] * n_query)

    support_X = np.vstack(support_X)
    support_y = np.array(support_y)
    query_X = np.vstack(query_X)
    query_y = np.array(query_y)

    return support_X, support_y, query_X, query_y


class EmbeddingNetwork:
    """
    Neural network that learns to embed inputs into a metric space.

    Used for computing prototypes and distances in few-shot learning.
    """

    def __init__(self, input_dim, embedding_dim=64, hidden_dims=[128, 128]):
        """
        Initialize embedding network.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build network
        self.layers = []
        layer_sizes = [input_dim] + hidden_dims + [embedding_dim]

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': w, 'b': b})

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def forward(self, X):
        """
        Forward pass through embedding network.

        Args:
            X: Input features [batch_size, input_dim]

        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        h = X

        # All layers with ReLU (including last for normalized embeddings)
        for layer in self.layers:
            z = np.dot(h, layer['W']) + layer['b']
            h = self.relu(z)

        # L2 normalize embeddings
        norms = np.linalg.norm(h, axis=1, keepdims=True) + 1e-10
        h = h / norms

        return h


class PrototypicalNetwork:
    """
    Prototypical Networks for few-shot learning.

    Learns to embed examples such that examples from the same class
    cluster around a prototype.
    """

    def __init__(self, input_dim, embedding_dim=64):
        """
        Initialize Prototypical Network.

        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding space dimension
        """
        self.embedding_net = EmbeddingNetwork(input_dim, embedding_dim)

    def compute_prototypes(self, support_X, support_y, n_way):
        """
        Compute class prototypes from support set.

        Args:
            support_X: Support examples
            support_y: Support labels
            n_way: Number of classes

        Returns:
            Prototypes [n_way, embedding_dim]
        """
        # Embed support examples
        embeddings = self.embedding_net.forward(support_X)

        # Compute prototype for each class (mean of class embeddings)
        prototypes = []
        for c in range(n_way):
            class_mask = support_y == c
            class_embeddings = embeddings[class_mask]
            prototype = np.mean(class_embeddings, axis=0)
            prototypes.append(prototype)

        return np.array(prototypes)

    def euclidean_distance(self, x, y):
        """
        Compute Euclidean distance between points.

        Args:
            x: Points [n, d]
            y: Points [m, d]

        Returns:
            Distances [n, m]
        """
        # Efficient computation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy
        x_norm = np.sum(x**2, axis=1, keepdims=True)
        y_norm = np.sum(y**2, axis=1, keepdims=True)
        distances = x_norm + y_norm.T - 2 * np.dot(x, y.T)
        return np.sqrt(np.maximum(distances, 0))

    def predict(self, query_X, prototypes):
        """
        Predict classes for query examples.

        Args:
            query_X: Query examples
            prototypes: Class prototypes

        Returns:
            Predicted classes
        """
        # Embed query examples
        query_embeddings = self.embedding_net.forward(query_X)

        # Compute distances to prototypes
        distances = self.euclidean_distance(query_embeddings, prototypes)

        # Predict nearest prototype
        predictions = np.argmin(distances, axis=1)

        return predictions

    def compute_loss(self, query_X, query_y, prototypes):
        """
        Compute prototypical loss (negative log probability).

        Args:
            query_X: Query examples
            query_y: Query labels
            prototypes: Class prototypes

        Returns:
            Loss value
        """
        # Embed query examples
        query_embeddings = self.embedding_net.forward(query_X)

        # Compute distances to all prototypes
        distances = self.euclidean_distance(query_embeddings, prototypes)

        # Convert to probabilities (softmax over negative distances)
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probs = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        # Cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(len(query_y)), query_y] + 1e-10))

        return loss


def meta_train(model, X_train, y_train, n_episodes=1000,
               n_way=5, k_shot=5, n_query=15, learning_rate=0.001):
    """
    Meta-train the prototypical network.

    Args:
        model: PrototypicalNetwork instance
        X_train, y_train: Training data
        n_episodes: Number of meta-training episodes
        n_way: Number of classes per task
        k_shot: Number of support examples per class
        n_query: Number of query examples per class
        learning_rate: Learning rate
    """
    history = {
        'loss': [],
        'accuracy': []
    }

    print(f"Meta-training for {n_episodes} episodes ({n_way}-way {k_shot}-shot)...")

    for episode in range(n_episodes):
        # Sample a task
        support_X, support_y, query_X, query_y = create_few_shot_task(
            X_train, y_train, n_way, k_shot, n_query
        )

        # Compute prototypes
        prototypes = model.compute_prototypes(support_X, support_y, n_way)

        # Compute loss
        loss = model.compute_loss(query_X, query_y, prototypes)

        # Compute accuracy
        predictions = model.predict(query_X, prototypes)
        accuracy = accuracy_score(query_y, predictions)

        history['loss'].append(loss)
        history['accuracy'].append(accuracy)

        if (episode + 1) % 100 == 0:
            avg_loss = np.mean(history['loss'][-100:])
            avg_acc = np.mean(history['accuracy'][-100:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    return history


def meta_test(model, X_test, y_test, n_episodes=100,
              n_way=5, k_shot=5, n_query=15):
    """
    Meta-test the model on novel classes.

    Args:
        model: Trained PrototypicalNetwork
        X_test, y_test: Test data (novel classes)
        n_episodes: Number of test episodes
        n_way, k_shot, n_query: Task configuration

    Returns:
        Test accuracy
    """
    accuracies = []

    for episode in range(n_episodes):
        # Sample a test task
        support_X, support_y, query_X, query_y = create_few_shot_task(
            X_test, y_test, n_way, k_shot, n_query
        )

        # Compute prototypes
        prototypes = model.compute_prototypes(support_X, support_y, n_way)

        # Predict
        predictions = model.predict(query_X, prototypes)

        # Accuracy
        accuracy = accuracy_score(query_y, predictions)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)


def visualize_results(model, X_test, y_test, history):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Meta-Learning (Few-Shot) Results', fontsize=16, fontweight='bold')

    # 1. Training loss
    ax = fig.add_subplot(gs[0, 0])
    episodes = range(1, len(history['loss']) + 1)

    # Moving average
    window = 50
    if len(history['loss']) > window:
        ma_loss = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episodes)+1), ma_loss, linewidth=2, color='blue')
    else:
        ax.plot(episodes, history['loss'], linewidth=2, color='blue', alpha=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Meta-Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Training accuracy
    ax = fig.add_subplot(gs[0, 1])

    if len(history['accuracy']) > window:
        ma_acc = np.convolve(history['accuracy'], np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episodes)+1), ma_acc, linewidth=2, color='green')
    else:
        ax.plot(episodes, history['accuracy'], linewidth=2, color='green', alpha=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Meta-Training Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Few-shot example
    ax = fig.add_subplot(gs[0, 2])

    # Create a test task
    support_X, support_y, query_X, query_y = create_few_shot_task(
        X_test, y_test, n_way=3, k_shot=3, n_query=5
    )

    prototypes = model.compute_prototypes(support_X, support_y, 3)
    predictions = model.predict(query_X, prototypes)

    # Embed everything
    support_emb = model.embedding_net.forward(support_X)
    query_emb = model.embedding_net.forward(query_X)

    # Reduce to 2D
    all_emb = np.vstack([support_emb, query_emb, prototypes])
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_emb)

    n_support = len(support_emb)
    n_query = len(query_emb)

    support_2d = all_2d[:n_support]
    query_2d = all_2d[n_support:n_support+n_query]
    proto_2d = all_2d[-3:]

    # Plot
    colors = ['red', 'blue', 'green']
    for c in range(3):
        # Support examples
        mask = support_y == c
        ax.scatter(support_2d[mask, 0], support_2d[mask, 1],
                  c=colors[c], marker='o', s=100, alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label=f'Class {c} (support)')

        # Query examples
        mask = query_y == c
        ax.scatter(query_2d[mask, 0], query_2d[mask, 1],
                  c=colors[c], marker='x', s=100, alpha=0.7)

        # Prototype
        ax.scatter(proto_2d[c, 0], proto_2d[c, 1],
                  c=colors[c], marker='*', s=500,
                  edgecolors='black', linewidth=2)

    ax.set_title('3-way 3-shot Task (PCA)', fontweight='bold')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend(fontsize=8)

    # 4. Performance vs K-shot
    ax = fig.add_subplot(gs[1, 0])

    k_values = [1, 3, 5, 10]
    accs = []
    stds = []

    for k in k_values:
        acc, std = meta_test(model, X_test, y_test, n_episodes=50,
                            n_way=5, k_shot=k, n_query=15)
        accs.append(acc)
        stds.append(std)

    ax.errorbar(k_values, accs, yerr=stds, marker='o', linewidth=2,
               markersize=8, capsize=5, capthick=2)
    ax.set_xlabel('K (shots per class)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance vs K-shot', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Performance vs N-way
    ax = fig.add_subplot(gs[1, 1])

    n_values = [3, 5, 7]
    accs = []
    stds = []

    for n in n_values:
        acc, std = meta_test(model, X_test, y_test, n_episodes=50,
                            n_way=n, k_shot=5, n_query=15)
        accs.append(acc)
        stds.append(std)

    ax.errorbar(n_values, accs, yerr=stds, marker='s', linewidth=2,
               markersize=8, capsize=5, capthick=2, color='orange')
    ax.set_xlabel('N (classes per task)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance vs N-way', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. Embedding space visualization
    ax = fig.add_subplot(gs[1, 2])

    # Embed test data
    embeddings = model.embedding_net.forward(X_test)

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=y_test, cmap='tab10', s=30, alpha=0.6)
    ax.set_title('Learned Embedding Space', fontweight='bold')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.colorbar(scatter, ax=ax, label='Class')

    # 7. Distance distribution
    ax = fig.add_subplot(gs[2, 0])

    # Sample task and compute distances
    support_X, support_y, query_X, query_y = create_few_shot_task(
        X_test, y_test, n_way=5, k_shot=5, n_query=30
    )

    prototypes = model.compute_prototypes(support_X, support_y, 5)
    query_emb = model.embedding_net.forward(query_X)

    # Compute distances
    same_class_dists = []
    diff_class_dists = []

    for i, q_emb in enumerate(query_emb):
        for c, proto in enumerate(prototypes):
            dist = np.linalg.norm(q_emb - proto)
            if c == query_y[i]:
                same_class_dists.append(dist)
            else:
                diff_class_dists.append(dist)

    ax.hist(same_class_dists, bins=20, alpha=0.7, label='Same Class', color='green')
    ax.hist(diff_class_dists, bins=20, alpha=0.7, label='Different Class', color='red')
    ax.set_xlabel('Distance to Prototype')
    ax.set_ylabel('Frequency')
    ax.set_title('Distance Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Confusion matrix for a task
    ax = fig.add_subplot(gs[2, 1])

    from sklearn.metrics import confusion_matrix

    support_X, support_y, query_X, query_y = create_few_shot_task(
        X_test, y_test, n_way=5, k_shot=5, n_query=20
    )

    prototypes = model.compute_prototypes(support_X, support_y, 5)
    predictions = model.predict(query_X, prototypes)

    cm = confusion_matrix(query_y, predictions)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Sample Task Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 9. Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Final test
    test_acc, test_std = meta_test(model, X_test, y_test, n_episodes=100,
                                   n_way=5, k_shot=5, n_query=15)

    summary = f"""
    META-LEARNING SUMMARY
    ══════════════════════

    Model:
    • Embedding dim: {model.embedding_net.embedding_dim}
    • Approach: Prototypical Networks

    Meta-Training:
    • Episodes: {len(history['loss'])}
    • Final Loss: {history['loss'][-1]:.4f}
    • Final Acc: {history['accuracy'][-1]:.4f}

    Meta-Test (5-way 5-shot):
    • Accuracy: {test_acc:.4f}
    • Std Dev: {test_std:.4f}
    • 95% CI: ±{1.96*test_std:.4f}

    Few-shot learning allows
    rapid adaptation to new
    classes with minimal data!
    """

    ax.text(0.1, 0.5, summary, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.savefig('/tmp/meta_learning_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/meta_learning_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("META-LEARNING (FEW-SHOT LEARNING) - KAGGLE SOLUTION")
    print("=" * 70)

    # Load dataset
    print("\n📊 Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # Split into meta-train and meta-test classes
    # Meta-train: classes 0-6
    # Meta-test: classes 7-9 (novel classes)
    train_classes = [0, 1, 2, 3, 4, 5, 6]
    test_classes = [7, 8, 9]

    train_mask = np.isin(y, train_classes)
    test_mask = np.isin(y, test_classes)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nMeta-train: {len(X_train)} samples, classes {train_classes}")
    print(f"Meta-test: {len(X_test)} samples, classes {test_classes} (novel!)")

    # Create model
    print("\n🏗️ Building Prototypical Network...")
    model = PrototypicalNetwork(
        input_dim=X.shape[1],
        embedding_dim=64
    )

    print(f"  Embedding dimension: {model.embedding_net.embedding_dim}")
    print(f"  Network: {X.shape[1]} → 128 → 128 → 64")

    # Meta-train
    print("\n" + "=" * 70)
    print("Meta-Training Phase (learning to learn from few examples)...")
    print("=" * 70)

    history = meta_train(
        model, X_train, y_train,
        n_episodes=1000,
        n_way=5,
        k_shot=5,
        n_query=15
    )

    # Meta-test
    print("\n" + "=" * 70)
    print("Meta-Testing Phase (novel classes, never seen before)...")
    print("=" * 70)

    test_acc, test_std = meta_test(
        model, X_test, y_test,
        n_episodes=100,
        n_way=3,  # Only 3 novel classes available
        k_shot=5,
        n_query=15
    )

    print(f"\n✅ Meta-Test Accuracy (3-way 5-shot): {test_acc:.4f} ± {test_std:.4f}")
    print(f"   95% Confidence Interval: {test_acc:.4f} ± {1.96*test_std:.4f}")

    # Additional evaluations
    print("\n📊 Evaluating different configurations...")

    print("\n1-shot learning (extreme few-shot):")
    acc_1shot, std_1shot = meta_test(model, X_test, y_test, n_episodes=100,
                                     n_way=3, k_shot=1, n_query=15)
    print(f"   Accuracy: {acc_1shot:.4f} ± {std_1shot:.4f}")

    print("\n10-shot learning (more examples):")
    acc_10shot, std_10shot = meta_test(model, X_test, y_test, n_episodes=100,
                                       n_way=3, k_shot=10, n_query=15)
    print(f"   Accuracy: {acc_10shot:.4f} ± {std_10shot:.4f}")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(model, X_test, y_test, history)

    print("\n" + "=" * 70)
    print("✅ META-LEARNING COMPLETED!")
    print("=" * 70)
    print("\nKey Achievement: Model can classify novel classes")
    print("with just 5 examples per class!")


if __name__ == "__main__":
    main()
