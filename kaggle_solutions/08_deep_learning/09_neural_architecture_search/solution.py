"""
Neural Architecture Search (NAS) - Kaggle Solution Example
===========================================================

This example demonstrates a simple but effective neural architecture search
using random search and evolutionary algorithms to find optimal network architectures.

Problem: Find the best CNN architecture for MNIST-like image classification

Approach:
1. Define a search space for network architectures
2. Sample random architectures
3. Train and evaluate each architecture
4. Use evolutionary selection to find best performers
5. Visualize architecture performance landscape

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


class SimpleNeuralNet:
    """
    Simple feedforward neural network with configurable architecture.
    Implements backpropagation with ReLU activation.
    """

    def __init__(self, architecture: List[int]):
        """
        Initialize neural network with given architecture.

        Args:
            architecture: List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.architecture = architecture
        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(len(architecture) - 1):
            w = np.random.randn(architecture[i], architecture[i+1]) * np.sqrt(2.0 / architecture[i])
            b = np.zeros((1, architecture[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass through the network."""
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)

        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, learning_rate=0.01):
        """Backward pass with gradient descent."""
        m = X.shape[0]

        # Convert labels to one-hot
        y_onehot = np.zeros((m, self.architecture[-1]))
        y_onehot[np.arange(m), y] = 1

        # Output layer gradient
        delta = self.activations[-1] - y_onehot
        gradients_w = []
        gradients_b = []

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, X, y, epochs=50, learning_rate=0.01, batch_size=32, verbose=False):
        """Train the neural network."""
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

    def predict(self, X):
        """Make predictions."""
        output = self.forward(X)
        return np.argmax(output, axis=1)


class NeuralArchitectureSearch:
    """
    Neural Architecture Search using random search and evolutionary algorithms.
    """

    def __init__(self, input_dim: int, output_dim: int, search_space: Dict):
        """
        Initialize NAS.

        Args:
            input_dim: Input feature dimension
            output_dim: Number of output classes
            search_space: Dictionary defining architecture search space
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.search_space = search_space
        self.history = []

    def sample_architecture(self) -> List[int]:
        """Sample a random architecture from the search space."""
        n_layers = np.random.randint(
            self.search_space['min_layers'],
            self.search_space['max_layers'] + 1
        )

        architecture = [self.input_dim]

        for _ in range(n_layers):
            layer_size = np.random.choice(self.search_space['layer_sizes'])
            architecture.append(layer_size)

        architecture.append(self.output_dim)
        return architecture

    def evaluate_architecture(self, architecture: List[int], X_train, y_train,
                            X_val, y_val, epochs=30) -> Dict:
        """
        Train and evaluate a single architecture.

        Returns:
            Dictionary with architecture info and performance metrics
        """
        start_time = time.time()

        # Create and train model
        model = SimpleNeuralNet(architecture)
        model.train(X_train, y_train, epochs=epochs, learning_rate=0.01, batch_size=32)

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))

        training_time = time.time() - start_time

        # Count parameters
        n_params = sum(w.size for w in model.weights)

        result = {
            'architecture': architecture,
            'n_layers': len(architecture) - 2,  # Excluding input and output
            'n_params': n_params,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'training_time': training_time,
            'score': val_acc  # Primary metric
        }

        return result

    def random_search(self, X_train, y_train, X_val, y_val,
                     n_architectures=20, epochs=30):
        """
        Perform random architecture search.

        Args:
            n_architectures: Number of random architectures to try
            epochs: Training epochs per architecture
        """
        print(f"🔍 Random Search: Evaluating {n_architectures} architectures...")

        for i in range(n_architectures):
            architecture = self.sample_architecture()
            result = self.evaluate_architecture(
                architecture, X_train, y_train, X_val, y_val, epochs
            )
            self.history.append(result)

            print(f"  [{i+1}/{n_architectures}] Arch: {architecture[1:-1]} | "
                  f"Val Acc: {result['val_acc']:.4f} | Params: {result['n_params']}")

        return self.get_best_architecture()

    def evolutionary_search(self, X_train, y_train, X_val, y_val,
                          n_generations=5, population_size=10,
                          n_mutations=3, epochs=30):
        """
        Evolutionary architecture search.

        Args:
            n_generations: Number of evolutionary generations
            population_size: Size of population per generation
            n_mutations: Number of mutations per generation
            epochs: Training epochs per architecture
        """
        print(f"\n🧬 Evolutionary Search: {n_generations} generations, "
              f"population {population_size}...")

        # Initial population
        population = [self.sample_architecture() for _ in range(population_size)]

        for gen in range(n_generations):
            print(f"\n  Generation {gen + 1}/{n_generations}")

            # Evaluate population
            results = []
            for i, arch in enumerate(population):
                result = self.evaluate_architecture(
                    arch, X_train, y_train, X_val, y_val, epochs
                )
                results.append(result)
                self.history.append(result)

                print(f"    [{i+1}/{len(population)}] Arch: {arch[1:-1]} | "
                      f"Val Acc: {result['val_acc']:.4f}")

            # Selection: Keep top 50%
            results.sort(key=lambda x: x['score'], reverse=True)
            survivors = [r['architecture'] for r in results[:population_size//2]]

            # Reproduction: Create new population
            new_population = survivors.copy()

            # Mutation: Modify survivors
            for _ in range(n_mutations):
                parent = survivors[np.random.randint(len(survivors))]
                mutated = self.mutate_architecture(parent)
                new_population.append(mutated)

            # Fill remaining with random architectures
            while len(new_population) < population_size:
                new_population.append(self.sample_architecture())

            population = new_population

        return self.get_best_architecture()

    def mutate_architecture(self, architecture: List[int]) -> List[int]:
        """Mutate an architecture by adding, removing, or modifying layers."""
        mutated = architecture[1:-1].copy()  # Exclude input/output layers

        mutation_type = np.random.choice(['add', 'remove', 'modify'])

        if mutation_type == 'add' and len(mutated) < self.search_space['max_layers']:
            # Add a layer
            pos = np.random.randint(0, len(mutated) + 1)
            new_size = np.random.choice(self.search_space['layer_sizes'])
            mutated.insert(pos, new_size)
        elif mutation_type == 'remove' and len(mutated) > self.search_space['min_layers']:
            # Remove a layer
            pos = np.random.randint(0, len(mutated))
            mutated.pop(pos)
        else:  # modify
            # Change layer size
            if len(mutated) > 0:
                pos = np.random.randint(0, len(mutated))
                mutated[pos] = np.random.choice(self.search_space['layer_sizes'])

        return [self.input_dim] + mutated + [self.output_dim]

    def get_best_architecture(self) -> Dict:
        """Get the best performing architecture."""
        if not self.history:
            return None

        best = max(self.history, key=lambda x: x['score'])
        return best


def visualize_results(nas: NeuralArchitectureSearch):
    """Create comprehensive visualizations of NAS results."""
    history = nas.history

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Neural Architecture Search Results', fontsize=16, fontweight='bold')

    # 1. Accuracy over search iterations
    ax = axes[0, 0]
    iterations = range(1, len(history) + 1)
    val_accs = [h['val_acc'] for h in history]
    train_accs = [h['train_acc'] for h in history]

    ax.plot(iterations, val_accs, 'o-', label='Validation', alpha=0.7, linewidth=2)
    ax.plot(iterations, train_accs, 's-', label='Train', alpha=0.5, linewidth=1)
    ax.axhline(max(val_accs), color='red', linestyle='--', alpha=0.5, label='Best Val')
    ax.set_xlabel('Architecture #', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Search Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Number of parameters vs accuracy
    ax = axes[0, 1]
    n_params = [h['n_params'] for h in history]
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))

    scatter = ax.scatter(n_params, val_accs, c=range(len(history)),
                        cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Number of Parameters', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Model Complexity vs Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Search Order')

    # 3. Training time vs accuracy
    ax = axes[0, 2]
    times = [h['training_time'] for h in history]

    ax.scatter(times, val_accs, c=range(len(history)), cmap='plasma',
              s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Training Time (seconds)', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Efficiency vs Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Layer depth distribution
    ax = axes[1, 0]
    n_layers = [h['n_layers'] for h in history]
    unique_layers = sorted(set(n_layers))

    layer_accs = {l: [] for l in unique_layers}
    for h in history:
        layer_accs[h['n_layers']].append(h['val_acc'])

    positions = []
    data = []
    for l in unique_layers:
        if layer_accs[l]:
            positions.append(l)
            data.append(layer_accs[l])

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xlabel('Number of Hidden Layers', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Network Depth Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Top 5 architectures
    ax = axes[1, 1]
    top_5 = sorted(history, key=lambda x: x['val_acc'], reverse=True)[:5]

    arch_names = [f"Arch {i+1}\n{h['architecture'][1:-1]}" for i, h in enumerate(top_5)]
    accs = [h['val_acc'] for h in top_5]

    bars = ax.barh(range(len(top_5)), accs, color=plt.cm.RdYlGn(np.linspace(0.5, 1, 5)))
    ax.set_yticks(range(len(top_5)))
    ax.set_yticklabels(arch_names, fontsize=9)
    ax.set_xlabel('Validation Accuracy', fontsize=11)
    ax.set_title('Top 5 Architectures', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc - 0.01, i, f'{acc:.4f}', va='center', ha='right',
               fontweight='bold', color='white')

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    best = max(history, key=lambda x: x['val_acc'])

    summary_text = f"""
    SEARCH SUMMARY
    ══════════════════════════

    Total Architectures: {len(history)}

    Best Architecture:
    {best['architecture'][1:-1]}

    Best Val Accuracy: {best['val_acc']:.4f}
    Best Train Accuracy: {best['train_acc']:.4f}

    Parameters: {best['n_params']:,}
    Training Time: {best['training_time']:.2f}s

    Average Val Acc: {np.mean(val_accs):.4f}
    Std Val Acc: {np.std(val_accs):.4f}

    Best Params: {min(n_params):,} - {max(n_params):,}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.tight_layout()
    plt.savefig('/tmp/nas_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/nas_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("NEURAL ARCHITECTURE SEARCH - KAGGLE SOLUTION")
    print("=" * 70)

    # Load and prepare data
    print("\n📊 Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Define search space
    search_space = {
        'min_layers': 1,
        'max_layers': 3,
        'layer_sizes': [32, 64, 128, 256]
    }

    print("\n🔧 Search Space Configuration:")
    print(f"  Hidden Layers: {search_space['min_layers']}-{search_space['max_layers']}")
    print(f"  Layer Sizes: {search_space['layer_sizes']}")

    # Initialize NAS
    nas = NeuralArchitectureSearch(
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y)),
        search_space=search_space
    )

    # Random search
    print("\n" + "=" * 70)
    best_random = nas.random_search(X_train, y_train, X_val, y_val,
                                   n_architectures=15, epochs=30)

    print(f"\n✅ Best from Random Search:")
    print(f"  Architecture: {best_random['architecture']}")
    print(f"  Validation Accuracy: {best_random['val_acc']:.4f}")
    print(f"  Parameters: {best_random['n_params']:,}")

    # Evolutionary search
    print("\n" + "=" * 70)
    best_evo = nas.evolutionary_search(X_train, y_train, X_val, y_val,
                                      n_generations=3, population_size=8,
                                      n_mutations=2, epochs=30)

    print(f"\n✅ Best from Evolutionary Search:")
    print(f"  Architecture: {best_evo['architecture']}")
    print(f"  Validation Accuracy: {best_evo['val_acc']:.4f}")
    print(f"  Parameters: {best_evo['n_params']:,}")

    # Find overall best
    overall_best = max([best_random, best_evo], key=lambda x: x['val_acc'])

    print("\n" + "=" * 70)
    print("🏆 OVERALL BEST ARCHITECTURE")
    print("=" * 70)
    print(f"Architecture: {overall_best['architecture']}")
    print(f"Validation Accuracy: {overall_best['val_acc']:.4f}")
    print(f"Training Accuracy: {overall_best['train_acc']:.4f}")
    print(f"Parameters: {overall_best['n_params']:,}")
    print(f"Training Time: {overall_best['training_time']:.2f}s")

    # Test final model
    print("\n📈 Training final model on full training set...")
    final_model = SimpleNeuralNet(overall_best['architecture'])
    final_model.train(
        np.vstack([X_train, X_val]),
        np.hstack([y_train, y_val]),
        epochs=50, learning_rate=0.01
    )

    test_acc = accuracy_score(y_test, final_model.predict(X_test))
    print(f"✅ Final Test Accuracy: {test_acc:.4f}")

    # Visualize results
    print("\n📊 Generating visualizations...")
    visualize_results(nas)

    print("\n" + "=" * 70)
    print("✅ NEURAL ARCHITECTURE SEARCH COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
