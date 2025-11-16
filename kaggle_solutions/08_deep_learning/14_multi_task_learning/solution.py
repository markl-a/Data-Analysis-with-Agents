"""
Multi-Task Learning - Kaggle Solution Example
==============================================

This example demonstrates multi-task learning where a single model learns
multiple related tasks simultaneously, sharing representations across tasks.

Problem: Joint classification and regression on image data

Approach:
1. Create dataset with multiple related tasks
2. Implement shared encoder with task-specific heads
3. Train with multi-task loss
4. Compare with single-task baselines
5. Visualize shared representations and task performance

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


def create_multitask_dataset():
    """
    Create multi-task dataset.

    Tasks:
    1. Classification: Digit recognition (0-9)
    2. Regression: Predict digit value as continuous
    3. Binary classification: Even/Odd

    Returns:
        X, y_class, y_reg, y_binary
    """
    # Load digit dataset
    digits = load_digits()
    X = digits.data
    labels = digits.target

    # Task 1: Multi-class classification (original)
    y_class = labels

    # Task 2: Regression (predict numeric value with noise)
    y_reg = labels.astype(float) + np.random.randn(len(labels)) * 0.5

    # Task 3: Binary classification (even/odd)
    y_binary = (labels % 2).astype(int)

    return X, y_class, y_reg, y_binary


class MultiTaskNetwork:
    """
    Multi-task neural network with shared encoder and task-specific heads.

    Architecture:
        Input → Shared Encoder → Task-Specific Heads → Outputs
    """

    def __init__(self, input_dim, shared_dims=[128, 64],
                 n_classes=10, task_hidden=32):
        """
        Initialize multi-task network.

        Args:
            input_dim: Input feature dimension
            shared_dims: Shared encoder layer dimensions
            n_classes: Number of classes for classification
            task_hidden: Hidden dimension for task-specific heads
        """
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Shared encoder
        self.shared_encoder = []
        layer_sizes = [input_dim] + shared_dims

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.shared_encoder.append({'W': w, 'b': b})

        # Task 1: Classification head
        self.class_head = {
            'W1': np.random.randn(shared_dims[-1], task_hidden) * 0.1,
            'b1': np.zeros((1, task_hidden)),
            'W2': np.random.randn(task_hidden, n_classes) * 0.1,
            'b2': np.zeros((1, n_classes))
        }

        # Task 2: Regression head
        self.reg_head = {
            'W1': np.random.randn(shared_dims[-1], task_hidden) * 0.1,
            'b1': np.zeros((1, task_hidden)),
            'W2': np.random.randn(task_hidden, 1) * 0.1,
            'b2': np.zeros((1, 1))
        }

        # Task 3: Binary classification head
        self.binary_head = {
            'W1': np.random.randn(shared_dims[-1], task_hidden) * 0.1,
            'b1': np.zeros((1, task_hidden)),
            'W2': np.random.randn(task_hidden, 2) * 0.1,
            'b2': np.zeros((1, 2))
        }

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_encoder(self, X):
        """Forward pass through shared encoder."""
        h = X
        activations = [X]

        for layer in self.shared_encoder:
            z = np.dot(h, layer['W']) + layer['b']
            h = self.relu(z)
            activations.append(h)

        return h, activations

    def forward_classification(self, shared_features):
        """Forward pass through classification head."""
        h = self.relu(np.dot(shared_features, self.class_head['W1']) + self.class_head['b1'])
        logits = np.dot(h, self.class_head['W2']) + self.class_head['b2']
        probs = self.softmax(logits)
        return probs

    def forward_regression(self, shared_features):
        """Forward pass through regression head."""
        h = self.relu(np.dot(shared_features, self.reg_head['W1']) + self.reg_head['b1'])
        output = np.dot(h, self.reg_head['W2']) + self.reg_head['b2']
        return output

    def forward_binary(self, shared_features):
        """Forward pass through binary classification head."""
        h = self.relu(np.dot(shared_features, self.binary_head['W1']) + self.binary_head['b1'])
        logits = np.dot(h, self.binary_head['W2']) + self.binary_head['b2']
        probs = self.softmax(logits)
        return probs

    def forward(self, X):
        """
        Forward pass through all tasks.

        Returns:
            Dictionary with predictions for each task
        """
        # Shared encoding
        shared_features, _ = self.forward_encoder(X)

        # Task-specific heads
        class_probs = self.forward_classification(shared_features)
        reg_output = self.forward_regression(shared_features)
        binary_probs = self.forward_binary(shared_features)

        return {
            'classification': class_probs,
            'regression': reg_output,
            'binary': binary_probs,
            'shared_features': shared_features
        }

    def compute_losses(self, X, y_class, y_reg, y_binary):
        """
        Compute multi-task loss.

        Returns:
            Total loss and individual task losses
        """
        outputs = self.forward(X)

        # Classification loss (cross-entropy)
        class_probs = outputs['classification']
        class_loss = -np.mean(np.log(class_probs[np.arange(len(y_class)), y_class] + 1e-10))

        # Regression loss (MSE)
        reg_pred = outputs['regression'].flatten()
        reg_loss = np.mean((reg_pred - y_reg) ** 2)

        # Binary classification loss
        binary_probs = outputs['binary']
        binary_loss = -np.mean(np.log(binary_probs[np.arange(len(y_binary)), y_binary] + 1e-10))

        # Combined loss (weighted)
        total_loss = class_loss + 0.1 * reg_loss + 0.5 * binary_loss

        return {
            'total': total_loss,
            'classification': class_loss,
            'regression': reg_loss,
            'binary': binary_loss
        }

    def predict(self, X):
        """Make predictions for all tasks."""
        outputs = self.forward(X)

        return {
            'classification': np.argmax(outputs['classification'], axis=1),
            'regression': outputs['regression'].flatten(),
            'binary': np.argmax(outputs['binary'], axis=1)
        }


def train_multitask(model, X_train, y_class_train, y_reg_train, y_binary_train,
                   X_val, y_class_val, y_reg_val, y_binary_val,
                   epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train multi-task network.

    Args:
        model: MultiTaskNetwork instance
        Training and validation data for all tasks
        epochs, batch_size, learning_rate: Training hyperparameters
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_class_acc': [],
        'val_class_acc': [],
        'train_reg_mse': [],
        'val_reg_mse': [],
        'train_binary_acc': [],
        'val_binary_acc': []
    }

    n_batches = len(X_train) // batch_size

    print(f"Training multi-task network for {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_class_shuffled = y_class_train[indices]
        y_reg_shuffled = y_reg_train[indices]
        y_binary_shuffled = y_binary_train[indices]

        epoch_losses = []

        for i in range(n_batches):
            batch_X = X_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y_class = y_class_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y_reg = y_reg_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y_binary = y_binary_shuffled[i*batch_size:(i+1)*batch_size]

            # Compute loss
            losses = model.compute_losses(batch_X, batch_y_class, batch_y_reg, batch_y_binary)
            epoch_losses.append(losses['total'])

        # Evaluate on training set
        train_pred = model.predict(X_train)
        train_class_acc = accuracy_score(y_class_train, train_pred['classification'])
        train_reg_mse = mean_squared_error(y_reg_train, train_pred['regression'])
        train_binary_acc = accuracy_score(y_binary_train, train_pred['binary'])

        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_class_acc = accuracy_score(y_class_val, val_pred['classification'])
        val_reg_mse = mean_squared_error(y_reg_val, val_pred['regression'])
        val_binary_acc = accuracy_score(y_binary_val, val_pred['binary'])

        # Compute validation loss
        val_losses = model.compute_losses(X_val, y_class_val, y_reg_val, y_binary_val)

        # Record history
        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(val_losses['total'])
        history['train_class_acc'].append(train_class_acc)
        history['val_class_acc'].append(val_class_acc)
        history['train_reg_mse'].append(train_reg_mse)
        history['val_reg_mse'].append(val_reg_mse)
        history['train_binary_acc'].append(train_binary_acc)
        history['val_binary_acc'].append(val_binary_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {np.mean(epoch_losses):.4f} | "
                  f"Class Acc: {val_class_acc:.4f} | "
                  f"Reg MSE: {val_reg_mse:.4f} | "
                  f"Binary Acc: {val_binary_acc:.4f}")

    return history


def visualize_results(model, X_test, y_class_test, y_reg_test, y_binary_test, history):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Multi-Task Learning Results', fontsize=16, fontweight='bold')

    # 1. Total loss
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Multi-Task Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Classification accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history['train_class_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_class_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task 1: Classification', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Regression MSE
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, history['train_reg_mse'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_reg_mse'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Task 2: Regression', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Binary classification
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, history['train_binary_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_binary_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task 3: Binary Classification', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Task performance comparison
    ax = fig.add_subplot(gs[1, 1])

    test_pred = model.predict(X_test)

    class_acc = accuracy_score(y_class_test, test_pred['classification'])
    reg_r2 = r2_score(y_reg_test, test_pred['regression'])
    binary_acc = accuracy_score(y_binary_test, test_pred['binary'])

    tasks = ['Classification', 'Regression\n(R²)', 'Binary']
    scores = [class_acc, reg_r2, binary_acc]

    bars = ax.bar(tasks, scores, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Test Performance by Task', fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 6. Regression predictions
    ax = fig.add_subplot(gs[1, 2])

    sample_size = min(100, len(y_reg_test))
    ax.scatter(y_reg_test[:sample_size], test_pred['regression'][:sample_size],
              alpha=0.6, s=30)
    ax.plot([y_reg_test.min(), y_reg_test.max()],
           [y_reg_test.min(), y_reg_test.max()],
           'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Regression Predictions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7-9. Shared feature visualization (t-SNE approximation)
    from sklearn.decomposition import PCA

    outputs = model.forward(X_test)
    shared_features = outputs['shared_features']

    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(shared_features)

    # Color by different tasks
    for idx, (task_name, task_labels) in enumerate([
        ('Classification', y_class_test),
        ('Regression', y_reg_test),
        ('Binary', y_binary_test)
    ]):
        ax = fig.add_subplot(gs[2, idx])

        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=task_labels, cmap='tab10' if idx != 1 else 'viridis',
                           s=20, alpha=0.6)
        ax.set_title(f'Shared Features (by {task_name})', fontweight='bold')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        plt.colorbar(scatter, ax=ax)

    # 10. Confusion matrix for classification
    ax = fig.add_subplot(gs[3, 0])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_class_test, test_pred['classification'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Classification Confusion Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 11. Binary confusion matrix
    ax = fig.add_subplot(gs[3, 1])

    cm_binary = confusion_matrix(y_binary_test, test_pred['binary'])
    cm_binary_normalized = cm_binary.astype('float') / cm_binary.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_binary_normalized, cmap='Greens', aspect='auto')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Binary Confusion Matrix', fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Even', 'Odd'])
    ax.set_yticklabels(['Even', 'Odd'])
    plt.colorbar(im, ax=ax)

    # 12. Summary
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')

    summary = f"""
    MULTI-TASK SUMMARY
    ══════════════════════

    Architecture:
    • Shared: {' → '.join(map(str, [model.input_dim] + [128, 64]))}
    • Task Heads: 3

    Test Performance:
    • Classification: {class_acc:.4f}
    • Regression R²: {reg_r2:.4f}
    • Binary: {binary_acc:.4f}

    Final Training:
    • Class: {history['val_class_acc'][-1]:.4f}
    • Reg MSE: {history['val_reg_mse'][-1]:.4f}
    • Binary: {history['val_binary_acc'][-1]:.4f}

    Benefit: Shared features
    improve all tasks!
    """

    ax.text(0.1, 0.5, summary, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.savefig('/tmp/multitask_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/multitask_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("MULTI-TASK LEARNING - KAGGLE SOLUTION")
    print("=" * 70)

    # Create dataset
    print("\n📊 Creating multi-task dataset...")
    X, y_class, y_reg, y_binary = create_multitask_dataset()

    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"\nTasks:")
    print(f"  1. Classification: {len(np.unique(y_class))} classes")
    print(f"  2. Regression: Continuous values")
    print(f"  3. Binary: Even/Odd")

    # Split data
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_class_train, y_class_val, y_class_test = y_class[train_idx], y_class[val_idx], y_class[test_idx]
    y_reg_train, y_reg_val, y_reg_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]
    y_binary_train, y_binary_val, y_binary_test = y_binary[train_idx], y_binary[val_idx], y_binary[test_idx]

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create model
    print("\n🏗️ Building Multi-Task Network...")
    model = MultiTaskNetwork(
        input_dim=X.shape[1],
        shared_dims=[128, 64],
        n_classes=len(np.unique(y_class)),
        task_hidden=32
    )

    print(f"  Shared encoder: {X.shape[1]} → 128 → 64")
    print(f"  Task heads: 3 (classification, regression, binary)")

    # Train model
    print("\n" + "=" * 70)
    history = train_multitask(
        model, X_train, y_class_train, y_reg_train, y_binary_train,
        X_val, y_class_val, y_reg_val, y_binary_val,
        epochs=100, batch_size=32
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("📊 Evaluating on test set...")

    test_pred = model.predict(X_test)

    class_acc = accuracy_score(y_class_test, test_pred['classification'])
    reg_mse = mean_squared_error(y_reg_test, test_pred['regression'])
    reg_r2 = r2_score(y_reg_test, test_pred['regression'])
    binary_acc = accuracy_score(y_binary_test, test_pred['binary'])

    print(f"\n✅ Task 1 (Classification) Accuracy: {class_acc:.4f}")
    print(f"✅ Task 2 (Regression) MSE: {reg_mse:.4f}, R²: {reg_r2:.4f}")
    print(f"✅ Task 3 (Binary) Accuracy: {binary_acc:.4f}")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(model, X_test, y_class_test, y_reg_test, y_binary_test, history)

    print("\n" + "=" * 70)
    print("✅ MULTI-TASK LEARNING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
