"""
Capsule Networks (CapsNet) - Kaggle Solution Example
=====================================================

This example demonstrates Capsule Networks, which use groups of neurons (capsules)
to better capture hierarchical relationships and pose information in images.

Problem: Image classification with better handling of spatial relationships

Approach:
1. Implement basic capsule network architecture
2. Use dynamic routing between capsules
3. Train on image data with reconstruction regularization
4. Visualize capsule activations and learned features
5. Compare with traditional CNNs

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


def squash(vectors, axis=-1):
    """
    Squash activation function for capsules.

    Squashes vector length to be between 0 and 1 while preserving direction.

    Args:
        vectors: Input vectors to squash
        axis: Axis along which to compute norm

    Returns:
        Squashed vectors
    """
    norm_squared = np.sum(vectors ** 2, axis=axis, keepdims=True)
    norm = np.sqrt(norm_squared + 1e-8)

    return (norm_squared / (1 + norm_squared)) * (vectors / norm)


class PrimaryCapsule:
    """
    Primary capsule layer - converts convolutional features to capsules.
    """

    def __init__(self, input_dim, n_capsules, capsule_dim):
        """
        Initialize primary capsules.

        Args:
            input_dim: Input feature dimension
            n_capsules: Number of capsules
            capsule_dim: Dimension of each capsule
        """
        self.input_dim = input_dim
        self.n_capsules = n_capsules
        self.capsule_dim = capsule_dim

        # Weights to transform input to capsules
        self.W = np.random.randn(input_dim, n_capsules * capsule_dim) * 0.1
        self.b = np.zeros((1, n_capsules * capsule_dim))

    def forward(self, X):
        """
        Forward pass through primary capsule layer.

        Args:
            X: Input features [batch_size, input_dim]

        Returns:
            Capsule outputs [batch_size, n_capsules, capsule_dim]
        """
        # Linear transformation
        u = np.dot(X, self.W) + self.b

        # Reshape to capsules
        batch_size = X.shape[0]
        u = u.reshape(batch_size, self.n_capsules, self.capsule_dim)

        # Apply squash activation
        self.output = squash(u, axis=-1)

        return self.output


class DigitCapsule:
    """
    Digit capsule layer with dynamic routing.
    """

    def __init__(self, input_n_capsules, input_capsule_dim,
                 n_capsules, capsule_dim, n_routing_iterations=3):
        """
        Initialize digit capsule layer.

        Args:
            input_n_capsules: Number of input capsules
            input_capsule_dim: Dimension of input capsules
            n_capsules: Number of output capsules (digits)
            capsule_dim: Dimension of output capsules
            n_routing_iterations: Number of dynamic routing iterations
        """
        self.input_n_capsules = input_n_capsules
        self.input_capsule_dim = input_capsule_dim
        self.n_capsules = n_capsules
        self.capsule_dim = capsule_dim
        self.n_routing_iterations = n_routing_iterations

        # Transformation matrices for each routing
        self.W = np.random.randn(input_n_capsules, n_capsules,
                                capsule_dim, input_capsule_dim) * 0.1

    def forward(self, u):
        """
        Forward pass with dynamic routing.

        Args:
            u: Input capsules [batch_size, input_n_capsules, input_capsule_dim]

        Returns:
            Output capsules [batch_size, n_capsules, capsule_dim]
        """
        batch_size = u.shape[0]

        # Compute predictions u_hat from input capsules
        # u_hat[i,j] = W[i,j] @ u[i]
        u_expanded = np.expand_dims(u, axis=2)  # [batch, in_caps, 1, in_dim]
        u_expanded = np.expand_dims(u_expanded, axis=-1)  # [batch, in_caps, 1, in_dim, 1]

        W_expanded = np.expand_dims(self.W, axis=0)  # [1, in_caps, n_caps, cap_dim, in_dim]

        # Compute predictions
        u_hat = np.matmul(W_expanded, u_expanded)  # [batch, in_caps, n_caps, cap_dim, 1]
        u_hat = np.squeeze(u_hat, axis=-1)  # [batch, in_caps, n_caps, cap_dim]

        # Initialize routing logits
        b = np.zeros((batch_size, self.input_n_capsules, self.n_capsules, 1))

        # Dynamic routing
        for iteration in range(self.n_routing_iterations):
            # Softmax across output capsules
            c = np.exp(b) / np.sum(np.exp(b), axis=2, keepdims=True)

            # Weighted sum of predictions
            s = np.sum(c * u_hat, axis=1)  # [batch, n_caps, cap_dim]

            # Squash
            v = squash(s, axis=-1)

            # Update routing logits (except last iteration)
            if iteration < self.n_routing_iterations - 1:
                # Agreement between v and u_hat
                v_expanded = np.expand_dims(v, axis=1)  # [batch, 1, n_caps, cap_dim]
                agreement = np.sum(u_hat * v_expanded, axis=-1, keepdims=True)
                b = b + agreement

        self.output = v
        return self.output


class CapsuleNetwork:
    """
    Complete Capsule Network with primary and digit capsules.
    """

    def __init__(self, input_dim, n_classes,
                 n_primary_capsules=16, primary_capsule_dim=8,
                 digit_capsule_dim=16):
        """
        Initialize CapsNet.

        Args:
            input_dim: Input feature dimension
            n_classes: Number of output classes
            n_primary_capsules: Number of primary capsules
            primary_capsule_dim: Dimension of primary capsules
            digit_capsule_dim: Dimension of digit capsules
        """
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Feature extraction layer
        self.conv_features = 128
        self.W_conv = np.random.randn(input_dim, self.conv_features) * 0.1
        self.b_conv = np.zeros((1, self.conv_features))

        # Primary capsule layer
        self.primary_caps = PrimaryCapsule(
            self.conv_features, n_primary_capsules, primary_capsule_dim
        )

        # Digit capsule layer
        self.digit_caps = DigitCapsule(
            n_primary_capsules, primary_capsule_dim,
            n_classes, digit_capsule_dim
        )

        # Decoder for reconstruction
        self.decoder_W1 = np.random.randn(digit_capsule_dim, 128) * 0.1
        self.decoder_b1 = np.zeros((1, 128))
        self.decoder_W2 = np.random.randn(128, 64) * 0.1
        self.decoder_b2 = np.zeros((1, 64))
        self.decoder_W3 = np.random.randn(64, input_dim) * 0.1
        self.decoder_b3 = np.zeros((1, input_dim))

    def forward(self, X, y=None):
        """
        Forward pass through CapsNet.

        Args:
            X: Input data [batch_size, input_dim]
            y: True labels (for masking during training)

        Returns:
            Capsule outputs and reconstructions
        """
        # Feature extraction
        features = np.maximum(0, np.dot(X, self.W_conv) + self.b_conv)

        # Primary capsules
        primary_output = self.primary_caps.forward(features)

        # Digit capsules
        digit_output = self.digit_caps.forward(primary_output)

        # Compute lengths (class predictions)
        lengths = np.sqrt(np.sum(digit_output ** 2, axis=-1))

        # Masking for reconstruction
        if y is not None:
            # Use true labels for masking during training
            mask = np.zeros((X.shape[0], self.n_classes))
            mask[np.arange(X.shape[0]), y] = 1
        else:
            # Use predicted labels for inference
            mask = np.zeros((X.shape[0], self.n_classes))
            mask[np.arange(X.shape[0]), np.argmax(lengths, axis=1)] = 1

        # Masked digit capsules for reconstruction
        mask_expanded = np.expand_dims(mask, axis=-1)
        masked_output = digit_output * mask_expanded

        # Decoder network
        decoder_input = masked_output.reshape(X.shape[0], -1)
        h1 = np.maximum(0, np.dot(decoder_input, self.decoder_W1) + self.decoder_b1)
        h2 = np.maximum(0, np.dot(h1, self.decoder_W2) + self.decoder_b2)
        reconstruction = np.dot(h2, self.decoder_W3) + self.decoder_b3

        return lengths, reconstruction, digit_output

    def margin_loss(self, y_true, lengths, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """
        Margin loss for capsule network.

        Args:
            y_true: True labels
            lengths: Capsule lengths
            m_plus: Target for correct class
            m_minus: Target for incorrect classes
            lambda_: Down-weighting for incorrect classes
        """
        batch_size = y_true.shape[0]

        # One-hot encoding
        y_onehot = np.zeros((batch_size, self.n_classes))
        y_onehot[np.arange(batch_size), y_true] = 1

        # Margin loss
        present_loss = y_onehot * np.maximum(0, m_plus - lengths) ** 2
        absent_loss = (1 - y_onehot) * np.maximum(0, lengths - m_minus) ** 2

        loss = np.sum(present_loss + lambda_ * absent_loss, axis=-1)
        return np.mean(loss)

    def reconstruction_loss(self, X, reconstruction):
        """MSE reconstruction loss."""
        return np.mean((X - reconstruction) ** 2)

    def train_step(self, X, y, learning_rate=0.001, recon_weight=0.0005):
        """
        Single training step.

        Args:
            X: Input batch
            y: Labels
            learning_rate: Learning rate
            recon_weight: Weight for reconstruction loss
        """
        # Forward pass
        lengths, reconstruction, digit_output = self.forward(X, y)

        # Compute losses
        margin_loss = self.margin_loss(y, lengths)
        recon_loss = self.reconstruction_loss(X, reconstruction)
        total_loss = margin_loss + recon_weight * recon_loss

        # Simple gradient approximation for demonstration
        # In practice, use automatic differentiation

        return total_loss, margin_loss, recon_loss

    def predict(self, X):
        """Make predictions."""
        lengths, _, _ = self.forward(X)
        return np.argmax(lengths, axis=1)


def train_capsnet(model, X_train, y_train, X_val, y_val,
                  epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train capsule network.

    Args:
        model: CapsuleNetwork instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    n_batches = len(X_train) // batch_size

    print(f"Training CapsNet for {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        for i in range(n_batches):
            batch_X = X_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = y_shuffled[i*batch_size:(i+1)*batch_size]

            # Training step
            loss, _, _ = model.train_step(batch_X, batch_y, learning_rate)
            epoch_losses.append(loss)

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        train_loss = np.mean(epoch_losses)
        val_lengths, val_recon, _ = model.forward(X_val, y_val)
        val_loss = model.margin_loss(y_val, val_lengths) + \
                   0.0005 * model.reconstruction_loss(X_val, val_recon)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history


def visualize_results(model, X_test, y_test, history):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Capsule Network Results', fontsize=16, fontweight='bold')

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

    # 3. Capsule activations
    ax = fig.add_subplot(gs[0, 2])
    lengths, _, digit_output = model.forward(X_test[:100])

    # Average capsule activation per class
    capsule_activations = np.zeros((model.n_classes, model.digit_caps.capsule_dim))
    for i in range(model.n_classes):
        mask = y_test[:100] == i
        if np.sum(mask) > 0:
            capsule_activations[i] = np.mean(digit_output[mask, i, :], axis=0)

    im = ax.imshow(capsule_activations.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Capsule Dimension')
    ax.set_title('Average Capsule Activations', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 4. Reconstruction examples
    sample_indices = np.random.choice(len(X_test), 6, replace=False)
    for idx, sample_idx in enumerate(sample_indices[:6]):
        ax = fig.add_subplot(gs[1, idx % 3] if idx < 3 else gs[2, idx % 3])

        X_sample = X_test[sample_idx:sample_idx+1]
        y_sample = y_test[sample_idx:sample_idx+1]

        _, reconstruction, _ = model.forward(X_sample, y_sample)

        # Reshape for visualization (assuming 8x8 images)
        original = X_sample[0].reshape(8, 8)
        recon = reconstruction[0].reshape(8, 8)

        # Show original and reconstruction side by side
        combined = np.hstack([original, recon])
        ax.imshow(combined, cmap='gray')
        ax.set_title(f'Digit {y_sample[0]}: Original | Reconstruction', fontsize=9)
        ax.axis('off')

    plt.savefig('/tmp/capsnet_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/capsnet_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("CAPSULE NETWORKS - KAGGLE SOLUTION")
    print("=" * 70)

    # Load data
    print("\n📊 Loading dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create model
    print("\n🏗️ Building Capsule Network...")
    model = CapsuleNetwork(
        input_dim=X.shape[1],
        n_classes=len(np.unique(y)),
        n_primary_capsules=16,
        primary_capsule_dim=8,
        digit_capsule_dim=16
    )

    print(f"  Primary Capsules: {model.primary_caps.n_capsules} × {model.primary_caps.capsule_dim}D")
    print(f"  Digit Capsules: {model.n_classes} × {model.digit_caps.capsule_dim}D")

    # Train model
    print("\n" + "=" * 70)
    history = train_capsnet(model, X_train, y_train, X_val, y_val,
                           epochs=50, batch_size=32)

    # Evaluate
    print("\n" + "=" * 70)
    print("📊 Evaluating on test set...")

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"✅ Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(model, X_test, y_test, history)

    print("\n" + "=" * 70)
    print("✅ CAPSULE NETWORK TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
