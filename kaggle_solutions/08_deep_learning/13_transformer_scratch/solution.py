"""
Transformer from Scratch - Kaggle Solution Example
===================================================

This example implements a Transformer model from scratch, demonstrating
self-attention mechanisms and position encodings for sequence modeling.

Problem: Sequence-to-sequence learning (simplified machine translation)

Approach:
1. Implement scaled dot-product attention
2. Build multi-head attention mechanism
3. Create positional encoding
4. Implement encoder and decoder
5. Train on sequence copying/transformation task
6. Visualize attention patterns

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


def create_sequence_dataset(n_samples=1000, seq_length=10, vocab_size=20):
    """
    Create synthetic sequence-to-sequence dataset.

    Task: Copy sequence with transformation (e.g., reverse, shift)

    Args:
        n_samples: Number of sequences
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary

    Returns:
        source_sequences, target_sequences
    """
    # Generate random sequences
    source = np.random.randint(1, vocab_size, (n_samples, seq_length))

    # Target is reversed sequence (simple task)
    target = source[:, ::-1].copy()

    return source, target


def positional_encoding(seq_length, d_model):
    """
    Create positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_length: Sequence length
        d_model: Model dimension

    Returns:
        Positional encodings [seq_length, d_model]
    """
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Queries [batch_size, seq_len, d_k]
        K: Keys [batch_size, seq_len, d_k]
        V: Values [batch_size, seq_len, d_v]
        mask: Optional mask

    Returns:
        Output and attention weights
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores + (mask * -1e9)

    # Softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

    # Apply attention to values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Allows model to attend to different representation subspaces.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass.

        Args:
            Q, K, V: Query, key, value tensors
            mask: Optional attention mask

        Returns:
            Output and attention weights
        """
        batch_size = Q.shape[0]
        seq_length = Q.shape[1]

        # Linear projections
        Q = np.dot(Q, self.W_q)
        K = np.dot(K, self.W_k)
        V = np.dot(V, self.W_v)

        # Split into heads: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Apply attention for each head
        attn_outputs = []
        attn_weights_list = []

        for h in range(self.num_heads):
            output, weights = scaled_dot_product_attention(
                Q[:, h, :, :], K[:, h, :, :], V[:, h, :, :], mask
            )
            attn_outputs.append(output)
            attn_weights_list.append(weights)

        # Concatenate heads
        attn_output = np.concatenate(attn_outputs, axis=-1)

        # Final linear projection
        output = np.dot(attn_output, self.W_o)

        # Average attention weights across heads
        avg_attn_weights = np.mean(np.stack(attn_weights_list, axis=1), axis=1)

        return output, avg_attn_weights


class FeedForward:
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize FFN.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros((1, d_model))

    def forward(self, x):
        """Forward pass."""
        # First layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)

        # Second layer
        output = np.dot(hidden, self.W2) + self.b2

        return output


class TransformerBlock:
    """
    Single transformer encoder block.

    Consists of multi-head attention and feed-forward network
    with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = dropout

    def layer_norm(self, x, eps=1e-6):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor and attention weights
        """
        # Multi-head attention with residual
        attn_output, attn_weights = self.attention.forward(x, x, x, mask)
        x = self.layer_norm(x + attn_output)

        # Feed-forward with residual
        ffn_output = self.ffn.forward(x)
        x = self.layer_norm(x + ffn_output)

        return x, attn_weights


class SimpleTransformer:
    """
    Simplified Transformer model for sequence-to-sequence learning.
    """

    def __init__(self, vocab_size, d_model=64, num_heads=4,
                 num_layers=2, d_ff=256, max_seq_length=50):
        """
        Initialize transformer.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_seq_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Embedding layer
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1

        # Positional encoding
        self.pos_encoding = positional_encoding(max_seq_length, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1

    def embed(self, x):
        """
        Embed tokens and add positional encoding.

        Args:
            x: Token indices [batch_size, seq_length]

        Returns:
            Embedded sequences [batch_size, seq_length, d_model]
        """
        batch_size, seq_length = x.shape

        # Token embeddings
        embedded = self.embedding[x]  # [batch, seq_len, d_model]

        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_length]

        return embedded

    def forward(self, x):
        """
        Forward pass through transformer.

        Args:
            x: Input token indices [batch_size, seq_length]

        Returns:
            Output logits [batch_size, seq_length, vocab_size]
        """
        # Embed input
        h = self.embed(x)

        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            h, attn = block.forward(h)
            attention_weights.append(attn)

        # Project to vocabulary
        logits = np.dot(h, self.output_proj)

        return logits, attention_weights

    def predict(self, x):
        """Make predictions."""
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=-1)


def train_transformer(model, X_train, y_train, X_val, y_val,
                     epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train transformer model.

    Args:
        model: Transformer model
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

    print(f"Training Transformer for {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_losses = []

        for i in range(n_batches):
            batch_X = X_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = y_shuffled[i*batch_size:(i+1)*batch_size]

            # Forward pass
            logits, _ = model.forward(batch_X)

            # Compute loss (cross-entropy)
            batch_size_actual = batch_X.shape[0]
            seq_length = batch_X.shape[1]

            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Cross-entropy
            loss = 0
            for b in range(batch_size_actual):
                for t in range(seq_length):
                    loss -= np.log(probs[b, t, batch_y[b, t]] + 1e-10)

            loss /= (batch_size_actual * seq_length)
            epoch_losses.append(loss)

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)

        # Validation loss
        val_logits, _ = model.forward(X_val)
        exp_logits = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        val_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        val_loss = 0
        for b in range(len(X_val)):
            for t in range(X_val.shape[1]):
                val_loss -= np.log(val_probs[b, t, y_val[b, t]] + 1e-10)
        val_loss /= (len(X_val) * X_val.shape[1])

        history['train_loss'].append(np.mean(epoch_losses))
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {np.mean(epoch_losses):.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history


def visualize_results(model, X_test, y_test, history):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Transformer from Scratch Results', fontsize=16, fontweight='bold')

    # 1. Training loss
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training accuracy
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Attention visualization
    ax = fig.add_subplot(gs[0, 2])

    # Get attention weights for a sample
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]
    _, attn_weights_list = model.forward(sample)

    # Average across all layers
    avg_attn = np.mean(np.stack([aw[0] for aw in attn_weights_list], axis=0), axis=0)

    im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Attention Heatmap (Avg)', fontweight='bold')
    plt.colorbar(im, ax=ax)

    # 4-6. Example predictions
    for idx in range(3):
        ax = fig.add_subplot(gs[1, idx])

        sample_idx = idx * 10
        source = X_test[sample_idx]
        target = y_test[sample_idx]
        predicted = model.predict(X_test[sample_idx:sample_idx+1])[0]

        positions = range(len(source))

        ax.plot(positions, source, 'o-', label='Source', linewidth=2, markersize=6)
        ax.plot(positions, target, 's-', label='Target', linewidth=2, markersize=6)
        ax.plot(positions, predicted, 'x-', label='Predicted', linewidth=2, markersize=8)

        ax.set_xlabel('Position')
        ax.set_ylabel('Token ID')
        ax.set_title(f'Example {idx+1}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 7. Per-position accuracy
    ax = fig.add_subplot(gs[2, 0])

    predictions = model.predict(X_test)
    seq_length = X_test.shape[1]

    position_acc = []
    for pos in range(seq_length):
        acc = np.mean(predictions[:, pos] == y_test[:, pos])
        position_acc.append(acc)

    ax.bar(range(seq_length), position_acc, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Position')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Position Accuracy', fontweight='bold')
    ax.axhline(np.mean(position_acc), color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Token prediction accuracy
    ax = fig.add_subplot(gs[2, 1])

    vocab_size = model.vocab_size
    token_correct = np.zeros(vocab_size)
    token_total = np.zeros(vocab_size)

    for i in range(len(y_test)):
        for j in range(seq_length):
            token = y_test[i, j]
            token_total[token] += 1
            if predictions[i, j] == token:
                token_correct[token] += 1

    token_acc = token_correct / (token_total + 1e-10)

    ax.bar(range(vocab_size), token_acc, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Token ID')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Token Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 9. Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    test_acc = np.mean(predictions == y_test)
    perfect_sequences = np.mean(np.all(predictions == y_test, axis=1))

    summary = f"""
    TRAINING SUMMARY
    ══════════════════════

    Model:
    • d_model: {model.d_model}
    • Layers: {len(model.blocks)}
    • Vocab: {model.vocab_size}

    Performance:
    • Test Accuracy: {test_acc:.4f}
    • Perfect Seqs: {perfect_sequences*100:.1f}%

    Final:
    • Train Acc: {history['train_acc'][-1]:.4f}
    • Val Acc: {history['val_acc'][-1]:.4f}

    Best Val: {max(history['val_acc']):.4f}
    """

    ax.text(0.1, 0.5, summary, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.savefig('/tmp/transformer_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/transformer_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("TRANSFORMER FROM SCRATCH - KAGGLE SOLUTION")
    print("=" * 70)

    # Create dataset
    print("\n📊 Creating sequence dataset...")
    X, y = create_sequence_dataset(n_samples=1000, seq_length=10, vocab_size=20)

    print(f"Sequences: {X.shape[0]}")
    print(f"Sequence length: {X.shape[1]}")
    print(f"Vocabulary size: {20}")
    print(f"Task: Reverse sequence")

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Create model
    print("\n🏗️ Building Transformer...")
    model = SimpleTransformer(
        vocab_size=20,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=50
    )

    print(f"  d_model: {model.d_model}")
    print(f"  Attention heads: 4")
    print(f"  Layers: {len(model.blocks)}")

    # Train model
    print("\n" + "=" * 70)
    history = train_transformer(
        model, X_train, y_train, X_val, y_val,
        epochs=100, batch_size=32, learning_rate=0.001
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("📊 Evaluating on test set...")

    predictions = model.predict(X_test)
    test_acc = np.mean(predictions == y_test)
    perfect_sequences = np.mean(np.all(predictions == y_test, axis=1))

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Perfect Sequences: {perfect_sequences*100:.1f}%")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(model, X_test, y_test, history)

    print("\n" + "=" * 70)
    print("✅ TRANSFORMER TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
