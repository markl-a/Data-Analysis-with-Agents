"""
LSTM for Text Generation - Kaggle Solution
==========================================
Generate text character-by-character using LSTM networks.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class TextGenerator:
    """Character-level text generation using LSTM."""

    def __init__(self, vocab_size, embedding_dim=64, lstm_units=128):
        """Initialize Text Generator.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of character embeddings
            lstm_units: Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        # Build model
        self.model = self.build_model()

    def build_model(self):
        """Build LSTM model for text generation.

        Returns:
            LSTM model
        """
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim,
                           name='embedding'),
            layers.LSTM(self.lstm_units, return_sequences=True,
                       name='lstm_1'),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units, name='lstm_2'),
            layers.Dropout(0.2),
            layers.Dense(self.lstm_units, activation='relu', name='dense'),
            layers.Dense(self.vocab_size, activation='softmax', name='output')
        ], name='text_generator')

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train the model.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print("Starting LSTM training...")
        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        return history

    def generate_text(self, seed_text, char_to_idx, idx_to_char,
                     seq_length, n_chars=200, temperature=1.0):
        """Generate text given a seed.

        Args:
            seed_text: Initial text to start generation
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
            seq_length: Length of input sequences
            n_chars: Number of characters to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text
        """
        generated_text = seed_text
        current_seq = seed_text[-seq_length:]

        for _ in range(n_chars):
            # Encode current sequence
            x = np.zeros((1, seq_length))
            for i, char in enumerate(current_seq):
                if char in char_to_idx:
                    x[0, i] = char_to_idx[char]

            # Predict next character
            predictions = self.model.predict(x, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            # Sample next character
            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = idx_to_char[next_idx]

            # Update sequence
            generated_text += next_char
            current_seq = current_seq[1:] + next_char

        return generated_text


def create_training_data(text, seq_length=40):
    """Create training sequences from text.

    Args:
        text: Input text
        seq_length: Length of input sequences

    Returns:
        X, y, char_to_idx, idx_to_char
    """
    print("Creating training data...")

    # Create character mappings
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"Total characters: {len(text)}")
    print(f"Unique characters: {len(chars)}")

    # Create sequences
    sequences = []
    next_chars = []

    for i in range(len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])

    print(f"Number of sequences: {len(sequences)}")

    # Convert to numerical arrays
    X = np.zeros((len(sequences), seq_length), dtype=np.int32)
    y = np.zeros(len(sequences), dtype=np.int32)

    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t] = char_to_idx[char]
        y[i] = char_to_idx[next_chars[i]]

    return X, y, char_to_idx, idx_to_char, chars


def create_sample_text():
    """Create sample text for training.

    Returns:
        Sample text string
    """
    text = """
    The quick brown fox jumps over the lazy dog. The dog barks at the moon.
    The moon shines bright in the night sky. Stars twinkle like diamonds.
    A wise old owl sits in the tree. The tree stands tall in the forest.
    The forest is full of mysteries and wonders. Birds sing beautiful songs.
    The river flows gently through the valley. Fish swim in the crystal water.
    Mountains rise majestically in the distance. Clouds drift slowly overhead.
    The sun sets painting the sky in orange and red. Night falls softly.
    A gentle breeze whispers through the leaves. Nature speaks in silent beauty.
    The cat prowls quietly in the garden. Butterflies dance among flowers.
    Bees buzz collecting nectar from blooms. The garden is a paradise.
    Time passes like sand through fingers. Memories fade but love remains.
    Hope springs eternal in the human heart. Dreams guide us forward.
    Knowledge is power and wisdom is strength. Learning never ends.
    The journey of life is full of surprises. Every day brings new lessons.
    Friendship is a precious gift. Kindness costs nothing but means everything.
    The world is a book to explore. Adventure awaits the brave.
    Stars guide sailors across the ocean. The sea holds many secrets.
    Ancient ruins tell stories of the past. History repeats itself.
    Music soothes the troubled soul. Art expresses what words cannot.
    Love conquers all obstacles. Faith moves mountains.
    """
    return text.strip()


def visualize_results(history, generated_samples):
    """Visualize training results and generated text.

    Args:
        history: Training history
        generated_samples: List of generated text samples
    """
    print("Generating visualizations...")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'lstm_training_curves.png'")

    # Save generated text
    with open('generated_text_samples.txt', 'w') as f:
        for i, sample in enumerate(generated_samples):
            f.write(f"\n{'='*60}\n")
            f.write(f"Sample {i+1}:\n")
            f.write(f"{'='*60}\n")
            f.write(sample)
            f.write("\n")

    print("Generated text samples saved to 'generated_text_samples.txt'")


def main():
    """Main execution function."""
    print("=" * 60)
    print("LSTM for Text Generation - Kaggle Solution")
    print("=" * 60)

    # Create sample text
    print("\nCreating sample text...")
    text = create_sample_text()

    # Prepare training data
    seq_length = 40
    X, y, char_to_idx, idx_to_char, chars = create_training_data(text, seq_length)

    # Split into train and validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nTraining set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")

    # Initialize model
    print("\nInitializing LSTM model...")
    generator = TextGenerator(
        vocab_size=len(chars),
        embedding_dim=64,
        lstm_units=128
    )

    # Print model summary
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    generator.model.summary()

    # Train model
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = generator.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=128
    )

    # Generate text samples
    print("\n" + "=" * 60)
    print("GENERATING TEXT SAMPLES")
    print("=" * 60)

    seed_texts = [
        "The quick brown fox",
        "The moon shines",
        "A wise old owl"
    ]

    temperatures = [0.5, 1.0, 1.5]
    generated_samples = []

    for seed in seed_texts:
        print(f"\nSeed: '{seed}'")
        for temp in temperatures:
            generated = generator.generate_text(
                seed, char_to_idx, idx_to_char,
                seq_length, n_chars=200, temperature=temp
            )
            print(f"\nTemperature {temp}:")
            print(generated[:150] + "...")
            generated_samples.append(f"Seed: '{seed}' | Temperature: {temp}\n{generated}")

    # Visualize results
    visualize_results(history, generated_samples)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Vocabulary Size: {len(chars)}")
    print(f"Total Sequences: {len(X)}")
    print(f"Sequence Length: {seq_length}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Total Training Epochs: {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
