"""
Attention Mechanism for Translation - Kaggle Solution
====================================================
Implement sequence-to-sequence translation with attention mechanism.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class BahdanauAttention(layers.Layer):
    """Bahdanau attention mechanism."""

    def __init__(self, units):
        """Initialize attention layer.

        Args:
            units: Number of hidden units
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        """Compute attention weights and context vector.

        Args:
            query: Decoder hidden state (batch_size, hidden_dim)
            values: Encoder outputs (batch_size, seq_len, hidden_dim)

        Returns:
            context_vector, attention_weights
        """
        # Expand query to (batch_size, 1, hidden_dim)
        query_with_time_axis = tf.expand_dims(query, 1)

        # Calculate attention scores
        # score shape: (batch_size, seq_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))

        # Attention weights shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector shape: (batch_size, hidden_dim)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(keras.Model):
    """Encoder with GRU."""

    def __init__(self, vocab_size, embedding_dim, enc_units):
        """Initialize encoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            enc_units: Number of encoder units
        """
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(enc_units,
                             return_sequences=True,
                             return_state=True)

    def call(self, x, hidden):
        """Forward pass.

        Args:
            x: Input sequences
            hidden: Initial hidden state

        Returns:
            output, state
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size):
        """Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Zero-initialized hidden state
        """
        return tf.zeros((batch_size, self.enc_units))


class Decoder(keras.Model):
    """Decoder with attention."""

    def __init__(self, vocab_size, embedding_dim, dec_units):
        """Initialize decoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            dec_units: Number of decoder units
        """
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(dec_units,
                             return_sequences=True,
                             return_state=True)
        self.fc = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        """Forward pass.

        Args:
            x: Input token
            hidden: Previous hidden state
            enc_output: Encoder outputs

        Returns:
            predictions, state, attention_weights
        """
        # Get context vector from attention
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # Embed input
        x = self.embedding(x)

        # Concatenate embedding and context vector
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass through GRU
        output, state = self.gru(x)

        # Reshape output
        output = tf.reshape(output, (-1, output.shape[2]))

        # Generate predictions
        x = self.fc(output)

        return x, state, attention_weights


def create_synthetic_translation_data(n_samples=1000):
    """Create synthetic translation dataset.

    Simple number-to-word translation task.

    Args:
        n_samples: Number of training samples

    Returns:
        source_texts, target_texts
    """
    print(f"Creating {n_samples} translation pairs...")

    # Number words
    numbers = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    source_texts = []
    target_texts = []

    for _ in range(n_samples):
        # Generate random number sequence
        length = np.random.randint(2, 6)
        num_seq = ''.join([str(np.random.randint(0, 10)) for _ in range(length)])

        # Convert to words
        word_seq = ' '.join([numbers[digit] for digit in num_seq])

        source_texts.append(num_seq)
        target_texts.append('<start> ' + word_seq + ' <end>')

    print(f"Created {n_samples} translation pairs")
    return source_texts, target_texts


def tokenize_data(texts):
    """Tokenize text data.

    Args:
        texts: List of text strings

    Returns:
        tokenizer, sequences, max_length
    """
    tokenizer = keras.preprocessing.text.Tokenizer(
        filters='',
        char_level=True
    )
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    max_length = max(len(seq) for seq in sequences)
    sequences = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post'
    )

    return tokenizer, sequences, max_length


def train_step(inp, targ, encoder, decoder, optimizer, loss_object):
    """Single training step.

    Args:
        inp: Input sequence
        targ: Target sequence
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        loss_object: Loss function

    Returns:
        batch_loss
    """
    batch_size = inp.shape[0]
    loss = 0

    with tf.GradientTape() as tape:
        enc_hidden = encoder.initialize_hidden_state(batch_size)
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        # Teacher forcing - feed target as next input
        for t in range(1, targ.shape[1]):
            dec_input = tf.expand_dims(targ[:, t-1], 1)

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_object(targ[:, t], predictions)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def translate(sentence, encoder, decoder, source_tokenizer, target_tokenizer,
              max_length_source, max_length_target):
    """Translate a sentence.

    Args:
        sentence: Input sentence
        encoder: Encoder model
        decoder: Decoder model
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        max_length_source: Max source length
        max_length_target: Max target length

    Returns:
        Translated sentence and attention weights
    """
    # Encode input
    inputs = source_tokenizer.texts_to_sequences([sentence])
    inputs = keras.preprocessing.sequence.pad_sequences(
        inputs, maxlen=max_length_source, padding='post'
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    attention_plot = np.zeros((max_length_target, max_length_source))

    enc_hidden = encoder.initialize_hidden_state(1)
    enc_output, enc_hidden = encoder(inputs, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<']], 0)

    for t in range(max_length_target):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_output
        )

        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        if predicted_id == 0:
            break

        result += target_tokenizer.index_word.get(predicted_id, '')

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, attention_plot[:t]


def main():
    """Main execution function."""
    print("=" * 60)
    print("Attention Mechanism for Translation - Kaggle Solution")
    print("=" * 60)

    # Create synthetic data
    source_texts, target_texts = create_synthetic_translation_data(n_samples=1000)

    # Split data
    source_train, source_val, target_train, target_val = train_test_split(
        source_texts, target_texts, test_size=0.2, random_state=42
    )

    print(f"\nTraining samples: {len(source_train)}")
    print(f"Validation samples: {len(source_val)}")

    # Tokenize
    print("\nTokenizing data...")
    source_tokenizer, source_sequences_train, max_length_source = tokenize_data(source_train)
    target_tokenizer, target_sequences_train, max_length_target = tokenize_data(target_train)

    source_val_sequences = keras.preprocessing.sequence.pad_sequences(
        source_tokenizer.texts_to_sequences(source_val),
        maxlen=max_length_source, padding='post'
    )

    print(f"Source vocabulary size: {len(source_tokenizer.word_index) + 1}")
    print(f"Target vocabulary size: {len(target_tokenizer.word_index) + 1}")

    # Model parameters
    BUFFER_SIZE = len(source_sequences_train)
    BATCH_SIZE = 64
    embedding_dim = 64
    units = 128
    vocab_inp_size = len(source_tokenizer.word_index) + 1
    vocab_tar_size = len(target_tokenizer.word_index) + 1

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (source_sequences_train, target_sequences_train)
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Initialize models
    print("\nInitializing encoder and decoder...")
    encoder = Encoder(vocab_inp_size, embedding_dim, units)
    decoder = Decoder(vocab_tar_size, embedding_dim, units)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    EPOCHS = 20
    history = {'loss': []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0

        for batch, (inp, targ) in enumerate(dataset):
            batch_loss = train_step(inp, targ, encoder, decoder, optimizer, loss_object)
            total_loss += batch_loss

        epoch_loss = total_loss / (batch + 1)
        history['loss'].append(float(epoch_loss))

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")

    # Test translations
    print("\n" + "=" * 60)
    print("TRANSLATION EXAMPLES")
    print("=" * 60)

    test_samples = source_val[:5]
    for i, source in enumerate(test_samples):
        target_actual = target_val[i]
        translation, _ = translate(
            source, encoder, decoder,
            source_tokenizer, target_tokenizer,
            max_length_source, max_length_target
        )

        print(f"\nExample {i+1}:")
        print(f"  Input:    {source}")
        print(f"  Expected: {target_actual}")
        print(f"  Predicted: {translation}")

    # Plot training curve
    print("\nGenerating visualization...")
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('attention_training_curve.png', dpi=300, bbox_inches='tight')
    print("Training curve saved to 'attention_training_curve.png'")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Loss: {history['loss'][-1]:.4f}")
    print(f"Total Epochs: {EPOCHS}")
    print(f"Encoder Units: {units}")
    print(f"Decoder Units: {units}")
    print(f"Embedding Dimension: {embedding_dim}")
    print("=" * 60)


if __name__ == "__main__":
    main()
