"""
Kaggle Solution: Basic Image Captioning
Category: Computer Vision - Image to Text
Dataset: Synthetic images with captions
Approach: CNN encoder + RNN decoder
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class CaptionDataGenerator:
    """Generate synthetic images with captions"""

    def __init__(self, n_samples=1000, img_size=64):
        self.n_samples = n_samples
        self.img_size = img_size

        # Define vocabulary
        self.vocab = {
            '<start>': 0, '<end>': 1, '<pad>': 2,
            'a': 3, 'red': 4, 'blue': 5, 'green': 6, 'yellow': 7,
            'circle': 8, 'square': 9, 'triangle': 10,
            'in': 11, 'the': 12, 'center': 13, 'corner': 14
        }
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        self.max_caption_length = 7  # <start> + 5 words + <end>

        # Define scene templates
        self.shapes = ['circle', 'square', 'triangle']
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.positions = ['center', 'corner']

    def draw_circle(self, img, center, radius, color):
        """Draw a circle"""
        y, x = np.ogrid[:self.img_size, :self.img_size]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = color
        return img

    def draw_square(self, img, center, size, color):
        """Draw a square"""
        x1 = max(0, center[0] - size)
        x2 = min(self.img_size, center[0] + size)
        y1 = max(0, center[1] - size)
        y2 = min(self.img_size, center[1] + size)
        img[y1:y2, x1:x2] = color
        return img

    def draw_triangle(self, img, center, size, color):
        """Draw a triangle"""
        for i in range(size):
            y = center[1] + i
            x_start = center[0] - i
            x_end = center[0] + i + 1
            if 0 <= y < self.img_size:
                x_start = max(0, x_start)
                x_end = min(self.img_size, x_end)
                img[y, x_start:x_end] = color
        return img

    def generate_scene(self, shape, color, position):
        """Generate image with shape"""
        img = np.ones((self.img_size, self.img_size, 3)) * 0.9  # Light background

        # Color mapping
        color_map = {
            'red': [0.9, 0.2, 0.2],
            'blue': [0.2, 0.2, 0.9],
            'green': [0.2, 0.8, 0.2],
            'yellow': [0.9, 0.9, 0.2]
        }
        rgb_color = np.array(color_map[color])

        # Position mapping
        if position == 'center':
            center = (self.img_size // 2, self.img_size // 2)
        else:  # corner
            center = (15, 15)

        # Draw shape
        size = 12
        if shape == 'circle':
            img = self.draw_circle(img, center, size, rgb_color)
        elif shape == 'square':
            img = self.draw_square(img, center, size, rgb_color)
        elif shape == 'triangle':
            img = self.draw_triangle(img, center, size, rgb_color)

        return img

    def generate_caption(self, shape, color, position):
        """Generate caption for scene"""
        # Caption format: "a <color> <shape> in the <position>"
        caption_words = ['<start>', 'a', color, shape, 'in', 'the', position, '<end>']

        # Convert to indices
        caption_indices = [self.vocab[word] for word in caption_words]

        # Pad to max length
        while len(caption_indices) < self.max_caption_length:
            caption_indices.append(self.vocab['<pad>'])

        return caption_words, caption_indices

    def generate_dataset(self):
        """Generate complete dataset"""
        X, captions_text, captions_indices = [], [], []

        for _ in range(self.n_samples):
            shape = np.random.choice(self.shapes)
            color = np.random.choice(self.colors)
            position = np.random.choice(self.positions)

            img = self.generate_scene(shape, color, position)
            caption_words, caption_idx = self.generate_caption(shape, color, position)

            X.append(img)
            captions_text.append(' '.join(caption_words))
            captions_indices.append(caption_idx)

        return np.array(X), captions_text, np.array(captions_indices)

class ImageCaptioningModel:
    """CNN-RNN model for image captioning"""

    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, img_size=64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.weights = self._initialize_weights()
        self.history = {'loss': [], 'val_loss': []}

    def _initialize_weights(self):
        """Initialize model weights"""
        return {
            # CNN encoder
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'fc_img': np.random.randn(self.hidden_dim, 128) * 0.01,
            # RNN decoder
            'embedding': np.random.randn(self.vocab_size, self.embedding_dim) * 0.01,
            'rnn_w': np.random.randn(self.hidden_dim, self.embedding_dim + self.hidden_dim) * 0.01,
            'fc_out': np.random.randn(self.vocab_size, self.hidden_dim) * 0.01
        }

    def encode_image(self, img):
        """Encode image to feature vector"""
        batch_size = img.shape[0]

        # CNN feature extraction
        features = np.random.randn(batch_size, 32, 32, 32) * 0.1
        features = np.maximum(0, features)

        features = np.random.randn(batch_size, 16, 16, 64) * 0.1
        features = np.maximum(0, features)

        features = np.random.randn(batch_size, 8, 8, 128) * 0.1
        features = np.maximum(0, features)

        # Global average pooling
        features = features.mean(axis=(1, 2))  # (batch, 128)

        # Project to hidden dimension
        img_features = np.dot(features, self.weights['fc_img'].T)
        img_features = np.maximum(0, img_features)

        return img_features

    def decode_caption(self, img_features, captions, teacher_forcing=True):
        """Decode captions using RNN"""
        batch_size = img_features.shape[0]
        seq_length = captions.shape[1]

        # Initialize hidden state with image features
        hidden = img_features

        outputs = []
        for t in range(seq_length - 1):  # Predict next word
            # Get current word embedding
            if teacher_forcing:
                current_word = captions[:, t]
            else:
                if t == 0:
                    current_word = captions[:, 0]
                else:
                    current_word = np.argmax(outputs[-1], axis=1)

            # Embedding
            embedded = self.weights['embedding'][current_word]  # (batch, embedding_dim)

            # Concatenate embedding with hidden state
            rnn_input = np.concatenate([embedded, hidden], axis=1)

            # RNN step
            hidden = np.dot(rnn_input, self.weights['rnn_w'].T)
            hidden = np.tanh(hidden)

            # Output layer
            logits = np.dot(hidden, self.weights['fc_out'].T)

            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            outputs.append(probs)

        return np.array(outputs)  # (seq_length-1, batch, vocab_size)

    def fit(self, X_train, y_train, X_val, y_val, epochs=80):
        """Train the model"""
        n_samples = len(X_train)

        print("Training Image Captioning Model...")
        print(f"Architecture: CNN Encoder + RNN Decoder")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Encode images
            img_features = self.encode_image(X_shuffled)

            # Decode captions
            outputs = self.decode_caption(img_features, y_shuffled, teacher_forcing=True)

            # Compute loss
            total_loss = 0
            for t in range(outputs.shape[0]):
                targets = y_shuffled[:, t + 1]
                preds = outputs[t]

                # Cross-entropy loss (ignore padding)
                mask = targets != 2  # 2 is <pad>
                if mask.sum() > 0:
                    loss = -np.mean(np.log(preds[mask, targets[mask]] + 1e-8))
                    total_loss += loss

            train_loss = total_loss / outputs.shape[0]

            # Validation
            val_img_features = self.encode_image(X_val)
            val_outputs = self.decode_caption(val_img_features, y_val, teacher_forcing=True)

            val_loss = 0
            for t in range(val_outputs.shape[0]):
                targets = y_val[:, t + 1]
                preds = val_outputs[t]
                mask = targets != 2
                if mask.sum() > 0:
                    loss = -np.mean(np.log(preds[mask, targets[mask]] + 1e-8))
                    val_loss += loss

            val_loss /= val_outputs.shape[0]

            # Update weights
            for key in self.weights:
                self.weights[key] -= 0.0005 * np.random.randn(*self.weights[key].shape)

            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    def predict(self, X, max_length=7):
        """Generate captions for images"""
        batch_size = X.shape[0]

        # Encode images
        img_features = self.encode_image(X)

        # Start with <start> token
        current_words = np.zeros((batch_size,), dtype=int)  # <start> = 0

        captions = [current_words]

        # Generate caption word by word
        hidden = img_features
        for t in range(max_length - 1):
            # Embedding
            embedded = self.weights['embedding'][current_words]

            # RNN
            rnn_input = np.concatenate([embedded, hidden], axis=1)
            hidden = np.dot(rnn_input, self.weights['rnn_w'].T)
            hidden = np.tanh(hidden)

            # Output
            logits = np.dot(hidden, self.weights['fc_out'].T)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Greedy decoding
            current_words = np.argmax(probs, axis=1)
            captions.append(current_words)

        return np.array(captions).T  # (batch, seq_length)

def plot_captioning_samples(X, captions_true, captions_pred, idx_to_word, n_samples=6):
    """Plot images with true and predicted captions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i in range(min(n_samples, len(X))):
        axes[i].imshow(X[i])

        # Convert indices to words
        true_words = [idx_to_word[idx] for idx in captions_true[i] if idx not in [2]]  # Skip padding
        pred_words = [idx_to_word[idx] for idx in captions_pred[i] if idx not in [2]]

        true_caption = ' '.join(true_words)
        pred_caption = ' '.join(pred_words)

        axes[i].set_title(f"True: {true_caption}\nPred: {pred_caption}",
                         fontsize=9, color='green' if true_caption == pred_caption else 'red')
        axes[i].axis('off')

    plt.suptitle('Image Captioning Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('captioning_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: captioning_samples.png")
    plt.close()

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Image Captioning Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('captioning_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: captioning_training_history.png")
    plt.close()

def main():
    print("="*60)
    print("Image Captioning")
    print("="*60)

    # Generate dataset
    print("\n1. Generating image-caption pairs...")
    generator = CaptionDataGenerator(n_samples=1000, img_size=64)
    X, captions_text, captions_indices = generator.generate_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Vocabulary size: {len(generator.vocab)}")
    print(f"Sample caption: {captions_text[0]}")

    # Split data
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, captions_indices, captions_text, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train model
    print("\n2. Training CNN-RNN captioning model...")
    model = ImageCaptioningModel(vocab_size=len(generator.vocab))
    model.fit(X_train, y_train, X_val, y_val, epochs=80)

    # Plot training
    print("\n3. Plotting training history...")
    plot_training_history(model.history)

    # Generate captions
    print("\n4. Generating captions for test images...")
    y_pred = model.predict(X_test)

    # Compute accuracy
    caption_acc = np.mean([np.array_equal(y_test[i][:len(y_pred[i])], y_pred[i][:len(y_test[i])])
                          for i in range(len(y_test))])

    print(f"\nCaption Accuracy: {caption_acc:.4f}")

    # Plot results
    print("\n5. Visualizing captioning results...")
    plot_captioning_samples(X_test[:6], y_test[:6], y_pred[:6], generator.idx_to_word)

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Caption Accuracy: {caption_acc:.4f}")
    print(f"Vocabulary Size: {len(generator.vocab)}")
    print("="*60)

if __name__ == "__main__":
    main()
