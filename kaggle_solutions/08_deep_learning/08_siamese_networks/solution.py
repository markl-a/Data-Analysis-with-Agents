"""
Siamese Networks for Similarity Learning - Kaggle Solution
==========================================================
Learn similarity metrics using Siamese neural networks.
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


class SiameseNetwork:
    """Siamese network for learning similarity."""

    def __init__(self, input_shape=(28, 28, 1), embedding_dim=128):
        """Initialize Siamese network.

        Args:
            input_shape: Shape of input images
            embedding_dim: Dimension of embedding vector
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

        # Build embedding network
        self.embedding_network = self.build_embedding_network()

        # Build Siamese model
        self.model = self.build_siamese_model()

    def build_embedding_network(self):
        """Build embedding network (shared weights).

        Returns:
            Embedding network model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.embedding_dim, activation=None),  # No activation
            layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # L2 normalize
        ], name='embedding_network')

        return model

    def build_siamese_model(self):
        """Build Siamese model with contrastive loss.

        Returns:
            Siamese model
        """
        # Input layers
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')

        # Generate embeddings using shared network
        embedding_a = self.embedding_network(input_a)
        embedding_b = self.embedding_network(input_b)

        # Calculate L2 distance
        l2_distance = layers.Lambda(
            lambda embeddings: tf.sqrt(
                tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)
            ),
            name='l2_distance'
        )([embedding_a, embedding_b])

        # Build model
        model = keras.Model(
            inputs=[input_a, input_b],
            outputs=l2_distance,
            name='siamese_network'
        )

        return model

    def contrastive_loss(self, y_true, y_pred, margin=1.0):
        """Contrastive loss function.

        Args:
            y_true: True labels (1 for similar, 0 for dissimilar)
            y_pred: Predicted distances
            margin: Margin for dissimilar pairs

        Returns:
            Contrastive loss
        """
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        loss = tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
        return loss

    def compile_model(self, learning_rate=0.001):
        """Compile the model.

        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.contrastive_loss,
            metrics=[self.accuracy_metric]
        )

    @staticmethod
    def accuracy_metric(y_true, y_pred, threshold=0.5):
        """Calculate accuracy based on threshold.

        Args:
            y_true: True labels
            y_pred: Predicted distances
            threshold: Distance threshold

        Returns:
            Accuracy
        """
        y_true = tf.cast(y_true, tf.float32)
        predictions = tf.cast(y_pred < threshold, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))

    def train(self, pairs_train, labels_train, pairs_val, labels_val,
              epochs=20, batch_size=32):
        """Train the Siamese network.

        Args:
            pairs_train: Training image pairs
            labels_train: Training labels (1=similar, 0=dissimilar)
            pairs_val: Validation image pairs
            labels_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print("Starting Siamese network training...")
        start_time = time.time()

        history = self.model.fit(
            [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
            validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        return history

    def predict_similarity(self, image_a, image_b):
        """Predict similarity between two images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            Distance (lower = more similar)
        """
        if len(image_a.shape) == 3:
            image_a = np.expand_dims(image_a, 0)
        if len(image_b.shape) == 3:
            image_b = np.expand_dims(image_b, 0)

        distance = self.model.predict([image_a, image_b], verbose=0)
        return float(distance[0])

    def get_embedding(self, image):
        """Get embedding for an image.

        Args:
            image: Input image

        Returns:
            Embedding vector
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        embedding = self.embedding_network.predict(image, verbose=0)
        return embedding[0]


def create_synthetic_images(n_samples=500, img_size=28):
    """Create synthetic images with different patterns.

    Args:
        n_samples: Number of samples per class
        img_size: Image size

    Returns:
        images, labels
    """
    print(f"Creating {n_samples * 3} synthetic images...")

    images = []
    labels = []

    # Class 0: Circles
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size))
        center_x = np.random.randint(img_size // 3, 2 * img_size // 3)
        center_y = np.random.randint(img_size // 3, 2 * img_size // 3)
        radius = np.random.randint(5, 10)

        y, x = np.ogrid[:img_size, :img_size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = 1.0

        images.append(img)
        labels.append(0)

    # Class 1: Squares
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size))
        x1 = np.random.randint(5, 15)
        y1 = np.random.randint(5, 15)
        size = np.random.randint(8, 14)
        x2 = min(x1 + size, img_size)
        y2 = min(y1 + size, img_size)
        img[y1:y2, x1:x2] = 1.0

        images.append(img)
        labels.append(1)

    # Class 2: Triangles
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size))
        center_x = np.random.randint(img_size // 3, 2 * img_size // 3)
        top_y = np.random.randint(5, 10)
        bottom_y = np.random.randint(18, 23)

        for y in range(top_y, bottom_y):
            width = int((y - top_y) * 0.5)
            x1 = max(0, center_x - width)
            x2 = min(img_size, center_x + width)
            img[y, x1:x2] = 1.0

        images.append(img)
        labels.append(2)

    images = np.array(images).reshape(-1, img_size, img_size, 1)
    labels = np.array(labels)

    print(f"Created {len(images)} images across 3 classes")
    return images, labels


def create_pairs(images, labels, n_pairs=1000):
    """Create pairs of images with labels.

    Args:
        images: Array of images
        labels: Array of labels
        n_pairs: Number of pairs to create

    Returns:
        pairs, pair_labels
    """
    print(f"Creating {n_pairs} image pairs...")

    pairs = []
    pair_labels = []

    n_classes = len(np.unique(labels))

    # Create similar pairs (same class)
    for _ in range(n_pairs // 2):
        # Select random class
        class_idx = np.random.randint(0, n_classes)
        class_images = images[labels == class_idx]

        # Select two random images from same class
        if len(class_images) >= 2:
            idx = np.random.choice(len(class_images), 2, replace=False)
            pairs.append([class_images[idx[0]], class_images[idx[1]]])
            pair_labels.append(1)  # Similar

    # Create dissimilar pairs (different classes)
    for _ in range(n_pairs // 2):
        # Select two different classes
        class_indices = np.random.choice(n_classes, 2, replace=False)

        img1 = images[labels == class_indices[0]][np.random.randint(0, np.sum(labels == class_indices[0]))]
        img2 = images[labels == class_indices[1]][np.random.randint(0, np.sum(labels == class_indices[1]))]

        pairs.append([img1, img2])
        pair_labels.append(0)  # Dissimilar

    pairs = np.array(pairs)
    pair_labels = np.array(pair_labels)

    # Shuffle
    indices = np.random.permutation(len(pairs))
    pairs = pairs[indices]
    pair_labels = pair_labels[indices]

    print(f"Created {len(pairs)} pairs ({np.sum(pair_labels)} similar, {len(pairs) - np.sum(pair_labels)} dissimilar)")
    return pairs, pair_labels


def visualize_results(siamese, test_images, test_labels, history):
    """Visualize results.

    Args:
        siamese: Trained Siamese network
        test_images: Test images
        test_labels: Test labels
        history: Training history
    """
    print("\nGenerating visualizations...")

    # Create test pairs
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(4):
        # Similar pair
        class_idx = i % 3
        class_images = test_images[test_labels == class_idx]
        if len(class_images) >= 2:
            img1, img2 = class_images[:2]
            distance = siamese.predict_similarity(img1, img2)

            axes[i, 0].imshow(img1.reshape(28, 28), cmap='gray')
            axes[i, 0].set_title('Image 1', fontsize=10)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(img2.reshape(28, 28), cmap='gray')
            axes[i, 1].set_title('Image 2', fontsize=10)
            axes[i, 1].axis('off')

            axes[i, 2].text(0.5, 0.5, f'Distance:\n{distance:.3f}\n\nSimilar',
                          ha='center', va='center', fontsize=12, color='green')
            axes[i, 2].axis('off')

        # Dissimilar pair
        class_indices = np.random.choice(3, 2, replace=False)
        img1 = test_images[test_labels == class_indices[0]][0]
        img2 = test_images[test_labels == class_indices[1]][0]
        distance = siamese.predict_similarity(img1, img2)

        axes[i, 3].text(0.5, 0.5, f'Distance:\n{distance:.3f}\n\nDissimilar',
                       ha='center', va='center', fontsize=12, color='red')
        axes[i, 3].axis('off')

    plt.suptitle('Siamese Network Similarity Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('siamese_predictions.png', dpi=300, bbox_inches='tight')
    print("Predictions saved to 'siamese_predictions.png'")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Contrastive Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history.history['accuracy_metric'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy_metric'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Pair Classification Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('siamese_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'siamese_training_curves.png'")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Siamese Networks for Similarity Learning - Kaggle Solution")
    print("=" * 60)

    # Create synthetic dataset
    images, labels = create_synthetic_images(n_samples=500, img_size=28)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Test: {len(X_test)} images")

    # Create pairs
    pairs_train, labels_train = create_pairs(X_train, y_train, n_pairs=2000)
    pairs_val, labels_val = create_pairs(X_test, y_test, n_pairs=400)

    # Initialize Siamese network
    print("\nInitializing Siamese network...")
    siamese = SiameseNetwork(input_shape=(28, 28, 1), embedding_dim=128)
    siamese.compile_model(learning_rate=0.001)

    # Print model summary
    print("\n" + "=" * 60)
    print("EMBEDDING NETWORK ARCHITECTURE")
    print("=" * 60)
    siamese.embedding_network.summary()

    print("\n" + "=" * 60)
    print("SIAMESE MODEL ARCHITECTURE")
    print("=" * 60)
    siamese.model.summary()

    # Train model
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = siamese.train(
        pairs_train, labels_train,
        pairs_val, labels_val,
        epochs=20,
        batch_size=32
    )

    # Test similarity predictions
    print("\n" + "=" * 60)
    print("SIMILARITY EXAMPLES")
    print("=" * 60)

    # Similar pairs
    for i in range(3):
        class_images = X_test[y_test == i][:2]
        if len(class_images) == 2:
            distance = siamese.predict_similarity(class_images[0], class_images[1])
            print(f"\nSimilar pair (Class {i}):")
            print(f"  Distance: {distance:.4f}")

    # Dissimilar pairs
    for i in range(3):
        class_indices = [(i, (i+1) % 3)]
        for idx1, idx2 in class_indices:
            img1 = X_test[y_test == idx1][0]
            img2 = X_test[y_test == idx2][0]
            distance = siamese.predict_similarity(img1, img2)
            print(f"\nDissimilar pair (Class {idx1} vs Class {idx2}):")
            print(f"  Distance: {distance:.4f}")

    # Visualize results
    visualize_results(siamese, X_test, y_test, history)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Embedding Dimension: {siamese.embedding_dim}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy_metric'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy_metric'][-1]:.4f}")
    print(f"Total Training Epochs: {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
