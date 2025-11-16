"""
Transfer Learning with Pre-trained Models - Kaggle Solution
==========================================================
Use pre-trained models for image classification with transfer learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class TransferLearningClassifier:
    """Transfer learning classifier using pre-trained models."""

    def __init__(self, base_model_name='MobileNetV2', num_classes=3,
                 input_shape=(224, 224, 3), fine_tune_layers=0):
        """Initialize transfer learning classifier.

        Args:
            base_model_name: Name of base model (MobileNetV2, VGG16, ResNet50)
            num_classes: Number of output classes
            input_shape: Shape of input images
            fine_tune_layers: Number of layers to fine-tune (0 = freeze all)
        """
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.fine_tune_layers = fine_tune_layers

        # Build model
        self.model = self.build_model()

    def build_model(self):
        """Build transfer learning model.

        Returns:
            Transfer learning model
        """
        # Load pre-trained base model
        if self.base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.base_model_name == 'VGG16':
            base_model = VGG16(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {self.base_model_name}")

        # Freeze base model layers
        base_model.trainable = False

        # Build model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='transfer_learning_model')

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def unfreeze_base_model(self, fine_tune_layers):
        """Unfreeze top layers of base model for fine-tuning.

        Args:
            fine_tune_layers: Number of layers to unfreeze
        """
        # Get base model (first layer)
        base_model = self.model.layers[0]

        # Unfreeze last N layers
        if fine_tune_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False

            # Recompile with lower learning rate for fine-tuning
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            print(f"\nUnfrozen last {fine_tune_layers} layers for fine-tuning")

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print(f"\nStarting training with {self.base_model_name}...")
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

    def predict(self, X):
        """Make predictions.

        Args:
            X: Input images

        Returns:
            Predictions
        """
        return self.model.predict(X, verbose=0)


def create_synthetic_dataset(n_samples=300, img_size=224):
    """Create synthetic image dataset with 3 classes.

    Args:
        n_samples: Number of samples per class
        img_size: Image size

    Returns:
        images, labels
    """
    print(f"Creating synthetic dataset ({n_samples} samples per class)...")

    images = []
    labels = []

    # Class 0: Images with mostly red channel
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size, 3))
        img[:, :, 0] = np.random.uniform(0.6, 1.0, (img_size, img_size))  # Red
        img[:, :, 1] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Green
        img[:, :, 2] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Blue

        # Add some random shapes
        center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
        center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
        radius = np.random.randint(20, 50)

        y, x = np.ogrid[:img_size, :img_size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = [0.9, 0.1, 0.1]

        images.append(img)
        labels.append(0)

    # Class 1: Images with mostly green channel
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size, 3))
        img[:, :, 0] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Red
        img[:, :, 1] = np.random.uniform(0.6, 1.0, (img_size, img_size))  # Green
        img[:, :, 2] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Blue

        # Add rectangle
        x1 = np.random.randint(20, 80)
        y1 = np.random.randint(20, 80)
        x2 = np.random.randint(x1 + 50, img_size - 20)
        y2 = np.random.randint(y1 + 50, img_size - 20)
        img[y1:y2, x1:x2] = [0.1, 0.9, 0.1]

        images.append(img)
        labels.append(1)

    # Class 2: Images with mostly blue channel
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size, 3))
        img[:, :, 0] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Red
        img[:, :, 1] = np.random.uniform(0.0, 0.3, (img_size, img_size))  # Green
        img[:, :, 2] = np.random.uniform(0.6, 1.0, (img_size, img_size))  # Blue

        # Add diagonal line
        for i in range(0, img_size, 2):
            if i + 10 < img_size:
                img[i:i+10, i:i+10] = [0.1, 0.1, 0.9]

        images.append(img)
        labels.append(2)

    images = np.array(images)
    labels = np.array(labels)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    print(f"Created dataset with {len(images)} images")
    return images, labels


def visualize_results(X_test, y_test, predictions, history, class_names):
    """Visualize results.

    Args:
        X_test: Test images
        y_test: True labels
        predictions: Predicted labels
        history: Training history
        class_names: Class names
    """
    print("\nGenerating visualizations...")

    # Plot sample predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        axes[i].imshow(X_test[i])
        pred_class = np.argmax(predictions[i])
        true_class = y_test[i]

        color = 'green' if pred_class == true_class else 'red'
        axes[i].set_title(
            f'True: {class_names[true_class]}\nPred: {class_names[pred_class]}',
            color=color, fontsize=10
        )
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transfer_learning_predictions.png', dpi=300, bbox_inches='tight')
    print("Predictions saved to 'transfer_learning_predictions.png'")

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
    plt.savefig('transfer_learning_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'transfer_learning_curves.png'")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Transfer Learning with Pre-trained Models - Kaggle Solution")
    print("=" * 60)

    # Create synthetic dataset
    images, labels = create_synthetic_dataset(n_samples=300, img_size=224)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Test: {len(X_test)} images")

    class_names = ['Red Class', 'Green Class', 'Blue Class']

    # Initialize model
    print("\nInitializing transfer learning model...")
    classifier = TransferLearningClassifier(
        base_model_name='MobileNetV2',
        num_classes=3,
        input_shape=(224, 224, 3),
        fine_tune_layers=0
    )

    # Print model summary
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    classifier.model.summary()

    # Train model
    print("\n" + "=" * 60)
    print("TRAINING (Feature Extraction)")
    print("=" * 60)
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    test_loss, test_acc = classifier.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Make predictions
    predictions = classifier.predict(X_test)

    # Visualize results
    visualize_results(X_test, y_test, predictions, history, class_names)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Base Model: {classifier.base_model_name}")
    print(f"Number of Classes: {classifier.num_classes}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Total Training Epochs: {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
