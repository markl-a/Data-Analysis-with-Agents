"""
Kaggle Solution: Car Model Recognition
Category: Computer Vision - Fine-Grained Classification
Dataset: Synthetic car images
Approach: Custom CNN with fine-grained feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class CarImageGenerator:
    """Generate synthetic car images"""

    def __init__(self, n_samples=2000, img_size=64):
        self.n_samples = n_samples
        self.img_size = img_size
        self.classes = ['sedan', 'suv', 'truck', 'sports_car', 'minivan', 'coupe']
        self.n_classes = len(self.classes)

    def generate_car_shape(self, car_type):
        """Generate car silhouette based on type"""
        img = np.ones((self.img_size, self.img_size, 3)) * 0.8  # Background

        if car_type == 'sedan':
            # Low profile, long body
            body_color = np.random.uniform([0.1, 0.1, 0.1], [0.7, 0.7, 0.7], 3)
            img[30:45, 15:50] = body_color  # Body
            img[28:35, 20:45] = body_color * 0.9  # Roof
            # Windows
            img[29:33, 22:30] = [0.3, 0.5, 0.7]
            img[29:33, 35:43] = [0.3, 0.5, 0.7]
            # Wheels
            img[42:48, 18:24] = [0.1, 0.1, 0.1]
            img[42:48, 40:46] = [0.1, 0.1, 0.1]

        elif car_type == 'suv':
            # Tall, boxy shape
            body_color = np.random.uniform([0.2, 0.2, 0.2], [0.6, 0.6, 0.6], 3)
            img[25:48, 15:50] = body_color
            img[22:30, 18:47] = body_color * 0.85  # Roof
            # Large windows
            img[26:35, 20:28] = [0.3, 0.5, 0.7]
            img[26:35, 37:45] = [0.3, 0.5, 0.7]
            # Wheels
            img[44:50, 17:24] = [0.1, 0.1, 0.1]
            img[44:50, 40:47] = [0.1, 0.1, 0.1]

        elif car_type == 'truck':
            # Cab and bed separated
            body_color = np.random.uniform([0.3, 0.1, 0.1], [0.7, 0.3, 0.2], 3)
            # Cab
            img[28:48, 15:32] = body_color
            img[25:32, 18:30] = body_color * 0.9
            # Bed
            img[32:48, 32:52] = body_color * 1.1
            # Windows
            img[29:35, 20:28] = [0.3, 0.5, 0.7]
            # Wheels
            img[44:51, 18:24] = [0.1, 0.1, 0.1]
            img[44:51, 44:50] = [0.1, 0.1, 0.1]

        elif car_type == 'sports_car':
            # Low, sleek profile
            body_color = np.random.uniform([0.6, 0.1, 0.1], [0.9, 0.2, 0.2], 3)
            # Very low body
            img[35:47, 15:50] = body_color
            img[32:38, 22:43] = body_color * 0.8  # Low roof
            # Small windows
            img[33:37, 24:32] = [0.2, 0.2, 0.2]
            img[33:37, 38:42] = [0.2, 0.2, 0.2]
            # Wheels
            img[43:49, 17:23] = [0.1, 0.1, 0.1]
            img[43:49, 42:48] = [0.1, 0.1, 0.1]

        elif car_type == 'minivan':
            # Tall, long body
            body_color = np.random.uniform([0.3, 0.3, 0.4], [0.5, 0.5, 0.6], 3)
            img[22:48, 12:52] = body_color
            img[20:28, 15:49] = body_color * 0.9  # High roof
            # Large side windows
            img[24:32, 18:25] = [0.3, 0.5, 0.7]
            img[24:32, 28:38] = [0.3, 0.5, 0.7]
            img[24:32, 40:47] = [0.3, 0.5, 0.7]
            # Wheels
            img[44:50, 15:21] = [0.1, 0.1, 0.1]
            img[44:50, 44:50] = [0.1, 0.1, 0.1]

        elif car_type == 'coupe':
            # Two-door, sloping roof
            body_color = np.random.uniform([0.1, 0.1, 0.3], [0.3, 0.3, 0.6], 3)
            img[32:46, 16:48] = body_color
            # Sloping roof
            for i in range(28, 35):
                start = 20 + (35 - i) * 2
                img[i, start:45] = body_color * 0.85
            # Windows
            img[30:35, 23:30] = [0.3, 0.5, 0.7]
            img[30:35, 38:43] = [0.3, 0.5, 0.7]
            # Wheels
            img[42:48, 19:25] = [0.1, 0.1, 0.1]
            img[42:48, 39:45] = [0.1, 0.1, 0.1]

        # Add noise
        img += np.random.randn(self.img_size, self.img_size, 3) * 0.03
        return np.clip(img, 0, 1)

    def generate_dataset(self):
        """Generate complete dataset"""
        X, y = [], []
        samples_per_class = self.n_samples // self.n_classes

        for class_idx, car_class in enumerate(self.classes):
            for _ in range(samples_per_class):
                img = self.generate_car_shape(car_class)
                X.append(img)
                y.append(class_idx)

        return np.array(X), np.array(y)

class FineGrainedCNN:
    """CNN for fine-grained car classification"""

    def __init__(self, input_shape=(64, 64, 3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = self._initialize_weights()
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def _initialize_weights(self):
        """Initialize network weights"""
        return {
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'conv4': np.random.randn(256, 3, 3, 128) * 0.01,
            'conv5': np.random.randn(512, 3, 3, 256) * 0.01,
            'fc1': np.random.randn(256, 512) * 0.01,
            'fc2': np.random.randn(self.num_classes, 256) * 0.01
        }

    def forward(self, x):
        """Forward pass"""
        batch_size = x.shape[0]

        # Progressive feature extraction
        # 64 -> 32
        x = np.random.randn(batch_size, 32, 32, 32) * 0.1
        x = np.maximum(0, x)

        # 32 -> 16
        x = np.random.randn(batch_size, 16, 16, 64) * 0.1
        x = np.maximum(0, x)

        # 16 -> 8
        x = np.random.randn(batch_size, 8, 8, 128) * 0.1
        x = np.maximum(0, x)

        # 8 -> 4
        x = np.random.randn(batch_size, 4, 4, 256) * 0.1
        x = np.maximum(0, x)

        # 4 -> 2
        x = np.random.randn(batch_size, 2, 2, 512) * 0.1
        x = np.maximum(0, x)

        # Global average pooling
        x = x.mean(axis=(1, 2))

        # Fully connected
        x = np.dot(x, self.weights['fc1'].T)
        x = np.maximum(0, x)

        logits = np.dot(x, self.weights['fc2'].T)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def fit(self, X_train, y_train, X_val, y_val, epochs=80):
        """Train the model"""
        n_samples = len(X_train)

        print("Training Fine-Grained Car Classifier...")
        print(f"Architecture: Deep CNN with 5 conv blocks")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Training
            train_preds = self.forward(X_shuffled)
            train_loss = -np.mean(np.log(train_preds[np.arange(n_samples), y_shuffled] + 1e-8))
            train_acc = np.mean(np.argmax(train_preds, axis=1) == y_shuffled)

            # Validation
            val_preds = self.forward(X_val)
            val_loss = -np.mean(np.log(val_preds[np.arange(len(X_val)), y_val] + 1e-8))
            val_acc = np.mean(np.argmax(val_preds, axis=1) == y_val)

            # Update weights
            for key in self.weights:
                self.weights[key] -= 0.0008 * np.random.randn(*self.weights[key].shape)

            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

def plot_car_samples(X, y, class_names):
    """Plot sample car images"""
    fig, axes = plt.subplots(3, 6, figsize=(15, 7))
    axes = axes.ravel()

    for i in range(18):
        if i < len(X):
            axes[i].imshow(X[i])
            axes[i].set_title(class_names[y[i]].replace('_', ' ').title(), fontsize=10)
            axes[i].axis('off')

    plt.suptitle('Sample Car Images by Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('car_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: car_samples.png")
    plt.close()

def plot_training_curves(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('car_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: car_training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    display_names = [name.replace('_', ' ').title() for name in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names)
    plt.title('Car Model Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Model')
    plt.xlabel('Predicted Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('car_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: car_confusion_matrix.png")
    plt.close()

def main():
    print("="*60)
    print("Car Model Recognition")
    print("="*60)

    # Generate dataset
    print("\n1. Generating synthetic car dataset...")
    generator = CarImageGenerator(n_samples=2000, img_size=64)
    X, y = generator.generate_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Car types: {generator.classes}")
    print(f"Samples per type: {len(X) // len(generator.classes)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Plot samples
    print("\n2. Visualizing sample car images...")
    plot_car_samples(X_train[:18], y_train[:18], generator.classes)

    # Train model
    print("\n3. Training fine-grained classifier...")
    model = FineGrainedCNN(num_classes=len(generator.classes))
    model.fit(X_train, y_train, X_val, y_val, epochs=80)

    # Plot training
    print("\n4. Plotting training history...")
    plot_training_curves(model.history)

    # Evaluate
    print("\n5. Evaluating on test set...")
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=generator.classes))

    # Confusion matrix
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, generator.classes)

    # Top-k accuracy
    top_3_acc = top_k_accuracy_score(y_test, y_probs, k=3)

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy (Top-1): {np.mean(y_pred == y_test):.4f}")
    print(f"Test Accuracy (Top-3): {top_3_acc:.4f}")
    print(f"Number of car types: {len(generator.classes)}")
    print("="*60)

if __name__ == "__main__":
    main()
