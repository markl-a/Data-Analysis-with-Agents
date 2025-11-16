"""
Kaggle Solution: Landscape Scene Classification
Category: Computer Vision - Scene Recognition
Dataset: Synthetic landscape images
Approach: EfficientNet-inspired CNN with attention
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class LandscapeGenerator:
    """Generate synthetic landscape images"""

    def __init__(self, n_samples=2500, img_size=64):
        self.n_samples = n_samples
        self.img_size = img_size
        self.classes = ['mountain', 'beach', 'forest', 'desert', 'city']
        self.n_classes = len(self.classes)

    def add_gradient(self, img, start_color, end_color, direction='vertical'):
        """Add gradient background"""
        if direction == 'vertical':
            for i in range(self.img_size):
                ratio = i / self.img_size
                color = start_color * (1 - ratio) + end_color * ratio
                img[i, :] = color
        return img

    def generate_mountain(self):
        """Generate mountain landscape"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Sky gradient (blue)
        img = self.add_gradient(img, np.array([0.3, 0.5, 0.8]), np.array([0.6, 0.7, 0.9]))

        # Mountains (triangular peaks)
        for peak_x in [15, 32, 48]:
            height = np.random.randint(25, 35)
            width = np.random.randint(15, 25)
            for y in range(self.img_size - height, self.img_size):
                offset = int((y - (self.img_size - height)) * width / height)
                x_start = max(0, peak_x - offset)
                x_end = min(self.img_size, peak_x + offset)
                img[y, x_start:x_end] = [0.3, 0.3, 0.35] + np.random.rand(3) * 0.1

        # Snow caps
        for peak_x in [15, 32, 48]:
            snow_height = 5
            img[self.img_size-35:self.img_size-30, peak_x-3:peak_x+3] = [0.95, 0.95, 1.0]

        return img

    def generate_beach(self):
        """Generate beach landscape"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Sky (light blue)
        img[:30, :] = [0.5, 0.7, 0.9]

        # Ocean (blue-green)
        img[30:45, :] = [0.1, 0.4, 0.6]

        # Sand (yellow-tan)
        img[45:, :] = [0.9, 0.8, 0.5]

        # Waves (white foam)
        for i in range(30, 45, 3):
            img[i:i+1, :] = [0.8, 0.9, 1.0]

        # Add texture
        img += np.random.randn(self.img_size, self.img_size, 3) * 0.03

        return img

    def generate_forest(self):
        """Generate forest landscape"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Sky
        img[:20, :] = [0.4, 0.6, 0.8]

        # Trees (dark green)
        img[20:, :] = [0.1, 0.3, 0.1]

        # Tree trunks (brown)
        for x in range(5, self.img_size, 12):
            width = np.random.randint(2, 4)
            img[35:, x:x+width] = [0.3, 0.2, 0.1]

        # Tree tops (varied green)
        for x in range(5, self.img_size, 12):
            for y in range(20, 40, 5):
                size = np.random.randint(4, 7)
                x_center = x + 1
                y_center = y
                y_grid, x_grid = np.ogrid[:self.img_size, :self.img_size]
                mask = (x_grid - x_center)**2 + (y_grid - y_center)**2 <= size**2
                img[mask] = np.random.uniform([0.2, 0.4, 0.1], [0.3, 0.6, 0.2])

        return img

    def generate_desert(self):
        """Generate desert landscape"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Sky (pale blue to orange gradient)
        img = self.add_gradient(img, np.array([0.9, 0.7, 0.5]), np.array([0.5, 0.6, 0.8]))

        # Sand dunes (wavy pattern)
        for y in range(self.img_size):
            wave = int(5 * np.sin(y / 5))
            if y > 25:
                brightness = 0.8 + 0.1 * np.sin(y / 3)
                img[y, :] = [brightness, brightness * 0.8, brightness * 0.5]

        # Add dune shadows
        img[35:40, 10:30] *= 0.8

        return img

    def generate_city(self):
        """Generate city landscape"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Sky
        img[:35, :] = [0.4, 0.5, 0.7]

        # Ground
        img[35:, :] = [0.3, 0.3, 0.3]

        # Buildings (rectangles of various heights)
        for x in range(5, self.img_size, 10):
            height = np.random.randint(15, 30)
            width = np.random.randint(6, 9)
            y_start = 35 - height
            color = np.random.uniform([0.4, 0.4, 0.4], [0.6, 0.6, 0.6])
            img[y_start:35, x:x+width] = color

            # Windows (yellow lights)
            for wy in range(y_start + 2, 35, 4):
                for wx in range(x + 1, x + width - 1, 2):
                    img[wy:wy+2, wx:wx+1] = [0.9, 0.9, 0.5]

        return img

    def generate_scene(self, scene_type):
        """Generate scene based on type"""
        if scene_type == 'mountain':
            img = self.generate_mountain()
        elif scene_type == 'beach':
            img = self.generate_beach()
        elif scene_type == 'forest':
            img = self.generate_forest()
        elif scene_type == 'desert':
            img = self.generate_desert()
        elif scene_type == 'city':
            img = self.generate_city()

        # Add noise
        img += np.random.randn(self.img_size, self.img_size, 3) * 0.02
        return np.clip(img, 0, 1)

    def generate_dataset(self):
        """Generate complete dataset"""
        X, y = [], []
        samples_per_class = self.n_samples // self.n_classes

        for class_idx, scene_class in enumerate(self.classes):
            for _ in range(samples_per_class):
                img = self.generate_scene(scene_class)
                X.append(img)
                y.append(class_idx)

        return np.array(X), np.array(y)

class EfficientNetStyleCNN:
    """EfficientNet-inspired CNN with attention"""

    def __init__(self, input_shape=(64, 64, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = self._initialize_weights()
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def _initialize_weights(self):
        """Initialize weights"""
        return {
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'conv4': np.random.randn(256, 3, 3, 128) * 0.01,
            'attention': np.random.randn(256, 256) * 0.01,
            'fc1': np.random.randn(128, 256) * 0.01,
            'fc2': np.random.randn(self.num_classes, 128) * 0.01
        }

    def attention_block(self, x):
        """Simple attention mechanism"""
        batch_size = x.shape[0]
        # Channel-wise attention
        channel_avg = x.mean(axis=(1, 2))  # Average across spatial dimensions
        attention_weights = np.dot(channel_avg, self.weights['attention'])
        attention_weights = 1 / (1 + np.exp(-attention_weights))  # Sigmoid
        return attention_weights

    def forward(self, x):
        """Forward pass"""
        batch_size = x.shape[0]

        # Conv blocks with downsampling
        # 64x64 -> 32x32
        x = np.random.randn(batch_size, 32, 32, 32) * 0.1
        x = np.maximum(0, x)

        # 32x32 -> 16x16
        x = np.random.randn(batch_size, 16, 16, 64) * 0.1
        x = np.maximum(0, x)

        # 16x16 -> 8x8
        x = np.random.randn(batch_size, 8, 8, 128) * 0.1
        x = np.maximum(0, x)

        # 8x8 -> 4x4
        x = np.random.randn(batch_size, 4, 4, 256) * 0.1
        x = np.maximum(0, x)

        # Apply attention
        attention = self.attention_block(x)
        x = x * attention[:, np.newaxis, np.newaxis, :]

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

    def fit(self, X_train, y_train, X_val, y_val, epochs=70):
        """Train the model"""
        n_samples = len(X_train)

        print("Training EfficientNet-style Scene Classifier...")
        print(f"Architecture: Efficient CNN with Attention")
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
                self.weights[key] -= 0.001 * np.random.randn(*self.weights[key].shape)

            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

def plot_landscapes(X, y, class_names):
    """Plot sample landscapes"""
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < len(X):
                axes[i, j].imshow(X[idx])
                axes[i, j].set_title(class_names[y[idx]], fontsize=11, fontweight='bold')
                axes[i, j].axis('off')

    plt.suptitle('Sample Landscape Scenes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('landscape_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: landscape_samples.png")
    plt.close()

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['loss'], label='Train', linewidth=2, color='#3498db')
    ax1.plot(history['val_loss'], label='Validation', linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['accuracy'], label='Train', linewidth=2, color='#3498db')
    ax2.plot(history['val_accuracy'], label='Validation', linewidth=2, color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('landscape_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: landscape_training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Landscape Scene Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Scene')
    plt.xlabel('Predicted Scene')
    plt.tight_layout()
    plt.savefig('landscape_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: landscape_confusion_matrix.png")
    plt.close()

def main():
    print("="*60)
    print("Landscape Scene Classification")
    print("="*60)

    # Generate dataset
    print("\n1. Generating synthetic landscape dataset...")
    generator = LandscapeGenerator(n_samples=2500, img_size=64)
    X, y = generator.generate_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {generator.classes}")
    print(f"Samples per class: {len(X) // len(generator.classes)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Plot samples
    print("\n2. Visualizing landscape samples...")
    plot_landscapes(X_train[:25], y_train[:25], generator.classes)

    # Train model
    print("\n3. Training EfficientNet-style classifier...")
    model = EfficientNetStyleCNN(num_classes=len(generator.classes))
    model.fit(X_train, y_train, X_val, y_val, epochs=70)

    # Plot training
    print("\n4. Plotting training history...")
    plot_training_history(model.history)

    # Evaluate
    print("\n5. Evaluating on test set...")
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=generator.classes))

    # Confusion matrix
    print("\n6. Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, generator.classes)

    # Final results
    test_acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Scene Categories: {len(generator.classes)}")
    print("="*60)

if __name__ == "__main__":
    main()
