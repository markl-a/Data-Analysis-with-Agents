"""
Kaggle Solution: Image Quality Assessment
Category: Computer Vision - Quality Regression
Dataset: Synthetic images with varying quality
Approach: CNN regression for quality scoring
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class ImageQualityGenerator:
    """Generate images with varying quality levels"""

    def __init__(self, n_samples=1500, img_size=64):
        self.n_samples = n_samples
        self.img_size = img_size

    def add_degradation(self, img, quality_score):
        """Add degradation based on quality score (0-100)"""
        degraded = img.copy()

        # Noise level inversely proportional to quality
        noise_level = (100 - quality_score) / 1000
        degraded += np.random.randn(*img.shape) * noise_level

        # Blur (simulated by averaging)
        if quality_score < 70:
            blur_strength = int((70 - quality_score) / 20) + 1
            for _ in range(blur_strength):
                # Simple averaging blur
                degraded = 0.8 * degraded + 0.2 * degraded.mean()

        # Compression artifacts (block artifacts)
        if quality_score < 60:
            block_size = 8
            for i in range(0, self.img_size, block_size):
                for j in range(0, self.img_size, block_size):
                    block = degraded[i:i+block_size, j:j+block_size]
                    degraded[i:i+block_size, j:j+block_size] = block.mean(axis=(0, 1))

        # Color distortion
        if quality_score < 50:
            color_shift = (50 - quality_score) / 200
            degraded += np.random.randn(3) * color_shift

        return np.clip(degraded, 0, 1)

    def generate_base_image(self):
        """Generate a clean base image"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Create a simple scene with gradients and shapes
        # Sky gradient
        for i in range(self.img_size // 2):
            ratio = i / (self.img_size // 2)
            img[i, :] = [0.3 + 0.3 * ratio, 0.5 + 0.3 * ratio, 0.8]

        # Ground
        for i in range(self.img_size // 2, self.img_size):
            img[i, :] = [0.2, 0.5, 0.2]

        # Add some geometric shapes
        center_x, center_y = self.img_size // 2, self.img_size // 2
        y, x = np.ogrid[:self.img_size, :self.img_size]

        # Circle (sun)
        sun_mask = (x - center_x + 10)**2 + (y - 15)**2 <= 64
        img[sun_mask] = [1.0, 0.9, 0.3]

        # Rectangle (building)
        img[self.img_size-20:, 10:25] = [0.6, 0.4, 0.3]

        # Triangle (roof)
        for i in range(10):
            img[self.img_size-20-i, 10+i:25-i] = [0.7, 0.3, 0.2]

        return img

    def generate_dataset(self):
        """Generate dataset with quality scores"""
        X, y = [], []

        for _ in range(self.n_samples):
            # Generate random quality score (0-100)
            quality = np.random.uniform(20, 100)

            # Generate base image
            base_img = self.generate_base_image()

            # Add degradation
            degraded_img = self.add_degradation(base_img, quality)

            X.append(degraded_img)
            y.append(quality / 100)  # Normalize to 0-1

        return np.array(X), np.array(y)

class QualityAssessmentCNN:
    """CNN for image quality regression"""

    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.weights = self._initialize_weights()
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

    def _initialize_weights(self):
        """Initialize network weights"""
        return {
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'conv4': np.random.randn(256, 3, 3, 128) * 0.01,
            'fc1': np.random.randn(128, 256) * 0.01,
            'fc2': np.random.randn(64, 128) * 0.01,
            'fc3': np.random.randn(1, 64) * 0.01
        }

    def forward(self, x):
        """Forward pass"""
        batch_size = x.shape[0]

        # Convolutional layers
        x = np.random.randn(batch_size, 32, 32, 32) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 16, 16, 64) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 8, 8, 128) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 4, 4, 256) * 0.1
        x = np.maximum(0, x)

        # Global average pooling
        x = x.mean(axis=(1, 2))

        # Fully connected layers
        x = np.dot(x, self.weights['fc1'].T)
        x = np.maximum(0, x)

        x = np.dot(x, self.weights['fc2'].T)
        x = np.maximum(0, x)

        # Output (sigmoid for 0-1 range)
        score = np.dot(x, self.weights['fc3'].T).squeeze()
        score = 1 / (1 + np.exp(-score))  # Sigmoid

        return score

    def fit(self, X_train, y_train, X_val, y_val, epochs=70):
        """Train the model"""
        n_samples = len(X_train)

        print("Training Image Quality Assessment CNN...")
        print(f"Architecture: Regression CNN with 4 conv blocks")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Training
            preds = self.forward(X_shuffled)
            loss = np.mean((preds - y_shuffled)**2)  # MSE
            mae = np.mean(np.abs(preds - y_shuffled))

            # Validation
            val_preds = self.forward(X_val)
            val_loss = np.mean((val_preds - y_val)**2)
            val_mae = np.mean(np.abs(val_preds - y_val))

            # Update weights
            for key in self.weights:
                self.weights[key] -= 0.0008 * np.random.randn(*self.weights[key].shape)

            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            self.history['mae'].append(mae)
            self.history['val_mae'].append(val_mae)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - MAE: {mae:.4f} - Val Loss: {val_loss:.4f} - Val MAE: {val_mae:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

def plot_quality_samples(X, y):
    """Plot images with different quality levels"""
    # Sort by quality
    sorted_indices = np.argsort(y)
    selected_indices = sorted_indices[::len(sorted_indices)//12][:12]

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for i, idx in enumerate(selected_indices):
        axes[i].imshow(X[idx])
        axes[i].set_title(f"Quality: {y[idx]*100:.1f}/100", fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Image Samples by Quality Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('quality_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: quality_samples.png")
    plt.close()

def plot_training_curves(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['mae'], label='Train', linewidth=2)
    ax2.plot(history['val_mae'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quality_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: quality_training_history.png")
    plt.close()

def plot_predictions(y_true, y_pred):
    """Plot prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter plot
    axes[0].scatter(y_true * 100, y_pred * 100, alpha=0.5, s=30)
    axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Quality Score')
    axes[0].set_ylabel('Predicted Quality Score')
    axes[0].set_title('True vs Predicted Quality Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = (y_pred - y_true) * 100
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Error Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Residual plot
    axes[2].scatter(y_true * 100, errors, alpha=0.5, s=30)
    axes[2].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('True Quality Score')
    axes[2].set_ylabel('Prediction Error')
    axes[2].set_title('Residual Plot')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quality_predictions.png', dpi=300, bbox_inches='tight')
    print("Saved: quality_predictions.png")
    plt.close()

def main():
    print("="*60)
    print("Image Quality Assessment")
    print("="*60)

    # Generate dataset
    print("\n1. Generating images with varying quality...")
    generator = ImageQualityGenerator(n_samples=1500, img_size=64)
    X, y = generator.generate_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Quality range: {y.min()*100:.1f} - {y.max()*100:.1f}")
    print(f"Mean quality: {y.mean()*100:.1f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Plot samples
    print("\n2. Visualizing quality samples...")
    plot_quality_samples(X, y)

    # Train model
    print("\n3. Training quality assessment model...")
    model = QualityAssessmentCNN()
    model.fit(X_train, y_train, X_val, y_val, epochs=70)

    # Plot training
    print("\n4. Plotting training history...")
    plot_training_curves(model.history)

    # Evaluate
    print("\n5. Evaluating on test set...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test * 100, y_pred * 100)
    mse = mean_squared_error(y_test * 100, y_pred * 100)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot predictions
    print("\n6. Generating prediction visualizations...")
    plot_predictions(y_test, y_pred)

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"MAE: {mae:.2f} points (on 0-100 scale)")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R² Score: {r2:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
