"""
MNIST Digit Classification - Computer Vision Introduction

This module implements handwritten digit recognition using both traditional
machine learning and deep learning approaches.

Dataset: https://www.kaggle.com/competitions/digit-recognizer
Difficulty: ⭐⭐ Intermediate Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class MNISTClassifier:
    """MNIST Handwritten Digit Classifier."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None

    def create_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic MNIST-like dataset."""
        np.random.seed(42)
        n_samples = 5000
        n_pixels = 784  # 28 x 28

        # Generate patterns for each digit
        images = []
        labels = []

        for _ in range(n_samples):
            digit = np.random.randint(0, 10)

            # Create a 28x28 image with noise
            image = np.zeros((28, 28))

            # Add digit-specific patterns
            if digit == 0:
                # Circle-like pattern
                for i in range(5, 23):
                    for j in range(5, 23):
                        if 8 <= ((i-14)**2 + (j-14)**2)**0.5 <= 12:
                            image[i, j] = 200 + np.random.randint(-30, 30)
            elif digit == 1:
                # Vertical line
                image[4:24, 12:16] = 200 + np.random.randint(-30, 30, (20, 4))
            elif digit == 2:
                # S-like curve
                image[4:8, 8:20] = 200
                image[8:14, 16:20] = 200
                image[12:16, 8:20] = 200
                image[14:20, 8:12] = 200
                image[20:24, 8:20] = 200
            elif digit == 3:
                # Three horizontal lines connected
                image[4:8, 8:20] = 200
                image[12:16, 8:20] = 200
                image[20:24, 8:20] = 200
                image[4:24, 16:20] = 200
            elif digit == 4:
                # L-shape with vertical
                image[4:16, 6:10] = 200
                image[12:16, 6:22] = 200
                image[4:24, 18:22] = 200
            elif digit == 5:
                # S-like (reversed)
                image[4:8, 8:20] = 200
                image[4:14, 8:12] = 200
                image[12:16, 8:20] = 200
                image[14:24, 16:20] = 200
                image[20:24, 8:20] = 200
            elif digit == 6:
                # Circle with tail
                for i in range(10, 24):
                    for j in range(6, 22):
                        if 5 <= ((i-17)**2 + (j-14)**2)**0.5 <= 8:
                            image[i, j] = 200
                image[4:17, 6:10] = 200
            elif digit == 7:
                # Angle
                image[4:8, 6:22] = 200
                image[4:24, 18:22] = 200
            elif digit == 8:
                # Two circles
                for i in range(4, 14):
                    for j in range(6, 22):
                        if 3 <= ((i-9)**2 + (j-14)**2)**0.5 <= 6:
                            image[i, j] = 200
                for i in range(14, 24):
                    for j in range(6, 22):
                        if 3 <= ((i-19)**2 + (j-14)**2)**0.5 <= 6:
                            image[i, j] = 200
            elif digit == 9:
                # Circle with tail going down
                for i in range(4, 16):
                    for j in range(6, 22):
                        if 4 <= ((i-10)**2 + (j-14)**2)**0.5 <= 7:
                            image[i, j] = 200
                image[10:24, 16:20] = 200

            # Add noise
            noise = np.random.normal(0, 20, (28, 28))
            image = np.clip(image + noise, 0, 255)

            # Random slight rotation effect (shift some pixels)
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            image = np.roll(np.roll(image, shift_x, axis=1), shift_y, axis=0)

            images.append(image.flatten())
            labels.append(digit)

        return np.array(images), np.array(labels)

    def plot_samples(self, X: np.ndarray, y: np.ndarray, output_dir: str = '.') -> None:
        """Visualize sample digits."""
        fig, axes = plt.subplots(4, 10, figsize=(15, 6))
        fig.suptitle('MNIST Sample Images', fontsize=14)

        for digit in range(10):
            # Get samples of this digit
            digit_indices = np.where(y == digit)[0][:4]

            for i, idx in enumerate(digit_indices):
                ax = axes[i, digit]
                img = X[idx].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_title(str(digit))

        plt.tight_layout()
        plt.savefig(f'{output_dir}/mnist_samples.png', dpi=300, bbox_inches='tight')
        print(f"Samples saved to {output_dir}/mnist_samples.png")
        plt.close()

    def plot_analysis(self, X: np.ndarray, y: np.ndarray, output_dir: str = '.') -> None:
        """Generate dataset analysis visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('MNIST Dataset Analysis', fontsize=16)

        # Digit distribution
        unique, counts = np.unique(y, return_counts=True)
        axes[0, 0].bar(unique, counts, color='steelblue')
        axes[0, 0].set_title('Digit Distribution')
        axes[0, 0].set_xlabel('Digit')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(10))

        # Average pixel intensity
        pixel_means = X.mean(axis=0).reshape(28, 28)
        im = axes[0, 1].imshow(pixel_means, cmap='hot')
        axes[0, 1].set_title('Average Pixel Intensity')
        plt.colorbar(im, ax=axes[0, 1])

        # Pixel variance
        pixel_vars = X.var(axis=0).reshape(28, 28)
        im = axes[0, 2].imshow(pixel_vars, cmap='hot')
        axes[0, 2].set_title('Pixel Variance')
        plt.colorbar(im, ax=axes[0, 2])

        # Average image per digit
        avg_images = []
        for digit in range(10):
            digit_images = X[y == digit]
            avg_images.append(digit_images.mean(axis=0).reshape(28, 28))

        # Show first 5 average digits
        for i in range(5):
            axes[1, 0].imshow(np.hstack([avg_images[i*2], avg_images[i*2+1]]),
                             cmap='gray' if i == 0 else None)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Average Digit Images')

        # Pixel distribution
        axes[1, 1].hist(X.flatten(), bins=50, color='gray', alpha=0.7)
        axes[1, 1].set_title('Pixel Value Distribution')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')

        # Non-zero pixels per image
        non_zero_counts = (X > 50).sum(axis=1)
        axes[1, 2].hist(non_zero_counts, bins=30, color='coral')
        axes[1, 2].set_title('Non-Zero Pixels per Image')
        axes[1, 2].set_xlabel('Count')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/mnist_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/mnist_analysis.png")
        plt.close()

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train multiple classifiers."""
        # Normalize data
        X_scaled = X_train / 255.0

        print("\nTraining models...")

        # Logistic Regression
        print("  - Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        )
        self.models['Logistic Regression'].fit(X_scaled, y_train)

        # KNN
        print("  - Training KNN...")
        self.models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        self.models['KNN'].fit(X_scaled, y_train)

        # Random Forest
        print("  - Training Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_scaled, y_train)

        # MLP
        print("  - Training MLP...")
        self.models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=100,
            random_state=42, early_stopping=True
        )
        self.models['MLP'].fit(X_scaled, y_train)

        if XGBOOST_AVAILABLE:
            print("  - Training XGBoost...")
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='mlogloss'
            )
            self.models['XGBoost'].fit(X_scaled, y_train)

        print(f"\nTrained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        X_scaled = X_test / 255.0
        results = []

        print("\n=== Model Evaluation ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                'Model': name,
                'Accuracy': acc
            })

            print(f"{name}: Accuracy={acc:.4f}")

        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize classification results."""
        X_scaled = X_test / 255.0
        y_pred = self.best_model.predict(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Model comparison
        results_df.set_index('Model')['Accuracy'].plot(
            kind='bar', ax=axes[0, 0], color='steelblue'
        )
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim([0.8, 1.0])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        axes[1, 0].bar(range(10), class_acc, color='green')
        axes[1, 0].set_title('Per-Digit Accuracy')
        axes[1, 0].set_xlabel('Digit')
        axes[1, 0].set_xticks(range(10))
        axes[1, 0].set_ylim([0.8, 1.0])

        # Show misclassified examples
        misclassified = np.where(y_pred != y_test)[0][:16]
        if len(misclassified) > 0:
            for i, idx in enumerate(misclassified[:16]):
                ax_sub = axes[1, 1].inset_axes([
                    (i % 4) * 0.25, 1 - ((i // 4) + 1) * 0.25, 0.24, 0.24
                ])
                ax_sub.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                ax_sub.set_title(f'{y_test[idx]}→{y_pred[idx]}', fontsize=8)
                ax_sub.axis('off')
        axes[1, 1].set_title('Misclassified Examples (True→Pred)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/mnist_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/mnist_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("MNIST DIGIT CLASSIFICATION")
    print("=" * 70)

    classifier = MNISTClassifier()

    # Create data
    X, y = classifier.create_sample_data()
    print(f"\nDataset: {X.shape}, 10 digit classes")

    # Visualizations
    classifier.plot_samples(X, y)
    classifier.plot_analysis(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")

    # Train and evaluate
    classifier.train_models(X_train, y_train)
    results = classifier.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    classifier.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best Accuracy: {results.iloc[0]['Accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
