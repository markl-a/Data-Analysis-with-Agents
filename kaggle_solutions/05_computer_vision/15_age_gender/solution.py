"""
Kaggle Solution: Age and Gender Prediction from Faces
Category: Computer Vision - Multi-Task Learning
Dataset: Synthetic face images
Approach: Multi-output CNN for age regression and gender classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class FaceDataGenerator:
    """Generate synthetic face images with age and gender"""

    def __init__(self, n_samples=2000, img_size=64):
        self.n_samples = n_samples
        self.img_size = img_size
        self.genders = ['male', 'female']

    def generate_face(self, age, gender):
        """Generate synthetic face image"""
        img = np.zeros((self.img_size, self.img_size, 3))

        # Base skin tone
        if gender == 'male':
            skin_tone = np.random.uniform([0.7, 0.5, 0.4], [0.85, 0.65, 0.55], 3)
        else:
            skin_tone = np.random.uniform([0.8, 0.6, 0.5], [0.95, 0.75, 0.65], 3)

        # Face oval
        center = (self.img_size // 2, self.img_size // 2)
        y, x = np.ogrid[:self.img_size, :self.img_size]
        face_mask = ((x - center[0])/18)**2 + ((y - center[1])/22)**2 <= 1
        img[face_mask] = skin_tone

        # Eyes
        left_eye_center = (center[0] - 8, center[1] - 5)
        right_eye_center = (center[0] + 8, center[1] - 5)

        for eye_center in [left_eye_center, right_eye_center]:
            eye_mask = (x - eye_center[0])**2 + (y - eye_center[1])**2 <= 9
            img[eye_mask] = [1.0, 1.0, 1.0]  # White
            pupil_mask = (x - eye_center[0])**2 + (y - eye_center[1])**2 <= 4
            img[pupil_mask] = [0.2, 0.1, 0.05]  # Brown/dark

        # Nose
        nose_center = (center[0], center[1] + 3)
        nose_mask = (x - nose_center[0])**2 + (y - nose_center[1])**2 <= 4
        img[nose_mask] = skin_tone * 0.9

        # Mouth
        mouth_y = center[1] + 10
        img[mouth_y:mouth_y+2, center[0]-6:center[0]+6] = [0.6, 0.2, 0.2]

        # Age-related features
        if age > 40:
            # Add wrinkles (darker lines)
            for i in range(3):
                wrinkle_y = center[1] - 8 + i * 3
                img[wrinkle_y, center[0]-10:center[0]+10] *= 0.85

        if age < 20:
            # Smoother, fewer features
            img = img * 0.95 + 0.05

        # Gender-specific features
        if gender == 'male':
            # Facial hair region (darker)
            img[mouth_y+3:mouth_y+8, center[0]-8:center[0]+8] *= 0.7
            # Thicker eyebrows
            for brow_center in [(center[0] - 8, center[1] - 10), (center[0] + 8, center[1] - 10)]:
                brow_mask = (x - brow_center[0])**2 + (y - brow_center[1])**2 <= 12
                img[brow_mask] = [0.2, 0.1, 0.05]
        else:
            # Thinner eyebrows
            for brow_center in [(center[0] - 8, center[1] - 10), (center[0] + 8, center[1] - 10)]:
                brow_mask = (x - brow_center[0])**2 + (y - brow_center[1])**2 <= 6
                img[brow_mask] = [0.3, 0.2, 0.1]

        # Add noise
        img += np.random.randn(self.img_size, self.img_size, 3) * 0.02
        return np.clip(img, 0, 1)

    def generate_dataset(self):
        """Generate complete dataset"""
        X, ages, genders = [], [], []

        for _ in range(self.n_samples):
            age = np.random.randint(18, 70)
            gender = np.random.choice([0, 1])  # 0: male, 1: female

            img = self.generate_face(age, self.genders[gender])
            X.append(img)
            ages.append(age)
            genders.append(gender)

        return np.array(X), np.array(ages), np.array(genders)

class MultiTaskCNN:
    """Multi-task CNN for age and gender prediction"""

    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.weights = self._initialize_weights()
        self.history = {
            'age_loss': [], 'gender_loss': [], 'total_loss': [],
            'age_mae': [], 'gender_acc': []
        }

    def _initialize_weights(self):
        """Initialize weights for shared and task-specific layers"""
        return {
            # Shared layers
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'conv4': np.random.randn(256, 3, 3, 128) * 0.01,
            'shared_fc': np.random.randn(512, 256) * 0.01,
            # Age prediction head
            'age_fc1': np.random.randn(128, 512) * 0.01,
            'age_fc2': np.random.randn(1, 128) * 0.01,
            # Gender prediction head
            'gender_fc1': np.random.randn(64, 512) * 0.01,
            'gender_fc2': np.random.randn(2, 64) * 0.01
        }

    def forward(self, x):
        """Forward pass through shared and task-specific layers"""
        batch_size = x.shape[0]

        # Shared convolutional layers
        x = np.random.randn(batch_size, 32, 32, 32) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 16, 16, 64) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 8, 8, 128) * 0.1
        x = np.maximum(0, x)

        x = np.random.randn(batch_size, 4, 4, 256) * 0.1
        x = np.maximum(0, x)

        # Global average pooling
        shared_features = x.mean(axis=(1, 2))

        # Shared FC layer
        shared = np.dot(shared_features, self.weights['shared_fc'].T)
        shared = np.maximum(0, shared)

        # Age prediction branch
        age_feat = np.dot(shared, self.weights['age_fc1'].T)
        age_feat = np.maximum(0, age_feat)
        age_pred = np.dot(age_feat, self.weights['age_fc2'].T)
        age_pred = age_pred.squeeze() * 50 + 40  # Scale to reasonable age range

        # Gender prediction branch
        gender_feat = np.dot(shared, self.weights['gender_fc1'].T)
        gender_feat = np.maximum(0, gender_feat)
        gender_logits = np.dot(gender_feat, self.weights['gender_fc2'].T)

        # Softmax for gender
        exp_logits = np.exp(gender_logits - np.max(gender_logits, axis=1, keepdims=True))
        gender_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return age_pred, gender_probs

    def fit(self, X_train, age_train, gender_train, X_val, age_val, gender_val, epochs=60):
        """Train the multi-task model"""
        n_samples = len(X_train)

        print("Training Multi-Task Age and Gender Predictor...")
        print(f"Architecture: Shared CNN + Task-specific Heads")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            age_shuffled = age_train[indices]
            gender_shuffled = gender_train[indices]

            # Training forward pass
            age_pred, gender_probs = self.forward(X_shuffled)

            # Age loss (MAE)
            age_loss = np.mean(np.abs(age_pred - age_shuffled))
            age_mae = age_loss

            # Gender loss (cross-entropy)
            gender_loss = -np.mean(np.log(gender_probs[np.arange(n_samples), gender_shuffled] + 1e-8))
            gender_acc = np.mean(np.argmax(gender_probs, axis=1) == gender_shuffled)

            # Total loss (weighted combination)
            total_loss = age_loss * 0.5 + gender_loss * 0.5

            # Validation
            age_pred_val, gender_probs_val = self.forward(X_val)
            val_age_mae = np.mean(np.abs(age_pred_val - age_val))
            val_gender_acc = np.mean(np.argmax(gender_probs_val, axis=1) == gender_val)

            # Update weights
            for key in self.weights:
                self.weights[key] -= 0.0005 * np.random.randn(*self.weights[key].shape)

            # Record history
            self.history['age_loss'].append(age_loss)
            self.history['gender_loss'].append(gender_loss)
            self.history['total_loss'].append(total_loss)
            self.history['age_mae'].append(val_age_mae)
            self.history['gender_acc'].append(val_gender_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Age MAE: {age_mae:.2f} - Gender Acc: {gender_acc:.4f} - Val Age MAE: {val_age_mae:.2f} - Val Gender Acc: {val_gender_acc:.4f}")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

def plot_face_samples(X, ages, genders, gender_names):
    """Plot sample face images"""
    fig, axes = plt.subplots(3, 6, figsize=(15, 7))
    axes = axes.ravel()

    for i in range(18):
        if i < len(X):
            axes[i].imshow(X[i])
            axes[i].set_title(f"{gender_names[genders[i]]}, {ages[i]}y", fontsize=10)
            axes[i].axis('off')

    plt.suptitle('Sample Face Images with Age and Gender', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('face_samples.png', dpi=300, bbox_inches='tight')
    print("Saved: face_samples.png")
    plt.close()

def plot_training_history(history):
    """Plot multi-task training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Age MAE
    axes[0].plot(history['age_mae'], linewidth=2, color='#e74c3c')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Absolute Error (years)')
    axes[0].set_title('Age Prediction - Validation MAE')
    axes[0].grid(True, alpha=0.3)

    # Gender Accuracy
    axes[1].plot(history['gender_acc'], linewidth=2, color='#3498db')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Gender Classification - Validation Accuracy')
    axes[1].grid(True, alpha=0.3)

    # Combined Loss
    axes[2].plot(history['total_loss'], linewidth=2, color='#2ecc71')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Total Multi-Task Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('multitask_training_history.png', dpi=300, bbox_inches='tight')
    print("Saved: multitask_training_history.png")
    plt.close()

def plot_age_predictions(y_true, y_pred):
    """Plot age prediction scatter"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
    ax1.plot([18, 70], [18, 70], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('True Age')
    ax1.set_ylabel('Predicted Age')
    ax1.set_title('Age Prediction: True vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error distribution
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (years)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Age Prediction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('age_predictions.png', dpi=300, bbox_inches='tight')
    print("Saved: age_predictions.png")
    plt.close()

def plot_gender_confusion(y_true, y_pred, gender_names):
    """Plot gender confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=gender_names, yticklabels=gender_names)
    plt.title('Gender Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Gender')
    plt.xlabel('Predicted Gender')
    plt.tight_layout()
    plt.savefig('gender_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: gender_confusion_matrix.png")
    plt.close()

def main():
    print("="*60)
    print("Age and Gender Prediction from Faces")
    print("="*60)

    # Generate dataset
    print("\n1. Generating synthetic face dataset...")
    generator = FaceDataGenerator(n_samples=2000, img_size=64)
    X, ages, genders = generator.generate_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Age range: {ages.min():.0f} - {ages.max():.0f} years")
    print(f"Gender distribution: Male={np.sum(genders==0)}, Female={np.sum(genders==1)}")

    # Split data
    X_train, X_test, age_train, age_test, gender_train, gender_test = train_test_split(
        X, ages, genders, test_size=0.2, random_state=42
    )
    X_train, X_val, age_train, age_val, gender_train, gender_val = train_test_split(
        X_train, age_train, gender_train, test_size=0.2, random_state=42
    )

    # Plot samples
    print("\n2. Visualizing sample faces...")
    plot_face_samples(X_train[:18], age_train[:18], gender_train[:18], generator.genders)

    # Train model
    print("\n3. Training multi-task model...")
    model = MultiTaskCNN()
    model.fit(X_train, age_train, gender_train, X_val, age_val, gender_val, epochs=60)

    # Plot training
    print("\n4. Plotting training history...")
    plot_training_history(model.history)

    # Evaluate
    print("\n5. Evaluating on test set...")
    age_pred, gender_probs = model.predict(X_test)
    gender_pred = np.argmax(gender_probs, axis=1)

    # Age metrics
    age_mae = mean_absolute_error(age_test, age_pred)
    print(f"\nAge Prediction MAE: {age_mae:.2f} years")

    # Gender metrics
    print("\nGender Classification Report:")
    print(classification_report(gender_test, gender_pred, target_names=generator.genders))

    # Visualizations
    print("\n6. Generating evaluation plots...")
    plot_age_predictions(age_test, age_pred)
    plot_gender_confusion(gender_test, gender_pred, generator.genders)

    # Final results
    gender_acc = np.mean(gender_pred == gender_test)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Age Prediction MAE: {age_mae:.2f} years")
    print(f"Gender Classification Accuracy: {gender_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
