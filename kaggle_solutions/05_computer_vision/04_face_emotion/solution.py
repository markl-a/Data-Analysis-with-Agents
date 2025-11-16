"""
Face Emotion Recognition - 7 Emotion Classification
Kaggle-style solution for detecting emotions from facial expressions

This solution demonstrates:
- Synthetic facial emotion data generation
- CNN architecture for multi-class classification
- Data augmentation techniques
- Training with validation
- Visualization of results and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Face Emotion Recognition - Kaggle Solution")
print("=" * 60)

# Configuration
IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SAMPLES_PER_CLASS = 200
EPOCHS = 25
BATCH_SIZE = 32


def generate_synthetic_face_emotion_data(num_samples_per_class, img_size):
    """
    Generate synthetic facial emotion data with distinct patterns
    Each emotion has characteristic features
    """
    print(f"\nGenerating {num_samples_per_class * NUM_CLASSES} synthetic face images...")

    images = []
    labels = []

    for emotion_idx, emotion in enumerate(EMOTIONS):
        for _ in range(num_samples_per_class):
            # Create base face structure
            img = np.ones((img_size, img_size)) * 0.5

            # Add face oval
            y, x = np.ogrid[:img_size, :img_size]
            center_y, center_x = img_size // 2, img_size // 2
            face_mask = ((x - center_x) ** 2 / (img_size // 3) ** 2 +
                        (y - center_y) ** 2 / (img_size // 2.5) ** 2) <= 1
            img[face_mask] = 0.7

            # Eyes (consistent across emotions)
            eye_y = img_size // 3
            left_eye_x = img_size // 3
            right_eye_x = 2 * img_size // 3
            eye_radius = img_size // 15

            for eye_x in [left_eye_x, right_eye_x]:
                eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= eye_radius ** 2
                img[eye_mask] = 0.2

            # Emotion-specific features
            if emotion_idx == 0:  # Angry
                # Furrowed brows (angled down toward center)
                for i in range(img_size // 6, img_size // 3):
                    img[eye_y - img_size // 10:eye_y - img_size // 12, i] = 0.1
                    img[eye_y - img_size // 10:eye_y - img_size // 12, img_size - i] = 0.1
                # Mouth (downturned)
                mouth_y = 2 * img_size // 3
                for i in range(img_size // 3, 2 * img_size // 3):
                    offset = abs(i - center_x) // 10
                    img[mouth_y + offset:mouth_y + offset + 2, i] = 0.1

            elif emotion_idx == 1:  # Disgust
                # Raised upper lip
                mouth_y = 2 * img_size // 3
                for i in range(img_size // 3, 2 * img_size // 3):
                    img[mouth_y - 3:mouth_y, i] = 0.2
                # Wrinkled nose
                nose_y = center_y + img_size // 10
                img[nose_y:nose_y + 2, center_x - 2:center_x + 2] = 0.3

            elif emotion_idx == 2:  # Fear
                # Wide eyes (larger circles)
                for eye_x in [left_eye_x, right_eye_x]:
                    fear_eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= (eye_radius * 1.5) ** 2
                    img[fear_eye_mask] = 0.15
                # Open mouth (circle)
                mouth_y = 2 * img_size // 3
                mouth_mask = ((x - center_x) ** 2 + (y - mouth_y) ** 2) <= (img_size // 12) ** 2
                img[mouth_mask] = 0.1

            elif emotion_idx == 3:  # Happy
                # Raised cheeks
                for cheek_x in [img_size // 4, 3 * img_size // 4]:
                    cheek_y = center_y + img_size // 8
                    cheek_mask = ((x - cheek_x) ** 2 + (y - cheek_y) ** 2) <= (img_size // 10) ** 2
                    img[cheek_mask] = 0.8
                # Smile (upturned mouth)
                mouth_y = 2 * img_size // 3
                for i in range(img_size // 3, 2 * img_size // 3):
                    offset = abs(i - center_x) // 10
                    img[mouth_y - offset:mouth_y - offset + 2, i] = 0.1

            elif emotion_idx == 4:  # Sad
                # Drooping eyebrows
                for i in range(img_size // 6, img_size // 3):
                    offset = abs(i - img_size // 4) // 15
                    img[eye_y - img_size // 12 + offset, i] = 0.1
                    img[eye_y - img_size // 12 + offset, img_size - i] = 0.1
                # Frown (downturned mouth)
                mouth_y = 2 * img_size // 3
                for i in range(img_size // 3, 2 * img_size // 3):
                    offset = abs(i - center_x) // 8
                    img[mouth_y + offset:mouth_y + offset + 2, i] = 0.1

            elif emotion_idx == 5:  # Surprise
                # Wide eyes (very large)
                for eye_x in [left_eye_x, right_eye_x]:
                    surprise_eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= (eye_radius * 1.8) ** 2
                    img[surprise_eye_mask] = 0.1
                # Open mouth (oval)
                mouth_y = 2 * img_size // 3
                mouth_mask = ((x - center_x) ** 2 / (img_size // 15) ** 2 +
                            (y - mouth_y) ** 2 / (img_size // 10) ** 2) <= 1
                img[mouth_mask] = 0.1
                # Raised eyebrows
                for i in range(img_size // 6, img_size // 3):
                    img[eye_y - img_size // 8, i] = 0.1
                    img[eye_y - img_size // 8, img_size - i] = 0.1

            else:  # Neutral
                # Straight mouth
                mouth_y = 2 * img_size // 3
                img[mouth_y:mouth_y + 2, img_size // 3:2 * img_size // 3] = 0.1
                # Normal eyebrows
                for i in range(img_size // 6, img_size // 3):
                    img[eye_y - img_size // 10, i] = 0.1
                    img[eye_y - img_size // 10, img_size - i] = 0.1

            # Add noise
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)

            images.append(img)
            labels.append(emotion_idx)

    return np.array(images), np.array(labels)


def build_emotion_cnn(input_shape, num_classes):
    """Build CNN architecture for emotion recognition"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Generate synthetic data
X, y = generate_synthetic_face_emotion_data(SAMPLES_PER_CLASS, IMG_SIZE)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Emotion distribution: {np.bincount(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build model
print("\nBuilding CNN model...")
model = build_emotion_cnn((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Train model
print("\nTraining model...")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=EMOTIONS))

# Visualizations
fig = plt.figure(figsize=(18, 12))

# Plot 1: Training history
ax1 = plt.subplot(3, 4, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 4, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Confusion matrix
ax3 = plt.subplot(3, 4, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax3)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot 3: Sample predictions
for i in range(9):
    ax = plt.subplot(3, 4, i + 4)
    idx = np.random.randint(0, len(X_test))
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    true_label = EMOTIONS[y_test[idx]]
    pred_label = EMOTIONS[y_pred[idx]]
    color = 'green' if y_test[idx] == y_pred[idx] else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/04_face_emotion/emotion_results.png',
            dpi=300, bbox_inches='tight')
print("\nResults saved to 'emotion_results.png'")

print("\n" + "=" * 60)
print("Face Emotion Recognition Complete!")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print("=" * 60)
