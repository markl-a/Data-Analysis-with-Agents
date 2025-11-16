"""
Traffic Sign Recognition - Autonomous Driving
Kaggle-style solution for classifying traffic signs

This solution demonstrates:
- Synthetic traffic sign data generation
- CNN for multi-class traffic sign classification
- Data augmentation for robustness
- Real-world variation handling (lighting, rotation, occlusion)
- High-accuracy requirements for safety-critical applications
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

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Traffic Sign Recognition - Kaggle Solution")
print("=" * 60)

# Configuration
IMG_SIZE = 64
NUM_CLASSES = 8
SIGN_CLASSES = [
    'Stop',
    'Yield',
    'Speed_Limit_50',
    'Speed_Limit_80',
    'No_Entry',
    'Turn_Right',
    'Turn_Left',
    'Pedestrian_Crossing'
]
SAMPLES_PER_CLASS = 150
EPOCHS = 35
BATCH_SIZE = 32


def generate_traffic_sign_data(samples_per_class, img_size, num_classes):
    """
    Generate synthetic traffic sign images with various shapes and colors
    """
    print(f"\nGenerating {samples_per_class * num_classes} traffic sign images...")

    images = []
    labels = []

    for sign_idx, sign_name in enumerate(SIGN_CLASSES):
        for _ in range(samples_per_class):
            # Create RGB image with sky/road background
            background_color = np.random.uniform(0.6, 0.9, 3)  # Light background
            img = np.ones((img_size, img_size, 3)) * background_color

            y, x = np.ogrid[:img_size, :img_size]
            center_x, center_y = img_size // 2, img_size // 2

            # Sign size (with variation)
            sign_radius = int(img_size * np.random.uniform(0.35, 0.42))

            # Generate sign based on type
            if sign_idx == 0:  # Stop - Red octagon with white text
                # Octagon shape
                angles = np.linspace(0, 2 * np.pi, 9)
                for i in range(8):
                    x1 = center_x + int(sign_radius * np.cos(angles[i]))
                    y1 = center_y + int(sign_radius * np.sin(angles[i]))
                    x2 = center_x + int(sign_radius * np.cos(angles[i + 1]))
                    y2 = center_y + int(sign_radius * np.sin(angles[i + 1]))

                    # Fill triangle between center and edge
                    for py in range(img_size):
                        for px in range(img_size):
                            # Simple octagon approximation with circle
                            dist = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
                            if dist <= sign_radius:
                                img[py, px] = [0.8, 0.1, 0.1]  # Red

                # White border
                for py in range(img_size):
                    for px in range(img_size):
                        dist = np.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
                        if sign_radius - 3 <= dist <= sign_radius:
                            img[py, px] = [0.95, 0.95, 0.95]  # White

                # White "STOP" text simulation (horizontal bars)
                text_y = center_y
                img[text_y - 3:text_y + 3, center_x - 10:center_x + 10] = [0.95, 0.95, 0.95]

            elif sign_idx == 1:  # Yield - Red triangle pointing down
                # Triangle
                triangle_height = int(sign_radius * 1.5)
                for dy in range(-triangle_height // 2, triangle_height // 2):
                    y_pos = center_y + dy
                    width = int((triangle_height // 2 - abs(dy)) * 0.8)
                    if 0 <= y_pos < img_size:
                        for dx in range(-width, width):
                            x_pos = center_x + dx
                            if 0 <= x_pos < img_size:
                                if dy < 0:  # Top part
                                    img[y_pos, x_pos] = [0.9, 0.9, 0.9]  # White
                                else:  # Bottom part
                                    img[y_pos, x_pos] = [0.85, 0.1, 0.1]  # Red

            elif sign_idx in [2, 3]:  # Speed Limit - Red circle with white background
                # White circle
                circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= sign_radius ** 2
                img[circle_mask] = [0.95, 0.95, 0.95]

                # Red border
                border_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= sign_radius ** 2) & \
                             ((x - center_x) ** 2 + (y - center_y) ** 2 >= (sign_radius - 4) ** 2)
                img[border_mask] = [0.85, 0.1, 0.1]

                # Black number (50 or 80)
                if sign_idx == 2:  # 50
                    img[center_y - 8:center_y + 8, center_x - 8:center_x - 2] = [0.1, 0.1, 0.1]
                    img[center_y - 8:center_y - 4, center_x - 2:center_x + 4] = [0.1, 0.1, 0.1]
                    img[center_y - 2:center_y + 2, center_x - 2:center_x + 4] = [0.1, 0.1, 0.1]
                else:  # 80
                    img[center_y - 8:center_y + 8, center_x - 8:center_x - 2] = [0.1, 0.1, 0.1]
                    img[center_y - 8:center_y - 4, center_x - 2:center_x + 4] = [0.1, 0.1, 0.1]
                    img[center_y + 4:center_y + 8, center_x - 2:center_x + 4] = [0.1, 0.1, 0.1]
                    img[center_y - 2:center_y + 2, center_x - 2:center_x + 4] = [0.1, 0.1, 0.1]

            elif sign_idx == 4:  # No Entry - Red circle with white bar
                # Red circle
                circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= sign_radius ** 2
                img[circle_mask] = [0.85, 0.1, 0.1]

                # White horizontal bar
                img[center_y - 4:center_y + 4,
                    center_x - sign_radius + 5:center_x + sign_radius - 5] = [0.95, 0.95, 0.95]

            elif sign_idx == 5:  # Turn Right - Blue circle with white arrow
                # Blue circle
                circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= sign_radius ** 2
                img[circle_mask] = [0.1, 0.3, 0.8]

                # White arrow (simplified)
                # Vertical part
                img[center_y - sign_radius // 2:center_y + sign_radius // 3,
                    center_x - 3:center_x + 3] = [0.95, 0.95, 0.95]
                # Horizontal part (right)
                img[center_y - sign_radius // 3:center_y - sign_radius // 3 + 6,
                    center_x:center_x + sign_radius // 2] = [0.95, 0.95, 0.95]

            elif sign_idx == 6:  # Turn Left - Blue circle with white arrow
                # Blue circle
                circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= sign_radius ** 2
                img[circle_mask] = [0.1, 0.3, 0.8]

                # White arrow (simplified)
                # Vertical part
                img[center_y - sign_radius // 2:center_y + sign_radius // 3,
                    center_x - 3:center_x + 3] = [0.95, 0.95, 0.95]
                # Horizontal part (left)
                img[center_y - sign_radius // 3:center_y - sign_radius // 3 + 6,
                    center_x - sign_radius // 2:center_x] = [0.95, 0.95, 0.95]

            else:  # Pedestrian Crossing - Blue triangle with pedestrian symbol
                # Blue triangle (pointing up)
                triangle_height = int(sign_radius * 1.5)
                for dy in range(-triangle_height // 2, triangle_height // 2):
                    y_pos = center_y + dy
                    width = int((triangle_height // 2 - abs(dy)) * 0.8)
                    if 0 <= y_pos < img_size:
                        for dx in range(-width, width):
                            x_pos = center_x + dx
                            if 0 <= x_pos < img_size:
                                img[y_pos, x_pos] = [0.1, 0.3, 0.8]

                # White pedestrian (stick figure)
                # Head
                head_y, head_x = center_y - 8, center_x
                head_mask = (x - head_x) ** 2 + (y - head_y) ** 2 <= 16
                img[head_mask] = [0.95, 0.95, 0.95]
                # Body
                img[center_y - 4:center_y + 8, center_x - 1:center_x + 2] = [0.95, 0.95, 0.95]

            # Add realistic variations
            # Random brightness
            brightness = np.random.uniform(0.85, 1.15)
            img = np.clip(img * brightness, 0, 1)

            # Add noise
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)

            # Random slight blur (motion blur effect)
            if np.random.random() > 0.7:
                from scipy.ndimage import gaussian_filter
                sigma = np.random.uniform(0.3, 0.8)
                for c in range(3):
                    img[:, :, c] = gaussian_filter(img[:, :, c], sigma=sigma)

            images.append(img)
            labels.append(sign_idx)

    return np.array(images), np.array(labels)


def build_traffic_sign_classifier(input_shape, num_classes):
    """
    Build CNN for traffic sign classification
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Generate data
X, y = generate_traffic_sign_data(SAMPLES_PER_CLASS, IMG_SIZE, NUM_CLASSES)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to categorical
y_train_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build model
print("\nBuilding traffic sign classifier...")
model = build_traffic_sign_classifier((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Data augmentation (robust for real-world conditions)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.1
)

# Train
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
print(classification_report(y_test, y_pred, target_names=SIGN_CLASSES))

# Visualizations
fig = plt.figure(figsize=(20, 12))

# Training curves
ax1 = plt.subplot(3, 6, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 6, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrix
ax3 = plt.subplot(3, 6, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=[s.replace('_', '\n') for s in SIGN_CLASSES],
            yticklabels=[s.replace('_', '\n') for s in SIGN_CLASSES],
            ax=ax3, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)

# Per-class accuracy
ax4 = plt.subplot(3, 6, 4)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['green' if acc > 0.9 else 'orange' if acc > 0.8 else 'red' for acc in per_class_acc]
plt.barh(range(NUM_CLASSES), per_class_acc, color=colors)
plt.yticks(range(NUM_CLASSES), [s.replace('_', ' ') for s in SIGN_CLASSES], fontsize=8)
plt.xlabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.xlim([0, 1])
plt.grid(True, alpha=0.3, axis='x')

# Sample predictions
sample_indices = np.random.choice(len(X_test), 14, replace=False)

for i, idx in enumerate(sample_indices):
    ax = plt.subplot(3, 6, i + 5)
    plt.imshow(X_test[idx])

    true_label = SIGN_CLASSES[y_test[idx]].replace('_', ' ')
    pred_label = SIGN_CLASSES[y_pred[idx]].replace('_', ' ')
    color = 'green' if y_test[idx] == y_pred[idx] else 'red'

    plt.title(f'True: {true_label}\nPred: {pred_label}',
              fontsize=7, color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/10_traffic_sign/traffic_sign_results.png',
            dpi=300, bbox_inches='tight')
print("\nResults saved to 'traffic_sign_results.png'")

print("\n" + "=" * 60)
print("Traffic Sign Recognition Complete!")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print("=" * 60)
