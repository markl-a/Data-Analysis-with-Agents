"""
Basic Image Segmentation - Pixel-wise Classification
Kaggle-style solution for semantic segmentation

This solution demonstrates:
- Synthetic segmentation data generation
- U-Net architecture for pixel-wise classification
- Semantic segmentation (multiple object classes)
- Dice coefficient metric
- Segmentation mask visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Basic Image Segmentation - Kaggle Solution")
print("=" * 60)

# Configuration
IMG_SIZE = 128
NUM_CLASSES = 4  # Background, Circle, Square, Triangle
CLASS_NAMES = ['Background', 'Circle', 'Square', 'Triangle']
NUM_SAMPLES = 600
EPOCHS = 30
BATCH_SIZE = 16


def generate_segmentation_data(num_samples, img_size, num_classes):
    """
    Generate synthetic images with multiple objects and corresponding segmentation masks
    """
    print(f"\nGenerating {num_samples} segmentation samples...")

    images = []
    masks = []

    for _ in range(num_samples):
        # Create image and mask
        img = np.ones((img_size, img_size)) * 0.8  # Light background
        mask = np.zeros((img_size, img_size, num_classes))  # One-hot encoded
        mask[:, :, 0] = 1  # Initially all background

        # Add 1-3 random objects
        num_objects = np.random.randint(1, 4)

        for obj_idx in range(num_objects):
            # Random object type (1=Circle, 2=Square, 3=Triangle)
            obj_type = np.random.randint(1, num_classes)

            # Random size
            obj_size = np.random.randint(img_size // 8, img_size // 4)

            # Random position
            x_pos = np.random.randint(5, img_size - obj_size - 5)
            y_pos = np.random.randint(5, img_size - obj_size - 5)

            # Random color/intensity
            intensity = np.random.uniform(0.2, 0.6)

            y, x = np.ogrid[:img_size, :img_size]

            if obj_type == 1:  # Circle
                center_x = x_pos + obj_size // 2
                center_y = y_pos + obj_size // 2
                radius = obj_size // 2
                obj_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

            elif obj_type == 2:  # Square
                obj_mask = np.zeros((img_size, img_size), dtype=bool)
                obj_mask[y_pos:y_pos + obj_size, x_pos:x_pos + obj_size] = True

            else:  # Triangle
                obj_mask = np.zeros((img_size, img_size), dtype=bool)
                for i in range(obj_size):
                    start = x_pos + (obj_size - i) // 2
                    end = start + i
                    if y_pos + i < img_size:
                        obj_mask[y_pos + i, max(0, start):min(img_size, end)] = True

            # Apply object to image
            img[obj_mask] = intensity

            # Update mask (remove background where object is)
            mask[obj_mask, 0] = 0
            mask[obj_mask, obj_type] = 1

        # Add noise to image
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def build_unet(input_shape, num_classes):
    """
    Build U-Net architecture for semantic segmentation
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder (downsampling path)
    # Block 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # Block 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Block 3
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder (upsampling path)
    # Block 5
    up5 = layers.UpSampling2D((2, 2))(conv4)
    up5 = layers.concatenate([up5, conv3], axis=-1)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    # Block 6
    up6 = layers.UpSampling2D((2, 2))(conv5)
    up6 = layers.concatenate([up6, conv2], axis=-1)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    # Block 7
    up7 = layers.UpSampling2D((2, 2))(conv6)
    up7 = layers.concatenate([up7, conv1], axis=-1)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv7)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice coefficient metric for segmentation
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)


# Generate data
X, y_masks = generate_segmentation_data(NUM_SAMPLES, IMG_SIZE, NUM_CLASSES)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Images shape: {X.shape}")
print(f"Masks shape: {y_masks.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_masks, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build model
print("\nBuilding U-Net model...")
model = build_unet((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', dice_coefficient]
)

print("\nModel Summary:")
model.summary()

# Train model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
test_loss, test_accuracy, test_dice = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Dice Coefficient: {test_dice:.4f}")

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=-1)
y_test_labels = np.argmax(y_test, axis=-1)

# Calculate per-class IoU
ious_per_class = []
for class_idx in range(NUM_CLASSES):
    intersection = np.sum((y_pred_labels == class_idx) & (y_test_labels == class_idx))
    union = np.sum((y_pred_labels == class_idx) | (y_test_labels == class_idx))
    iou = intersection / union if union > 0 else 0
    ious_per_class.append(iou)
    print(f"{CLASS_NAMES[class_idx]} IoU: {iou:.4f}")

mean_iou = np.mean(ious_per_class)
print(f"\nMean IoU: {mean_iou:.4f}")

# Visualizations
fig = plt.figure(figsize=(18, 12))

# Training curves
ax1 = plt.subplot(3, 5, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Pixel Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 5, 2)
plt.plot(history.history['dice_coefficient'], label='Train Dice')
plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()
plt.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 5, 3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Per-class IoU
ax4 = plt.subplot(3, 5, 4)
plt.bar(CLASS_NAMES, ious_per_class, color=['gray', 'blue', 'green', 'red'])
plt.title('Per-Class IoU')
plt.ylabel('IoU Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Sample segmentations
sample_indices = np.random.choice(len(X_test), 10, replace=False)

for i, idx in enumerate(sample_indices):
    # Original image
    ax = plt.subplot(3, 5, i + 6)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    if i < 5:
        plt.title('Original')
    plt.axis('off')

    # True mask (on same subplot)
    true_mask = y_test_labels[idx]
    plt.imshow(true_mask, cmap='tab10', alpha=0.4, vmin=0, vmax=NUM_CLASSES-1)

    # Predicted mask overlay
    pred_mask = y_pred_labels[idx]
    # Show prediction as contour
    plt.contour(pred_mask, levels=np.arange(NUM_CLASSES), colors='red',
                linewidths=1.5, alpha=0.8)

    if i == 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='red', label='Predicted'),
            Patch(facecolor='blue', alpha=0.4, label='True Mask')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=7)

plt.tight_layout()
plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/06_image_segmentation/segmentation_results.png',
            dpi=300, bbox_inches='tight')
print("\nResults saved to 'segmentation_results.png'")

print("\n" + "=" * 60)
print("Image Segmentation Complete!")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Dice Coefficient: {test_dice:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print("=" * 60)
