"""
Simple Object Detection - Detecting and Localizing Objects
Kaggle-style solution for object detection with bounding boxes

This solution demonstrates:
- Synthetic object detection data generation
- CNN architecture for both classification and localization
- Multi-task learning (class + bounding box)
- Visualization of detections
- IoU (Intersection over Union) metric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Simple Object Detection - Kaggle Solution")
print("=" * 60)

# Configuration
IMG_SIZE = 64
NUM_CLASSES = 4
OBJECTS = ['Circle', 'Square', 'Triangle', 'Star']
NUM_SAMPLES = 800
EPOCHS = 30
BATCH_SIZE = 32


def generate_synthetic_object_data(num_samples, img_size):
    """
    Generate synthetic images with geometric objects and bounding boxes
    Returns images, class labels, and bounding boxes (x, y, width, height)
    """
    print(f"\nGenerating {num_samples} synthetic object detection images...")

    images = []
    labels = []
    bboxes = []  # Format: [x_min, y_min, width, height] normalized to [0, 1]

    for _ in range(num_samples):
        img = np.ones((img_size, img_size)) * 0.9  # Light background

        # Random object type
        obj_type = np.random.randint(0, NUM_CLASSES)

        # Random object size (20-40% of image)
        obj_size = np.random.randint(img_size // 5, img_size // 2)

        # Random position (ensure object fits in image)
        margin = 5
        x_pos = np.random.randint(margin, img_size - obj_size - margin)
        y_pos = np.random.randint(margin, img_size - obj_size - margin)

        # Draw object
        if obj_type == 0:  # Circle
            y, x = np.ogrid[:img_size, :img_size]
            center_x = x_pos + obj_size // 2
            center_y = y_pos + obj_size // 2
            radius = obj_size // 2
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            img[mask] = 0.2

        elif obj_type == 1:  # Square
            img[y_pos:y_pos + obj_size, x_pos:x_pos + obj_size] = 0.2

        elif obj_type == 2:  # Triangle
            for i in range(obj_size):
                start = x_pos + (obj_size - i) // 2
                end = start + i
                img[y_pos + i, start:end] = 0.2

        else:  # Star (simplified 5-pointed star)
            center_x = x_pos + obj_size // 2
            center_y = y_pos + obj_size // 2
            # Draw diamond shape as simplified star
            for i in range(obj_size):
                if i < obj_size // 2:
                    offset = i
                else:
                    offset = obj_size - i - 1
                start = center_x - offset
                end = center_x + offset
                img[y_pos + i, max(0, start):min(img_size, end)] = 0.2

            # Add cross for star effect
            img[center_y - obj_size // 4:center_y + obj_size // 4,
                center_x - 1:center_x + 1] = 0.1
            img[center_y - 1:center_y + 1,
                center_x - obj_size // 4:center_x + obj_size // 4] = 0.1

        # Calculate bounding box (normalized)
        bbox = [
            x_pos / img_size,
            y_pos / img_size,
            obj_size / img_size,
            obj_size / img_size
        ]

        # Add noise
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

        images.append(img)
        labels.append(obj_type)
        bboxes.append(bbox)

    return np.array(images), np.array(labels), np.array(bboxes)


def build_detection_model(input_shape, num_classes):
    """
    Build CNN for object detection (classification + localization)
    Multi-task output: class probabilities + bounding box coordinates
    """
    inputs = layers.Input(shape=input_shape)

    # Shared convolutional base
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Classification head
    class_output = layers.Dense(num_classes, activation='softmax', name='class')(x)

    # Bounding box regression head
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)

    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])

    return model


def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two bounding boxes"""
    # box format: [x, y, width, height]
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]

    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# Generate data
X, y_class, y_bbox = generate_synthetic_object_data(NUM_SAMPLES, IMG_SIZE)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Dataset shape: {X.shape}")
print(f"Class labels shape: {y_class.shape}")
print(f"Bounding boxes shape: {y_bbox.shape}")

# Split data
X_train, X_test, y_class_train, y_class_test, y_bbox_train, y_bbox_test = train_test_split(
    X, y_class, y_bbox, test_size=0.2, random_state=42, stratify=y_class
)

# Convert class labels to categorical
y_class_train_cat = keras.utils.to_categorical(y_class_train, NUM_CLASSES)
y_class_test_cat = keras.utils.to_categorical(y_class_test, NUM_CLASSES)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build model
print("\nBuilding detection model...")
model = build_detection_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

# Compile with multiple losses
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'class': 'categorical_crossentropy',
        'bbox': 'mse'
    },
    loss_weights={
        'class': 1.0,
        'bbox': 1.0
    },
    metrics={
        'class': 'accuracy',
        'bbox': 'mae'
    }
)

print("\nModel Summary:")
model.summary()

# Train model
print("\nTraining model...")
history = model.fit(
    X_train,
    {'class': y_class_train_cat, 'bbox': y_bbox_train},
    validation_data=(X_test, {'class': y_class_test_cat, 'bbox': y_bbox_test}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
results = model.evaluate(X_test, {'class': y_class_test_cat, 'bbox': y_bbox_test}, verbose=0)
print(f"Test Classification Accuracy: {results[3]:.4f}")
print(f"Test Bbox MAE: {results[4]:.4f}")

# Predictions
y_pred_class, y_pred_bbox = model.predict(X_test)
y_pred_class_labels = np.argmax(y_pred_class, axis=1)

# Calculate IoU
ious = [calculate_iou(true, pred) for true, pred in zip(y_bbox_test, y_pred_bbox)]
mean_iou = np.mean(ious)
print(f"\nMean IoU: {mean_iou:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred_class_labels, target_names=OBJECTS))

# Visualizations
fig = plt.figure(figsize=(18, 10))

# Training curves
ax1 = plt.subplot(2, 4, 1)
plt.plot(history.history['class_accuracy'], label='Train Class Acc')
plt.plot(history.history['val_class_accuracy'], label='Val Class Acc')
plt.title('Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 4, 2)
plt.plot(history.history['bbox_mae'], label='Train Bbox MAE')
plt.plot(history.history['val_bbox_mae'], label='Val Bbox MAE')
plt.title('Bounding Box MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 4, 3)
plt.hist(ious, bins=20, edgecolor='black')
plt.axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.3f}')
plt.title('IoU Distribution')
plt.xlabel('IoU Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample detections
for i in range(5):
    ax = plt.subplot(2, 4, i + 4)
    idx = np.random.randint(0, len(X_test))

    plt.imshow(X_test[idx].squeeze(), cmap='gray')

    # True bounding box (green)
    true_bbox = y_bbox_test[idx] * IMG_SIZE
    rect_true = patches.Rectangle(
        (true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
        linewidth=2, edgecolor='green', facecolor='none', label='True'
    )
    ax.add_patch(rect_true)

    # Predicted bounding box (red)
    pred_bbox = y_pred_bbox[idx] * IMG_SIZE
    rect_pred = patches.Rectangle(
        (pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3],
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred'
    )
    ax.add_patch(rect_pred)

    true_label = OBJECTS[y_class_test[idx]]
    pred_label = OBJECTS[y_pred_class_labels[idx]]
    iou_score = ious[idx]

    plt.title(f'True: {true_label} | Pred: {pred_label}\nIoU: {iou_score:.3f}', fontsize=9)
    if i == 0:
        plt.legend(loc='upper right', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/05_object_detection/detection_results.png',
            dpi=300, bbox_inches='tight')
print("\nResults saved to 'detection_results.png'")

print("\n" + "=" * 60)
print("Object Detection Complete!")
print(f"Classification Accuracy: {results[3]:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
print("=" * 60)
