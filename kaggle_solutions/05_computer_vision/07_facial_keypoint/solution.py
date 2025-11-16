"""
Facial Keypoint Detection - Landmark Localization
Kaggle-style solution for detecting facial landmarks/keypoints

This solution demonstrates:
- Synthetic facial keypoint data generation
- CNN regression for coordinate prediction
- 15 facial keypoints (30 coordinates)
- Visualization of predicted keypoints
- MSE and MAE metrics for regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Facial Keypoint Detection - Kaggle Solution")
print("=" * 60)

# Configuration
IMG_SIZE = 96
NUM_KEYPOINTS = 15  # 15 keypoints = 30 coordinates (x, y)
NUM_SAMPLES = 1000
EPOCHS = 40
BATCH_SIZE = 32

# Keypoint names
KEYPOINT_NAMES = [
    'left_eye_center', 'right_eye_center',
    'left_eye_inner', 'left_eye_outer',
    'right_eye_inner', 'right_eye_outer',
    'left_eyebrow_inner', 'left_eyebrow_outer',
    'right_eyebrow_inner', 'right_eyebrow_outer',
    'nose_tip', 'mouth_left', 'mouth_right',
    'mouth_center_top', 'mouth_center_bottom'
]


def generate_facial_keypoints_data(num_samples, img_size):
    """
    Generate synthetic face images with corresponding keypoint coordinates
    Returns images and keypoint coordinates (normalized to [0, 1])
    """
    print(f"\nGenerating {num_samples} facial keypoint samples...")

    images = []
    keypoints_list = []

    for _ in range(num_samples):
        # Create base face
        img = np.ones((img_size, img_size)) * 0.5

        # Random face parameters
        face_center_x = img_size // 2 + np.random.randint(-5, 5)
        face_center_y = img_size // 2 + np.random.randint(-5, 5)
        face_width = img_size // 2.5 + np.random.randint(-5, 5)
        face_height = img_size // 2 + np.random.randint(-5, 5)

        # Draw face oval
        y, x = np.ogrid[:img_size, :img_size]
        face_mask = ((x - face_center_x) ** 2 / face_width ** 2 +
                     (y - face_center_y) ** 2 / face_height ** 2) <= 1
        img[face_mask] = 0.7

        # Initialize keypoints array (15 keypoints = 30 values)
        keypoints = np.zeros(NUM_KEYPOINTS * 2)

        # Eye positions
        eye_y = face_center_y - img_size // 8 + np.random.randint(-3, 3)
        left_eye_x = face_center_x - img_size // 6 + np.random.randint(-2, 2)
        right_eye_x = face_center_x + img_size // 6 + np.random.randint(-2, 2)
        eye_radius = img_size // 20

        # Left eye
        left_eye_mask = (x - left_eye_x) ** 2 + (y - eye_y) ** 2 <= eye_radius ** 2
        img[left_eye_mask] = 0.2
        keypoints[0] = left_eye_x  # left_eye_center x
        keypoints[1] = eye_y       # left_eye_center y
        keypoints[4] = left_eye_x - eye_radius  # left_eye_inner x
        keypoints[5] = eye_y                     # left_eye_inner y
        keypoints[6] = left_eye_x + eye_radius  # left_eye_outer x
        keypoints[7] = eye_y                     # left_eye_outer y

        # Right eye
        right_eye_mask = (x - right_eye_x) ** 2 + (y - eye_y) ** 2 <= eye_radius ** 2
        img[right_eye_mask] = 0.2
        keypoints[2] = right_eye_x  # right_eye_center x
        keypoints[3] = eye_y        # right_eye_center y
        keypoints[8] = right_eye_x + eye_radius  # right_eye_inner x
        keypoints[9] = eye_y                      # right_eye_inner y
        keypoints[10] = right_eye_x - eye_radius  # right_eye_outer x
        keypoints[11] = eye_y                      # right_eye_outer y

        # Eyebrows
        eyebrow_y = eye_y - img_size // 12
        # Left eyebrow
        keypoints[12] = left_eye_x - eye_radius // 2  # left_eyebrow_inner x
        keypoints[13] = eyebrow_y                      # left_eyebrow_inner y
        keypoints[14] = left_eye_x + eye_radius       # left_eyebrow_outer x
        keypoints[15] = eyebrow_y                      # left_eyebrow_outer y

        # Draw left eyebrow
        for i in range(int(keypoints[12]), int(keypoints[14])):
            if 0 <= i < img_size and 0 <= eyebrow_y < img_size:
                img[eyebrow_y:eyebrow_y + 2, i] = 0.2

        # Right eyebrow
        keypoints[16] = right_eye_x - eye_radius       # right_eyebrow_inner x
        keypoints[17] = eyebrow_y                       # right_eyebrow_inner y
        keypoints[18] = right_eye_x + eye_radius // 2  # right_eyebrow_outer x
        keypoints[19] = eyebrow_y                       # right_eyebrow_outer y

        # Draw right eyebrow
        for i in range(int(keypoints[16]), int(keypoints[18])):
            if 0 <= i < img_size and 0 <= eyebrow_y < img_size:
                img[eyebrow_y:eyebrow_y + 2, i] = 0.2

        # Nose
        nose_y = face_center_y + img_size // 10 + np.random.randint(-2, 2)
        nose_x = face_center_x + np.random.randint(-2, 2)
        keypoints[20] = nose_x  # nose_tip x
        keypoints[21] = nose_y  # nose_tip y

        # Draw nose
        img[nose_y - 2:nose_y + 2, nose_x - 1:nose_x + 1] = 0.3

        # Mouth
        mouth_y = face_center_y + img_size // 4 + np.random.randint(-3, 3)
        mouth_width = img_size // 6

        keypoints[22] = face_center_x - mouth_width  # mouth_left x
        keypoints[23] = mouth_y                       # mouth_left y
        keypoints[24] = face_center_x + mouth_width  # mouth_right x
        keypoints[25] = mouth_y                       # mouth_right y
        keypoints[26] = face_center_x                 # mouth_center_top x
        keypoints[27] = mouth_y - 3                   # mouth_center_top y
        keypoints[28] = face_center_x                 # mouth_center_bottom x
        keypoints[29] = mouth_y + 3                   # mouth_center_bottom y

        # Draw mouth
        img[mouth_y:mouth_y + 2,
            int(keypoints[22]):int(keypoints[24])] = 0.2

        # Add noise
        noise = np.random.normal(0, 0.03, img.shape)
        img = np.clip(img + noise, 0, 1)

        # Normalize keypoints to [0, 1]
        keypoints_normalized = keypoints / img_size

        images.append(img)
        keypoints_list.append(keypoints_normalized)

    return np.array(images), np.array(keypoints_list)


def build_keypoint_detector(input_shape, num_outputs):
    """
    Build CNN for keypoint regression
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense layers for regression
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_outputs)  # No activation for regression
    ])

    return model


# Generate data
X, y_keypoints = generate_facial_keypoints_data(NUM_SAMPLES, IMG_SIZE)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Images shape: {X.shape}")
print(f"Keypoints shape: {y_keypoints.shape}")
print(f"Number of coordinates per sample: {y_keypoints.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_keypoints, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build model
print("\nBuilding keypoint detection model...")
model = build_keypoint_detector((IMG_SIZE, IMG_SIZE, 1), NUM_KEYPOINTS * 2)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
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
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Predictions
y_pred = model.predict(X_test)

# Calculate pixel-level error (denormalize)
pixel_errors = np.abs((y_pred - y_test) * IMG_SIZE)
mean_pixel_error = np.mean(pixel_errors)
print(f"\nMean Pixel Error: {mean_pixel_error:.2f} pixels")

# Visualizations
fig = plt.figure(figsize=(18, 12))

# Training curves
ax1 = plt.subplot(3, 5, 1)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 5, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

# Pixel error distribution
ax3 = plt.subplot(3, 5, 3)
plt.hist(pixel_errors.flatten(), bins=50, edgecolor='black')
plt.axvline(mean_pixel_error, color='red', linestyle='--',
            label=f'Mean: {mean_pixel_error:.2f}px')
plt.title('Pixel Error Distribution')
plt.xlabel('Error (pixels)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample predictions
sample_indices = np.random.choice(len(X_test), 12, replace=False)

for i, idx in enumerate(sample_indices):
    ax = plt.subplot(3, 5, i + 4)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')

    # Denormalize keypoints
    true_kp = y_test[idx] * IMG_SIZE
    pred_kp = y_pred[idx] * IMG_SIZE

    # Plot true keypoints (green)
    for j in range(NUM_KEYPOINTS):
        plt.plot(true_kp[j * 2], true_kp[j * 2 + 1], 'go', markersize=4)

    # Plot predicted keypoints (red)
    for j in range(NUM_KEYPOINTS):
        plt.plot(pred_kp[j * 2], pred_kp[j * 2 + 1], 'rx', markersize=4)

    # Calculate error for this sample
    sample_error = np.mean(np.abs(true_kp - pred_kp))

    plt.title(f'Error: {sample_error:.2f}px', fontsize=9)
    if i == 0:
        plt.plot([], [], 'go', label='True', markersize=4)
        plt.plot([], [], 'rx', label='Pred', markersize=4)
        plt.legend(loc='upper right', fontsize=7)
    plt.axis('off')

plt.tight_layout()
plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/07_facial_keypoint/keypoint_results.png',
            dpi=300, bbox_inches='tight')
print("\nResults saved to 'keypoint_results.png'")

print("\n" + "=" * 60)
print("Facial Keypoint Detection Complete!")
print(f"Test MAE: {test_mae:.6f} (normalized)")
print(f"Mean Pixel Error: {mean_pixel_error:.2f} pixels")
print("=" * 60)
