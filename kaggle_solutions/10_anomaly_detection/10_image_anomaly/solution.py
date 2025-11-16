"""
Image Anomaly Detection
Detects defective or anomalous images using PCA reconstruction and statistical methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_image_data(n_normal=500, n_anomaly=30, img_size=28):
    """Generate synthetic grayscale image data"""

    # Normal images: circles with slight variations
    normal_images = []
    for _ in range(n_normal):
        img = np.zeros((img_size, img_size))
        center = (img_size // 2, img_size // 2)
        radius = np.random.randint(8, 12)

        # Draw circle
        for i in range(img_size):
            for j in range(img_size):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist <= radius:
                    img[i, j] = 1.0

        # Add noise
        noise = np.random.normal(0, 0.05, (img_size, img_size))
        img = np.clip(img + noise, 0, 1)

        normal_images.append(img.flatten())

    # Anomalous images: various defects
    anomaly_images = []
    anomaly_types = []

    for _ in range(n_anomaly):
        img = np.zeros((img_size, img_size))
        anomaly_type = np.random.choice(['scratch', 'spot', 'incomplete', 'distorted'])
        anomaly_types.append(anomaly_type)

        if anomaly_type == 'scratch':
            # Circle with scratch
            center = (img_size // 2, img_size // 2)
            radius = np.random.randint(8, 12)
            for i in range(img_size):
                for j in range(img_size):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist <= radius:
                        img[i, j] = 1.0

            # Add scratch
            scratch_row = np.random.randint(5, img_size - 5)
            img[scratch_row:scratch_row+2, :] = 0  # Horizontal scratch

        elif anomaly_type == 'spot':
            # Circle with spots/defects
            center = (img_size // 2, img_size // 2)
            radius = np.random.randint(8, 12)
            for i in range(img_size):
                for j in range(img_size):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist <= radius:
                        img[i, j] = 1.0

            # Add spots
            n_spots = np.random.randint(2, 5)
            for _ in range(n_spots):
                spot_i = np.random.randint(0, img_size)
                spot_j = np.random.randint(0, img_size)
                img[spot_i:spot_i+3, spot_j:spot_j+3] = 0

        elif anomaly_type == 'incomplete':
            # Incomplete circle (missing section)
            center = (img_size // 2, img_size // 2)
            radius = np.random.randint(8, 12)
            missing_angle_start = np.random.uniform(0, 2*np.pi)
            missing_angle_range = np.random.uniform(np.pi/3, np.pi)

            for i in range(img_size):
                for j in range(img_size):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    angle = np.arctan2(i - center[0], j - center[1])

                    # Check if in missing section
                    angle_diff = (angle - missing_angle_start) % (2 * np.pi)
                    if dist <= radius and not (angle_diff < missing_angle_range):
                        img[i, j] = 1.0

        else:  # distorted
            # Ellipse instead of circle
            center = (img_size // 2, img_size // 2)
            radius_x = np.random.randint(6, 10)
            radius_y = np.random.randint(12, 16)

            for i in range(img_size):
                for j in range(img_size):
                    dist = ((i - center[0])/radius_x)**2 + ((j - center[1])/radius_y)**2
                    if dist <= 1:
                        img[i, j] = 1.0

        # Add noise
        noise = np.random.normal(0, 0.05, (img_size, img_size))
        img = np.clip(img + noise, 0, 1)

        anomaly_images.append(img.flatten())

    # Combine
    X = np.vstack([normal_images, anomaly_images])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    return X, y, img_size

def visualize_samples(X, y, img_size, n_samples=10):
    """Visualize sample images"""
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
    fig.suptitle('Sample Images', fontsize=16)

    # Normal samples
    normal_indices = np.where(y == 0)[0][:n_samples]
    for i, idx in enumerate(normal_indices):
        axes[0, i].imshow(X[idx].reshape(img_size, img_size), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Normal', fontsize=10)

    # Anomaly samples
    anomaly_indices = np.where(y == 1)[0][:n_samples]
    for i, idx in enumerate(anomaly_indices):
        axes[1, i].imshow(X[idx].reshape(img_size, img_size), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Anomaly', fontsize=10)

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("Saved: sample_images.png")

def plot_reconstruction_examples(X, X_reconstructed, y, img_size):
    """Plot original vs reconstructed images"""
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    fig.suptitle('PCA Reconstruction: Original vs Reconstructed', fontsize=16)

    # Show both normal and anomaly examples
    normal_idx = np.where(y == 0)[0][:4]
    anomaly_idx = np.where(y == 1)[0][:4]
    indices = np.concatenate([normal_idx, anomaly_idx])

    for col, idx in enumerate(indices):
        # Original
        axes[0, col].imshow(X[idx].reshape(img_size, img_size), cmap='gray')
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('Original', fontsize=10)

        # Reconstructed
        axes[1, col].imshow(X_reconstructed[idx].reshape(img_size, img_size), cmap='gray')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel('Reconstructed', fontsize=10)

        # Difference
        diff = np.abs(X[idx] - X_reconstructed[idx])
        axes[2, col].imshow(diff.reshape(img_size, img_size), cmap='hot')
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel('Difference', fontsize=10)

        # Label
        label = 'Normal' if y[idx] == 0 else 'Anomaly'
        axes[0, col].set_title(label, fontsize=9)

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: reconstruction_comparison.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def main():
    print("Image Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic image data...")
    X, y, img_size = generate_image_data(n_normal=500, n_anomaly=30)
    print(f"Total images: {len(X)}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Anomalies: {y.sum():.0f} ({y.mean()*100:.2f}%)")

    # Visualize samples
    print("\nVisualizing sample images...")
    visualize_samples(X, y, img_size)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Method 1: PCA Reconstruction Error
    print("\n" + "="*60)
    print("Training PCA for Reconstruction...")

    # Use only normal images for PCA
    X_train = X_scaled[y == 0]

    # PCA with enough components to capture main patterns
    n_components = 50
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)

    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Reconstruct all images
    X_transformed = pca.transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_transformed)

    # Calculate reconstruction error
    reconstruction_errors = np.mean((X_scaled - X_reconstructed)**2, axis=1)

    # Set threshold (95th percentile of normal data)
    normal_errors = reconstruction_errors[y == 0]
    threshold_pca = np.percentile(normal_errors, 95)

    y_pred_pca = (reconstruction_errors > threshold_pca).astype(int)
    metrics_dict['PCA Reconstruction'] = evaluate_detector(y, y_pred_pca, "PCA Reconstruction")

    # Visualize reconstructions
    X_reconstructed_original = scaler.inverse_transform(X_reconstructed)
    plot_reconstruction_examples(X, X_reconstructed_original, y, img_size)

    # Plot reconstruction errors
    fig, ax = plt.subplots(figsize=(12, 5))
    normal_errors_all = reconstruction_errors[y == 0]
    anomaly_errors = reconstruction_errors[y == 1]

    ax.hist(normal_errors_all, bins=50, alpha=0.6, label='Normal', density=True)
    ax.hist(anomaly_errors, bins=30, alpha=0.6, label='Anomaly', density=True)
    ax.axvline(threshold_pca, color='r', linestyle='--', label=f'Threshold: {threshold_pca:.4f}')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title('PCA Reconstruction Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('reconstruction_error_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: reconstruction_error_distribution.png")

    # Method 2: Isolation Forest on PCA components
    print("\n" + "="*60)
    print("Training Isolation Forest on PCA Features...")

    iso_forest = IsolationForest(contamination=0.056, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_transformed)
    y_pred_if = (y_pred_if == -1).astype(int)

    metrics_dict['Isolation Forest'] = evaluate_detector(y, y_pred_if, "Isolation Forest")

    # Method 3: Simple pixel-based statistics
    print("\n" + "="*60)
    print("Applying Pixel Statistics Method...")

    # Calculate mean and std from normal images
    normal_mean = np.mean(X_scaled[y == 0], axis=0)
    normal_std = np.std(X_scaled[y == 0], axis=0) + 1e-6

    # Calculate max z-score for each image
    z_scores = np.abs((X_scaled - normal_mean) / normal_std)
    max_z_scores = np.max(z_scores, axis=1)

    threshold_stat = np.percentile(max_z_scores[y == 0], 95)
    y_pred_stat = (max_z_scores > threshold_stat).astype(int)

    metrics_dict['Pixel Statistics'] = evaluate_detector(y, y_pred_stat, "Pixel Statistics")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Detection Results Comparison', fontsize=16)

    methods = [
        ('PCA Reconstruction', y_pred_pca),
        ('Isolation Forest', y_pred_if),
        ('Pixel Statistics', y_pred_stat)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Performance comparison
    ax = axes[1, 1]
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: detection_results.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- PCA reconstruction captures normal pattern structure")
    print("- Works well for texture and structural defects")
    print("- Isolation Forest adds robustness on PCA features")
    print("- For production, consider deep learning autoencoders")
    print("- Adjust threshold based on quality control requirements")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
