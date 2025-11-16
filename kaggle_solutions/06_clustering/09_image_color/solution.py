"""
Image Color Quantization using Clustering
Reduce the number of colors in an image while preserving visual quality
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


class ImageColorQuantization:
    """Perform color quantization using clustering algorithms"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_synthetic_image(self, width=200, height=200):
        """
        Generate a synthetic colorful image
        Creates regions with different colors to simulate a real image
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Create multiple colored regions
        # Sky region (blue gradient)
        for i in range(height // 3):
            image[i, :, :] = [135 - i // 2, 206 - i // 2, 235]  # Sky blue gradient

        # Sun region (yellow circle)
        center_x, center_y = width // 4, height // 6
        radius = 25
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = [255, 220, 0]  # Yellow

        # Grass region (green)
        grass_start = height // 3
        for i in range(grass_start, 2 * height // 3):
            color_var = np.random.randint(-20, 20, width)
            for j in range(width):
                image[i, j, :] = [34 + color_var[j], 139 + color_var[j], 34]

        # Flower field (mixed colors)
        flower_start = 2 * height // 3
        flowers = [
            [255, 0, 0],    # Red
            [255, 192, 203], # Pink
            [255, 255, 0],   # Yellow
            [255, 165, 0],   # Orange
            [138, 43, 226],  # Purple
        ]
        for i in range(flower_start, height):
            for j in range(width):
                if np.random.random() > 0.7:
                    image[i, j, :] = flowers[np.random.randint(0, len(flowers))]
                else:
                    image[i, j, :] = [34, 139, 34]  # Green background

        # Add some trees (brown and green)
        for tree_x in [width // 6, width // 2, 5 * width // 6]:
            # Trunk
            trunk_width = 8
            trunk_height = 40
            trunk_start = grass_start + 20
            image[trunk_start:trunk_start + trunk_height,
                  tree_x - trunk_width // 2:tree_x + trunk_width // 2] = [139, 69, 19]  # Brown

            # Foliage (green circle)
            foliage_radius = 25
            mask_tree = (x - tree_x)**2 + (y - trunk_start)**2 <= foliage_radius**2
            current_green = image[mask_tree].copy()
            image[mask_tree] = [0, 128, 0]  # Dark green

        # Add noise for realism
        noise = np.random.randint(-15, 15, (height, width, 3))
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def quantize_colors(self, image, n_colors=16, method='kmeans'):
        """
        Perform color quantization using clustering

        Args:
            image: Input image (H, W, 3)
            n_colors: Number of colors in output
            method: 'kmeans' or 'minibatch'
        """
        h, w, d = image.shape
        image_array = image.reshape(-1, 3).astype(np.float64)

        # Sample pixels for faster computation
        if len(image_array) > 10000:
            image_array_sample = shuffle(image_array, random_state=self.random_state, n_samples=10000)
        else:
            image_array_sample = image_array

        # Perform clustering
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_colors, random_state=self.random_state,
                           n_init=10, max_iter=100)
            kmeans.fit(image_array_sample)
            labels = kmeans.predict(image_array)
            centers = kmeans.cluster_centers_
        else:  # minibatch
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=self.random_state,
                                    batch_size=100, n_init=10, max_iter=100)
            kmeans.fit(image_array_sample)
            labels = kmeans.predict(image_array)
            centers = kmeans.cluster_centers_

        # Reconstruct quantized image
        quantized = centers[labels].reshape(h, w, d).astype(np.uint8)

        return quantized, centers, labels

    def compare_color_counts(self, image, color_counts=[4, 8, 16, 32, 64]):
        """Compare quantization with different color counts"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Original image
        axes[0].imshow(image)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        axes[0].set_title(f'Original\n{unique_colors} unique colors', fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Quantized versions
        for idx, n_colors in enumerate(color_counts, 1):
            if idx >= len(axes):
                break

            quantized, centers, labels = self.quantize_colors(image, n_colors=n_colors)
            axes[idx].imshow(quantized)
            axes[idx].set_title(f'{n_colors} Colors\nCompression: {100*(1-n_colors/unique_colors):.1f}%',
                               fontsize=11, fontweight='bold')
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('/tmp/color_quantization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_color_palette(self, centers, labels, image_shape):
        """Visualize the extracted color palette"""
        # Count pixels per color
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_pixels = image_shape[0] * image_shape[1]
        percentages = (counts / total_pixels) * 100

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Color palette
        palette_height = 100
        palette = np.zeros((palette_height, len(centers) * 50, 3), dtype=np.uint8)
        for i, idx in enumerate(sorted_indices):
            color = centers[idx].astype(np.uint8)
            palette[:, i*50:(i+1)*50] = color

        ax1.imshow(palette)
        ax1.set_title('Extracted Color Palette (Sorted by Frequency)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Bar chart of color distribution
        colors_norm = centers[sorted_indices] / 255.0
        bars = ax2.bar(range(len(centers)), percentages[sorted_indices], color=colors_norm)
        ax2.set_xlabel('Color Index', fontsize=12)
        ax2.set_ylabel('Percentage of Pixels (%)', fontsize=12)
        ax2.set_title('Color Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages[sorted_indices])):
            if pct > 2:  # Only label significant colors
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('/tmp/color_palette.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_compression_quality(self, original, color_counts=[4, 8, 16, 32, 64]):
        """Analyze quality vs compression trade-off"""
        mse_scores = []
        compression_ratios = []
        silhouette_scores = []

        original_flat = original.reshape(-1, 3).astype(np.float64)
        original_colors = len(np.unique(original_flat, axis=0))

        for n_colors in color_counts:
            # Quantize
            quantized, centers, labels = self.quantize_colors(original, n_colors=n_colors)
            quantized_flat = quantized.reshape(-1, 3).astype(np.float64)

            # MSE (lower is better)
            mse = np.mean((original_flat - quantized_flat) ** 2)
            mse_scores.append(mse)

            # Compression ratio
            compression = 100 * (1 - n_colors / original_colors)
            compression_ratios.append(compression)

            # Silhouette score (sample for speed)
            if len(original_flat) > 5000:
                sample_idx = np.random.choice(len(original_flat), 5000, replace=False)
                score = silhouette_score(original_flat[sample_idx], labels[sample_idx])
            else:
                score = silhouette_score(original_flat, labels)
            silhouette_scores.append(score)

        # Plot analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MSE vs Colors
        ax1.plot(color_counts, mse_scores, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Colors', fontsize=12)
        ax1.set_ylabel('Mean Squared Error', fontsize=12)
        ax1.set_title('Reconstruction Error vs Color Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)

        # Silhouette vs Colors
        ax2.plot(color_counts, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Colors', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Clustering Quality vs Color Count', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig('/tmp/compression_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return mse_scores, silhouette_scores

    def compare_algorithms(self, image, n_colors=16):
        """Compare K-Means vs MiniBatch K-Means"""
        import time

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # K-Means
        start = time.time()
        quantized_km, centers_km, labels_km = self.quantize_colors(
            image, n_colors=n_colors, method='kmeans')
        time_km = time.time() - start

        axes[1].imshow(quantized_km)
        axes[1].set_title(f'K-Means\n{n_colors} colors, {time_km:.2f}s',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # MiniBatch K-Means
        start = time.time()
        quantized_mb, centers_mb, labels_mb = self.quantize_colors(
            image, n_colors=n_colors, method='minibatch')
        time_mb = time.time() - start

        axes[2].imshow(quantized_mb)
        axes[2].set_title(f'MiniBatch K-Means\n{n_colors} colors, {time_mb:.2f}s',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('/tmp/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nAlgorithm Comparison:")
        print(f"  K-Means:          {time_km:.3f}s")
        print(f"  MiniBatch:        {time_mb:.3f}s")
        print(f"  Speedup:          {time_km/time_mb:.2f}x")


def main():
    print("="*80)
    print("IMAGE COLOR QUANTIZATION USING CLUSTERING")
    print("="*80)

    # Initialize
    quantizer = ImageColorQuantization(random_state=42)

    # Generate synthetic image
    print("\n[1/5] Generating synthetic image...")
    image = quantizer.generate_synthetic_image(width=200, height=200)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
    print(f"Unique colors in original: {unique_colors}")

    # Display original
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title('Original Synthetic Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/original_image.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Compare different color counts
    print("\n[2/5] Comparing different color counts...")
    quantizer.compare_color_counts(image, color_counts=[4, 8, 16, 32, 64])

    # Analyze with 16 colors
    print("\n[3/5] Analyzing color palette (16 colors)...")
    quantized, centers, labels = quantizer.quantize_colors(image, n_colors=16)
    quantizer.visualize_color_palette(centers, labels, image.shape)

    # Quality analysis
    print("\n[4/5] Analyzing compression quality...")
    mse_scores, sil_scores = quantizer.analyze_compression_quality(
        image, color_counts=[4, 8, 16, 32, 64])

    print("\nQuality Metrics:")
    print(f"{'Colors':<10} {'MSE':<12} {'Silhouette':<12}")
    print("-" * 40)
    for n, mse, sil in zip([4, 8, 16, 32, 64], mse_scores, sil_scores):
        print(f"{n:<10} {mse:<12.2f} {sil:<12.4f}")

    # Compare algorithms
    print("\n[5/5] Comparing K-Means vs MiniBatch K-Means...")
    quantizer.compare_algorithms(image, n_colors=16)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nApplications:")
    print("  - Image compression for web/mobile")
    print("  - Palette extraction for design")
    print("  - Reducing image file sizes")
    print("  - Creating stylized/artistic effects")
    print("  - Improving image processing speed")


if __name__ == "__main__":
    main()
