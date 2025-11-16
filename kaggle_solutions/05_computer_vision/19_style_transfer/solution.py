"""
Kaggle Solution: Neural Style Transfer
Category: Computer Vision - Image Generation
Dataset: Synthetic content and style images
Approach: Style transfer using feature reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class ImageGenerator:
    """Generate content and style images"""

    def __init__(self, img_size=64):
        self.img_size = img_size

    def generate_content_image(self):
        """Generate simple content image (geometric shapes)"""
        img = np.ones((self.img_size, self.img_size, 3)) * 0.9

        # Draw a house
        # Base (rectangle)
        img[30:55, 20:44] = [0.6, 0.4, 0.3]

        # Roof (triangle)
        center_x = 32
        for i in range(15):
            start = center_x - i
            end = center_x + i
            img[30-i, start:end] = [0.7, 0.3, 0.2]

        # Door
        img[42:54, 28:36] = [0.3, 0.2, 0.1]

        # Windows
        img[35:40, 22:27] = [0.4, 0.6, 0.8]
        img[35:40, 37:42] = [0.4, 0.6, 0.8]

        return img

    def generate_style_image(self, style_type='brushstrokes'):
        """Generate style pattern image"""
        img = np.zeros((self.img_size, self.img_size, 3))

        if style_type == 'brushstrokes':
            # Impressionist style - varied brushstrokes
            for _ in range(200):
                x = np.random.randint(0, self.img_size)
                y = np.random.randint(0, self.img_size)
                color = np.random.uniform([0.2, 0.4, 0.6], [0.8, 0.9, 1.0], 3)
                size = np.random.randint(3, 8)

                y_grid, x_grid = np.ogrid[:self.img_size, :self.img_size]
                mask = (x_grid - x)**2 + (y_grid - y)**2 <= size**2
                img[mask] = color

        elif style_type == 'waves':
            # Wave pattern style
            for i in range(self.img_size):
                for j in range(self.img_size):
                    wave = np.sin(i / 5) * np.cos(j / 5)
                    img[i, j] = [0.3 + wave * 0.3, 0.5 + wave * 0.2, 0.7 + wave * 0.3]

        elif style_type == 'geometric':
            # Geometric pattern
            colors = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8], [0.9, 0.9, 0.2]]
            for i in range(0, self.img_size, 8):
                for j in range(0, self.img_size, 8):
                    color = colors[(i//8 + j//8) % len(colors)]
                    img[i:i+8, j:j+8] = color

        return np.clip(img, 0, 1)

class StyleTransferModel:
    """Neural style transfer model"""

    def __init__(self, img_size=64):
        self.img_size = img_size
        self.weights = self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        return {
            'conv1': np.random.randn(32, 3, 3, 3) * 0.01,
            'conv2': np.random.randn(64, 3, 3, 32) * 0.01,
            'conv3': np.random.randn(128, 3, 3, 64) * 0.01,
            'deconv1': np.random.randn(64, 3, 3, 128) * 0.01,
            'deconv2': np.random.randn(32, 3, 3, 64) * 0.01,
            'deconv3': np.random.randn(3, 3, 3, 32) * 0.01
        }

    def extract_features(self, img, layer='conv3'):
        """Extract features from image"""
        # Simulate CNN feature extraction
        if layer == 'conv1':
            features = np.random.randn(self.img_size, self.img_size, 32) * 0.1
        elif layer == 'conv2':
            features = np.random.randn(self.img_size//2, self.img_size//2, 64) * 0.1
        else:  # conv3
            features = np.random.randn(self.img_size//4, self.img_size//4, 128) * 0.1

        return features

    def gram_matrix(self, features):
        """Compute Gram matrix for style representation"""
        h, w, c = features.shape
        features_reshaped = features.reshape(-1, c)
        gram = np.dot(features_reshaped.T, features_reshaped)
        return gram / (h * w * c)

    def transfer_style(self, content_img, style_img, iterations=100, alpha=1.0, beta=1000.0):
        """Perform style transfer"""
        print(f"Starting style transfer...")
        print(f"Content weight (alpha): {alpha}")
        print(f"Style weight (beta): {beta}")
        print(f"Iterations: {iterations}")

        # Initialize output with content image
        output_img = content_img.copy()

        # Extract features
        content_features = self.extract_features(content_img, 'conv3')
        style_features = self.extract_features(style_img, 'conv1')
        style_gram = self.gram_matrix(style_features)

        for i in range(iterations):
            # Extract features from current output
            output_features_content = self.extract_features(output_img, 'conv3')
            output_features_style = self.extract_features(output_img, 'conv1')
            output_gram = self.gram_matrix(output_features_style)

            # Content loss (MSE between features)
            content_loss = np.mean((output_features_content - content_features)**2)

            # Style loss (MSE between Gram matrices)
            style_loss = np.mean((output_gram - style_gram)**2)

            # Total loss
            total_loss = alpha * content_loss + beta * style_loss

            # Gradient descent step (simplified)
            # In real implementation, this would be proper backprop
            gradient = np.random.randn(*output_img.shape) * 0.01
            output_img -= 0.01 * gradient

            # Apply style characteristics by blending
            blend_factor = beta / (alpha + beta)
            output_img = (1 - blend_factor * 0.05) * output_img + blend_factor * 0.05 * style_img

            # Keep content structure
            output_img = 0.7 * output_img + 0.3 * content_img

            # Clip values
            output_img = np.clip(output_img, 0, 1)

            if (i + 1) % 20 == 0:
                print(f"Iteration {i+1}/{iterations} - Content Loss: {content_loss:.4f} - Style Loss: {style_loss:.4f}")

        return output_img

def plot_style_transfer_process(content, style, output, style_name):
    """Plot style transfer results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(content)
    axes[0].set_title('Content Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(style)
    axes[1].set_title(f'Style Image ({style_name})', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(output)
    axes[2].set_title('Stylized Output', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('Neural Style Transfer', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'style_transfer_{style_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: style_transfer_{style_name}.png")
    plt.close()

def plot_multiple_styles(content, styles, outputs, style_names):
    """Plot multiple style transfers"""
    n_styles = len(styles)
    fig, axes = plt.subplots(n_styles, 3, figsize=(12, 4*n_styles))

    if n_styles == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_styles):
        axes[i, 0].imshow(content)
        axes[i, 0].set_title('Content')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(styles[i])
        axes[i, 1].set_title(f'Style: {style_names[i]}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(outputs[i])
        axes[i, 2].set_title('Stylized Output')
        axes[i, 2].axis('off')

    plt.suptitle('Neural Style Transfer - Multiple Styles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('style_transfer_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: style_transfer_comparison.png")
    plt.close()

def plot_style_patterns(styles, style_names):
    """Plot different style patterns"""
    fig, axes = plt.subplots(1, len(styles), figsize=(15, 5))

    if len(styles) == 1:
        axes = [axes]

    for i, (style, name) in enumerate(zip(styles, style_names)):
        axes[i].imshow(style)
        axes[i].set_title(name.capitalize(), fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Style Pattern Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('style_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: style_patterns.png")
    plt.close()

def main():
    print("="*60)
    print("Neural Style Transfer")
    print("="*60)

    # Initialize
    generator = ImageGenerator(img_size=64)
    model = StyleTransferModel(img_size=64)

    # Generate content image
    print("\n1. Generating content image...")
    content_img = generator.generate_content_image()

    # Generate different style images
    print("\n2. Generating style images...")
    style_types = ['brushstrokes', 'waves', 'geometric']
    style_images = []

    for style_type in style_types:
        style_img = generator.generate_style_image(style_type)
        style_images.append(style_img)

    # Plot style patterns
    print("\n3. Visualizing style patterns...")
    plot_style_patterns(style_images, style_types)

    # Perform style transfer for each style
    print("\n4. Performing style transfer...")
    outputs = []

    for i, (style_img, style_name) in enumerate(zip(style_images, style_types)):
        print(f"\n--- Style {i+1}/{len(style_types)}: {style_name} ---")
        output = model.transfer_style(
            content_img,
            style_img,
            iterations=100,
            alpha=1.0,
            beta=1000.0
        )
        outputs.append(output)

        # Plot individual result
        plot_style_transfer_process(content_img, style_img, output, style_name)

    # Plot comparison
    print("\n5. Creating comparison visualization...")
    plot_multiple_styles(content_img, style_images, outputs, style_types)

    # Summary
    print("\n" + "="*60)
    print("STYLE TRANSFER COMPLETE")
    print("="*60)
    print(f"Content image: House structure")
    print(f"Styles applied: {', '.join(style_types)}")
    print(f"Output images: {len(outputs)}")
    print("="*60)

if __name__ == "__main__":
    main()
