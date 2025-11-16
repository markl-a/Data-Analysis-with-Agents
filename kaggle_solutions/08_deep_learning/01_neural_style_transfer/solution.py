"""
Neural Style Transfer - Kaggle Solution
========================================
Transfer artistic style from one image to another using VGG19 features.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing import image as kp_image
import matplotlib.pyplot as plt
from PIL import Image
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class NeuralStyleTransfer:
    """Neural Style Transfer using VGG19 pretrained model."""

    def __init__(self, img_height=400, img_width=400):
        """Initialize Neural Style Transfer.

        Args:
            img_height: Height of output image
            img_width: Width of output image
        """
        self.img_height = img_height
        self.img_width = img_width

        # Layers for content and style representations
        self.content_layers = ['block5_conv2']
        self.style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        # Build model
        self.model = self.build_model()

    def build_model(self):
        """Build VGG19 model for extracting features."""
        # Load VGG19 without top layers
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Get outputs from specified layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs

        # Build model
        model = keras.Model(vgg.input, model_outputs)
        return model

    def preprocess_image(self, img_array):
        """Preprocess image for VGG19."""
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess for VGG19
        img_array = vgg19.preprocess_input(img_array)
        return img_array

    def deprocess_image(self, processed_img):
        """Convert preprocessed image back to displayable format."""
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)

        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # BGR to RGB
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def load_and_process_image(self, img_array):
        """Load and process image."""
        img = tf.image.resize(img_array, (self.img_height, self.img_width))
        img = self.preprocess_image(img)
        return img

    def get_feature_representations(self, content_img, style_img):
        """Extract content and style features."""
        # Preprocess images
        content_image = self.load_and_process_image(content_img)
        style_image = self.load_and_process_image(style_img)

        # Get features
        content_outputs = self.model(content_image)
        style_outputs = self.model(style_image)

        # Separate style and content features
        style_features = [style_layer for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer for content_layer in content_outputs[self.num_style_layers:]]

        return style_features, content_features

    def compute_content_loss(self, content_features, target_features):
        """Compute content loss."""
        return tf.reduce_mean(tf.square(content_features - target_features))

    def gram_matrix(self, input_tensor):
        """Compute Gram matrix for style representation."""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def compute_style_loss(self, style_features, target_features):
        """Compute style loss using Gram matrices."""
        style_gram = self.gram_matrix(style_features)
        target_gram = self.gram_matrix(target_features)
        return tf.reduce_mean(tf.square(style_gram - target_gram))

    def compute_loss(self, outputs, style_targets, content_targets,
                     style_weight=1e-2, content_weight=1e4):
        """Compute total loss."""
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Content loss
        content_loss = tf.add_n([self.compute_content_loss(content_outputs[i], content_targets[i])
                                  for i in range(self.num_content_layers)])
        content_loss *= content_weight / self.num_content_layers

        # Style loss
        style_loss = tf.add_n([self.compute_style_loss(style_outputs[i], style_targets[i])
                                for i in range(self.num_style_layers)])
        style_loss *= style_weight / self.num_style_layers

        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss

    @tf.function()
    def train_step(self, image, style_targets, content_targets, optimizer,
                   style_weight=1e-2, content_weight=1e4):
        """Single training step."""
        with tf.GradientTape() as tape:
            outputs = self.model(image)
            loss, content_loss, style_loss = self.compute_loss(
                outputs, style_targets, content_targets,
                style_weight, content_weight
            )

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, -103.939, 151.061))

        return loss, content_loss, style_loss

    def transfer_style(self, content_img, style_img, epochs=10,
                       steps_per_epoch=100, style_weight=1e-2,
                       content_weight=1e4, learning_rate=5.0):
        """Transfer style from style_img to content_img.

        Args:
            content_img: Content image array
            style_img: Style image array
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            learning_rate: Learning rate for optimizer

        Returns:
            Generated image and loss history
        """
        # Get target features
        style_targets, content_targets = self.get_feature_representations(
            content_img, style_img
        )

        # Initialize generated image with content image
        generated_image = self.load_and_process_image(content_img)
        generated_image = tf.Variable(generated_image, dtype=tf.float32)

        # Optimizer
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        print("Starting style transfer...")
        history = {'total_loss': [], 'content_loss': [], 'style_loss': []}

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()

            for step in range(steps_per_epoch):
                loss, content_loss, style_loss = self.train_step(
                    generated_image, style_targets, content_targets,
                    optimizer, style_weight, content_weight
                )

                history['total_loss'].append(float(loss))
                history['content_loss'].append(float(content_loss))
                history['style_loss'].append(float(style_loss))

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {loss:.2f} (Content: {content_loss:.2f}, Style: {style_loss:.2f}) - "
                  f"Time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"\nStyle transfer completed in {total_time:.2f}s")

        return generated_image, history


def create_sample_images():
    """Create sample content and style images."""
    # Create a simple content image (landscape with shapes)
    content = np.ones((400, 400, 3), dtype=np.uint8) * 200

    # Add some geometric shapes
    content[100:300, 150:250] = [100, 150, 200]  # Blue rectangle
    cv2_available = False
    try:
        import cv2
        cv2_available = True
    except:
        pass

    if cv2_available:
        import cv2
        cv2.circle(content, (300, 200), 50, (200, 100, 100), -1)  # Red circle
    else:
        # Manual circle drawing
        y, x = np.ogrid[:400, :400]
        mask = (x - 300)**2 + (y - 200)**2 <= 50**2
        content[mask] = [200, 100, 100]

    # Create a style image (artistic pattern)
    style = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(0, 400, 20):
        for j in range(0, 400, 20):
            color = [(i * j) % 255, (i + j) % 255, (255 - i) % 255]
            style[i:i+10, j:j+10] = color

    return content.astype(np.float32), style.astype(np.float32)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Neural Style Transfer - Kaggle Solution")
    print("=" * 60)

    # Create sample images
    print("\nCreating sample images...")
    content_img, style_img = create_sample_images()

    # Initialize style transfer
    nst = NeuralStyleTransfer(img_height=400, img_width=400)

    # Perform style transfer
    generated_image, history = nst.transfer_style(
        content_img, style_img,
        epochs=5,
        steps_per_epoch=50,
        style_weight=1e-2,
        content_weight=1e4,
        learning_rate=5.0
    )

    # Convert generated image back to displayable format
    final_image = nst.deprocess_image(generated_image.numpy())

    # Visualize results
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Content image
    axes[0, 0].imshow(content_img.astype(np.uint8))
    axes[0, 0].set_title('Content Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Style image
    axes[0, 1].imshow(style_img.astype(np.uint8))
    axes[0, 1].set_title('Style Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Generated image
    axes[1, 0].imshow(final_image)
    axes[1, 0].set_title('Generated Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Loss curves
    axes[1, 1].plot(history['total_loss'], label='Total Loss', linewidth=2)
    axes[1, 1].plot(history['content_loss'], label='Content Loss', linewidth=2, alpha=0.7)
    axes[1, 1].plot(history['style_loss'], label='Style Loss', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neural_style_transfer_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'neural_style_transfer_results.png'")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Total Loss: {history['total_loss'][-1]:.2f}")
    print(f"Final Content Loss: {history['content_loss'][-1]:.2f}")
    print(f"Final Style Loss: {history['style_loss'][-1]:.2f}")
    print(f"Total Iterations: {len(history['total_loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
