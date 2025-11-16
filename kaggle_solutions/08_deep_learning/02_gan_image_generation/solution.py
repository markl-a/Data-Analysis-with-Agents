"""
GAN for Image Generation - Kaggle Solution
==========================================
Generate synthetic images using Generative Adversarial Networks (GANs).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class GAN:
    """Generative Adversarial Network for image generation."""

    def __init__(self, latent_dim=100, img_shape=(28, 28, 1)):
        """Initialize GAN.

        Args:
            latent_dim: Dimension of latent noise vector
            img_shape: Shape of generated images
        """
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_rows, self.img_cols, self.channels = img_shape

        # Build generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Build combined model (stacked generator and discriminator)
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = keras.Model(z, validity)
        self.combined.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

    def build_generator(self):
        """Build generator network.

        Returns:
            Generator model
        """
        model = keras.Sequential([
            # Input layer
            layers.Dense(128 * 7 * 7, activation='relu', input_dim=self.latent_dim),
            layers.Reshape((7, 7, 128)),
            layers.BatchNormalization(momentum=0.8),

            # Upsample to 14x14
            layers.UpSampling2D(),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(momentum=0.8),

            # Upsample to 28x28
            layers.UpSampling2D(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.Activation('relu'),
            layers.BatchNormalization(momentum=0.8),

            # Output layer
            layers.Conv2D(self.channels, kernel_size=3, padding='same'),
            layers.Activation('tanh')
        ], name='generator')

        return model

    def build_discriminator(self):
        """Build discriminator network.

        Returns:
            Discriminator model
        """
        model = keras.Sequential([
            # Input layer
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
                         input_shape=self.img_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            # Hidden layers
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            layers.BatchNormalization(momentum=0.8),

            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            layers.BatchNormalization(momentum=0.8),

            layers.Conv2D(256, kernel_size=3, strides=1, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),

            # Output layer
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        return model

    def train(self, X_train, epochs=10000, batch_size=32, sample_interval=1000):
        """Train the GAN.

        Args:
            X_train: Training images
            epochs: Number of training epochs
            batch_size: Batch size
            sample_interval: Interval for sampling images

        Returns:
            Training history
        """
        # Rescale images to [-1, 1]
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Training history
        history = {
            'd_loss': [],
            'd_acc': [],
            'g_loss': []
        }

        print("Starting GAN training...")
        start_time = time.time()

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_imgs = X_train[idx]

            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_imgs = self.generator.predict(noise, verbose=0)

            # Train discriminator on real and fake images
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Save history
            history['d_loss'].append(d_loss[0])
            history['d_acc'].append(100 * d_loss[1])
            history['g_loss'].append(g_loss)

            # Print progress
            if epoch % sample_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{epochs} - "
                      f"D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% - "
                      f"G loss: {g_loss:.4f} - "
                      f"Time: {elapsed_time:.2f}s")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        return history

    def generate_images(self, n_samples=25):
        """Generate sample images.

        Args:
            n_samples: Number of images to generate

        Returns:
            Generated images
        """
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_images = self.generator.predict(noise, verbose=0)

        # Rescale images to [0, 1]
        generated_images = 0.5 * generated_images + 0.5

        return generated_images


def create_synthetic_dataset(n_samples=10000):
    """Create synthetic image dataset.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Array of synthetic images
    """
    print("Creating synthetic dataset...")

    images = []
    for _ in range(n_samples):
        # Create 28x28 image with random shapes
        img = np.zeros((28, 28))

        # Random circle
        center_x = np.random.randint(7, 21)
        center_y = np.random.randint(7, 21)
        radius = np.random.randint(3, 7)

        y, x = np.ogrid[:28, :28]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask] = 1.0

        # Add some noise
        img += np.random.normal(0, 0.1, (28, 28))
        img = np.clip(img, 0, 1)

        images.append(img)

    images = np.array(images)
    images = images.reshape(n_samples, 28, 28, 1) * 255

    print(f"Created {n_samples} synthetic images")
    return images


def visualize_results(real_images, generated_images, history):
    """Visualize training results.

    Args:
        real_images: Real training images
        generated_images: Generated images
        history: Training history
    """
    fig = plt.figure(figsize=(15, 10))

    # Plot real images
    print("Generating visualization...")
    for i in range(5):
        plt.subplot(4, 5, i + 1)
        plt.imshow(real_images[i].reshape(28, 28), cmap='gray')
        plt.title('Real Image')
        plt.axis('off')

    # Plot generated images
    for i in range(5):
        plt.subplot(4, 5, i + 6)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.title('Generated Image')
        plt.axis('off')

    # Plot more generated images
    for i in range(10):
        plt.subplot(4, 5, i + 11)
        plt.imshow(generated_images[i + 5].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('gan_generated_images.png', dpi=300, bbox_inches='tight')
    print("Generated images saved to 'gan_generated_images.png'")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Discriminator loss
    axes[0].plot(history['d_loss'], linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Discriminator Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Discriminator accuracy
    axes[1].plot(history['d_acc'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Discriminator Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Generator loss
    axes[2].plot(history['g_loss'], linewidth=2, color='red')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Generator Loss', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gan_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'gan_training_curves.png'")


def main():
    """Main execution function."""
    print("=" * 60)
    print("GAN for Image Generation - Kaggle Solution")
    print("=" * 60)

    # Create synthetic dataset
    X_train = create_synthetic_dataset(n_samples=5000)

    # Initialize GAN
    print("\nInitializing GAN...")
    gan = GAN(latent_dim=100, img_shape=(28, 28, 1))

    # Print model summaries
    print("\n" + "=" * 60)
    print("GENERATOR ARCHITECTURE")
    print("=" * 60)
    gan.generator.summary()

    print("\n" + "=" * 60)
    print("DISCRIMINATOR ARCHITECTURE")
    print("=" * 60)
    gan.discriminator.summary()

    # Train GAN
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = gan.train(
        X_train,
        epochs=3000,
        batch_size=32,
        sample_interval=500
    )

    # Generate images
    print("\nGenerating sample images...")
    generated_images = gan.generate_images(n_samples=25)

    # Visualize results
    visualize_results(X_train[:5] / 255.0, generated_images, history)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Discriminator Loss: {history['d_loss'][-1]:.4f}")
    print(f"Final Discriminator Accuracy: {history['d_acc'][-1]:.2f}%")
    print(f"Final Generator Loss: {history['g_loss'][-1]:.4f}")
    print(f"Total Training Epochs: {len(history['d_loss'])}")
    print(f"Number of Generated Images: {len(generated_images)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
