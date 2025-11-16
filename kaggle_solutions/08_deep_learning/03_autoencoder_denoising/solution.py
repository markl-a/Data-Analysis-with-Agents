"""
Autoencoder for Image Denoising - Kaggle Solution
=================================================
Use autoencoders to remove noise from images.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class DenoisingAutoencoder:
    """Autoencoder for image denoising."""

    def __init__(self, input_shape=(28, 28, 1), encoding_dim=32):
        """Initialize Denoising Autoencoder.

        Args:
            input_shape: Shape of input images
            encoding_dim: Dimension of encoded representation
        """
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim

        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        # Build full autoencoder
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        """Build encoder network.

        Returns:
            Encoder model
        """
        encoder_input = layers.Input(shape=self.input_shape, name='encoder_input')

        # Encoding layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # Bottleneck
        x = layers.Flatten()(x)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(x)

        encoder = keras.Model(encoder_input, encoded, name='encoder')
        return encoder

    def build_decoder(self):
        """Build decoder network.

        Returns:
            Decoder model
        """
        decoder_input = layers.Input(shape=(self.encoding_dim,), name='decoder_input')

        # Calculate the shape after encoder's max pooling
        x = layers.Dense(4 * 4 * 128, activation='relu')(decoder_input)
        x = layers.Reshape((4, 4, 128))(x)

        # Decoding layers
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        # Output layer
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same',
                                name='decoded')(x)

        # Crop to match input shape if necessary
        decoded = layers.Cropping2D(cropping=((2, 2), (2, 2)))(decoded)

        decoder = keras.Model(decoder_input, decoded, name='decoder')
        return decoder

    def build_autoencoder(self):
        """Build complete autoencoder.

        Returns:
            Autoencoder model
        """
        autoencoder_input = layers.Input(shape=self.input_shape)
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        autoencoder = keras.Model(autoencoder_input, decoded, name='autoencoder')

        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return autoencoder

    def train(self, X_train_noisy, X_train_clean, X_val_noisy, X_val_clean,
              epochs=50, batch_size=128):
        """Train the autoencoder.

        Args:
            X_train_noisy: Noisy training images
            X_train_clean: Clean training images
            X_val_noisy: Noisy validation images
            X_val_clean: Clean validation images
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print("Starting autoencoder training...")
        start_time = time.time()

        history = self.autoencoder.fit(
            X_train_noisy, X_train_clean,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_noisy, X_val_clean),
            verbose=1
        )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        return history

    def denoise(self, noisy_images):
        """Denoise images.

        Args:
            noisy_images: Noisy images

        Returns:
            Denoised images
        """
        return self.autoencoder.predict(noisy_images, verbose=0)

    def encode(self, images):
        """Encode images to latent representation.

        Args:
            images: Input images

        Returns:
            Encoded representations
        """
        return self.encoder.predict(images, verbose=0)


def create_synthetic_images(n_samples=5000, img_size=28):
    """Create synthetic clean images.

    Args:
        n_samples: Number of samples to generate
        img_size: Size of images

    Returns:
        Array of clean images
    """
    print(f"Creating {n_samples} synthetic images...")

    images = []
    for _ in range(n_samples):
        # Create image with random geometric shapes
        img = np.zeros((img_size, img_size))

        # Random number of shapes (1-3)
        n_shapes = np.random.randint(1, 4)

        for _ in range(n_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])

            if shape_type == 'circle':
                # Add circle
                center_x = np.random.randint(5, img_size - 5)
                center_y = np.random.randint(5, img_size - 5)
                radius = np.random.randint(2, 6)

                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = 1.0

            else:
                # Add rectangle
                x1 = np.random.randint(0, img_size - 10)
                y1 = np.random.randint(0, img_size - 10)
                width = np.random.randint(5, 12)
                height = np.random.randint(5, 12)

                x2 = min(x1 + width, img_size)
                y2 = min(y1 + height, img_size)

                img[y1:y2, x1:x2] = 1.0

        images.append(img)

    images = np.array(images)
    images = images.reshape(n_samples, img_size, img_size, 1)

    print(f"Created {n_samples} clean images")
    return images


def add_noise(images, noise_factor=0.5):
    """Add Gaussian noise to images.

    Args:
        images: Clean images
        noise_factor: Amount of noise to add

    Returns:
        Noisy images
    """
    noisy_images = images + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=images.shape
    )
    noisy_images = np.clip(noisy_images, 0.0, 1.0)
    return noisy_images


def visualize_results(clean_images, noisy_images, denoised_images, history):
    """Visualize denoising results.

    Args:
        clean_images: Original clean images
        noisy_images: Noisy images
        denoised_images: Denoised images
        history: Training history
    """
    print("Generating visualizations...")

    # Plot sample images
    n_samples = 5
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))

    for i in range(n_samples):
        # Clean images
        axes[0, i].imshow(clean_images[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[0, i].set_ylabel('Clean', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Noisy images
        axes[1, i].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[1, i].set_ylabel('Noisy', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')

        # Denoised images
        axes[2, i].imshow(denoised_images[i].reshape(28, 28), cmap='gray')
        if i == 0:
            axes[2, i].set_ylabel('Denoised', fontsize=12, fontweight='bold')
        axes[2, i].axis('off')

    plt.suptitle('Autoencoder Denoising Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=300, bbox_inches='tight')
    print("Denoising results saved to 'denoising_results.png'")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curves
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'training_curves.png'")

    # Calculate and display metrics
    mse = np.mean((clean_images - denoised_images) ** 2)
    mae = np.mean(np.abs(clean_images - denoised_images))
    psnr = 10 * np.log10(1.0 / mse)

    print(f"\nDenoising Performance Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Autoencoder for Image Denoising - Kaggle Solution")
    print("=" * 60)

    # Create synthetic dataset
    print("\nCreating dataset...")
    clean_images = create_synthetic_images(n_samples=5000, img_size=28)

    # Split into train and validation
    X_train_clean, X_val_clean = train_test_split(
        clean_images, test_size=0.2, random_state=42
    )

    # Add noise
    print("\nAdding noise to images...")
    X_train_noisy = add_noise(X_train_clean, noise_factor=0.5)
    X_val_noisy = add_noise(X_val_clean, noise_factor=0.5)

    print(f"Training set: {X_train_clean.shape[0]} images")
    print(f"Validation set: {X_val_clean.shape[0]} images")

    # Initialize autoencoder
    print("\nInitializing autoencoder...")
    autoencoder = DenoisingAutoencoder(input_shape=(28, 28, 1), encoding_dim=32)

    # Print model summary
    print("\n" + "=" * 60)
    print("AUTOENCODER ARCHITECTURE")
    print("=" * 60)
    autoencoder.autoencoder.summary()

    # Train autoencoder
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = autoencoder.train(
        X_train_noisy, X_train_clean,
        X_val_noisy, X_val_clean,
        epochs=30,
        batch_size=128
    )

    # Denoise test images
    print("\nDenoising validation images...")
    denoised_images = autoencoder.denoise(X_val_noisy[:5])

    # Visualize results
    visualize_results(X_val_clean[:5], X_val_noisy[:5], denoised_images, history)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Loss: {history.history['loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Final Training MAE: {history.history['mae'][-1]:.6f}")
    print(f"Final Validation MAE: {history.history['val_mae'][-1]:.6f}")
    print(f"Total Training Epochs: {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
