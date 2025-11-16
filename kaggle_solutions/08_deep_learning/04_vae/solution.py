"""
Variational Autoencoder (VAE) - Kaggle Solution
===============================================
Generate new images using Variational Autoencoders.
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


class Sampling(layers.Layer):
    """Sampling layer for VAE using reparameterization trick."""

    def call(self, inputs):
        """Sample from latent distribution.

        Args:
            inputs: [z_mean, z_log_var]

        Returns:
            Sampled latent vector
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE:
    """Variational Autoencoder for generative modeling."""

    def __init__(self, input_shape=(28, 28, 1), latent_dim=2):
        """Initialize VAE.

        Args:
            input_shape: Shape of input images
            latent_dim: Dimension of latent space
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build encoder, decoder, and full VAE
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):
        """Build encoder network.

        Returns:
            Encoder model
        """
        encoder_input = layers.Input(shape=self.input_shape, name='encoder_input')

        # Encoder layers
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(x)

        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)

        # Sample from latent distribution
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        """Build decoder network.

        Returns:
            Decoder model
        """
        latent_input = layers.Input(shape=(self.latent_dim,), name='decoder_input')

        # Decoder layers
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent_input)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        decoder_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

        decoder = keras.Model(latent_input, decoder_output, name='decoder')
        return decoder

    def build_vae(self):
        """Build complete VAE model.

        Returns:
            VAE model
        """
        # Custom training step for VAE
        class VAEModel(keras.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super().__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(
                    name="reconstruction_loss"
                )
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                ]

            def train_step(self, data):
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(data)
                    reconstruction = self.decoder(z)
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            keras.losses.binary_crossentropy(data, reconstruction),
                            axis=(1, 2)
                        )
                    )
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    total_loss = reconstruction_loss + kl_loss

                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)

                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }

            def test_step(self, data):
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction),
                        axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss

                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }

        vae = VAEModel(self.encoder, self.decoder)
        vae.compile(optimizer=keras.optimizers.Adam())

        return vae

    def train(self, X_train, X_val, epochs=30, batch_size=128):
        """Train the VAE.

        Args:
            X_train: Training images
            X_val: Validation images
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        print("Starting VAE training...")
        start_time = time.time()

        history = self.vae.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val,),
            verbose=1
        )

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        return history

    def generate(self, n_samples=25):
        """Generate new images by sampling from latent space.

        Args:
            n_samples: Number of images to generate

        Returns:
            Generated images
        """
        # Sample from standard normal distribution
        z_samples = np.random.normal(size=(n_samples, self.latent_dim))
        generated_images = self.decoder.predict(z_samples, verbose=0)
        return generated_images

    def encode(self, images):
        """Encode images to latent space.

        Args:
            images: Input images

        Returns:
            Latent representations [z_mean, z_log_var, z]
        """
        return self.encoder.predict(images, verbose=0)


def create_synthetic_images(n_samples=5000, img_size=28):
    """Create synthetic images.

    Args:
        n_samples: Number of samples
        img_size: Image size

    Returns:
        Array of images
    """
    print(f"Creating {n_samples} synthetic images...")

    images = []
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size))

        # Random shape type
        shape_type = np.random.choice(['circle', 'square', 'triangle'])

        if shape_type == 'circle':
            center_x = np.random.randint(8, 20)
            center_y = np.random.randint(8, 20)
            radius = np.random.randint(4, 8)

            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = 1.0

        elif shape_type == 'square':
            x1 = np.random.randint(5, 15)
            y1 = np.random.randint(5, 15)
            size = np.random.randint(8, 14)
            x2 = min(x1 + size, img_size)
            y2 = min(y1 + size, img_size)
            img[y1:y2, x1:x2] = 1.0

        else:  # triangle
            center_x = np.random.randint(8, 20)
            top_y = np.random.randint(5, 10)
            bottom_y = np.random.randint(18, 23)
            for y in range(top_y, bottom_y):
                width = int((y - top_y) * 0.5)
                x1 = max(0, center_x - width)
                x2 = min(img_size, center_x + width)
                img[y, x1:x2] = 1.0

        images.append(img)

    images = np.array(images)
    images = images.reshape(n_samples, img_size, img_size, 1)

    print(f"Created {n_samples} images")
    return images


def visualize_results(vae, X_test, history):
    """Visualize VAE results.

    Args:
        vae: Trained VAE model
        X_test: Test images
        history: Training history
    """
    print("Generating visualizations...")

    # Generate new images
    generated_images = vae.generate(n_samples=25)

    # Plot generated images
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            axes[i, j].imshow(generated_images[idx].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')

    plt.suptitle('Generated Images from VAE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_generated_images.png', dpi=300, bbox_inches='tight')
    print("Generated images saved to 'vae_generated_images.png'")

    # Visualize latent space (if 2D)
    if vae.latent_dim == 2:
        # Encode test images
        z_mean, _, _ = vae.encode(X_test)

        plt.figure(figsize=(10, 8))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=10)
        plt.xlabel('Latent Dimension 1', fontsize=12)
        plt.ylabel('Latent Dimension 2', fontsize=12)
        plt.title('Latent Space Distribution', fontsize=14, fontweight='bold')
        plt.colorbar(label='Sample Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('vae_latent_space.png', dpi=300, bbox_inches='tight')
        print("Latent space visualization saved to 'vae_latent_space.png'")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history.history['loss'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(history.history['reconstruction_loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_reconstruction_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL loss
    axes[2].plot(history.history['kl_loss'], label='Training', linewidth=2)
    axes[2].plot(history.history['val_kl_loss'], label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL Divergence', fontsize=12)
    axes[2].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vae_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved to 'vae_training_curves.png'")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Variational Autoencoder (VAE) - Kaggle Solution")
    print("=" * 60)

    # Create synthetic dataset
    print("\nCreating dataset...")
    images = create_synthetic_images(n_samples=5000, img_size=28)

    # Split into train and validation
    X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")

    # Initialize VAE
    print("\nInitializing VAE...")
    vae = VAE(input_shape=(28, 28, 1), latent_dim=2)

    # Print model summaries
    print("\n" + "=" * 60)
    print("ENCODER ARCHITECTURE")
    print("=" * 60)
    vae.encoder.summary()

    print("\n" + "=" * 60)
    print("DECODER ARCHITECTURE")
    print("=" * 60)
    vae.decoder.summary()

    # Train VAE
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = vae.train(X_train, X_val, epochs=30, batch_size=128)

    # Visualize results
    visualize_results(vae, X_val, history)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.4f}")
    print(f"Final KL Divergence: {history.history['kl_loss'][-1]:.4f}")
    print(f"Total Training Epochs: {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
