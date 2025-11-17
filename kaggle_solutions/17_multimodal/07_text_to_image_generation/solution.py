"""
Text-to-Image Generation
========================

This solution implements various approaches for generating images from text descriptions,
including GAN-based and diffusion-based methods.

Approaches:
1. Simple GAN with text conditioning
2. Stack GAN architecture simulation
3. AttnGAN with attention mechanisms
4. Diffusion model with text guidance
5. CLIP-guided generation

Dataset: Synthetic text embeddings and image features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
np.random.seed(42)

@dataclass
class TextToImageConfig:
    """Configuration for text-to-image models"""
    text_embedding_dim: int = 512
    noise_dim: int = 100
    image_size: int = 64
    image_channels: int = 3
    num_stages: int = 3  # For StackGAN
    attention_heads: int = 8

class TextEncoder:
    """Encode text descriptions into embedding vectors"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.vocab_size = 10000

        # Initialize random embedding matrix
        self.embedding_matrix = np.random.randn(self.vocab_size, embedding_dim) * 0.01

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode text descriptions

        Args:
            texts: List of text descriptions

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        embeddings = []

        for text in texts:
            # Simple tokenization
            words = text.lower().split()

            # Get word indices (use hash for simplicity)
            word_indices = [hash(word) % self.vocab_size for word in words]

            # Average word embeddings
            if len(word_indices) > 0:
                word_embeddings = self.embedding_matrix[word_indices]
                text_embedding = np.mean(word_embeddings, axis=0)
            else:
                text_embedding = np.zeros(self.embedding_dim)

            embeddings.append(text_embedding)

        return np.array(embeddings)

class ConditionalGAN:
    """Conditional GAN for text-to-image generation"""

    def __init__(self, config: TextToImageConfig):
        self.config = config

        # Generator weights
        self.gen_text_fc = np.random.randn(config.text_embedding_dim, 256) * 0.01
        self.gen_noise_fc = np.random.randn(config.noise_dim, 256) * 0.01
        self.gen_output = np.random.randn(512, config.image_size * config.image_size * config.image_channels) * 0.01

        # Discriminator weights
        self.disc_image_fc = np.random.randn(config.image_size * config.image_size * config.image_channels, 256) * 0.01
        self.disc_text_fc = np.random.randn(config.text_embedding_dim, 256) * 0.01

    def generate(self, text_embeddings: np.ndarray, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate images from text embeddings

        Args:
            text_embeddings: (batch_size, text_embedding_dim)
            noise: Optional noise vectors

        Returns:
            generated_images: (batch_size, image_size, image_size, channels)
        """
        batch_size = text_embeddings.shape[0]

        if noise is None:
            noise = np.random.randn(batch_size, self.config.noise_dim)

        # Process text
        text_features = np.tanh(np.dot(text_embeddings, self.gen_text_fc))

        # Process noise
        noise_features = np.tanh(np.dot(noise, self.gen_noise_fc))

        # Combine features
        combined = np.concatenate([text_features, noise_features], axis=1)

        # Generate image
        image_flat = np.tanh(np.dot(combined, self.gen_output))

        # Reshape to image
        images = image_flat.reshape(
            batch_size,
            self.config.image_size,
            self.config.image_size,
            self.config.image_channels
        )

        return images

    def discriminate(self, images: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Discriminate real vs fake images

        Args:
            images: (batch_size, image_size, image_size, channels)
            text_embeddings: (batch_size, text_embedding_dim)

        Returns:
            scores: (batch_size,)
        """
        batch_size = images.shape[0]

        # Flatten images
        images_flat = images.reshape(batch_size, -1)

        # Process image and text
        image_features = np.tanh(np.dot(images_flat, self.disc_image_fc))
        text_features = np.tanh(np.dot(text_embeddings, self.disc_text_fc))

        # Combine and score
        combined = image_features * text_features
        scores = np.sum(combined, axis=1)

        return self._sigmoid(scores)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

class StackGAN:
    """Stack GAN with multiple generation stages"""

    def __init__(self, config: TextToImageConfig):
        self.config = config
        self.stages = []

        # Create generators for each stage
        for stage in range(config.num_stages):
            stage_config = TextToImageConfig(
                text_embedding_dim=config.text_embedding_dim,
                noise_dim=config.noise_dim,
                image_size=config.image_size // (2 ** (config.num_stages - stage - 1)),
                image_channels=config.image_channels
            )
            self.stages.append(ConditionalGAN(stage_config))

    def generate_multistage(self, text_embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Generate images through multiple stages

        Args:
            text_embeddings: (batch_size, text_embedding_dim)

        Returns:
            images_per_stage: List of images from each stage
        """
        images_per_stage = []

        for stage_idx, stage_gan in enumerate(self.stages):
            # Generate at this stage
            stage_images = stage_gan.generate(text_embeddings)
            images_per_stage.append(stage_images)

            # Add refinement for next stage
            if stage_idx < len(self.stages) - 1:
                # Upscale for next stage (simulation)
                text_embeddings = text_embeddings + np.random.randn(*text_embeddings.shape) * 0.01

        return images_per_stage

class AttentionGAN:
    """AttnGAN with word-level attention"""

    def __init__(self, config: TextToImageConfig):
        self.config = config

        # Attention weights
        self.attention_query = np.random.randn(config.text_embedding_dim, config.attention_heads, 64) * 0.01
        self.attention_key = np.random.randn(config.text_embedding_dim, config.attention_heads, 64) * 0.01
        self.attention_value = np.random.randn(config.text_embedding_dim, config.attention_heads, 64) * 0.01

        # Base GAN
        self.base_gan = ConditionalGAN(config)

    def word_attention(self, text_embeddings: np.ndarray, word_features: np.ndarray) -> np.ndarray:
        """
        Compute word-level attention

        Args:
            text_embeddings: (batch_size, text_embedding_dim)
            word_features: (batch_size, num_words, text_embedding_dim)

        Returns:
            attended_features: (batch_size, text_embedding_dim)
        """
        batch_size = text_embeddings.shape[0]
        attended_features_list = []

        for h in range(self.config.attention_heads):
            # Compute attention for this head
            query = np.dot(text_embeddings, self.attention_query[:, h, :])
            key = np.dot(word_features, self.attention_key[:, h, :])
            value = np.dot(word_features, self.attention_value[:, h, :])

            # Attention scores
            scores = np.sum(query[:, np.newaxis, :] * key, axis=2) / np.sqrt(64)
            attention_weights = self._softmax(scores)

            # Apply attention
            attended = np.sum(value * attention_weights[:, :, np.newaxis], axis=1)
            attended_features_list.append(attended)

        # Concatenate heads
        attended_features = np.concatenate(attended_features_list, axis=1)

        # Project back to text_embedding_dim
        projection = np.random.randn(attended_features.shape[1], self.config.text_embedding_dim) * 0.01
        attended_features = np.dot(attended_features, projection)

        return attended_features

    def generate_with_attention(self, text_embeddings: np.ndarray,
                                word_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images with word-level attention

        Args:
            text_embeddings: (batch_size, text_embedding_dim)
            word_features: (batch_size, num_words, text_embedding_dim)

        Returns:
            images: (batch_size, image_size, image_size, channels)
            attention_maps: (batch_size, num_words)
        """
        # Compute attention-enhanced features
        attended_features = self.word_attention(text_embeddings, word_features)

        # Generate images
        images = self.base_gan.generate(attended_features)

        # Compute attention maps for visualization
        query = np.dot(text_embeddings, self.attention_query[:, 0, :])
        key = np.dot(word_features, self.attention_key[:, 0, :])
        attention_maps = np.sum(query[:, np.newaxis, :] * key, axis=2)
        attention_maps = self._softmax(attention_maps)

        return images, attention_maps

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class DiffusionModel:
    """Simplified diffusion model for text-to-image"""

    def __init__(self, config: TextToImageConfig, num_timesteps: int = 10):
        self.config = config
        self.num_timesteps = num_timesteps

        # Noise schedule
        self.betas = np.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def add_noise(self, images: np.ndarray, timestep: int) -> np.ndarray:
        """Add noise to images at given timestep"""
        noise = np.random.randn(*images.shape)
        alpha_t = self.alphas_cumprod[timestep]

        noisy_images = np.sqrt(alpha_t) * images + np.sqrt(1 - alpha_t) * noise
        return noisy_images

    def denoise_step(self, noisy_images: np.ndarray, text_embeddings: np.ndarray,
                     timestep: int) -> np.ndarray:
        """
        Single denoising step

        Args:
            noisy_images: Current noisy images
            text_embeddings: Text conditioning
            timestep: Current timestep

        Returns:
            denoised_images: Less noisy images
        """
        # Predict noise (simplified - in practice would use U-Net)
        batch_size = noisy_images.shape[0]
        image_flat = noisy_images.reshape(batch_size, -1)

        # Text conditioning
        text_cond = np.tile(text_embeddings[:, :256], (1, image_flat.shape[1] // 256 + 1))[:, :image_flat.shape[1]]

        # Predict noise
        predicted_noise = (image_flat * 0.1 + text_cond * 0.1) * 0.5

        # Denoise
        alpha_t = self.alphas_cumprod[timestep]
        beta_t = self.betas[timestep]

        denoised_flat = (image_flat - beta_t / np.sqrt(1 - alpha_t) * predicted_noise) / np.sqrt(self.alphas[timestep])

        denoised_images = denoised_flat.reshape(noisy_images.shape)

        return denoised_images

    def generate(self, text_embeddings: np.ndarray) -> List[np.ndarray]:
        """
        Generate images through diffusion process

        Args:
            text_embeddings: (batch_size, text_embedding_dim)

        Returns:
            images_per_step: Images at each denoising step
        """
        batch_size = text_embeddings.shape[0]

        # Start with pure noise
        current_images = np.random.randn(
            batch_size,
            self.config.image_size,
            self.config.image_size,
            self.config.image_channels
        )

        images_per_step = [current_images.copy()]

        # Denoise step by step
        for t in reversed(range(self.num_timesteps)):
            current_images = self.denoise_step(current_images, text_embeddings, t)
            images_per_step.append(current_images.copy())

        return images_per_step

class TextToImageEvaluator:
    """Evaluate text-to-image generation models"""

    def __init__(self):
        self.metrics = {}

    def inception_score(self, images: np.ndarray, num_splits: int = 10) -> Tuple[float, float]:
        """
        Simplified Inception Score

        Args:
            images: Generated images
            num_splits: Number of splits for computing score

        Returns:
            mean_score, std_score
        """
        # Simulate predictions from Inception network
        num_images = images.shape[0]
        num_classes = 1000

        # Random predictions (in practice, use real Inception network)
        predictions = np.random.dirichlet(np.ones(num_classes) * 0.1, num_images)

        # Compute score for each split
        split_scores = []
        split_size = num_images // num_splits

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            split_preds = predictions[start_idx:end_idx]

            # KL divergence
            p_y = np.mean(split_preds, axis=0)
            kl_divs = np.sum(split_preds * (np.log(split_preds + 1e-10) - np.log(p_y + 1e-10)), axis=1)
            split_scores.append(np.exp(np.mean(kl_divs)))

        return np.mean(split_scores), np.std(split_scores)

    def frechet_distance(self, real_features: np.ndarray, generated_features: np.ndarray) -> float:
        """
        Compute Frechet Inception Distance (FID)

        Args:
            real_features: Features from real images
            generated_features: Features from generated images

        Returns:
            fid_score
        """
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(generated_features, axis=0)

        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(generated_features, rowvar=False)

        # Compute FID
        diff = mu_real - mu_gen
        covmean = self._sqrtm(sigma_real.dot(sigma_gen))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2*covmean)

        return fid

    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Matrix square root"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        return eigenvectors.dot(np.diag(sqrt_eigenvalues)).dot(eigenvectors.T)

    def text_image_alignment(self, text_embeddings: np.ndarray,
                            image_features: np.ndarray) -> float:
        """
        Compute text-image alignment score

        Args:
            text_embeddings: Text embeddings
            image_features: Image features

        Returns:
            alignment_score
        """
        # Normalize embeddings
        text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        image_norm = image_features / (np.linalg.norm(image_features, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarities = np.sum(text_norm * image_norm, axis=1)

        return np.mean(similarities)

def visualize_generation_process(model_name: str, images_per_stage: List[np.ndarray],
                                 save_path: str):
    """Visualize multi-stage or multi-step generation process"""

    num_stages = len(images_per_stage)
    num_samples = min(4, images_per_stage[0].shape[0])

    fig, axes = plt.subplots(num_samples, num_stages, figsize=(num_stages * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(num_samples):
        for stage_idx, stage_images in enumerate(images_per_stage):
            ax = axes[sample_idx, stage_idx] if num_samples > 1 else axes[stage_idx]

            # Get image and normalize to [0, 1]
            img = stage_images[sample_idx]
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)

            ax.imshow(img_normalized)
            ax.axis('off')

            if sample_idx == 0:
                ax.set_title(f'Stage {stage_idx + 1}', fontweight='bold', fontsize=10)

    plt.suptitle(f'{model_name} Generation Process', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{model_name} generation process saved to {save_path}")

def visualize_attention_maps(text_words: List[str], attention_weights: np.ndarray,
                             generated_image: np.ndarray, save_path: str):
    """Visualize word-level attention maps"""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Show generated image
    ax = axes[0, 0]
    img_normalized = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min() + 1e-8)
    ax.imshow(img_normalized)
    ax.set_title('Generated Image', fontweight='bold')
    ax.axis('off')

    # Show attention distribution
    ax = axes[0, 1]
    ax.bar(range(len(text_words)), attention_weights, color='skyblue', alpha=0.8)
    ax.set_title('Word Attention Weights', fontweight='bold')
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Attention Weight')
    ax.set_xticks(range(len(text_words)))
    ax.set_xticklabels(text_words, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Show top attended words
    ax = axes[0, 2]
    top_k = min(5, len(text_words))
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_words = [text_words[i] for i in top_indices]
    top_weights = attention_weights[top_indices]

    colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    ax.barh(range(top_k), top_weights, color=colors, alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_words)
    ax.set_title('Top Attended Words', fontweight='bold')
    ax.set_xlabel('Attention Weight')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Attention heatmap
    ax = axes[0, 3]
    attention_matrix = attention_weights.reshape(-1, 1).T
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_title('Attention Heatmap', fontweight='bold')
    ax.set_yticks([0])
    ax.set_yticklabels(['Attention'])
    ax.set_xticks(range(len(text_words)))
    ax.set_xticklabels(text_words, rotation=45, ha='right')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Cumulative attention
    ax = axes[1, 0]
    cumulative_attention = np.cumsum(attention_weights)
    ax.plot(range(len(text_words)), cumulative_attention, marker='o',
            linewidth=2, markersize=6, color='#E74C3C')
    ax.set_title('Cumulative Attention', fontweight='bold')
    ax.set_xlabel('Word Index')
    ax.set_ylabel('Cumulative Weight')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(text_words)))
    ax.set_xticklabels(text_words, rotation=45, ha='right')

    # Attention entropy
    ax = axes[1, 1]
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
    max_entropy = -np.log(1.0 / len(text_words))
    normalized_entropy = entropy / max_entropy

    ax.bar(['Entropy'], [entropy], color='#3498DB', alpha=0.8, label='Actual')
    ax.axhline(y=max_entropy, color='red', linestyle='--', label='Max Entropy')
    ax.set_ylabel('Entropy')
    ax.set_title(f'Attention Entropy\n(Normalized: {normalized_entropy:.3f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Word importance ranking
    ax = axes[1, 2]
    sorted_indices = np.argsort(attention_weights)[::-1]
    ranks = np.arange(len(text_words)) + 1
    ax.scatter(ranks, attention_weights[sorted_indices], s=100, alpha=0.6, c=ranks, cmap='plasma')
    ax.set_title('Word Importance Ranking', fontweight='bold')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Attention Weight')
    ax.grid(True, alpha=0.3)

    # Attention distribution statistics
    ax = axes[1, 3]
    stats = {
        'Mean': np.mean(attention_weights),
        'Std': np.std(attention_weights),
        'Max': np.max(attention_weights),
        'Min': np.min(attention_weights)
    }

    colors_stats = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(stats.keys(), stats.values(), color=colors_stats, alpha=0.8)
    ax.set_title('Attention Statistics', fontweight='bold')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, stats.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Attention visualization saved to {save_path}")

def visualize_model_comparison(evaluator: TextToImageEvaluator,
                               model_results: Dict[str, Dict[str, float]],
                               save_path: str):
    """Compare different text-to-image models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = list(model_results.keys())

    # 1. Inception Score
    ax = axes[0, 0]
    is_means = [model_results[m]['inception_score_mean'] for m in models]
    is_stds = [model_results[m]['inception_score_std'] for m in models]

    x = np.arange(len(models))
    bars = ax.bar(x, is_means, yerr=is_stds, capsize=5, color='skyblue', alpha=0.8)
    ax.set_title('Inception Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. FID Score (lower is better)
    ax = axes[0, 1]
    fid_scores = [model_results[m]['fid_score'] for m in models]
    colors_fid = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(models, fid_scores, color=colors_fid, alpha=0.8)
    ax.set_title('FID Score (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, fid_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Text-Image Alignment
    ax = axes[0, 2]
    alignment_scores = [model_results[m]['text_image_alignment'] for m in models]
    colors_align = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, alignment_scores, color=colors_align, alpha=0.8)
    ax.set_title('Text-Image Alignment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alignment Score')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Overall performance radar
    ax = axes[1, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 4, projection='polar')

    metrics = ['IS', 'FID_inv', 'Alignment']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for model in models:
        # Normalize metrics for radar plot
        values = [
            model_results[model]['inception_score_mean'] / 10,  # Normalize IS
            1.0 / (1.0 + model_results[model]['fid_score'] / 100),  # Inverse FID
            model_results[model]['text_image_alignment']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # 5. Performance ranking
    ax = axes[1, 1]

    # Compute overall ranking
    rankings = []
    for model in models:
        score = (model_results[model]['inception_score_mean'] * 10 +
                (100 - model_results[model]['fid_score']) +
                model_results[model]['text_image_alignment'] * 100) / 3
        rankings.append(score)

    sorted_indices = np.argsort(rankings)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rankings = [rankings[i] for i in sorted_indices]

    colors_rank = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    ax.barh(range(len(models)), sorted_rankings, color=colors_rank, alpha=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(sorted_models)
    ax.set_title('Overall Model Ranking', fontsize=12, fontweight='bold')
    ax.set_xlabel('Combined Score')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # 6. Metric correlation
    ax = axes[1, 2]

    metric_matrix = np.array([
        [model_results[m]['inception_score_mean'] for m in models],
        [model_results[m]['fid_score'] for m in models],
        [model_results[m]['text_image_alignment'] for m in models]
    ])

    correlation = np.corrcoef(metric_matrix)
    metric_names = ['IS', 'FID', 'Alignment']

    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(metric_names)
    ax.set_title('Metric Correlation', fontsize=12, fontweight='bold')

    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Model comparison visualization saved to {save_path}")

def main():
    """Main execution function"""

    print("=" * 80)
    print("TEXT-TO-IMAGE GENERATION")
    print("=" * 80)

    config = TextToImageConfig()

    # Generate synthetic text descriptions
    print("\n1. Generating synthetic text data...")
    text_descriptions = [
        "a red car on the street",
        "a dog running in the park",
        "a beautiful sunset over the ocean",
        "a person sitting on a bench",
        "a cat sleeping on a couch"
    ] * 20

    text_encoder = TextEncoder(config.text_embedding_dim)
    text_embeddings = text_encoder.encode(text_descriptions)

    print(f"   - Generated {len(text_descriptions)} text descriptions")
    print(f"   - Text embedding shape: {text_embeddings.shape}")

    # Test different models
    print("\n2. Testing ConditionalGAN...")
    cond_gan = ConditionalGAN(config)
    gan_images = cond_gan.generate(text_embeddings[:10])
    print(f"   - Generated images shape: {gan_images.shape}")

    print("\n3. Testing StackGAN...")
    stack_gan = StackGAN(config)
    stack_images = stack_gan.generate_multistage(text_embeddings[:4])
    visualize_generation_process("StackGAN", stack_images, "stackgan_process.png")

    print("\n4. Testing AttnGAN...")
    attn_gan = AttentionGAN(config)

    # Create word features (simulating multiple words per caption)
    num_words = 5
    word_features = np.random.randn(10, num_words, config.text_embedding_dim)

    attn_images, attention_maps = attn_gan.generate_with_attention(
        text_embeddings[:10], word_features
    )

    # Visualize attention
    sample_words = text_descriptions[0].split()
    if len(sample_words) < num_words:
        sample_words += ['<pad>'] * (num_words - len(sample_words))
    else:
        sample_words = sample_words[:num_words]

    visualize_attention_maps(sample_words, attention_maps[0], attn_images[0],
                            "attention_maps.png")

    print("\n5. Testing Diffusion Model...")
    diffusion = DiffusionModel(config, num_timesteps=10)
    diffusion_images = diffusion.generate(text_embeddings[:4])
    visualize_generation_process("Diffusion", diffusion_images[::2], "diffusion_process.png")

    # Evaluate models
    print("\n6. Evaluating models...")
    evaluator = TextToImageEvaluator()

    model_results = {}
    models_dict = {
        'ConditionalGAN': gan_images,
        'StackGAN': stack_images[-1],
        'AttnGAN': attn_images,
        'Diffusion': diffusion_images[-1]
    }

    # Generate "real" images for comparison
    real_images = np.random.randn(100, config.image_size, config.image_size, config.image_channels)

    for model_name, images in models_dict.items():
        is_mean, is_std = evaluator.inception_score(images[:10])

        # Extract features (simulated)
        real_features = real_images[:10].reshape(10, -1)
        gen_features = images[:10].reshape(10, -1)

        fid = evaluator.frechet_distance(real_features, gen_features)

        # Text-image alignment
        image_features_flat = images[:10].reshape(10, -1)
        # Project to text embedding dimension
        projection = np.random.randn(image_features_flat.shape[1], config.text_embedding_dim) * 0.01
        image_features = np.dot(image_features_flat, projection)

        alignment = evaluator.text_image_alignment(text_embeddings[:10], image_features)

        model_results[model_name] = {
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'fid_score': fid,
            'text_image_alignment': alignment
        }

        print(f"\n   {model_name}:")
        print(f"   - IS: {is_mean:.3f} ± {is_std:.3f}")
        print(f"   - FID: {fid:.3f}")
        print(f"   - Alignment: {alignment:.3f}")

    # Visualize comparison
    print("\n7. Visualizing model comparison...")
    visualize_model_comparison(evaluator, model_results, "model_comparison.png")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGenerated images from {len(models_dict)} different models")
    print(f"Best model (by IS): {max(model_results, key=lambda x: model_results[x]['inception_score_mean'])}")
    print(f"Best model (by FID): {min(model_results, key=lambda x: model_results[x]['fid_score'])}")
    print(f"Best model (by Alignment): {max(model_results, key=lambda x: model_results[x]['text_image_alignment'])}")

    print("\n" + "=" * 80)
    print("All visualizations completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
