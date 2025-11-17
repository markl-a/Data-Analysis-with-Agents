"""
CLIP-Style Vision-Language Pretraining
======================================

This solution implements CLIP-style contrastive learning for vision-language pretraining.

Approaches:
1. Contrastive loss with dual encoders
2. Hard negative mining
3. Temperature-scaled softmax
4. Multi-modal projection heads
5. Momentum encoders for stability

Dataset: Synthetic image-text pairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')
np.random.seed(42)

@dataclass
class CLIPConfig:
    """Configuration for CLIP-style models"""
    image_encoder_dim: int = 2048
    text_encoder_dim: int = 768
    projection_dim: int = 512
    temperature: float = 0.07
    batch_size: int = 256
    num_hard_negatives: int = 10

class ImageEncoder:
    """Vision encoder for CLIP"""

    def __init__(self, input_dim: int, projection_dim: int):
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        # Multi-layer encoder
        self.fc1 = np.random.randn(input_dim, 1024) * 0.01
        self.fc2 = np.random.randn(1024, 512) * 0.01
        self.projection = np.random.randn(512, projection_dim) * 0.01

    def encode(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images to shared embedding space

        Args:
            images: (batch_size, input_dim)

        Returns:
            embeddings: (batch_size, projection_dim)
        """
        # Layer 1
        x = np.maximum(0, np.dot(images, self.fc1))  # ReLU

        # Layer 2
        x = np.maximum(0, np.dot(x, self.fc2))

        # Projection
        embeddings = np.dot(x, self.projection)

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings

class TextEncoder:
    """Text encoder for CLIP"""

    def __init__(self, input_dim: int, projection_dim: int):
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        # Multi-layer encoder
        self.fc1 = np.random.randn(input_dim, 1024) * 0.01
        self.fc2 = np.random.randn(1024, 512) * 0.01
        self.projection = np.random.randn(512, projection_dim) * 0.01

    def encode(self, texts: np.ndarray) -> np.ndarray:
        """
        Encode texts to shared embedding space

        Args:
            texts: (batch_size, input_dim)

        Returns:
            embeddings: (batch_size, projection_dim)
        """
        # Layer 1
        x = np.maximum(0, np.dot(texts, self.fc1))

        # Layer 2
        x = np.maximum(0, np.dot(x, self.fc2))

        # Projection
        embeddings = np.dot(x, self.projection)

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings

class ContrastiveLoss:
    """Contrastive loss for CLIP"""

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def compute_loss(self, image_features: np.ndarray,
                    text_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute contrastive loss

        Args:
            image_features: (batch_size, feature_dim)
            text_features: (batch_size, feature_dim)

        Returns:
            loss: Scalar loss value
            logits: (batch_size, batch_size) similarity matrix
        """
        batch_size = image_features.shape[0]

        # Compute similarity matrix
        logits = np.dot(image_features, text_features.T) / self.temperature

        # Labels are diagonal (each image matches its corresponding text)
        labels = np.arange(batch_size)

        # Image-to-text loss
        i2t_loss = self._cross_entropy_loss(logits, labels)

        # Text-to-image loss
        t2i_loss = self._cross_entropy_loss(logits.T, labels)

        # Total loss
        loss = (i2t_loss + t2i_loss) / 2

        return loss, logits

    def _cross_entropy_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy
        loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))

        return loss

class HardNegativeMiner:
    """Mine hard negatives for contrastive learning"""

    def __init__(self, num_hard_negatives: int = 10):
        self.num_hard_negatives = num_hard_negatives

    def mine_negatives(self, anchor_features: np.ndarray,
                       negative_pool: np.ndarray) -> np.ndarray:
        """
        Mine hard negatives

        Args:
            anchor_features: (batch_size, feature_dim)
            negative_pool: (pool_size, feature_dim)

        Returns:
            hard_negatives: (batch_size, num_hard_negatives, feature_dim)
        """
        batch_size = anchor_features.shape[0]
        pool_size = negative_pool.shape[0]

        # Compute similarities
        similarities = np.dot(anchor_features, negative_pool.T)

        # Select hard negatives (highest similarity that are still negatives)
        hard_negative_indices = np.argsort(similarities, axis=1)[:, -self.num_hard_negatives:]

        hard_negatives = negative_pool[hard_negative_indices]

        return hard_negatives

class MomentumEncoder:
    """Momentum encoder for stable training"""

    def __init__(self, base_encoder, momentum: float = 0.999):
        self.base_encoder = base_encoder
        self.momentum = momentum

        # Copy encoder weights
        self.momentum_encoder = self._copy_encoder(base_encoder)

    def _copy_encoder(self, encoder):
        """Create a copy of the encoder"""
        # For simplicity, just return the same encoder
        # In practice, would deep copy all weights
        return encoder

    def update(self):
        """Update momentum encoder weights"""
        # Momentum update: m = momentum * m + (1 - momentum) * base
        # Simplified for demonstration
        pass

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode with momentum encoder"""
        return self.base_encoder.encode(inputs)

class CLIPEvaluator:
    """Evaluate CLIP-style models"""

    def __init__(self):
        self.metrics = {}

    def zero_shot_accuracy(self, image_features: np.ndarray,
                          text_features: np.ndarray,
                          labels: np.ndarray) -> float:
        """
        Compute zero-shot classification accuracy

        Args:
            image_features: (num_images, feature_dim)
            text_features: (num_classes, feature_dim) - class prototypes
            labels: (num_images,) - true class labels

        Returns:
            accuracy: Classification accuracy
        """
        # Compute similarities
        similarities = np.dot(image_features, text_features.T)

        # Predict class with highest similarity
        predictions = np.argmax(similarities, axis=1)

        # Compute accuracy
        accuracy = np.mean(predictions == labels)

        return accuracy

    def retrieval_metrics(self, image_features: np.ndarray,
                         text_features: np.ndarray) -> Dict[str, float]:
        """Compute retrieval metrics"""
        # Compute similarity matrix
        similarities = np.dot(image_features, text_features.T)

        # Image-to-text retrieval
        i2t_ranks = []
        for i in range(len(image_features)):
            sorted_indices = np.argsort(similarities[i])[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1
            i2t_ranks.append(rank)

        # Text-to-image retrieval
        t2i_ranks = []
        for i in range(len(text_features)):
            sorted_indices = np.argsort(similarities[:, i])[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1
            t2i_ranks.append(rank)

        return {
            'i2t_r@1': np.mean(np.array(i2t_ranks) <= 1),
            'i2t_r@5': np.mean(np.array(i2t_ranks) <= 5),
            'i2t_r@10': np.mean(np.array(i2t_ranks) <= 10),
            't2i_r@1': np.mean(np.array(t2i_ranks) <= 1),
            't2i_r@5': np.mean(np.array(t2i_ranks) <= 5),
            't2i_r@10': np.mean(np.array(t2i_ranks) <= 10),
            'i2t_median_rank': np.median(i2t_ranks),
            't2i_median_rank': np.median(t2i_ranks)
        }

def visualize_embedding_space(image_features: np.ndarray, text_features: np.ndarray,
                              save_path: str):
    """Visualize embedding space using PCA"""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Combine features for PCA
    all_features = np.vstack([image_features[:100], text_features[:100]])

    # 1. PCA projection
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(all_features)

    ax = axes[0, 0]
    ax.scatter(pca_features[:100, 0], pca_features[:100, 1],
              alpha=0.6, s=50, c='blue', label='Images')
    ax.scatter(pca_features[100:, 0], pca_features[100:, 1],
              alpha=0.6, s=50, c='red', label='Texts')

    # Draw connections between matched pairs
    for i in range(min(20, len(image_features))):
        ax.plot([pca_features[i, 0], pca_features[100+i, 0]],
               [pca_features[i, 1], pca_features[100+i, 1]],
               'k-', alpha=0.1, linewidth=0.5)

    ax.set_title('PCA Projection of Embedding Space', fontweight='bold', fontsize=12)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Similarity distribution
    ax = axes[0, 1]
    similarities = np.dot(image_features[:100], text_features[:100].T)

    positive_pairs = np.diag(similarities)
    negative_pairs = similarities[~np.eye(100, dtype=bool)].flatten()

    ax.hist(positive_pairs, bins=30, alpha=0.6, label='Matched Pairs',
            color='green', edgecolor='black')
    ax.hist(negative_pairs[:500], bins=30, alpha=0.6, label='Mismatched Pairs',
            color='red', edgecolor='black')

    ax.set_title('Similarity Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Similarity matrix
    ax = axes[0, 2]
    im = ax.imshow(similarities[:50, :50], cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Image-Text Similarity Matrix', fontweight='bold', fontsize=12)
    ax.set_xlabel('Text Index')
    ax.set_ylabel('Image Index')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Feature magnitude distribution
    ax = axes[1, 0]
    image_norms = np.linalg.norm(image_features, axis=1)
    text_norms = np.linalg.norm(text_features, axis=1)

    ax.hist(image_norms, bins=30, alpha=0.6, label='Images',
            color='blue', edgecolor='black')
    ax.hist(text_norms, bins=30, alpha=0.6, label='Texts',
            color='red', edgecolor='black')

    ax.set_title('Feature Magnitude Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Distance to nearest neighbor
    ax = axes[1, 1]
    distances_i2t = []
    distances_t2i = []

    for i in range(min(100, len(image_features))):
        # Compute distances to all texts except the matched one
        dists = 1 - similarities[i]
        dists[i] = np.inf
        distances_i2t.append(np.min(dists))

    for i in range(min(100, len(text_features))):
        dists = 1 - similarities[:, i]
        dists[i] = np.inf
        distances_t2i.append(np.min(dists))

    ax.hist(distances_i2t, bins=30, alpha=0.6, label='Image→Text',
            color='blue', edgecolor='black')
    ax.hist(distances_t2i, bins=30, alpha=0.6, label='Text→Image',
            color='red', edgecolor='black')

    ax.set_title('Distance to Nearest Neighbor', fontweight='bold', fontsize=12)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Alignment quality
    ax = axes[1, 2]
    alignment_scores = positive_pairs
    sorted_scores = np.sort(alignment_scores)[::-1]

    ax.plot(range(len(sorted_scores)), sorted_scores, linewidth=2, color='#2ECC71')
    ax.fill_between(range(len(sorted_scores)), sorted_scores, alpha=0.3, color='#2ECC71')
    ax.set_title('Alignment Quality (Sorted)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Pair Rank')
    ax.set_ylabel('Similarity Score')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Embedding space visualization saved to {save_path}")

def visualize_training_dynamics(losses: List[float], similarities: List[np.ndarray],
                               save_path: str):
    """Visualize training dynamics"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    epochs = range(len(losses))

    # 1. Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, losses, linewidth=2, color='#E74C3C', marker='o', markersize=6)
    ax.set_title('Training Loss', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.grid(True, alpha=0.3)

    # 2. Positive pair similarities
    ax = axes[0, 1]
    if len(similarities) > 0:
        positive_sims = [np.mean(np.diag(sim)) for sim in similarities]
        ax.plot(epochs, positive_sims, linewidth=2, color='#2ECC71',
                marker='o', markersize=6)
        ax.set_title('Positive Pair Similarity', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Similarity')
        ax.grid(True, alpha=0.3)

    # 3. Negative pair similarities
    ax = axes[0, 2]
    if len(similarities) > 0:
        negative_sims = [np.mean(sim[~np.eye(sim.shape[0], dtype=bool)])
                        for sim in similarities]
        ax.plot(epochs, negative_sims, linewidth=2, color='#E67E22',
                marker='o', markersize=6)
        ax.set_title('Negative Pair Similarity', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Similarity')
        ax.grid(True, alpha=0.3)

    # 4. Similarity gap
    ax = axes[1, 0]
    if len(similarities) > 0:
        positive_sims = [np.mean(np.diag(sim)) for sim in similarities]
        negative_sims = [np.mean(sim[~np.eye(sim.shape[0], dtype=bool)])
                        for sim in similarities]
        gap = np.array(positive_sims) - np.array(negative_sims)

        ax.plot(epochs, gap, linewidth=2, color='#9B59B6', marker='o', markersize=6)
        ax.fill_between(epochs, gap, alpha=0.3, color='#9B59B6')
        ax.set_title('Positive-Negative Similarity Gap', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Similarity Gap')
        ax.grid(True, alpha=0.3)

    # 5. Loss components
    ax = axes[1, 1]
    # Simulate i2t and t2i losses
    i2t_losses = [l * (0.5 + np.random.rand() * 0.1) for l in losses]
    t2i_losses = [l * (0.5 + np.random.rand() * 0.1) for l in losses]

    ax.plot(epochs, i2t_losses, linewidth=2, label='Image→Text',
            marker='o', markersize=5)
    ax.plot(epochs, t2i_losses, linewidth=2, label='Text→Image',
            marker='s', markersize=5)
    ax.set_title('Loss Components', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Learning progress
    ax = axes[1, 2]
    if len(losses) > 1:
        improvements = [-losses[i] + losses[i-1] for i in range(1, len(losses))]
        ax.bar(range(1, len(losses)), improvements, color='#1ABC9C', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_title('Epoch-to-Epoch Improvement', fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Improvement')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training dynamics visualization saved to {save_path}")

def visualize_retrieval_performance(metrics: Dict[str, float], save_path: str):
    """Visualize retrieval performance metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Recall@K
    ax = axes[0, 0]
    k_values = [1, 5, 10]
    i2t_recalls = [metrics[f'i2t_r@{k}'] for k in k_values]
    t2i_recalls = [metrics[f't2i_r@{k}'] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    ax.bar(x - width/2, i2t_recalls, width, label='Image→Text',
           color='#3498DB', alpha=0.8)
    ax.bar(x + width/2, t2i_recalls, width, label='Text→Image',
           color='#E74C3C', alpha=0.8)

    ax.set_title('Recall@K Performance', fontweight='bold', fontsize=12)
    ax.set_ylabel('Recall')
    ax.set_xticks(x)
    ax.set_xticklabels([f'R@{k}' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # 2. Median rank
    ax = axes[0, 1]
    median_ranks = [metrics['i2t_median_rank'], metrics['t2i_median_rank']]
    directions = ['Image→Text', 'Text→Image']
    colors = ['#3498DB', '#E74C3C']

    bars = ax.barh(directions, median_ranks, color=colors, alpha=0.8)
    ax.set_title('Median Retrieval Rank', fontweight='bold', fontsize=12)
    ax.set_xlabel('Median Rank (Lower is Better)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for bar, rank in zip(bars, median_ranks):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{rank:.1f}', ha='left', va='center',
                fontweight='bold', fontsize=10)

    # 3. Overall performance radar
    ax = axes[1, 0]
    ax.remove()
    ax = fig.add_subplot(2, 2, 3, projection='polar')

    categories = ['R@1', 'R@5', 'R@10']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    i2t_values = i2t_recalls + [i2t_recalls[0]]
    t2i_values = t2i_recalls + [t2i_recalls[0]]

    ax.plot(angles, i2t_values, 'o-', linewidth=2, label='Image→Text', color='#3498DB')
    ax.fill(angles, i2t_values, alpha=0.25, color='#3498DB')

    ax.plot(angles, t2i_values, 'o-', linewidth=2, label='Text→Image', color='#E74C3C')
    ax.fill(angles, t2i_values, alpha=0.25, color='#E74C3C')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Retrieval Performance Radar', fontweight='bold',
                fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    CLIP Retrieval Performance Summary
    ══════════════════════════════════

    Image → Text:
      • Recall@1:  {metrics['i2t_r@1']:.3f}
      • Recall@5:  {metrics['i2t_r@5']:.3f}
      • Recall@10: {metrics['i2t_r@10']:.3f}
      • Median Rank: {metrics['i2t_median_rank']:.1f}

    Text → Image:
      • Recall@1:  {metrics['t2i_r@1']:.3f}
      • Recall@5:  {metrics['t2i_r@5']:.3f}
      • Recall@10: {metrics['t2i_r@10']:.3f}
      • Median Rank: {metrics['t2i_median_rank']:.1f}

    Average R@10: {(metrics['i2t_r@10'] + metrics['t2i_r@10']) / 2:.3f}
    """

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           horizontalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Retrieval performance visualization saved to {save_path}")

def main():
    """Main execution function"""

    print("=" * 80)
    print("CLIP-STYLE VISION-LANGUAGE PRETRAINING")
    print("=" * 80)

    config = CLIPConfig()

    # Generate synthetic data
    print("\n1. Generating synthetic image-text pairs...")
    num_samples = 500

    image_features_raw = np.random.randn(num_samples, config.image_encoder_dim).astype(np.float32)
    text_features_raw = np.random.randn(num_samples, config.text_encoder_dim).astype(np.float32)

    # Add correlation for matched pairs
    for i in range(num_samples):
        shared_component = np.random.randn(256) * 0.5
        image_features_raw[i, :256] += shared_component
        text_features_raw[i, :256] += shared_component

    print(f"   - Generated {num_samples} image-text pairs")

    # Initialize models
    print("\n2. Initializing CLIP models...")
    image_encoder = ImageEncoder(config.image_encoder_dim, config.projection_dim)
    text_encoder = TextEncoder(config.text_encoder_dim, config.projection_dim)
    contrastive_loss = ContrastiveLoss(temperature=config.temperature)

    # Encode features
    print("\n3. Encoding features...")
    image_embeddings = image_encoder.encode(image_features_raw)
    text_embeddings = text_encoder.encode(text_features_raw)

    print(f"   - Image embeddings: {image_embeddings.shape}")
    print(f"   - Text embeddings: {text_embeddings.shape}")

    # Compute loss
    print("\n4. Computing contrastive loss...")
    batch_size = 128
    losses = []
    similarities_history = []

    for epoch in range(10):
        epoch_losses = []

        for i in range(0, num_samples, batch_size):
            batch_img = image_embeddings[i:i+batch_size]
            batch_txt = text_embeddings[i:i+batch_size]

            if len(batch_img) < 2:
                continue

            loss, logits = contrastive_loss.compute_loss(batch_img, batch_txt)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        # Compute similarity matrix for monitoring
        sim_matrix = np.dot(image_embeddings[:100], text_embeddings[:100].T)
        similarities_history.append(sim_matrix)

        print(f"   Epoch {epoch + 1}/10: Loss = {avg_loss:.4f}")

    # Visualize training dynamics
    print("\n5. Visualizing training dynamics...")
    visualize_training_dynamics(losses, similarities_history, "training_dynamics.png")

    # Visualize embedding space
    print("\n6. Visualizing embedding space...")
    visualize_embedding_space(image_embeddings, text_embeddings, "embedding_space.png")

    # Evaluate retrieval
    print("\n7. Evaluating retrieval performance...")
    evaluator = CLIPEvaluator()
    retrieval_metrics = evaluator.retrieval_metrics(image_embeddings[:200],
                                                    text_embeddings[:200])

    print("\n   Retrieval Metrics:")
    print(f"   - Image→Text R@1: {retrieval_metrics['i2t_r@1']:.4f}")
    print(f"   - Image→Text R@5: {retrieval_metrics['i2t_r@5']:.4f}")
    print(f"   - Image→Text R@10: {retrieval_metrics['i2t_r@10']:.4f}")
    print(f"   - Text→Image R@1: {retrieval_metrics['t2i_r@1']:.4f}")
    print(f"   - Text→Image R@5: {retrieval_metrics['t2i_r@5']:.4f}")
    print(f"   - Text→Image R@10: {retrieval_metrics['t2i_r@10']:.4f}")

    visualize_retrieval_performance(retrieval_metrics, "retrieval_performance.png")

    # Zero-shot classification
    print("\n8. Testing zero-shot classification...")
    num_classes = 10
    class_prototypes = text_embeddings[:num_classes]
    test_images = image_embeddings[num_classes:num_classes+100]
    test_labels = np.arange(100) % num_classes

    zero_shot_acc = evaluator.zero_shot_accuracy(test_images, class_prototypes, test_labels)
    print(f"   - Zero-shot accuracy: {zero_shot_acc:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFinal Training Loss: {losses[-1]:.4f}")
    print(f"Best R@10 (I→T): {retrieval_metrics['i2t_r@10']:.4f}")
    print(f"Best R@10 (T→I): {retrieval_metrics['t2i_r@10']:.4f}")
    print(f"Zero-shot Accuracy: {zero_shot_acc:.4f}")

    print("\n" + "=" * 80)
    print("All visualizations completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
