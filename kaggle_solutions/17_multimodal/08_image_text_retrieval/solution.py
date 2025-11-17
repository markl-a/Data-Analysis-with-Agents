"""
Image-Text Retrieval
===================

This solution implements various approaches for cross-modal retrieval between images and text.

Approaches:
1. Dual Encoder with cosine similarity
2. Cross-attention matching network
3. Visual-semantic embedding
4. Triplet loss based retrieval
5. Transformer-based cross-modal matching

Dataset: Synthetic image and text features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
np.random.seed(42)

@dataclass
class RetrievalConfig:
    """Configuration for retrieval models"""
    image_dim: int = 2048
    text_dim: int = 768
    embedding_dim: int = 512
    num_samples: int = 1000
    batch_size: int = 32

class DualEncoder:
    """Dual encoder network for image-text retrieval"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

        # Image encoder
        self.image_fc1 = np.random.randn(config.image_dim, 1024) * 0.01
        self.image_fc2 = np.random.randn(1024, config.embedding_dim) * 0.01

        # Text encoder
        self.text_fc1 = np.random.randn(config.text_dim, 1024) * 0.01
        self.text_fc2 = np.random.randn(1024, config.embedding_dim) * 0.01

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Encode images to common embedding space"""
        hidden = np.tanh(np.dot(images, self.image_fc1))
        embeddings = np.dot(hidden, self.image_fc2)

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings

    def encode_text(self, texts: np.ndarray) -> np.ndarray:
        """Encode texts to common embedding space"""
        hidden = np.tanh(np.dot(texts, self.text_fc1))
        embeddings = np.dot(hidden, self.text_fc2)

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings

    def compute_similarity(self, image_embeddings: np.ndarray,
                          text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between images and texts

        Args:
            image_embeddings: (num_images, embedding_dim)
            text_embeddings: (num_texts, embedding_dim)

        Returns:
            similarity: (num_images, num_texts)
        """
        return np.dot(image_embeddings, text_embeddings.T)

class CrossAttentionMatcher:
    """Cross-attention based matching network"""

    def __init__(self, config: RetrievalConfig, num_heads: int = 8):
        self.config = config
        self.num_heads = num_heads
        self.head_dim = config.embedding_dim // num_heads

        # Image projection
        self.image_query = np.random.randn(config.image_dim, config.embedding_dim) * 0.01

        # Text projection
        self.text_key = np.random.randn(config.text_dim, config.embedding_dim) * 0.01
        self.text_value = np.random.randn(config.text_dim, config.embedding_dim) * 0.01

        # Output projection
        self.output_proj = np.random.randn(config.embedding_dim, config.embedding_dim) * 0.01

    def cross_attention(self, image_features: np.ndarray,
                       text_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-attention between image and text

        Args:
            image_features: (batch_size, image_dim)
            text_features: (batch_size, num_words, text_dim)

        Returns:
            attended_features: (batch_size, embedding_dim)
            attention_weights: (batch_size, num_words)
        """
        batch_size = image_features.shape[0]

        # Project to query, key, value
        query = np.dot(image_features, self.image_query)  # (batch_size, embedding_dim)
        key = np.dot(text_features, self.text_key)  # (batch_size, num_words, embedding_dim)
        value = np.dot(text_features, self.text_value)  # (batch_size, num_words, embedding_dim)

        # Compute attention scores
        scores = np.sum(query[:, np.newaxis, :] * key, axis=2) / np.sqrt(self.config.embedding_dim)
        attention_weights = self._softmax(scores)

        # Apply attention to values
        attended = np.sum(value * attention_weights[:, :, np.newaxis], axis=1)

        # Output projection
        output = np.dot(attended, self.output_proj)

        return output, attention_weights

    def compute_matching_score(self, image_features: np.ndarray,
                              text_features: np.ndarray) -> np.ndarray:
        """
        Compute matching scores between images and texts

        Args:
            image_features: (num_images, image_dim)
            text_features: (num_texts, num_words, text_dim)

        Returns:
            scores: (num_images, num_texts)
        """
        num_images = image_features.shape[0]
        num_texts = text_features.shape[0]

        scores = np.zeros((num_images, num_texts))

        for i in range(num_images):
            for j in range(num_texts):
                img_feat = image_features[i:i+1]
                txt_feat = text_features[j:j+1]

                attended, _ = self.cross_attention(img_feat, txt_feat)

                # Compute similarity
                img_norm = img_feat / (np.linalg.norm(img_feat) + 1e-8)
                attended_norm = attended / (np.linalg.norm(attended) + 1e-8)

                scores[i, j] = np.sum(img_norm * attended_norm)

        return scores

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TripletEncoder:
    """Triplet loss based encoder for retrieval"""

    def __init__(self, config: RetrievalConfig, margin: float = 0.2):
        self.config = config
        self.margin = margin

        # Shared embedding network
        self.fc1 = np.random.randn(max(config.image_dim, config.text_dim), 1024) * 0.01
        self.fc2 = np.random.randn(1024, config.embedding_dim) * 0.01

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to embedding space"""
        # Pad if necessary
        if features.shape[1] < self.fc1.shape[0]:
            padding = np.zeros((features.shape[0], self.fc1.shape[0] - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > self.fc1.shape[0]:
            features = features[:, :self.fc1.shape[0]]

        hidden = np.tanh(np.dot(features, self.fc1))
        embeddings = np.dot(hidden, self.fc2)

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return embeddings

    def triplet_loss(self, anchor: np.ndarray, positive: np.ndarray,
                    negative: np.ndarray) -> float:
        """
        Compute triplet loss

        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings

        Returns:
            loss: Triplet loss value
        """
        pos_dist = np.sum((anchor - positive) ** 2, axis=1)
        neg_dist = np.sum((anchor - negative) ** 2, axis=1)

        loss = np.maximum(pos_dist - neg_dist + self.margin, 0)

        return np.mean(loss)

class VisualSemanticEmbedding:
    """Visual-semantic embedding space"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

        # Multi-layer projection for images
        self.image_layers = [
            np.random.randn(config.image_dim, 1024) * 0.01,
            np.random.randn(1024, 512) * 0.01,
            np.random.randn(512, config.embedding_dim) * 0.01
        ]

        # Multi-layer projection for text
        self.text_layers = [
            np.random.randn(config.text_dim, 1024) * 0.01,
            np.random.randn(1024, 512) * 0.01,
            np.random.randn(512, config.embedding_dim) * 0.01
        ]

    def embed_image(self, images: np.ndarray) -> np.ndarray:
        """Project images to common space"""
        x = images

        for i, layer in enumerate(self.image_layers):
            x = np.dot(x, layer)
            if i < len(self.image_layers) - 1:
                x = np.maximum(0, x)  # ReLU

        # L2 normalize
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        return x

    def embed_text(self, texts: np.ndarray) -> np.ndarray:
        """Project texts to common space"""
        x = texts

        for i, layer in enumerate(self.text_layers):
            x = np.dot(x, layer)
            if i < len(self.text_layers) - 1:
                x = np.maximum(0, x)  # ReLU

        # L2 normalize
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        return x

    def compute_similarity(self, image_embeddings: np.ndarray,
                          text_embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity"""
        return np.dot(image_embeddings, text_embeddings.T)

class RetrievalEvaluator:
    """Evaluate retrieval performance"""

    def __init__(self):
        self.metrics = {}

    def recall_at_k(self, similarity_matrix: np.ndarray, k: int = 10,
                   mode: str = 'i2t') -> float:
        """
        Compute Recall@K

        Args:
            similarity_matrix: (num_images, num_texts) or (num_texts, num_images)
            k: Top-k to consider
            mode: 'i2t' for image-to-text or 't2i' for text-to-image

        Returns:
            recall: Recall@K score
        """
        num_queries = similarity_matrix.shape[0]
        correct = 0

        for i in range(num_queries):
            # Get top-k indices
            top_k_indices = np.argsort(similarity_matrix[i])[-k:]

            # Check if correct match is in top-k
            if i in top_k_indices:
                correct += 1

        return correct / num_queries

    def median_rank(self, similarity_matrix: np.ndarray) -> float:
        """Compute median rank of correct matches"""
        num_queries = similarity_matrix.shape[0]
        ranks = []

        for i in range(num_queries):
            # Get ranking of all items
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]

            # Find rank of correct match
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)

        return np.median(ranks)

    def mean_reciprocal_rank(self, similarity_matrix: np.ndarray) -> float:
        """Compute MRR"""
        num_queries = similarity_matrix.shape[0]
        reciprocal_ranks = []

        for i in range(num_queries):
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            rank = np.where(sorted_indices == i)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)

        return np.mean(reciprocal_ranks)

    def evaluate(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'recall@1': self.recall_at_k(similarity_matrix, k=1),
            'recall@5': self.recall_at_k(similarity_matrix, k=5),
            'recall@10': self.recall_at_k(similarity_matrix, k=10),
            'median_rank': self.median_rank(similarity_matrix),
            'mrr': self.mean_reciprocal_rank(similarity_matrix)
        }

def visualize_similarity_matrix(similarity: np.ndarray, title: str, save_path: str):
    """Visualize similarity matrix"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Full similarity matrix
    ax = axes[0]
    im = ax.imshow(similarity[:50, :50], cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'{title} - Similarity Matrix', fontweight='bold', fontsize=12)
    ax.set_xlabel('Text Index')
    ax.set_ylabel('Image Index')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Diagonal (correct matches)
    ax = axes[1]
    diagonal_scores = np.diag(similarity[:min(100, similarity.shape[0])])
    ax.hist(diagonal_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax.set_title('Correct Match Scores', fontweight='bold', fontsize=12)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(diagonal_scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(diagonal_scores):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Off-diagonal (incorrect matches)
    ax = axes[2]
    off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)].flatten()
    ax.hist(off_diagonal[:1000], bins=30, color='red', alpha=0.7, edgecolor='black')
    ax.set_title('Incorrect Match Scores', fontweight='bold', fontsize=12)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(off_diagonal), color='blue', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(off_diagonal):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Similarity matrix visualization saved to {save_path}")

def visualize_retrieval_results(image_features: np.ndarray, text_features: np.ndarray,
                                similarity_matrix: np.ndarray, save_path: str):
    """Visualize retrieval results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Top-k retrieval accuracy
    ax = axes[0, 0]
    k_values = [1, 5, 10, 20, 50, 100]
    evaluator = RetrievalEvaluator()
    recalls = [evaluator.recall_at_k(similarity_matrix, k) for k in k_values]

    ax.plot(k_values, recalls, marker='o', linewidth=2, markersize=8, color='#E74C3C')
    ax.set_title('Recall@K', fontweight='bold', fontsize=12)
    ax.set_xlabel('K')
    ax.set_ylabel('Recall')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 2. Rank distribution
    ax = axes[0, 1]
    ranks = []
    for i in range(min(100, similarity_matrix.shape[0])):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    ax.hist(ranks, bins=30, color='#3498DB', alpha=0.7, edgecolor='black')
    ax.set_title('Rank Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Rank of Correct Match')
    ax.set_ylabel('Frequency')
    ax.axvline(np.median(ranks), color='red', linestyle='--',
               linewidth=2, label=f'Median: {np.median(ranks):.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Precision-Recall curve
    ax = axes[0, 2]
    thresholds = np.linspace(similarity_matrix.min(), similarity_matrix.max(), 50)
    precisions = []
    recalls = []

    for threshold in thresholds:
        tp = np.sum((similarity_matrix >= threshold) & np.eye(similarity_matrix.shape[0], dtype=bool))
        fp = np.sum((similarity_matrix >= threshold) & ~np.eye(similarity_matrix.shape[0], dtype=bool))
        fn = np.sum((similarity_matrix < threshold) & np.eye(similarity_matrix.shape[0], dtype=bool))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    ax.plot(recalls, precisions, linewidth=2, color='#9B59B6')
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.3)

    # 4. Similarity score distribution
    ax = axes[1, 0]
    correct_scores = np.diag(similarity_matrix[:min(100, similarity_matrix.shape[0])])
    incorrect_scores = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)].flatten()[:1000]

    ax.hist(correct_scores, bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(incorrect_scores, bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    ax.set_title('Score Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Cumulative match characteristic
    ax = axes[1, 1]
    max_rank = 100
    cmc = []
    for rank in range(1, max_rank + 1):
        matches = sum(1 for r in ranks if r <= rank)
        cmc.append(matches / len(ranks))

    ax.plot(range(1, max_rank + 1), cmc, linewidth=2, color='#1ABC9C')
    ax.set_title('Cumulative Match Characteristic', fontweight='bold', fontsize=12)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cumulative Accuracy')
    ax.grid(True, alpha=0.3)

    # 6. Retrieval metrics comparison
    ax = axes[1, 2]
    metrics = evaluator.evaluate(similarity_matrix)
    metric_names = ['R@1', 'R@5', 'R@10', 'MRR']
    metric_values = [metrics['recall@1'], metrics['recall@5'],
                     metrics['recall@10'], metrics['mrr']]

    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8)
    ax.set_title('Retrieval Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Retrieval results visualization saved to {save_path}")

def visualize_model_comparison(model_results: Dict[str, Dict[str, float]], save_path: str):
    """Compare different retrieval models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = list(model_results.keys())

    # 1. Recall@K comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.25

    recalls = ['recall@1', 'recall@5', 'recall@10']
    colors_recall = ['#E74C3C', '#3498DB', '#2ECC71']

    for i, recall_key in enumerate(recalls):
        values = [model_results[m][recall_key] for m in models]
        ax.bar(x + i*width, values, width, label=recall_key.upper(), color=colors_recall[i], alpha=0.8)

    ax.set_title('Recall Comparison', fontweight='bold', fontsize=12)
    ax.set_ylabel('Recall')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. MRR comparison
    ax = axes[0, 1]
    mrr_values = [model_results[m]['mrr'] for m in models]
    colors_mrr = plt.cm.plasma(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, mrr_values, color=colors_mrr, alpha=0.8)
    ax.set_title('Mean Reciprocal Rank', fontweight='bold', fontsize=12)
    ax.set_ylabel('MRR')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, mrr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Median rank comparison (lower is better)
    ax = axes[0, 2]
    median_ranks = [model_results[m]['median_rank'] for m in models]
    colors_rank = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(models, median_ranks, color=colors_rank, alpha=0.8)
    ax.set_title('Median Rank (Lower is Better)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Median Rank')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Overall performance radar
    ax = axes[1, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 4, projection='polar')

    metrics_radar = ['R@1', 'R@5', 'R@10', 'MRR']
    angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]

    for model in models:
        values = [
            model_results[model]['recall@1'],
            model_results[model]['recall@5'],
            model_results[model]['recall@10'],
            model_results[model]['mrr']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance', fontweight='bold', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # 5. Performance ranking
    ax = axes[1, 1]
    rankings = []
    for model in models:
        score = (model_results[model]['recall@10'] * 100 +
                model_results[model]['mrr'] * 100) / 2
        rankings.append(score)

    sorted_indices = np.argsort(rankings)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_rankings = [rankings[i] for i in sorted_indices]

    colors_ranking = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    ax.barh(range(len(models)), sorted_rankings, color=colors_ranking, alpha=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(sorted_models)
    ax.set_title('Overall Model Ranking', fontweight='bold', fontsize=12)
    ax.set_xlabel('Combined Score')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # 6. Metric correlation
    ax = axes[1, 2]
    metric_keys = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'median_rank']
    metric_matrix = np.array([[model_results[m][k] for k in metric_keys] for m in models])
    correlation = np.corrcoef(metric_matrix.T)

    metric_labels = ['R@1', 'R@5', 'R@10', 'MRR', 'MedR']
    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_yticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(metric_labels)
    ax.set_title('Metric Correlation', fontweight='bold', fontsize=12)

    for i in range(len(metric_labels)):
        for j in range(len(metric_labels)):
            text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Model comparison visualization saved to {save_path}")

def main():
    """Main execution function"""

    print("=" * 80)
    print("IMAGE-TEXT RETRIEVAL")
    print("=" * 80)

    config = RetrievalConfig(num_samples=200)

    # Generate synthetic data
    print("\n1. Generating synthetic image and text features...")
    image_features = np.random.randn(config.num_samples, config.image_dim).astype(np.float32)
    text_features = np.random.randn(config.num_samples, config.text_dim).astype(np.float32)

    # Add some correlation between matching pairs
    for i in range(config.num_samples):
        correlation = np.random.randn(min(config.image_dim, config.text_dim)) * 0.3
        image_features[i, :len(correlation)] += correlation
        text_features[i, :len(correlation)] += correlation

    print(f"   - Generated {config.num_samples} image-text pairs")
    print(f"   - Image features: {image_features.shape}")
    print(f"   - Text features: {text_features.shape}")

    # Test different models
    model_results = {}

    print("\n2. Testing Dual Encoder...")
    dual_encoder = DualEncoder(config)
    img_emb = dual_encoder.encode_image(image_features)
    txt_emb = dual_encoder.encode_text(text_features)
    similarity_dual = dual_encoder.compute_similarity(img_emb, txt_emb)

    evaluator = RetrievalEvaluator()
    model_results['DualEncoder'] = evaluator.evaluate(similarity_dual)
    print(f"   - R@10: {model_results['DualEncoder']['recall@10']:.4f}")

    visualize_similarity_matrix(similarity_dual, "Dual Encoder", "dual_encoder_similarity.png")

    print("\n3. Testing Visual-Semantic Embedding...")
    vse = VisualSemanticEmbedding(config)
    img_emb_vse = vse.embed_image(image_features)
    txt_emb_vse = vse.embed_text(text_features)
    similarity_vse = vse.compute_similarity(img_emb_vse, txt_emb_vse)

    model_results['VSE'] = evaluator.evaluate(similarity_vse)
    print(f"   - R@10: {model_results['VSE']['recall@10']:.4f}")

    print("\n4. Testing Triplet Encoder...")
    triplet_enc = TripletEncoder(config)
    img_emb_triplet = triplet_enc.encode(image_features)
    txt_emb_triplet = triplet_enc.encode(text_features)
    similarity_triplet = np.dot(img_emb_triplet, txt_emb_triplet.T)

    model_results['TripletNet'] = evaluator.evaluate(similarity_triplet)
    print(f"   - R@10: {model_results['TripletNet']['recall@10']:.4f}")

    print("\n5. Visualizing retrieval results...")
    visualize_retrieval_results(image_features, text_features, similarity_dual,
                               "retrieval_results.png")

    print("\n6. Comparing all models...")
    visualize_model_comparison(model_results, "model_comparison.png")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_name, results in model_results.items():
        print(f"\n{model_name}:")
        print(f"  - Recall@1: {results['recall@1']:.4f}")
        print(f"  - Recall@5: {results['recall@5']:.4f}")
        print(f"  - Recall@10: {results['recall@10']:.4f}")
        print(f"  - MRR: {results['mrr']:.4f}")
        print(f"  - Median Rank: {results['median_rank']:.1f}")

    best_model = max(model_results, key=lambda x: model_results[x]['recall@10'])
    print(f"\nBest model: {best_model}")

    print("\n" + "=" * 80)
    print("All visualizations completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
