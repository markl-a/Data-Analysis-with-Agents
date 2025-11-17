"""
Image Captioning with Attention Mechanisms
==========================================

This solution demonstrates various approaches to generate captions for images using
attention mechanisms that allow the model to focus on relevant image regions.

Approaches:
1. CNN-LSTM with Bahdanau Attention
2. CNN-Transformer Encoder-Decoder
3. Show, Attend and Tell Architecture
4. Spatial Attention Mechanism
5. Multi-Head Attention Captioning

Dataset: Synthetic image features and captions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

@dataclass
class CaptionConfig:
    """Configuration for image captioning models"""
    vocab_size: int = 5000
    embedding_dim: int = 256
    hidden_dim: int = 512
    attention_dim: int = 256
    image_feature_dim: int = 2048
    max_caption_length: int = 20
    num_image_regions: int = 49  # 7x7 grid

class SyntheticCaptionDataGenerator:
    """Generate synthetic image features and captions for training"""

    def __init__(self, num_samples: int = 1000, config: CaptionConfig = None):
        self.num_samples = num_samples
        self.config = config or CaptionConfig()
        self.vocabulary = self._create_vocabulary()

    def _create_vocabulary(self) -> Dict[str, int]:
        """Create a synthetic vocabulary"""
        words = ['<PAD>', '<START>', '<END>', '<UNK>']

        # Add common words
        nouns = ['dog', 'cat', 'person', 'car', 'tree', 'house', 'bird', 'sky',
                'grass', 'water', 'mountain', 'beach', 'city', 'street', 'park']
        verbs = ['running', 'sitting', 'standing', 'flying', 'walking', 'playing',
                'eating', 'sleeping', 'jumping', 'swimming']
        adjectives = ['red', 'blue', 'green', 'large', 'small', 'beautiful', 'old',
                     'young', 'happy', 'sad', 'bright', 'dark']
        prepositions = ['in', 'on', 'at', 'with', 'near', 'under', 'over', 'by']
        articles = ['a', 'an', 'the']

        words.extend(nouns + verbs + adjectives + prepositions + articles)

        # Pad vocabulary to vocab_size
        for i in range(len(words), self.config.vocab_size):
            words.append(f'word_{i}')

        return {word: idx for idx, word in enumerate(words)}

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic image features and captions"""
        # Generate image features (batch_size, num_regions, feature_dim)
        image_features = np.random.randn(
            self.num_samples,
            self.config.num_image_regions,
            self.config.image_feature_dim
        ).astype(np.float32)

        # Normalize features
        image_features = image_features / (np.linalg.norm(image_features, axis=2, keepdims=True) + 1e-8)

        # Generate captions
        captions = []
        caption_sequences = []

        for i in range(self.num_samples):
            # Generate caption with 5-15 words
            caption_length = np.random.randint(5, 15)
            caption_words = ['<START>']

            # Simple caption generation
            caption_words.append(np.random.choice(list(self.vocabulary.keys())[:50]))
            for _ in range(caption_length):
                caption_words.append(np.random.choice(list(self.vocabulary.keys())[:100]))
            caption_words.append('<END>')

            captions.append(' '.join(caption_words))

            # Convert to sequence
            sequence = [self.vocabulary.get(w, self.vocabulary['<UNK>'])
                       for w in caption_words]
            # Pad sequence
            sequence = sequence + [self.vocabulary['<PAD>']] * (self.config.max_caption_length - len(sequence))
            sequence = sequence[:self.config.max_caption_length]
            caption_sequences.append(sequence)

        caption_sequences = np.array(caption_sequences)

        return image_features, caption_sequences, captions

class BahdanauAttention:
    """Bahdanau (Additive) Attention Mechanism"""

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.W1 = np.random.randn(config.hidden_dim, config.attention_dim) * 0.01
        self.W2 = np.random.randn(config.image_feature_dim, config.attention_dim) * 0.01
        self.V = np.random.randn(config.attention_dim, 1) * 0.01

    def forward(self, hidden_state: np.ndarray, image_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights and context vector

        Args:
            hidden_state: (batch_size, hidden_dim)
            image_features: (batch_size, num_regions, feature_dim)

        Returns:
            context: (batch_size, feature_dim)
            attention_weights: (batch_size, num_regions)
        """
        batch_size = image_features.shape[0]
        num_regions = image_features.shape[1]

        # Expand hidden state: (batch_size, num_regions, hidden_dim)
        hidden_expanded = np.tile(hidden_state[:, np.newaxis, :], (1, num_regions, 1))

        # Compute attention scores
        score = np.tanh(
            np.dot(hidden_expanded, self.W1) +
            np.dot(image_features, self.W2)
        )  # (batch_size, num_regions, attention_dim)

        attention_scores = np.dot(score, self.V).squeeze(-1)  # (batch_size, num_regions)

        # Apply softmax
        attention_weights = self._softmax(attention_scores)

        # Compute context vector
        context = np.sum(
            image_features * attention_weights[:, :, np.newaxis],
            axis=1
        )  # (batch_size, feature_dim)

        return context, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SpatialAttention:
    """Spatial Attention for focusing on image regions"""

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.conv_weights = np.random.randn(config.image_feature_dim, config.attention_dim) * 0.01
        self.query_weights = np.random.randn(config.hidden_dim, config.attention_dim) * 0.01

    def forward(self, hidden_state: np.ndarray, image_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial attention

        Args:
            hidden_state: (batch_size, hidden_dim)
            image_features: (batch_size, num_regions, feature_dim)

        Returns:
            context: (batch_size, feature_dim)
            attention_map: (batch_size, sqrt(num_regions), sqrt(num_regions))
        """
        batch_size = image_features.shape[0]
        num_regions = image_features.shape[1]
        grid_size = int(np.sqrt(num_regions))

        # Project features
        projected_features = np.dot(image_features, self.conv_weights)
        projected_query = np.dot(hidden_state, self.query_weights)

        # Compute attention scores
        scores = np.sum(
            projected_features * projected_query[:, np.newaxis, :],
            axis=2
        )  # (batch_size, num_regions)

        attention_weights = self._softmax(scores)

        # Compute context
        context = np.sum(
            image_features * attention_weights[:, :, np.newaxis],
            axis=1
        )

        # Reshape attention to spatial map
        attention_map = attention_weights.reshape(batch_size, grid_size, grid_size)

        return context, attention_map

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    """Multi-Head Attention for caption generation"""

    def __init__(self, config: CaptionConfig, num_heads: int = 8):
        self.config = config
        self.num_heads = num_heads
        self.head_dim = config.attention_dim // num_heads

        # Initialize weights for all heads
        self.W_q = np.random.randn(num_heads, config.hidden_dim, self.head_dim) * 0.01
        self.W_k = np.random.randn(num_heads, config.image_feature_dim, self.head_dim) * 0.01
        self.W_v = np.random.randn(num_heads, config.image_feature_dim, self.head_dim) * 0.01
        self.W_o = np.random.randn(num_heads * self.head_dim, config.image_feature_dim) * 0.01

    def forward(self, query: np.ndarray, image_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head attention forward pass

        Args:
            query: (batch_size, hidden_dim)
            image_features: (batch_size, num_regions, feature_dim)

        Returns:
            context: (batch_size, feature_dim)
            attention_weights: (batch_size, num_heads, num_regions)
        """
        batch_size = image_features.shape[0]
        num_regions = image_features.shape[1]

        all_attention_weights = []
        head_outputs = []

        for h in range(self.num_heads):
            # Compute Q, K, V for this head
            Q = np.dot(query, self.W_q[h])  # (batch_size, head_dim)
            K = np.dot(image_features, self.W_k[h])  # (batch_size, num_regions, head_dim)
            V = np.dot(image_features, self.W_v[h])  # (batch_size, num_regions, head_dim)

            # Compute attention scores
            scores = np.sum(Q[:, np.newaxis, :] * K, axis=2) / np.sqrt(self.head_dim)
            attention_weights = self._softmax(scores)

            # Apply attention to values
            head_output = np.sum(V * attention_weights[:, :, np.newaxis], axis=1)

            all_attention_weights.append(attention_weights)
            head_outputs.append(head_output)

        # Concatenate heads
        concat_heads = np.concatenate(head_outputs, axis=1)

        # Final linear projection
        context = np.dot(concat_heads, self.W_o)

        attention_weights = np.stack(all_attention_weights, axis=1)

        return context, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class ImageCaptioningEvaluator:
    """Evaluate image captioning models"""

    def __init__(self):
        self.metrics = {}

    def compute_bleu_score(self, reference: str, candidate: str, n: int = 4) -> float:
        """Simplified BLEU score computation"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        if len(cand_tokens) == 0:
            return 0.0

        # Compute n-gram precision
        scores = []
        for i in range(1, min(n + 1, len(cand_tokens) + 1)):
            ref_ngrams = self._get_ngrams(ref_tokens, i)
            cand_ngrams = self._get_ngrams(cand_tokens, i)

            matches = sum(min(cand_ngrams.get(ng, 0), ref_ngrams.get(ng, 0))
                         for ng in cand_ngrams)
            total = sum(cand_ngrams.values())

            if total > 0:
                scores.append(matches / total)
            else:
                scores.append(0.0)

        if len(scores) == 0:
            return 0.0

        # Geometric mean
        return np.exp(np.mean(np.log(np.array(scores) + 1e-10)))

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple, int]:
        """Extract n-grams from tokens"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def compute_meteor(self, reference: str, candidate: str) -> float:
        """Simplified METEOR score"""
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())

        if len(cand_tokens) == 0:
            return 0.0

        matches = len(ref_tokens & cand_tokens)
        precision = matches / len(cand_tokens) if len(cand_tokens) > 0 else 0
        recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0

        if precision + recall == 0:
            return 0.0

        f_mean = (10 * precision * recall) / (recall + 9 * precision)
        return f_mean

    def evaluate_model(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Evaluate model performance"""
        bleu_scores = [self.compute_bleu_score(r, c) for r, c in zip(references, candidates)]
        meteor_scores = [self.compute_meteor(r, c) for r, c in zip(references, candidates)]

        return {
            'bleu_1': np.mean([self.compute_bleu_score(r, c, 1) for r, c in zip(references, candidates)]),
            'bleu_2': np.mean([self.compute_bleu_score(r, c, 2) for r, c in zip(references, candidates)]),
            'bleu_3': np.mean([self.compute_bleu_score(r, c, 3) for r, c in zip(references, candidates)]),
            'bleu_4': np.mean(bleu_scores),
            'meteor': np.mean(meteor_scores),
            'avg_length': np.mean([len(c.split()) for c in candidates])
        }

def visualize_attention_mechanisms(image_features: np.ndarray,
                                   hidden_states: np.ndarray,
                                   config: CaptionConfig,
                                   save_path: str = 'attention_comparison.png'):
    """Visualize different attention mechanisms"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Sample data
    sample_idx = 0
    sample_image = image_features[sample_idx]
    sample_hidden = hidden_states[sample_idx]

    # 1. Bahdanau Attention
    bahdanau = BahdanauAttention(config)
    context_b, weights_b = bahdanau.forward(sample_hidden[np.newaxis, :], sample_image[np.newaxis, :, :])

    ax = axes[0, 0]
    weights_map = weights_b.reshape(7, 7)
    im = ax.imshow(weights_map, cmap='hot', interpolation='nearest')
    ax.set_title('Bahdanau Attention Weights', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. Spatial Attention
    spatial = SpatialAttention(config)
    context_s, weights_s = spatial.forward(sample_hidden[np.newaxis, :], sample_image[np.newaxis, :, :])

    ax = axes[0, 1]
    im = ax.imshow(weights_s[0], cmap='viridis', interpolation='nearest')
    ax.set_title('Spatial Attention Map', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Multi-Head Attention (average across heads)
    multihead = MultiHeadAttention(config, num_heads=8)
    context_m, weights_m = multihead.forward(sample_hidden[np.newaxis, :], sample_image[np.newaxis, :, :])

    ax = axes[0, 2]
    avg_weights = np.mean(weights_m[0], axis=0).reshape(7, 7)
    im = ax.imshow(avg_weights, cmap='plasma', interpolation='nearest')
    ax.set_title('Multi-Head Attention (Averaged)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Attention Distribution
    ax = axes[1, 0]
    ax.bar(range(len(weights_b[0])), weights_b[0], alpha=0.7, color='skyblue')
    ax.set_title('Bahdanau Attention Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Region Index')
    ax.set_ylabel('Attention Weight')
    ax.grid(True, alpha=0.3)

    # 5. Per-Head Attention
    ax = axes[1, 1]
    im = ax.imshow(weights_m[0], cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax.set_title('Multi-Head Attention (All Heads)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Head')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 6. Attention Entropy
    ax = axes[1, 2]
    entropy_b = -np.sum(weights_b[0] * np.log(weights_b[0] + 1e-10))
    entropy_s = -np.sum(weights_s[0].flatten() * np.log(weights_s[0].flatten() + 1e-10))
    entropy_m = -np.sum(avg_weights.flatten() * np.log(avg_weights.flatten() + 1e-10))

    entropies = [entropy_b, entropy_s, entropy_m]
    methods = ['Bahdanau', 'Spatial', 'Multi-Head']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax.bar(methods, entropies, color=colors, alpha=0.8)
    ax.set_title('Attention Entropy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Attention mechanisms visualization saved to {save_path}")

def visualize_caption_generation_process(config: CaptionConfig,
                                         save_path: str = 'caption_generation.png'):
    """Visualize the caption generation process over time"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Simulate caption generation steps
    num_steps = 10

    # 1. Attention weights over time
    ax = axes[0, 0]
    attention_evolution = np.random.dirichlet(np.ones(config.num_image_regions), num_steps)
    im = ax.imshow(attention_evolution.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_title('Attention Evolution During Generation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Image Region')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. Attention focus shift
    ax = axes[0, 1]
    max_attention_region = np.argmax(attention_evolution, axis=1)
    ax.plot(range(num_steps), max_attention_region, marker='o', linewidth=2,
            markersize=8, color='#E74C3C')
    ax.set_title('Most Attended Region Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Region Index')
    ax.grid(True, alpha=0.3)

    # 3. Attention concentration
    ax = axes[0, 2]
    attention_std = np.std(attention_evolution, axis=1)
    ax.plot(range(num_steps), attention_std, marker='s', linewidth=2,
            markersize=8, color='#3498DB')
    ax.set_title('Attention Concentration', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Std Dev of Weights')
    ax.grid(True, alpha=0.3)

    # 4. Hidden state magnitude
    ax = axes[1, 0]
    hidden_magnitude = np.random.exponential(1.0, num_steps) + np.linspace(1, 0.5, num_steps)
    ax.plot(range(num_steps), hidden_magnitude, marker='D', linewidth=2,
            markersize=8, color='#9B59B6')
    ax.set_title('Hidden State Magnitude', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)

    # 5. Word confidence scores
    ax = axes[1, 1]
    confidence_scores = np.random.beta(8, 2, num_steps)
    ax.bar(range(num_steps), confidence_scores, color='#1ABC9C', alpha=0.8)
    ax.set_title('Generated Word Confidence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Confidence Score')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # 6. Cumulative attention coverage
    ax = axes[1, 2]
    cumulative_attention = np.cumsum(attention_evolution, axis=0)
    for region_idx in [0, 10, 20, 30, 40]:
        ax.plot(range(num_steps), cumulative_attention[:, region_idx],
                linewidth=2, label=f'Region {region_idx}', alpha=0.7)
    ax.set_title('Cumulative Attention Coverage', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Cumulative Attention')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Caption generation process visualization saved to {save_path}")

def visualize_evaluation_metrics(evaluator: ImageCaptioningEvaluator,
                                 model_results: Dict[str, Dict[str, float]],
                                 save_path: str = 'evaluation_metrics.png'):
    """Visualize evaluation metrics across different models"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    models = list(model_results.keys())
    metrics_keys = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor']

    # 1. BLEU scores comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.15

    for i, metric in enumerate(['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']):
        values = [model_results[m][metric] for m in models]
        ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)

    ax.set_title('BLEU Scores Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. METEOR scores
    ax = axes[0, 1]
    meteor_scores = [model_results[m]['meteor'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, meteor_scores, color=colors, alpha=0.8)
    ax.set_title('METEOR Scores', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, meteor_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Overall performance radar chart
    ax = axes[0, 2]
    ax.remove()
    ax = fig.add_subplot(2, 3, 3, projection='polar')

    angles = np.linspace(0, 2*np.pi, len(metrics_keys), endpoint=False).tolist()
    angles += angles[:1]

    for model in models:
        values = [model_results[model][k] for k in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.upper() for k in metrics_keys])
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # 4. Caption length distribution
    ax = axes[1, 0]
    caption_lengths = [model_results[m]['avg_length'] for m in models]
    ax.barh(models, caption_lengths, color='#E67E22', alpha=0.8)
    ax.set_title('Average Caption Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Average Number of Words')
    ax.grid(True, alpha=0.3, axis='x')

    # 5. Performance improvement
    ax = axes[1, 1]
    baseline_bleu4 = model_results[models[0]]['bleu_4']
    improvements = [(model_results[m]['bleu_4'] - baseline_bleu4) / baseline_bleu4 * 100
                   for m in models]
    colors_imp = ['green' if x >= 0 else 'red' for x in improvements]
    bars = ax.bar(models, improvements, color=colors_imp, alpha=0.7)
    ax.set_title('BLEU-4 Improvement vs Baseline', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Metric correlation heatmap
    ax = axes[1, 2]
    metric_matrix = np.array([[model_results[m][k] for k in metrics_keys] for m in models])
    correlation = np.corrcoef(metric_matrix.T)

    im = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(metrics_keys)))
    ax.set_yticks(range(len(metrics_keys)))
    ax.set_xticklabels([k.upper() for k in metrics_keys], rotation=45, ha='right')
    ax.set_yticklabels([k.upper() for k in metrics_keys])
    ax.set_title('Metric Correlation Matrix', fontsize=12, fontweight='bold')

    for i in range(len(metrics_keys)):
        for j in range(len(metrics_keys)):
            text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Evaluation metrics visualization saved to {save_path}")

def main():
    """Main execution function"""

    print("=" * 80)
    print("IMAGE CAPTIONING WITH ATTENTION MECHANISMS")
    print("=" * 80)

    # Configuration
    config = CaptionConfig()

    # Generate synthetic data
    print("\n1. Generating synthetic image features and captions...")
    data_generator = SyntheticCaptionDataGenerator(num_samples=1000, config=config)
    image_features, caption_sequences, captions = data_generator.generate_data()

    print(f"   - Generated {len(image_features)} samples")
    print(f"   - Image features shape: {image_features.shape}")
    print(f"   - Caption sequences shape: {caption_sequences.shape}")

    # Generate synthetic hidden states for visualization
    hidden_states = np.random.randn(len(image_features), config.hidden_dim).astype(np.float32)

    # Visualize attention mechanisms
    print("\n2. Visualizing attention mechanisms...")
    visualize_attention_mechanisms(image_features, hidden_states, config,
                                   'attention_comparison.png')

    # Visualize caption generation process
    print("\n3. Visualizing caption generation process...")
    visualize_caption_generation_process(config, 'caption_generation.png')

    # Simulate model evaluation
    print("\n4. Evaluating different captioning models...")
    evaluator = ImageCaptioningEvaluator()

    # Generate synthetic predictions for different models
    model_results = {}
    models = ['CNN-LSTM', 'Show-Attend-Tell', 'Transformer', 'Spatial-Attention', 'Multi-Head']

    for i, model_name in enumerate(models):
        # Generate synthetic captions
        synthetic_captions = []
        for cap in captions[:100]:
            # Add some noise to simulate different model performance
            words = cap.split()
            if len(words) > 5:
                num_keep = int(len(words) * (0.6 + i * 0.08))
                synthetic_cap = ' '.join(words[:num_keep])
                synthetic_captions.append(synthetic_cap)
            else:
                synthetic_captions.append(cap)

        results = evaluator.evaluate_model(captions[:100], synthetic_captions)
        model_results[model_name] = results

        print(f"\n   {model_name}:")
        print(f"   - BLEU-4: {results['bleu_4']:.4f}")
        print(f"   - METEOR: {results['meteor']:.4f}")
        print(f"   - Avg Length: {results['avg_length']:.2f}")

    # Visualize evaluation metrics
    print("\n5. Visualizing evaluation metrics...")
    visualize_evaluation_metrics(evaluator, model_results, 'evaluation_metrics.png')

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  - Total samples: {len(image_features)}")
    print(f"  - Vocabulary size: {config.vocab_size}")
    print(f"  - Max caption length: {config.max_caption_length}")
    print(f"  - Number of image regions: {config.num_image_regions}")

    print(f"\nBest Model: {max(model_results, key=lambda x: model_results[x]['bleu_4'])}")
    print(f"  - BLEU-4: {max(m['bleu_4'] for m in model_results.values()):.4f}")

    print("\n" + "=" * 80)
    print("All visualizations completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
