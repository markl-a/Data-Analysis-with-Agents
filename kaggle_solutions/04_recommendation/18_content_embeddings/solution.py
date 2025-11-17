"""
Content-Based Recommendations with Embeddings
=============================================

This solution implements content-based filtering using neural embeddings
to capture semantic similarity between items for recommendation.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ContentEmbeddingRecommender:
    """Content-based recommender using item embeddings."""

    def __init__(self, embedding_dim=50, learning_rate=0.01, n_epochs=50):
        """Initialize content embedding recommender."""
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.item_embeddings = None
        self.item_ids = None
        self.item_features = None
        self.train_losses = []

    def fit(self, items_df, feature_columns):
        """Fit embeddings on item features."""
        np.random.seed(42)
        self.item_ids = items_df['item_id'].values
        self.item_features = items_df[feature_columns].values

        n_items = len(self.item_ids)
        n_features = self.item_features.shape[1]

        # Initialize embeddings
        self.item_embeddings = np.random.randn(n_items, self.embedding_dim) * 0.1

        # Initialize projection matrix
        W = np.random.randn(n_features, self.embedding_dim) * 0.1

        # Train to reconstruct features
        for epoch in range(self.n_epochs):
            # Forward pass
            reconstructed = self.item_embeddings.dot(W.T)

            # Calculate loss
            loss = np.mean((self.item_features - reconstructed) ** 2)
            self.train_losses.append(loss)

            # Backward pass
            error = reconstructed - self.item_features
            grad_embeddings = error.dot(W) * self.learning_rate
            grad_W = self.item_embeddings.T.dot(error).T * self.learning_rate

            # Update
            self.item_embeddings -= grad_embeddings
            W -= grad_W

            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {loss:.4f}")

    def get_similar_items(self, item_id, n_items=10):
        """Get similar items based on embedding similarity."""
        if item_id not in self.item_ids:
            return []

        item_idx = np.where(self.item_ids == item_id)[0][0]
        item_emb = self.item_embeddings[item_idx].reshape(1, -1)

        # Calculate cosine similarity
        similarities = cosine_similarity(item_emb, self.item_embeddings)[0]

        # Get top items
        top_indices = np.argsort(similarities)[::-1][1:n_items+1]

        return [(self.item_ids[i], similarities[i]) for i in top_indices]

    def recommend(self, user_history, n_recommendations=10):
        """Generate recommendations based on user history."""
        if not user_history:
            return []

        # Get embeddings of items in history
        history_indices = [
            np.where(self.item_ids == item_id)[0][0]
            for item_id in user_history if item_id in self.item_ids
        ]

        if not history_indices:
            return []

        # Average user profile
        user_profile = self.item_embeddings[history_indices].mean(axis=0).reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(user_profile, self.item_embeddings)[0]

        # Exclude history
        similarities[history_indices] = -1

        # Get top items
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        return [(self.item_ids[i], similarities[i]) for i in top_indices]


def generate_synthetic_data(n_items=300, n_users=500, n_features=20):
    """Generate synthetic item features and user interactions."""
    np.random.seed(42)

    # Generate item features (categorical clusters)
    n_clusters = 5
    items = []
    for i in range(n_items):
        cluster = i % n_clusters
        features = np.random.randn(n_features)
        # Add cluster bias
        features += np.random.randn(n_features) * 0.5 * (cluster + 1)
        features = (features - features.min()) / (features.max() - features.min())

        items.append({'item_id': i, **{f'feature_{j}': features[j] for j in range(n_features)}})

    items_df = pd.DataFrame(items)

    # Generate interactions
    interactions = []
    for user_id in range(n_users):
        preferred_cluster = user_id % n_clusters
        for _ in range(10):
            if np.random.rand() < 0.7:
                item_id = np.random.choice([i for i in range(n_items) if i % n_clusters == preferred_cluster])
            else:
                item_id = np.random.randint(0, n_items)

            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': np.random.randint(3, 6)
            })

    interactions_df = pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'item_id'])

    return items_df, interactions_df


def calculate_precision_recall_at_k(recommendations, relevant_items, k=10):
    """Calculate Precision@K and Recall@K."""
    if not recommendations or not relevant_items:
        return 0.0, 0.0
    top_k = set([item for item, _ in recommendations[:k]])
    relevant = set(relevant_items)
    hits = len(top_k & relevant)
    precision = hits / len(top_k)
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def calculate_ndcg_at_k(recommendations, relevant_items, k=10):
    """Calculate NDCG@K."""
    if not recommendations or not relevant_items:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, (item, _) in enumerate(recommendations[:k])
              if item in relevant_items)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    """Main execution function."""
    print("=" * 80)
    print("Content-Based Recommendations with Embeddings")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic data...")
    items_df, interactions_df = generate_synthetic_data(n_items=300, n_users=500)
    feature_cols = [col for col in items_df.columns if col.startswith('feature_')]
    print(f"   Generated {len(items_df)} items with {len(feature_cols)} features")
    print(f"   Generated {len(interactions_df)} interactions")

    # Train model
    print("\n2. Training embedding model...")
    recommender = ContentEmbeddingRecommender(embedding_dim=50, n_epochs=50)
    recommender.fit(items_df, feature_cols)

    # Split interactions
    train_int, test_int = train_test_split(interactions_df, test_size=0.2, random_state=42)

    # Evaluate
    print("\n3. Evaluating recommendations...")
    test_users = test_int['user_id'].unique()[:50]
    ranking_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}

    for user_id in test_users:
        user_history = train_int[train_int['user_id'] == user_id]['item_id'].tolist()
        test_items = test_int[test_int['user_id'] == user_id]['item_id'].tolist()

        if not user_history or not test_items:
            continue

        recommendations = recommender.recommend(user_history, n_recommendations=10)
        prec, rec = calculate_precision_recall_at_k(recommendations, test_items, 10)
        ndcg = calculate_ndcg_at_k(recommendations, test_items, 10)

        ranking_metrics['Precision@10'].append(prec)
        ranking_metrics['Recall@10'].append(rec)
        ranking_metrics['NDCG@10'].append(ndcg)

    for metric, values in ranking_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n4. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Training loss
    plt.subplot(3, 4, 1)
    plt.plot(recommender.train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training Loss', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 2: Embedding visualization (t-SNE)
    plt.subplot(3, 4, 2)
    if len(recommender.item_embeddings) > 50:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(recommender.item_embeddings)
        item_clusters = np.arange(len(items_df)) % 5
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=item_clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Item Embeddings (t-SNE)', fontweight='bold')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

    # Plot 3: Embedding heatmap
    plt.subplot(3, 4, 3)
    plt.imshow(recommender.item_embeddings[:50, :30], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Items')
    plt.title('Item Embeddings Heatmap', fontweight='bold')

    # Plot 4: Similarity matrix
    plt.subplot(3, 4, 4)
    sim_matrix = cosine_similarity(recommender.item_embeddings[:40])
    sns.heatmap(sim_matrix, cmap='YlOrRd', square=True, cbar_kws={'label': 'Similarity'})
    plt.title('Item Similarity Matrix', fontweight='bold')

    # Plot 5-12: Additional plots
    plt.subplot(3, 4, 5)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names if ranking_metrics[m]]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    plt.subplot(3, 4, 6)
    embedding_norms = np.linalg.norm(recommender.item_embeddings, axis=1)
    plt.hist(embedding_norms, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Embedding Norm')
    plt.ylabel('Frequency')
    plt.title('Embedding Magnitude Distribution', fontweight='bold')

    plt.subplot(3, 4, 7)
    feature_variance = items_df[feature_cols].var().values
    plt.bar(range(len(feature_variance)), feature_variance, alpha=0.8, color='coral')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.title('Feature Variance', fontweight='bold')

    plt.subplot(3, 4, 8)
    user_counts = interactions_df.groupby('user_id').size()
    plt.hist(user_counts, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Interactions per User')
    plt.ylabel('Frequency')
    plt.title('User Activity', fontweight='bold')

    plt.subplot(3, 4, 9)
    item_counts = interactions_df.groupby('item_id').size()
    plt.hist(item_counts, bins=20, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Interactions per Item')
    plt.ylabel('Frequency')
    plt.title('Item Popularity', fontweight='bold')

    plt.subplot(3, 4, 10)
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(recommender.item_embeddings)
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
               c=np.arange(len(items_df)) % 5, cmap='plasma', alpha=0.6)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Embeddings (PCA)', fontweight='bold')
    plt.colorbar(label='Cluster')

    plt.subplot(3, 4, 11)
    embedding_corr = np.corrcoef(recommender.item_embeddings[:30])
    sns.heatmap(embedding_corr, cmap='RdBu_r', center=0, square=True,
                cbar_kws={'label': 'Correlation'})
    plt.title('Item Embedding Correlation', fontweight='bold')

    plt.subplot(3, 4, 12)
    sample_item = items_df['item_id'].iloc[0]
    similar_items = recommender.get_similar_items(sample_item, n_items=10)
    items = [item for item, _ in similar_items]
    sims = [sim for _, sim in similar_items]
    plt.barh(range(len(items)), sims, alpha=0.8, color='teal')
    plt.yticks(range(len(items)), [f'Item {i}' for i in items])
    plt.xlabel('Similarity')
    plt.title(f'Similar to Item {sample_item}', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/18_content_embeddings/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
