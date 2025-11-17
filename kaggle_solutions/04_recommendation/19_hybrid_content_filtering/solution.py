"""
Hybrid Content-Based Filtering
==============================

This solution implements hybrid recommendation systems combining multiple
content-based features, collaborative filtering signals, and metadata.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HybridRecommender:
    """Hybrid recommender combining content and collaborative features."""

    def __init__(self, content_weight=0.5, cf_weight=0.3, metadata_weight=0.2):
        """Initialize hybrid recommender with feature weights."""
        self.content_weight = content_weight
        self.cf_weight = cf_weight
        self.metadata_weight = metadata_weight
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.content_sim = None
        self.cf_sim = None
        self.metadata_sim = None
        self.item_ids = None
        self.scaler = StandardScaler()

    def fit(self, items_df, ratings_df, content_col='description', metadata_cols=None):
        """Fit hybrid model."""
        self.item_ids = items_df['item_id'].values

        # Content-based similarity
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(items_df[content_col].fillna(''))
        self.content_sim = cosine_similarity(tfidf_matrix)

        # Collaborative filtering similarity (item-item)
        ratings_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        item_ratings = ratings_matrix.T.values
        self.cf_sim = cosine_similarity(item_ratings)

        # Metadata similarity
        if metadata_cols:
            metadata_features = items_df[metadata_cols].values
            metadata_features_scaled = self.scaler.fit_transform(metadata_features)
            self.metadata_sim = cosine_similarity(metadata_features_scaled)
        else:
            self.metadata_sim = np.eye(len(items_df))

        print(f"   Content similarity shape: {self.content_sim.shape}")
        print(f"   CF similarity shape: {self.cf_sim.shape}")
        print(f"   Metadata similarity shape: {self.metadata_sim.shape}")

    def get_hybrid_similarity(self):
        """Compute hybrid similarity matrix."""
        hybrid_sim = (
            self.content_weight * self.content_sim +
            self.cf_weight * self.cf_sim +
            self.metadata_weight * self.metadata_sim
        )
        return hybrid_sim

    def get_similar_items(self, item_id, n_items=10):
        """Get similar items using hybrid similarity."""
        if item_id not in self.item_ids:
            return []

        item_idx = np.where(self.item_ids == item_id)[0][0]
        hybrid_sim = self.get_hybrid_similarity()

        sim_scores = list(enumerate(hybrid_sim[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top_items = [(self.item_ids[i], score) for i, score in sim_scores[1:n_items+1]]
        return top_items

    def recommend(self, user_history, n_recommendations=10):
        """Generate recommendations based on user history."""
        if not user_history:
            return []

        history_indices = [
            np.where(self.item_ids == item_id)[0][0]
            for item_id in user_history if item_id in self.item_ids
        ]

        if not history_indices:
            return []

        hybrid_sim = self.get_hybrid_similarity()
        aggregated_scores = np.sum(hybrid_sim[history_indices], axis=0)
        aggregated_scores[history_indices] = -1

        top_indices = np.argsort(aggregated_scores)[::-1][:n_recommendations]
        return [(self.item_ids[idx], aggregated_scores[idx]) for idx in top_indices]


def generate_synthetic_data(n_items=300, n_users=500):
    """Generate synthetic item and interaction data."""
    np.random.seed(42)

    # Generate items
    categories = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']
    keywords = {
        'Action': ['action', 'adventure', 'thrilling', 'explosive', 'intense'],
        'Comedy': ['funny', 'hilarious', 'entertaining', 'comedy', 'humor'],
        'Drama': ['emotional', 'drama', 'powerful', 'moving', 'compelling'],
        'Sci-Fi': ['science', 'fiction', 'futuristic', 'technology', 'space'],
        'Romance': ['romantic', 'love', 'heartwarming', 'relationship', 'emotional']
    }

    items = []
    for i in range(n_items):
        category = categories[i % len(categories)]
        description = ' '.join(np.random.choice(keywords[category], size=10))
        items.append({
            'item_id': i,
            'description': description,
            'category': category,
            'year': 2000 + (i % 24),
            'popularity': np.random.randint(1, 100),
            'feature_1': np.random.rand(),
            'feature_2': np.random.rand()
        })

    items_df = pd.DataFrame(items)

    # Generate ratings
    ratings = []
    for user_id in range(n_users):
        preferred_category = categories[user_id % len(categories)]
        for _ in range(10):
            if np.random.rand() < 0.7:
                item = items_df[items_df['category'] == preferred_category].sample(1).iloc[0]
            else:
                item = items_df.sample(1).iloc[0]

            ratings.append({
                'user_id': user_id,
                'item_id': item['item_id'],
                'rating': np.random.randint(3, 6)
            })

    ratings_df = pd.DataFrame(ratings).drop_duplicates(subset=['user_id', 'item_id'])
    return items_df, ratings_df


def calculate_precision_recall_at_k(recommendations, relevant_items, k=10):
    """Calculate Precision@K and Recall@K."""
    if not recommendations or not relevant_items:
        return 0.0, 0.0
    top_k = set([item for item, _ in recommendations[:k]])
    relevant = set(relevant_items)
    hits = len(top_k & relevant)
    return hits / len(top_k), hits / len(relevant) if relevant else 0.0


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
    print("Hybrid Content-Based Filtering Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic data...")
    items_df, ratings_df = generate_synthetic_data(n_items=300, n_users=500)
    print(f"   Items: {len(items_df)}")
    print(f"   Ratings: {len(ratings_df)}")

    # Train hybrid model
    print("\n2. Training hybrid recommender...")
    recommender = HybridRecommender(content_weight=0.4, cf_weight=0.4, metadata_weight=0.2)
    recommender.fit(items_df, ratings_df, content_col='description',
                   metadata_cols=['year', 'popularity', 'feature_1', 'feature_2'])

    # Split data
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Evaluate
    print("\n3. Evaluating recommendations...")
    test_users = test_ratings['user_id'].unique()[:50]
    ranking_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}

    for user_id in test_users:
        user_history = train_ratings[train_ratings['user_id'] == user_id]['item_id'].tolist()
        test_items = test_ratings[test_ratings['user_id'] == user_id]['item_id'].tolist()

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

    # Plot 1: Content similarity heatmap
    plt.subplot(3, 4, 1)
    sns.heatmap(recommender.content_sim[:40, :40], cmap='YlOrRd', square=True,
                cbar_kws={'label': 'Similarity'})
    plt.title('Content Similarity', fontweight='bold')

    # Plot 2: CF similarity heatmap
    plt.subplot(3, 4, 2)
    sns.heatmap(recommender.cf_sim[:40, :40], cmap='Blues', square=True,
                cbar_kws={'label': 'Similarity'})
    plt.title('Collaborative Filtering Similarity', fontweight='bold')

    # Plot 3: Hybrid similarity
    plt.subplot(3, 4, 3)
    hybrid_sim = recommender.get_hybrid_similarity()
    sns.heatmap(hybrid_sim[:40, :40], cmap='Greens', square=True,
                cbar_kws={'label': 'Similarity'})
    plt.title('Hybrid Similarity', fontweight='bold')

    # Plot 4: Ranking metrics
    plt.subplot(3, 4, 4)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names if ranking_metrics[m]]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    # Plots 5-12: Additional analysis
    plt.subplot(3, 4, 5)
    items_df['category'].value_counts().plot(kind='bar', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Item Category Distribution', fontweight='bold')

    plt.subplot(3, 4, 6)
    item_counts = ratings_df.groupby('item_id').size()
    plt.hist(item_counts, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Ratings per Item')
    plt.ylabel('Frequency')
    plt.title('Item Popularity', fontweight='bold')

    plt.subplot(3, 4, 7)
    user_counts = ratings_df.groupby('user_id').size()
    plt.hist(user_counts, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Ratings per User')
    plt.ylabel('Frequency')
    plt.title('User Activity', fontweight='bold')

    plt.subplot(3, 4, 8)
    weights = [recommender.content_weight, recommender.cf_weight, recommender.metadata_weight]
    plt.pie(weights, labels=['Content', 'CF', 'Metadata'], autopct='%1.1f%%', startangle=90)
    plt.title('Feature Weights', fontweight='bold')

    plt.subplot(3, 4, 9)
    rating_dist = ratings_df['rating'].value_counts().sort_index()
    plt.bar(rating_dist.index, rating_dist.values, alpha=0.8, color='purple')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution', fontweight='bold')

    plt.subplot(3, 4, 10)
    year_dist = items_df['year'].value_counts().sort_index()
    plt.plot(year_dist.index, year_dist.values, marker='o', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Items by Year', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 11)
    sample_item = items_df['item_id'].iloc[0]
    similar_items = recommender.get_similar_items(sample_item, n_items=10)
    items = [item for item, _ in similar_items]
    sims = [sim for _, sim in similar_items]
    plt.barh(range(len(items)), sims, alpha=0.8, color='teal')
    plt.yticks(range(len(items)), [f'Item {i}' for i in items])
    plt.xlabel('Similarity')
    plt.title(f'Similar to Item {sample_item}', fontweight='bold')

    plt.subplot(3, 4, 12)
    plt.scatter(ranking_metrics['Recall@10'], ranking_metrics['Precision@10'], alpha=0.6)
    plt.xlabel('Recall@10')
    plt.ylabel('Precision@10')
    plt.title('Precision-Recall Trade-off', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/19_hybrid_content_filtering/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
