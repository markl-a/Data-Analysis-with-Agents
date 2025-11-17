"""
TF-IDF Based Content Recommendations
====================================

This solution implements content-based recommendation systems using TF-IDF
(Term Frequency-Inverse Document Frequency) for text-based item similarity
and recommendation generation.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TFIDFRecommender:
    """TF-IDF based content recommender system."""

    def __init__(self, max_features=1000, ngram_range=(1, 2), min_df=2):
        """
        Initialize TF-IDF recommender.

        Parameters:
        -----------
        max_features : int
            Maximum number of features
        ngram_range : tuple
            N-gram range for TF-IDF
        min_df : int
            Minimum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.item_similarity = None
        self.item_ids = None
        self.feature_names = None

    def fit(self, items_df, content_column='description'):
        """
        Fit TF-IDF model on item content.

        Parameters:
        -----------
        items_df : pd.DataFrame
            DataFrame with item_id and content columns
        content_column : str
            Column name containing item descriptions
        """
        self.item_ids = items_df['item_id'].values

        # Compute TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(
            items_df[content_column].fillna('')
        )

        # Compute item similarity matrix
        self.item_similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"   TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"   Number of features: {len(self.feature_names)}")

    def get_similar_items(self, item_id, n_items=10):
        """Get most similar items based on content."""
        if item_id not in self.item_ids:
            return []

        item_idx = np.where(self.item_ids == item_id)[0][0]

        # Get similarity scores
        sim_scores = list(enumerate(self.item_similarity[item_idx]))

        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top items (excluding the item itself)
        top_items = [(self.item_ids[i], score) for i, score in sim_scores[1:n_items+1]]

        return top_items

    def recommend(self, user_history, n_recommendations=10):
        """
        Generate recommendations based on user history.

        Parameters:
        -----------
        user_history : list
            List of item_ids the user has interacted with
        n_recommendations : int
            Number of recommendations
        """
        if not user_history:
            return []

        # Get indices of items in user history
        history_indices = [
            np.where(self.item_ids == item_id)[0][0]
            for item_id in user_history
            if item_id in self.item_ids
        ]

        if not history_indices:
            return []

        # Aggregate similarity scores
        aggregated_scores = np.sum(self.item_similarity[history_indices], axis=0)

        # Set scores of items in history to -1
        aggregated_scores[history_indices] = -1

        # Get top items
        top_indices = np.argsort(aggregated_scores)[::-1][:n_recommendations]

        recommendations = [
            (self.item_ids[idx], aggregated_scores[idx])
            for idx in top_indices
        ]

        return recommendations

    def get_top_features(self, item_id, n_features=10):
        """Get top TF-IDF features for an item."""
        if item_id not in self.item_ids:
            return []

        item_idx = np.where(self.item_ids == item_id)[0][0]

        # Get TF-IDF scores for this item
        feature_scores = self.tfidf_matrix[item_idx].toarray().flatten()

        # Get top features
        top_indices = np.argsort(feature_scores)[::-1][:n_features]

        top_features = [
            (self.feature_names[idx], feature_scores[idx])
            for idx in top_indices
            if feature_scores[idx] > 0
        ]

        return top_features


def generate_synthetic_items(n_items=300, n_categories=10):
    """Generate synthetic item descriptions."""
    np.random.seed(42)

    categories = [
        ['action', 'adventure', 'thriller', 'suspense', 'fast-paced'],
        ['comedy', 'funny', 'humor', 'entertaining', 'lighthearted'],
        ['drama', 'emotional', 'intense', 'compelling', 'character-driven'],
        ['science', 'fiction', 'futuristic', 'technology', 'space'],
        ['romance', 'love', 'relationship', 'heartwarming', 'emotional'],
        ['horror', 'scary', 'frightening', 'suspenseful', 'dark'],
        ['fantasy', 'magical', 'mythical', 'enchanting', 'otherworldly'],
        ['mystery', 'detective', 'investigation', 'puzzle', 'crime'],
        ['documentary', 'educational', 'informative', 'real-life', 'factual'],
        ['animation', 'animated', 'colorful', 'family-friendly', 'cartoon']
    ]

    adjectives = ['amazing', 'incredible', 'outstanding', 'excellent', 'brilliant',
                 'wonderful', 'fantastic', 'superb', 'remarkable', 'exceptional']

    items = []
    for i in range(n_items):
        # Select category
        category_idx = i % n_categories
        category_words = categories[category_idx]

        # Generate description
        n_words = np.random.randint(10, 30)
        description_words = []

        for _ in range(n_words):
            if np.random.rand() < 0.6:
                description_words.append(np.random.choice(category_words))
            else:
                description_words.append(np.random.choice(adjectives))

        description = ' '.join(description_words)

        items.append({
            'item_id': i,
            'description': description,
            'category': category_idx
        })

    return pd.DataFrame(items)


def generate_user_interactions(items_df, n_users=500, interactions_per_user=10):
    """Generate user-item interactions."""
    np.random.seed(42)

    interactions = []

    for user_id in range(n_users):
        # Users prefer certain categories
        preferred_category = user_id % 10

        for _ in range(interactions_per_user):
            # 70% chance to interact with preferred category
            if np.random.rand() < 0.7:
                category_items = items_df[items_df['category'] == preferred_category]
            else:
                category_items = items_df

            if len(category_items) > 0:
                item = category_items.sample(1).iloc[0]
                interactions.append({
                    'user_id': user_id,
                    'item_id': item['item_id'],
                    'rating': np.random.randint(3, 6)  # Ratings 3-5
                })

    return pd.DataFrame(interactions)


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

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, (item, _) in enumerate(recommendations[:k])
        if item in relevant_items
    )

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_diversity(recommendations, items_df):
    """Calculate diversity of recommendations."""
    if len(recommendations) < 2:
        return 0.0

    item_ids = [item for item, _ in recommendations]
    categories = items_df[items_df['item_id'].isin(item_ids)]['category'].values

    return len(set(categories)) / len(categories)


def main():
    """Main execution function."""
    print("=" * 80)
    print("TF-IDF Based Content Recommendation System")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic item data...")
    items_df = generate_synthetic_items(n_items=300, n_categories=10)
    print(f"   Generated {len(items_df)} items")
    print(f"   Categories: {items_df['category'].nunique()}")

    print("\n2. Generating user interactions...")
    interactions_df = generate_user_interactions(items_df, n_users=500)
    print(f"   Generated {len(interactions_df)} interactions")
    print(f"   Users: {interactions_df['user_id'].nunique()}")

    # Train TF-IDF model
    print("\n3. Training TF-IDF model...")
    recommender = TFIDFRecommender(max_features=500, ngram_range=(1, 2))
    recommender.fit(items_df, content_column='description')

    # Evaluate content similarity
    print("\n4. Analyzing item similarities...")
    sample_item = items_df['item_id'].iloc[0]
    similar_items = recommender.get_similar_items(sample_item, n_items=10)
    print(f"\n   Items similar to Item {sample_item}:")
    for item_id, score in similar_items[:5]:
        print(f"      Item {item_id}: {score:.4f}")

    # Split interactions
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42
    )

    # Generate recommendations
    print("\n5. Generating recommendations...")
    test_users = test_interactions['user_id'].unique()[:50]

    ranking_metrics = {
        'Precision@10': [], 'Recall@10': [], 'NDCG@10': [], 'Diversity': []
    }

    for user_id in test_users:
        # Get user history from train set
        user_history = train_interactions[
            train_interactions['user_id'] == user_id
        ]['item_id'].tolist()

        if not user_history:
            continue

        # Get test items (ground truth)
        test_items = test_interactions[
            test_interactions['user_id'] == user_id
        ]['item_id'].tolist()

        if not test_items:
            continue

        # Generate recommendations
        recommendations = recommender.recommend(user_history, n_recommendations=10)

        # Calculate metrics
        prec, rec = calculate_precision_recall_at_k(recommendations, test_items, 10)
        ndcg = calculate_ndcg_at_k(recommendations, test_items, 10)
        diversity = calculate_diversity(recommendations, items_df)

        ranking_metrics['Precision@10'].append(prec)
        ranking_metrics['Recall@10'].append(rec)
        ranking_metrics['NDCG@10'].append(ndcg)
        ranking_metrics['Diversity'].append(diversity)

    for metric, values in ranking_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n6. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Category distribution
    plt.subplot(3, 4, 1)
    items_df['category'].value_counts().sort_index().plot(kind='bar', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Number of Items')
    plt.title('Item Category Distribution', fontweight='bold')

    # Plot 2: TF-IDF feature importance
    plt.subplot(3, 4, 2)
    tfidf_means = recommender.tfidf_matrix.mean(axis=0).A1
    top_feature_indices = np.argsort(tfidf_means)[::-1][:15]
    top_features = [recommender.feature_names[i] for i in top_feature_indices]
    top_scores = [tfidf_means[i] for i in top_feature_indices]
    plt.barh(range(len(top_features)), top_scores, alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Average TF-IDF Score')
    plt.title('Top TF-IDF Features', fontweight='bold')

    # Plot 3: Item similarity heatmap
    plt.subplot(3, 4, 3)
    sample_size = 30
    sns.heatmap(recommender.item_similarity[:sample_size, :sample_size],
                cmap='YlOrRd', square=True, cbar_kws={'label': 'Similarity'})
    plt.title('Item Similarity Matrix (Sample)', fontweight='bold')

    # Plot 4: Ranking metrics
    plt.subplot(3, 4, 4)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    # Plot 5: Diversity distribution
    plt.subplot(3, 4, 5)
    plt.hist(ranking_metrics['Diversity'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Diversity Score')
    plt.ylabel('Frequency')
    plt.title('Recommendation Diversity', fontweight='bold')

    # Plot 6: User interaction distribution
    plt.subplot(3, 4, 6)
    user_counts = interactions_df.groupby('user_id').size()
    plt.hist(user_counts, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Interactions per User')
    plt.ylabel('Number of Users')
    plt.title('User Activity Distribution', fontweight='bold')

    # Plot 7: Item popularity
    plt.subplot(3, 4, 7)
    item_counts = interactions_df.groupby('item_id').size()
    plt.hist(item_counts, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Interactions per Item')
    plt.ylabel('Number of Items')
    plt.title('Item Popularity Distribution', fontweight='bold')

    # Plot 8: Category preference
    plt.subplot(3, 4, 8)
    category_counts = interactions_df.merge(items_df, on='item_id')['category'].value_counts().sort_index()
    plt.bar(range(len(category_counts)), category_counts.values, alpha=0.8, color='orange')
    plt.xticks(range(len(category_counts)), category_counts.index)
    plt.xlabel('Category')
    plt.ylabel('Number of Interactions')
    plt.title('Category Popularity', fontweight='bold')

    # Plot 9: TF-IDF matrix sparsity
    plt.subplot(3, 4, 9)
    sparsity = 1 - (recommender.tfidf_matrix.nnz / (recommender.tfidf_matrix.shape[0] * recommender.tfidf_matrix.shape[1]))
    plt.bar(['Non-zero', 'Zero'],
            [1-sparsity, sparsity],
            alpha=0.8, color=['green', 'lightgray'])
    plt.ylabel('Proportion')
    plt.title(f'TF-IDF Matrix Sparsity: {sparsity:.2%}', fontweight='bold')

    # Plot 10: Similarity score distribution
    plt.subplot(3, 4, 10)
    sim_scores = recommender.item_similarity[np.triu_indices_from(recommender.item_similarity, k=1)]
    plt.hist(sim_scores, bins=50, edgecolor='black', alpha=0.7, color='teal')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Item Similarity Distribution', fontweight='bold')

    # Plot 11: Top features for sample item
    plt.subplot(3, 4, 11)
    top_features_item = recommender.get_top_features(sample_item, n_features=10)
    if top_features_item:
        features = [f for f, _ in top_features_item]
        scores = [s for _, s in top_features_item]
        plt.barh(range(len(features)), scores, alpha=0.8, color='mediumpurple')
        plt.yticks(range(len(features)), features)
        plt.xlabel('TF-IDF Score')
        plt.title(f'Top Features for Item {sample_item}', fontweight='bold')

    # Plot 12: Precision-Recall scatter
    plt.subplot(3, 4, 12)
    plt.scatter(ranking_metrics['Recall@10'], ranking_metrics['Precision@10'], alpha=0.6)
    plt.xlabel('Recall@10')
    plt.ylabel('Precision@10')
    plt.title('Precision-Recall Trade-off', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/17_tfidf_recommendations/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
