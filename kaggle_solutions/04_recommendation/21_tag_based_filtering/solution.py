"""
Tag-Based Filtering Recommendation System
=========================================

This solution implements tag-based collaborative filtering using item tags,
user tag preferences, and tag similarity for recommendations.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class TagBasedRecommender:
    """Tag-based recommender system."""

    def __init__(self):
        """Initialize tag-based recommender."""
        self.item_tags = {}
        self.tag_items = defaultdict(set)
        self.tag_idf = {}
        self.item_tag_matrix = None
        self.item_ids = None

    def fit(self, items_df, tag_column='tags'):
        """Fit the recommender with item tags."""
        self.item_ids = items_df['item_id'].values

        # Process tags
        for _, row in items_df.iterrows():
            item_id = row['item_id']
            tags = row[tag_column]

            if isinstance(tags, str):
                tags = tags.split(',')
            elif not isinstance(tags, list):
                tags = []

            tags = [tag.strip().lower() for tag in tags]
            self.item_tags[item_id] = set(tags)

            for tag in tags:
                self.tag_items[tag].add(item_id)

        # Calculate tag IDF
        n_items = len(self.item_tags)
        for tag, items in self.tag_items.items():
            self.tag_idf[tag] = np.log(n_items / len(items))

        # Build item-tag matrix
        all_tags = sorted(self.tag_items.keys())
        n_tags = len(all_tags)

        self.item_tag_matrix = np.zeros((len(self.item_ids), n_tags))

        for i, item_id in enumerate(self.item_ids):
            if item_id in self.item_tags:
                for tag in self.item_tags[item_id]:
                    if tag in all_tags:
                        tag_idx = all_tags.index(tag)
                        self.item_tag_matrix[i, tag_idx] = self.tag_idf.get(tag, 0)

        self.all_tags = all_tags
        print(f"   Processed {len(self.item_tags)} items with {len(all_tags)} unique tags")

    def get_similar_items(self, item_id, n_items=10):
        """Get similar items based on tag similarity."""
        if item_id not in self.item_ids:
            return []

        item_idx = np.where(self.item_ids == item_id)[0][0]
        item_vector = self.item_tag_matrix[item_idx].reshape(1, -1)

        # Calculate similarity
        similarities = cosine_similarity(item_vector, self.item_tag_matrix)[0]

        # Get top items
        top_indices = np.argsort(similarities)[::-1][1:n_items+1]

        return [(self.item_ids[i], similarities[i]) for i in top_indices]

    def recommend(self, user_tag_preferences, n_recommendations=10, user_history=None):
        """Generate recommendations based on user tag preferences."""
        if not user_tag_preferences:
            return []

        # Build user profile from tag preferences
        user_vector = np.zeros(len(self.all_tags))
        for tag, weight in user_tag_preferences.items():
            if tag in self.all_tags:
                tag_idx = self.all_tags.index(tag)
                user_vector[tag_idx] = weight * self.tag_idf.get(tag, 0)

        user_vector = user_vector.reshape(1, -1)

        # Calculate similarities
        scores = cosine_similarity(user_vector, self.item_tag_matrix)[0]

        # Exclude user history
        if user_history:
            for item_id in user_history:
                if item_id in self.item_ids:
                    item_idx = np.where(self.item_ids == item_id)[0][0]
                    scores[item_idx] = -1

        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]

        return [(self.item_ids[i], scores[i]) for i in top_indices]

    def get_item_tags(self, item_id):
        """Get tags for an item."""
        return list(self.item_tags.get(item_id, set()))


def generate_synthetic_data(n_items=300, n_users=500):
    """Generate synthetic item and tag data."""
    np.random.seed(42)

    # Define tag categories
    tag_categories = {
        'genre': ['action', 'comedy', 'drama', 'scifi', 'romance', 'thriller', 'horror'],
        'mood': ['exciting', 'relaxing', 'inspiring', 'funny', 'emotional', 'dark'],
        'setting': ['urban', 'rural', 'space', 'historical', 'modern', 'fantasy'],
        'theme': ['adventure', 'family', 'love', 'friendship', 'mystery', 'survival']
    }

    # Generate items
    items = []
    for i in range(n_items):
        # Select tags from each category
        tags = []
        for category, options in tag_categories.items():
            n_tags = np.random.randint(1, 3)
            tags.extend(np.random.choice(options, size=n_tags, replace=False))

        items.append({
            'item_id': i,
            'tags': ','.join(tags),
            'popularity': np.random.randint(1, 100)
        })

    items_df = pd.DataFrame(items)

    # Generate user interactions
    interactions = []
    for user_id in range(n_users):
        # Users have preferred tags
        preferred_genre = np.random.choice(tag_categories['genre'])
        preferred_mood = np.random.choice(tag_categories['mood'])

        for _ in range(10):
            # 70% chance to interact with items having preferred tags
            if np.random.rand() < 0.7:
                matching_items = items_df[
                    items_df['tags'].str.contains(preferred_genre) |
                    items_df['tags'].str.contains(preferred_mood)
                ]
                if len(matching_items) > 0:
                    item = matching_items.sample(1).iloc[0]
                else:
                    item = items_df.sample(1).iloc[0]
            else:
                item = items_df.sample(1).iloc[0]

            interactions.append({
                'user_id': user_id,
                'item_id': item['item_id'],
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
    print("Tag-Based Filtering Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic data...")
    items_df, interactions_df = generate_synthetic_data(n_items=300, n_users=500)
    print(f"   Items: {len(items_df)}")
    print(f"   Interactions: {len(interactions_df)}")

    # Train recommender
    print("\n2. Training tag-based recommender...")
    recommender = TagBasedRecommender()
    recommender.fit(items_df, tag_column='tags')

    # Analyze tag distribution
    all_tags = []
    for tags in items_df['tags']:
        all_tags.extend(tags.split(','))
    tag_freq = Counter([tag.strip() for tag in all_tags])
    print(f"\n   Top 10 most frequent tags:")
    for tag, count in tag_freq.most_common(10):
        print(f"      {tag}: {count}")

    # Split interactions
    train_int, test_int = train_test_split(interactions_df, test_size=0.2, random_state=42)

    # Build user tag preferences from training data
    print("\n3. Building user tag preferences...")
    user_tag_prefs = defaultdict(lambda: defaultdict(float))

    for _, row in train_int.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']

        tags = recommender.get_item_tags(item_id)
        for tag in tags:
            user_tag_prefs[user_id][tag] += rating / len(tags)

    # Evaluate
    print("\n4. Evaluating recommendations...")
    test_users = test_int['user_id'].unique()[:50]
    ranking_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}

    for user_id in test_users:
        if user_id not in user_tag_prefs:
            continue

        user_history = train_int[train_int['user_id'] == user_id]['item_id'].tolist()
        test_items = test_int[test_int['user_id'] == user_id]['item_id'].tolist()

        if not test_items:
            continue

        recommendations = recommender.recommend(
            dict(user_tag_prefs[user_id]),
            n_recommendations=10,
            user_history=user_history
        )

        prec, rec = calculate_precision_recall_at_k(recommendations, test_items, 10)
        ndcg = calculate_ndcg_at_k(recommendations, test_items, 10)

        ranking_metrics['Precision@10'].append(prec)
        ranking_metrics['Recall@10'].append(rec)
        ranking_metrics['NDCG@10'].append(ndcg)

    for metric, values in ranking_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n5. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Tag frequency
    plt.subplot(3, 4, 1)
    top_tags = dict(tag_freq.most_common(15))
    plt.barh(range(len(top_tags)), list(top_tags.values()), alpha=0.8, color='skyblue')
    plt.yticks(range(len(top_tags)), list(top_tags.keys()))
    plt.xlabel('Frequency')
    plt.title('Top 15 Tags', fontweight='bold')

    # Plot 2: Tags per item
    plt.subplot(3, 4, 2)
    tags_per_item = [len(tags.split(',')) for tags in items_df['tags']]
    plt.hist(tags_per_item, bins=15, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Number of Tags')
    plt.ylabel('Frequency')
    plt.title('Tags per Item Distribution', fontweight='bold')

    # Plot 3: Tag co-occurrence heatmap
    plt.subplot(3, 4, 3)
    top_10_tags = [tag for tag, _ in tag_freq.most_common(10)]
    cooccurrence = np.zeros((10, 10))

    for tags in items_df['tags']:
        tag_list = [t.strip() for t in tags.split(',')]
        for i, tag1 in enumerate(top_10_tags):
            for j, tag2 in enumerate(top_10_tags):
                if tag1 in tag_list and tag2 in tag_list:
                    cooccurrence[i, j] += 1

    sns.heatmap(cooccurrence, xticklabels=top_10_tags, yticklabels=top_10_tags,
                cmap='YlOrRd', square=True, cbar_kws={'label': 'Co-occurrence'})
    plt.title('Tag Co-occurrence Matrix', fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Plot 4: Ranking metrics
    plt.subplot(3, 4, 4)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names if ranking_metrics[m]]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    # Plots 5-12
    plt.subplot(3, 4, 5)
    item_similarity = cosine_similarity(recommender.item_tag_matrix)
    sns.heatmap(item_similarity[:40, :40], cmap='Greens', square=True,
                cbar_kws={'label': 'Similarity'})
    plt.title('Item Similarity (Sample)', fontweight='bold')

    plt.subplot(3, 4, 6)
    tag_idf_values = list(recommender.tag_idf.values())
    plt.hist(tag_idf_values, bins=20, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('IDF Value')
    plt.ylabel('Frequency')
    plt.title('Tag IDF Distribution', fontweight='bold')

    plt.subplot(3, 4, 7)
    user_counts = interactions_df.groupby('user_id').size()
    plt.hist(user_counts, bins=20, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Interactions per User')
    plt.ylabel('Frequency')
    plt.title('User Activity', fontweight='bold')

    plt.subplot(3, 4, 8)
    item_counts = interactions_df.groupby('item_id').size()
    plt.hist(item_counts, bins=30, edgecolor='black', alpha=0.7, color='teal')
    plt.xlabel('Interactions per Item')
    plt.ylabel('Frequency')
    plt.title('Item Popularity', fontweight='bold')

    plt.subplot(3, 4, 9)
    rating_dist = interactions_df['rating'].value_counts().sort_index()
    plt.bar(rating_dist.index, rating_dist.values, alpha=0.8, color='pink')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution', fontweight='bold')

    plt.subplot(3, 4, 10)
    sample_item = items_df['item_id'].iloc[0]
    similar_items = recommender.get_similar_items(sample_item, n_items=10)
    items = [item for item, _ in similar_items]
    sims = [sim for _, sim in similar_items]
    plt.barh(range(len(items)), sims, alpha=0.8, color='mediumpurple')
    plt.yticks(range(len(items)), [f'Item {i}' for i in items])
    plt.xlabel('Similarity')
    plt.title(f'Similar to Item {sample_item}', fontweight='bold')

    plt.subplot(3, 4, 11)
    sparsity = 1 - (np.count_nonzero(recommender.item_tag_matrix) /
                    (recommender.item_tag_matrix.shape[0] * recommender.item_tag_matrix.shape[1]))
    plt.bar(['Non-zero', 'Zero'], [1-sparsity, sparsity], alpha=0.8, color=['green', 'lightgray'])
    plt.ylabel('Proportion')
    plt.title(f'Item-Tag Matrix Sparsity: {sparsity:.2%}', fontweight='bold')

    plt.subplot(3, 4, 12)
    plt.scatter(ranking_metrics['Recall@10'], ranking_metrics['Precision@10'], alpha=0.6)
    plt.xlabel('Recall@10')
    plt.ylabel('Precision@10')
    plt.title('Precision-Recall Trade-off', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/21_tag_based_filtering/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
