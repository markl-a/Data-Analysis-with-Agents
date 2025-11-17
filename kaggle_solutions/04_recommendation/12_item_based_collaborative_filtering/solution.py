"""
Item-Based Collaborative Filtering Recommendation System
========================================================

This solution implements comprehensive item-based collaborative filtering techniques,
including multiple similarity metrics, adjusted cosine similarity, and extensive
evaluation metrics for recommendation quality.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine, correlation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering Recommender.

    This class implements item-based collaborative filtering with various
    similarity metrics and prediction strategies.
    """

    def __init__(self, similarity_metric='adjusted_cosine', k_neighbors=20):
        """
        Initialize the Item-Based CF model.

        Parameters:
        -----------
        similarity_metric : str
            Similarity metric ('cosine', 'adjusted_cosine', 'pearson')
        k_neighbors : int
            Number of similar items to consider
        """
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.ratings_matrix = None
        self.user_means = None
        self.item_means = None

    def fit(self, ratings_df):
        """
        Fit the model on training data.

        Parameters:
        -----------
        ratings_df : pd.DataFrame
            DataFrame with columns: user_id, item_id, rating
        """
        # Create user-item matrix
        self.ratings_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # Calculate means
        self.user_means = self.ratings_matrix.apply(
            lambda row: row[row > 0].mean() if (row > 0).any() else 0,
            axis=1
        )

        self.item_means = self.ratings_matrix.apply(
            lambda col: col[col > 0].mean() if (col > 0).any() else 0,
            axis=0
        )

        # Calculate item similarity matrix
        self.item_similarity = self._calculate_similarity()

    def _calculate_similarity(self):
        """Calculate item-item similarity matrix."""
        n_items = self.ratings_matrix.shape[1]
        similarity = np.zeros((n_items, n_items))

        for i in range(n_items):
            for j in range(i, n_items):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    item1 = self.ratings_matrix.iloc[:, i].values
                    item2 = self.ratings_matrix.iloc[:, j].values
                    sim = self._compute_similarity(item1, item2)
                    similarity[i, j] = sim
                    similarity[j, i] = sim

        return similarity

    def _compute_similarity(self, item1, item2):
        """Compute similarity between two items."""
        # Find common users who rated both items
        mask = (item1 > 0) & (item2 > 0)

        if not mask.any():
            return 0.0

        if self.similarity_metric == 'cosine':
            denominator = np.linalg.norm(item1[mask]) * np.linalg.norm(item2[mask])
            if denominator == 0:
                return 0.0
            return np.dot(item1[mask], item2[mask]) / denominator

        elif self.similarity_metric == 'adjusted_cosine':
            # Adjust ratings by user means
            adjusted1 = item1.copy()
            adjusted2 = item2.copy()

            for idx in np.where(mask)[0]:
                user_id = self.ratings_matrix.index[idx]
                adjusted1[idx] = item1[idx] - self.user_means[user_id]
                adjusted2[idx] = item2[idx] - self.user_means[user_id]

            denominator = np.linalg.norm(adjusted1[mask]) * np.linalg.norm(adjusted2[mask])
            if denominator == 0:
                return 0.0
            return np.dot(adjusted1[mask], adjusted2[mask]) / denominator

        elif self.similarity_metric == 'pearson':
            mean1 = item1[mask].mean()
            mean2 = item2[mask].mean()
            centered1 = item1[mask] - mean1
            centered2 = item2[mask] - mean2
            denominator = np.linalg.norm(centered1) * np.linalg.norm(centered2)
            if denominator == 0:
                return 0.0
            return np.dot(centered1, centered2) / denominator

        return 0.0

    def predict(self, user_id, item_id, weighted=True):
        """
        Predict rating for a user-item pair.

        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
        weighted : bool
            Whether to use weighted average
        """
        if user_id not in self.ratings_matrix.index:
            return self.item_means.mean()

        if item_id not in self.ratings_matrix.columns:
            return self.user_means[user_id]

        item_idx = self.ratings_matrix.columns.get_loc(item_id)

        # Get items rated by the user
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            return self.item_means[item_id]

        # Get similarities with items rated by user
        similarities = []
        for other_item in rated_items:
            if other_item != item_id:
                other_idx = self.ratings_matrix.columns.get_loc(other_item)
                sim = self.item_similarity[item_idx, other_idx]
                if sim > 0:
                    similarities.append((other_item, sim))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k_neighbors]

        if not top_k:
            return self.item_means[item_id]

        # Calculate prediction
        if weighted:
            weighted_sum = sum(
                sim * self.ratings_matrix.loc[user_id, other_item]
                for other_item, sim in top_k
            )
            sim_sum = sum(sim for _, sim in top_k)

            if sim_sum == 0:
                return self.item_means[item_id]
            return weighted_sum / sim_sum
        else:
            return np.mean([
                self.ratings_matrix.loc[user_id, other_item]
                for other_item, _ in top_k
            ])

    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate top-N recommendations for a user.

        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations
        exclude_rated : bool
            Whether to exclude rated items
        """
        if user_id not in self.ratings_matrix.index:
            return []

        predictions = []
        for item_id in self.ratings_matrix.columns:
            if exclude_rated and self.ratings_matrix.loc[user_id, item_id] > 0:
                continue
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def get_similar_items(self, item_id, n_items=10):
        """Get most similar items to a given item."""
        if item_id not in self.ratings_matrix.columns:
            return []

        item_idx = self.ratings_matrix.columns.get_loc(item_id)
        similarities = []

        for other_idx, other_item in enumerate(self.ratings_matrix.columns):
            if other_item != item_id:
                sim = self.item_similarity[item_idx, other_idx]
                similarities.append((other_item, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_items]


def generate_synthetic_data(n_users=500, n_items=300, n_ratings=15000,
                           sparsity=0.9, rating_scale=(1, 5)):
    """Generate synthetic user-item rating data."""
    np.random.seed(42)

    # Generate user and item features
    user_features = np.random.randn(n_users, 10)
    item_features = np.random.randn(n_items, 10)

    # Generate ratings
    ratings = []
    n_ratings = int(n_users * n_items * (1 - sparsity))

    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)

        # Base rating from latent features
        base_rating = np.dot(user_features[user_id], item_features[item_id])
        rating = np.clip(
            3 + base_rating / 2 + np.random.normal(0, 0.5),
            rating_scale[0], rating_scale[1]
        )

        ratings.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': round(rating, 1)
        })

    df = pd.DataFrame(ratings)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    return df


def calculate_precision_recall_at_k(predictions, actuals, k=10):
    """Calculate Precision@K and Recall@K."""
    if len(predictions) == 0 or len(actuals) == 0:
        return 0.0, 0.0

    top_k = set([item for item, _ in predictions[:k]])
    relevant = set(actuals)

    hits = len(top_k & relevant)
    precision = hits / len(top_k) if len(top_k) > 0 else 0.0
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0

    return precision, recall


def calculate_ndcg_at_k(predictions, actuals, k=10):
    """Calculate Normalized Discounted Cumulative Gain at K."""
    if len(predictions) == 0 or len(actuals) == 0:
        return 0.0

    dcg = 0.0
    for i, (item, score) in enumerate(predictions[:k]):
        if item in actuals:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(actuals))))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_map(predictions, actuals, k=10):
    """Calculate Mean Average Precision."""
    if len(predictions) == 0 or len(actuals) == 0:
        return 0.0

    relevant = set(actuals)
    hits = 0
    sum_precisions = 0.0

    for i, (item, _) in enumerate(predictions[:k]):
        if item in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)

    return sum_precisions / min(k, len(relevant)) if len(relevant) > 0 else 0.0


def calculate_coverage(all_recommendations, total_items):
    """Calculate catalog coverage."""
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update([item for item, _ in recs])
    return len(recommended_items) / total_items


def calculate_diversity(recommendations):
    """Calculate intra-list diversity."""
    if len(recommendations) < 2:
        return 0.0

    items = [item for item, _ in recommendations]
    unique_items = len(set(items))
    return unique_items / len(items)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Item-Based Collaborative Filtering Recommendation System")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic rating data...")
    ratings_df = generate_synthetic_data(
        n_users=500, n_items=300, n_ratings=15000, sparsity=0.9
    )
    print(f"   Generated {len(ratings_df)} ratings")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")
    print(f"   Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train set: {len(train_df)} ratings")
    print(f"   Test set: {len(test_df)} ratings")

    # Train models with different similarity metrics
    print("\n3. Training Item-Based CF models...")
    models = {}
    metrics_results = {}

    for similarity in ['cosine', 'adjusted_cosine', 'pearson']:
        print(f"\n   Training with {similarity} similarity...")
        model = ItemBasedCF(similarity_metric=similarity, k_neighbors=20)
        model.fit(train_df)
        models[similarity] = model

        # Evaluate
        predictions = []
        actuals = []

        for _, row in test_df.head(1000).iterrows():
            pred = model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        metrics_results[similarity] = {'RMSE': rmse, 'MAE': mae}
        print(f"   {similarity} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Detailed evaluation
    print("\n4. Evaluating ranking metrics...")
    best_model = models['adjusted_cosine']

    ranking_metrics = {
        'Precision@5': [], 'Recall@5': [], 'NDCG@5': [], 'MAP@5': [],
        'Precision@10': [], 'Recall@10': [], 'NDCG@10': [], 'MAP@10': []
    }

    test_users = test_df['user_id'].unique()[:50]
    all_recs = []

    for user_id in test_users:
        user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()

        if len(user_test_items) == 0:
            continue

        recommendations = best_model.recommend(user_id, n_recommendations=10)
        all_recs.append(recommendations)

        for k in [5, 10]:
            prec, rec = calculate_precision_recall_at_k(recommendations, user_test_items, k)
            ndcg = calculate_ndcg_at_k(recommendations, user_test_items, k)
            map_score = calculate_map(recommendations, user_test_items, k)

            ranking_metrics[f'Precision@{k}'].append(prec)
            ranking_metrics[f'Recall@{k}'].append(rec)
            ranking_metrics[f'NDCG@{k}'].append(ndcg)
            ranking_metrics[f'MAP@{k}'].append(map_score)

    for metric, values in ranking_metrics.items():
        if len(values) > 0:
            print(f"   {metric}: {np.mean(values):.4f}")

    # Calculate coverage
    coverage = calculate_coverage(all_recs, ratings_df['item_id'].nunique())
    print(f"   Catalog Coverage: {coverage:.4f}")

    # Diversity analysis
    diversity_scores = [calculate_diversity(recs) for recs in all_recs]
    print(f"   Average Diversity: {np.mean(diversity_scores):.4f}")

    # Visualization
    print("\n5. Creating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Rating distribution
    plt.subplot(3, 4, 1)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.title('Rating Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Plot 2: Model comparison
    plt.subplot(3, 4, 2)
    models_list = list(metrics_results.keys())
    rmse_values = [metrics_results[m]['RMSE'] for m in models_list]
    mae_values = [metrics_results[m]['MAE'] for m in models_list]

    x = np.arange(len(models_list))
    width = 0.35
    plt.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    plt.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.xlabel('Similarity Metric')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, models_list, rotation=15)
    plt.legend()

    # Plot 3: Item similarity heatmap
    plt.subplot(3, 4, 3)
    sample_size = 30
    sim_sample = best_model.item_similarity[:sample_size, :sample_size]
    sns.heatmap(sim_sample, cmap='YlOrRd', center=0, square=True, cbar_kws={'label': 'Similarity'})
    plt.title('Item Similarity Matrix (Sample)', fontsize=12, fontweight='bold')

    # Plot 4: User activity
    plt.subplot(3, 4, 4)
    user_counts = ratings_df.groupby('user_id').size()
    plt.hist(user_counts, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('User Activity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Ratings per User')
    plt.ylabel('Number of Users')

    # Plot 5: Item popularity
    plt.subplot(3, 4, 5)
    item_counts = ratings_df.groupby('item_id').size()
    plt.hist(item_counts, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    plt.title('Item Popularity Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Ratings per Item')
    plt.ylabel('Number of Items')

    # Plot 6: Ranking metrics
    plt.subplot(3, 4, 6)
    metrics_to_plot = ['Precision@10', 'Recall@10', 'NDCG@10', 'MAP@10']
    metric_means = [np.mean(ranking_metrics[m]) for m in metrics_to_plot]
    plt.bar(range(len(metrics_to_plot)), metric_means, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45)
    plt.ylabel('Score')
    plt.title('Ranking Metrics @10', fontsize=12, fontweight='bold')

    # Plot 7: Precision-Recall trade-off
    plt.subplot(3, 4, 7)
    plt.scatter(ranking_metrics['Recall@10'], ranking_metrics['Precision@10'], alpha=0.6)
    plt.xlabel('Recall@10')
    plt.ylabel('Precision@10')
    plt.title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 8: Diversity scores
    plt.subplot(3, 4, 8)
    plt.hist(diversity_scores, bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
    plt.title('Recommendation Diversity', fontsize=12, fontweight='bold')
    plt.xlabel('Diversity Score')
    plt.ylabel('Frequency')

    # Plot 9: Rating density by item
    plt.subplot(3, 4, 9)
    item_avg_rating = ratings_df.groupby('item_id')['rating'].mean()
    plt.scatter(item_counts, item_avg_rating, alpha=0.5)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Average Rating')
    plt.title('Item Popularity vs Rating', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 10: Prediction error distribution
    plt.subplot(3, 4, 10)
    errors = np.array(predictions) - np.array(actuals[:len(predictions)])
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # Plot 11: Top similar items network (for a sample item)
    plt.subplot(3, 4, 11)
    sample_item = ratings_df['item_id'].value_counts().head(1).index[0]
    similar_items = best_model.get_similar_items(sample_item, n_items=10)
    items = [item for item, _ in similar_items]
    sims = [sim for _, sim in similar_items]
    plt.barh(range(len(items)), sims, alpha=0.8, color='teal')
    plt.yticks(range(len(items)), [f'Item {item}' for item in items])
    plt.xlabel('Similarity Score')
    plt.title(f'Top Similar Items to Item {sample_item}', fontsize=12, fontweight='bold')

    # Plot 12: Cumulative metrics
    plt.subplot(3, 4, 12)
    k_values = [1, 3, 5, 7, 10]
    ndcg_at_k = []
    for k in k_values:
        ndcg_scores = []
        for user_id in test_users[:30]:
            user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
            if len(user_test_items) > 0:
                recs = best_model.recommend(user_id, n_recommendations=k)
                ndcg = calculate_ndcg_at_k(recs, user_test_items, k)
                ndcg_scores.append(ndcg)
        ndcg_at_k.append(np.mean(ndcg_scores) if ndcg_scores else 0)

    plt.plot(k_values, ndcg_at_k, marker='o', linewidth=2, markersize=8)
    plt.xlabel('K')
    plt.ylabel('NDCG@K')
    plt.title('NDCG at Different K Values', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/12_item_based_collaborative_filtering/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization to analysis_plots.png")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
