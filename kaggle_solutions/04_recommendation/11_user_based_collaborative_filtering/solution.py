"""
User-Based Collaborative Filtering Recommendation System
=======================================================

This solution implements comprehensive user-based collaborative filtering techniques
for recommendation systems, including multiple similarity metrics, prediction algorithms,
and extensive evaluation metrics.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class UserBasedCF:
    """
    User-Based Collaborative Filtering Recommender.

    This class implements user-based collaborative filtering using various
    similarity metrics and prediction methods.
    """

    def __init__(self, similarity_metric='cosine', k_neighbors=20):
        """
        Initialize the User-Based CF model.

        Parameters:
        -----------
        similarity_metric : str
            Similarity metric to use ('cosine', 'pearson', 'jaccard')
        k_neighbors : int
            Number of similar users to consider
        """
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.user_similarity = None
        self.ratings_matrix = None
        self.user_means = None

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

        # Calculate user means (excluding zeros)
        self.user_means = self.ratings_matrix.apply(
            lambda row: row[row > 0].mean() if (row > 0).any() else 0,
            axis=1
        )

        # Calculate user similarity matrix
        self.user_similarity = self._calculate_similarity()

    def _calculate_similarity(self):
        """Calculate user-user similarity matrix."""
        n_users = self.ratings_matrix.shape[0]
        similarity = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(i, n_users):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    sim = self._compute_similarity(
                        self.ratings_matrix.iloc[i].values,
                        self.ratings_matrix.iloc[j].values
                    )
                    similarity[i, j] = sim
                    similarity[j, i] = sim

        return similarity

    def _compute_similarity(self, user1, user2):
        """Compute similarity between two users."""
        # Find common rated items
        mask = (user1 > 0) & (user2 > 0)

        if not mask.any():
            return 0.0

        if self.similarity_metric == 'cosine':
            denominator = np.linalg.norm(user1[mask]) * np.linalg.norm(user2[mask])
            if denominator == 0:
                return 0.0
            return np.dot(user1[mask], user2[mask]) / denominator

        elif self.similarity_metric == 'pearson':
            mean1 = user1[mask].mean()
            mean2 = user2[mask].mean()
            centered1 = user1[mask] - mean1
            centered2 = user2[mask] - mean2
            denominator = np.linalg.norm(centered1) * np.linalg.norm(centered2)
            if denominator == 0:
                return 0.0
            return np.dot(centered1, centered2) / denominator

        elif self.similarity_metric == 'jaccard':
            intersection = np.sum((user1 > 0) & (user2 > 0))
            union = np.sum((user1 > 0) | (user2 > 0))
            return intersection / union if union > 0 else 0.0

        return 0.0

    def predict(self, user_id, item_id, use_mean_centering=True):
        """
        Predict rating for a user-item pair.

        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
        use_mean_centering : bool
            Whether to use mean-centered predictions
        """
        if user_id not in self.ratings_matrix.index:
            return self.user_means.mean()

        if item_id not in self.ratings_matrix.columns:
            return self.user_means[user_id]

        user_idx = self.ratings_matrix.index.get_loc(user_id)

        # Get k most similar users who have rated the item
        item_ratings = self.ratings_matrix[item_id]
        rated_users = item_ratings[item_ratings > 0].index

        if len(rated_users) == 0:
            return self.user_means[user_id]

        # Get similarities with users who rated the item
        similarities = []
        for other_user in rated_users:
            if other_user != user_id:
                other_idx = self.ratings_matrix.index.get_loc(other_user)
                sim = self.user_similarity[user_idx, other_idx]
                similarities.append((other_user, sim))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.k_neighbors]

        if not top_k:
            return self.user_means[user_id]

        # Calculate weighted average
        if use_mean_centering:
            weighted_sum = 0
            sim_sum = 0
            for other_user, sim in top_k:
                if sim > 0:
                    rating_deviation = (
                        self.ratings_matrix.loc[other_user, item_id] -
                        self.user_means[other_user]
                    )
                    weighted_sum += sim * rating_deviation
                    sim_sum += abs(sim)

            if sim_sum == 0:
                return self.user_means[user_id]
            return self.user_means[user_id] + (weighted_sum / sim_sum)
        else:
            weighted_sum = sum(
                sim * self.ratings_matrix.loc[other_user, item_id]
                for other_user, sim in top_k if sim > 0
            )
            sim_sum = sum(abs(sim) for _, sim in top_k if sim > 0)

            if sim_sum == 0:
                return self.user_means[user_id]
            return weighted_sum / sim_sum

    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Generate top-N recommendations for a user.

        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
        exclude_rated : bool
            Whether to exclude already rated items
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


def generate_synthetic_data(n_users=500, n_items=300, n_ratings=15000,
                           sparsity=0.9, rating_scale=(1, 5)):
    """
    Generate synthetic user-item rating data.

    Parameters:
    -----------
    n_users : int
        Number of users
    n_items : int
        Number of items
    n_ratings : int
        Number of ratings to generate
    sparsity : float
        Sparsity of the matrix (0-1)
    rating_scale : tuple
        Min and max rating values
    """
    np.random.seed(42)

    # Generate user and item features
    user_features = np.random.randn(n_users, 10)
    item_features = np.random.randn(n_items, 10)

    # Generate ratings based on latent features
    ratings = []
    n_ratings = int(n_users * n_items * (1 - sparsity))

    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)

        # Base rating from latent features
        base_rating = np.dot(user_features[user_id], item_features[item_id])
        # Normalize to rating scale
        rating = np.clip(
            (base_rating - base_rating.min()) / (base_rating.max() - base_rating.min()) *
            (rating_scale[1] - rating_scale[0]) + rating_scale[0],
            rating_scale[0], rating_scale[1]
        )
        # Add noise
        rating = np.clip(
            rating + np.random.normal(0, 0.5),
            rating_scale[0], rating_scale[1]
        )

        ratings.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': round(rating, 1)
        })

    df = pd.DataFrame(ratings)
    # Remove duplicates, keeping last
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')

    return df


def calculate_precision_recall_at_k(predictions, actuals, k=10):
    """Calculate Precision@K and Recall@K."""
    if len(predictions) == 0 or len(actuals) == 0:
        return 0.0, 0.0

    top_k_predictions = set([item for item, _ in predictions[:k]])
    relevant_items = set(actuals)

    if len(top_k_predictions) == 0:
        return 0.0, 0.0

    hits = len(top_k_predictions & relevant_items)
    precision = hits / len(top_k_predictions)
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

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

    relevant_items = set(actuals)
    hits = 0
    sum_precisions = 0.0

    for i, (item, score) in enumerate(predictions[:k]):
        if item in relevant_items:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    return sum_precisions / min(k, len(relevant_items)) if len(relevant_items) > 0 else 0.0


def calculate_diversity(recommendations, item_features=None):
    """Calculate diversity of recommendations."""
    if len(recommendations) < 2:
        return 0.0

    items = [item for item, _ in recommendations]

    # If no features, use item IDs as simple diversity measure
    if item_features is None:
        return len(set(items)) / len(items)

    # Calculate pairwise diversity
    n = len(items)
    total_distance = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if items[i] in item_features and items[j] in item_features:
                distance = cosine(item_features[items[i]], item_features[items[j]])
                total_distance += distance
                count += 1

    return total_distance / count if count > 0 else 0.0


def main():
    """Main execution function."""
    print("=" * 80)
    print("User-Based Collaborative Filtering Recommendation System")
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
    print(f"   Sparsity: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()):.2%}")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"   Train set: {len(train_df)} ratings")
    print(f"   Test set: {len(test_df)} ratings")

    # Train models with different similarity metrics
    print("\n3. Training User-Based CF models...")
    models = {}
    metrics_results = {}

    for similarity in ['cosine', 'pearson', 'jaccard']:
        print(f"\n   Training with {similarity} similarity...")
        model = UserBasedCF(similarity_metric=similarity, k_neighbors=20)
        model.fit(train_df)
        models[similarity] = model

        # Evaluate on test set
        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            pred = model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        metrics_results[similarity] = {'RMSE': rmse, 'MAE': mae}
        print(f"   {similarity} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Evaluate ranking metrics
    print("\n4. Evaluating ranking metrics...")
    best_model = models['cosine']

    ranking_metrics = {
        'Precision@5': [], 'Recall@5': [], 'NDCG@5': [], 'MAP@5': [],
        'Precision@10': [], 'Recall@10': [], 'NDCG@10': [], 'MAP@10': []
    }

    # Sample users for evaluation
    test_users = test_df['user_id'].unique()[:50]

    for user_id in test_users:
        user_test_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()

        if len(user_test_items) == 0:
            continue

        recommendations = best_model.recommend(user_id, n_recommendations=10)

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

    # Visualization
    print("\n5. Creating visualizations...")

    # Plot 1: Rating distribution
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    ratings_df['rating'].hist(bins=20, edgecolor='black')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Plot 2: Model comparison
    plt.subplot(2, 3, 2)
    models_list = list(metrics_results.keys())
    rmse_values = [metrics_results[m]['RMSE'] for m in models_list]
    mae_values = [metrics_results[m]['MAE'] for m in models_list]

    x = np.arange(len(models_list))
    width = 0.35
    plt.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    plt.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.xlabel('Similarity Metric')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models_list)
    plt.legend()

    # Plot 3: User activity distribution
    plt.subplot(2, 3, 3)
    user_counts = ratings_df.groupby('user_id').size()
    plt.hist(user_counts, bins=30, edgecolor='black', alpha=0.7)
    plt.title('User Activity Distribution')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')

    # Plot 4: Item popularity
    plt.subplot(2, 3, 4)
    item_counts = ratings_df.groupby('item_id').size()
    plt.hist(item_counts, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.title('Item Popularity Distribution')
    plt.xlabel('Number of Ratings per Item')
    plt.ylabel('Number of Items')

    # Plot 5: Ranking metrics comparison
    plt.subplot(2, 3, 5)
    metrics_to_plot = ['Precision@10', 'Recall@10', 'NDCG@10', 'MAP@10']
    metric_means = [np.mean(ranking_metrics[m]) for m in metrics_to_plot]
    plt.bar(range(len(metrics_to_plot)), metric_means, alpha=0.8, color='green')
    plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45)
    plt.ylabel('Score')
    plt.title('Ranking Metrics Performance')

    # Plot 6: Similarity matrix heatmap (sample)
    plt.subplot(2, 3, 6)
    sample_size = 30
    sim_sample = best_model.user_similarity[:sample_size, :sample_size]
    sns.heatmap(sim_sample, cmap='coolwarm', center=0, square=True,
                cbar_kws={'label': 'Similarity'})
    plt.title('User Similarity Matrix (Sample)')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/11_user_based_collaborative_filtering/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization to analysis_plots.png")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
