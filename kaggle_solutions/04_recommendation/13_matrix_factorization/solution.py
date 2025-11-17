"""
Matrix Factorization for Recommendations (SVD & NMF)
====================================================

This solution implements comprehensive matrix factorization techniques including
Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF)
for building recommendation systems.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SVDRecommender:
    """
    SVD-based recommender system using matrix factorization.
    """

    def __init__(self, n_factors=50, random_state=42):
        """
        Initialize SVD recommender.

        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        random_state : int
            Random seed
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.U = None
        self.sigma = None
        self.Vt = None
        self.ratings_matrix = None
        self.user_mean = None
        self.predictions_matrix = None

    def fit(self, ratings_df):
        """Fit SVD model on training data."""
        # Create user-item matrix
        self.ratings_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # Calculate user means
        self.user_mean = self.ratings_matrix.apply(
            lambda row: row[row > 0].mean() if (row > 0).any() else 0,
            axis=1
        )

        # Mean-center the matrix
        ratings_mean_centered = self.ratings_matrix.sub(self.user_mean, axis=0)
        ratings_mean_centered = ratings_mean_centered.fillna(0)

        # Perform SVD
        n_factors = min(self.n_factors, min(self.ratings_matrix.shape) - 1)
        self.U, self.sigma, self.Vt = svds(ratings_mean_centered.values, k=n_factors)

        # Reconstruct predictions
        self.predictions_matrix = np.dot(
            np.dot(self.U, np.diag(self.sigma)),
            self.Vt
        ) + self.user_mean.values.reshape(-1, 1)

        self.predictions_df = pd.DataFrame(
            self.predictions_matrix,
            index=self.ratings_matrix.index,
            columns=self.ratings_matrix.columns
        )

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.predictions_df.index:
            return self.user_mean.mean()
        if item_id not in self.predictions_df.columns:
            return self.user_mean[user_id]

        return self.predictions_df.loc[user_id, item_id]

    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """Generate top-N recommendations."""
        if user_id not in self.predictions_df.index:
            return []

        user_predictions = self.predictions_df.loc[user_id]

        if exclude_rated:
            rated_items = self.ratings_matrix.loc[user_id]
            user_predictions = user_predictions[rated_items == 0]

        top_items = user_predictions.nlargest(n_recommendations)
        return [(item_id, score) for item_id, score in top_items.items()]


class NMFRecommender:
    """
    NMF-based recommender system.
    """

    def __init__(self, n_components=50, random_state=42, max_iter=200):
        """
        Initialize NMF recommender.

        Parameters:
        -----------
        n_components : int
            Number of latent components
        random_state : int
            Random seed
        max_iter : int
            Maximum iterations
        """
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.W = None
        self.H = None
        self.ratings_matrix = None
        self.predictions_df = None

    def fit(self, ratings_df):
        """Fit NMF model on training data."""
        # Create user-item matrix
        self.ratings_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        # Fit NMF
        self.model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            init='random'
        )

        self.W = self.model.fit_transform(self.ratings_matrix.values)
        self.H = self.model.components_

        # Generate predictions
        predictions = np.dot(self.W, self.H)
        self.predictions_df = pd.DataFrame(
            predictions,
            index=self.ratings_matrix.index,
            columns=self.ratings_matrix.columns
        )

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.predictions_df.index:
            return self.ratings_matrix.values.mean()
        if item_id not in self.predictions_df.columns:
            return self.ratings_matrix.loc[user_id].mean()

        return self.predictions_df.loc[user_id, item_id]

    def recommend(self, user_id, n_recommendations=10, exclude_rated=True):
        """Generate top-N recommendations."""
        if user_id not in self.predictions_df.index:
            return []

        user_predictions = self.predictions_df.loc[user_id]

        if exclude_rated:
            rated_items = self.ratings_matrix.loc[user_id]
            user_predictions = user_predictions[rated_items == 0]

        top_items = user_predictions.nlargest(n_recommendations)
        return [(item_id, score) for item_id, score in top_items.items()]


def generate_synthetic_data(n_users=500, n_items=300, sparsity=0.9, n_factors=10):
    """Generate synthetic rating data with latent factors."""
    np.random.seed(42)

    # Generate latent factors
    user_factors = np.random.rand(n_users, n_factors)
    item_factors = np.random.rand(n_items, n_factors)

    # Generate ratings matrix
    ratings_matrix = np.dot(user_factors, item_factors.T)

    # Scale to 1-5 range
    ratings_matrix = 1 + 4 * (ratings_matrix - ratings_matrix.min()) / (
        ratings_matrix.max() - ratings_matrix.min()
    )

    # Add noise
    ratings_matrix += np.random.normal(0, 0.3, ratings_matrix.shape)
    ratings_matrix = np.clip(ratings_matrix, 1, 5)

    # Create sparse matrix
    mask = np.random.rand(n_users, n_items) > sparsity
    ratings_list = []

    for i in range(n_users):
        for j in range(n_items):
            if mask[i, j]:
                ratings_list.append({
                    'user_id': i,
                    'item_id': j,
                    'rating': round(ratings_matrix[i, j], 1)
                })

    return pd.DataFrame(ratings_list)


def calculate_metrics(predictions, actuals):
    """Calculate RMSE and MAE."""
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return rmse, mae


def calculate_precision_recall_at_k(recommendations, relevant_items, k=10):
    """Calculate Precision@K and Recall@K."""
    if len(recommendations) == 0 or len(relevant_items) == 0:
        return 0.0, 0.0

    top_k = set([item for item, _ in recommendations[:k]])
    relevant = set(relevant_items)

    hits = len(top_k & relevant)
    precision = hits / len(top_k)
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0

    return precision, recall


def calculate_ndcg_at_k(recommendations, relevant_items, k=10):
    """Calculate NDCG@K."""
    if len(recommendations) == 0 or len(relevant_items) == 0:
        return 0.0

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, (item, _) in enumerate(recommendations[:k])
        if item in relevant_items
    )

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))

    return dcg / idcg if idcg > 0 else 0.0


def main():
    """Main execution function."""
    print("=" * 80)
    print("Matrix Factorization Recommendation System (SVD & NMF)")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic rating data...")
    ratings_df = generate_synthetic_data(n_users=500, n_items=300, sparsity=0.9)
    print(f"   Generated {len(ratings_df)} ratings")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")
    print(f"   Sparsity: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()):.2%}")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Train SVD model
    print("\n3. Training SVD models with different factors...")
    svd_results = {}

    for n_factors in [10, 20, 50, 100]:
        print(f"\n   Training SVD with {n_factors} factors...")
        svd_model = SVDRecommender(n_factors=n_factors)
        svd_model.fit(train_df)

        # Evaluate
        predictions = []
        actuals = []
        for _, row in test_df.head(1000).iterrows():
            pred = svd_model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse, mae = calculate_metrics(predictions, actuals)
        svd_results[n_factors] = {'RMSE': rmse, 'MAE': mae, 'model': svd_model}
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Train NMF model
    print("\n4. Training NMF models...")
    nmf_results = {}

    for n_components in [10, 20, 50, 100]:
        print(f"\n   Training NMF with {n_components} components...")
        nmf_model = NMFRecommender(n_components=n_components)
        nmf_model.fit(train_df)

        predictions = []
        actuals = []
        for _, row in test_df.head(1000).iterrows():
            pred = nmf_model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse, mae = calculate_metrics(predictions, actuals)
        nmf_results[n_components] = {'RMSE': rmse, 'MAE': mae, 'model': nmf_model}
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Ranking evaluation
    print("\n5. Evaluating ranking metrics...")
    best_svd = svd_results[50]['model']
    best_nmf = nmf_results[50]['model']

    test_users = test_df['user_id'].unique()[:50]

    svd_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}
    nmf_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}

    for user_id in test_users:
        relevant_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
        if not relevant_items:
            continue

        # SVD recommendations
        svd_recs = best_svd.recommend(user_id, n_recommendations=10)
        prec, rec = calculate_precision_recall_at_k(svd_recs, relevant_items, 10)
        ndcg = calculate_ndcg_at_k(svd_recs, relevant_items, 10)
        svd_metrics['Precision@10'].append(prec)
        svd_metrics['Recall@10'].append(rec)
        svd_metrics['NDCG@10'].append(ndcg)

        # NMF recommendations
        nmf_recs = best_nmf.recommend(user_id, n_recommendations=10)
        prec, rec = calculate_precision_recall_at_k(nmf_recs, relevant_items, 10)
        ndcg = calculate_ndcg_at_k(nmf_recs, relevant_items, 10)
        nmf_metrics['Precision@10'].append(prec)
        nmf_metrics['Recall@10'].append(rec)
        nmf_metrics['NDCG@10'].append(ndcg)

    print("\n   SVD Metrics:")
    for metric, values in svd_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    print("\n   NMF Metrics:")
    for metric, values in nmf_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n6. Creating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: SVD performance vs factors
    plt.subplot(3, 4, 1)
    factors = list(svd_results.keys())
    rmse_vals = [svd_results[f]['RMSE'] for f in factors]
    mae_vals = [svd_results[f]['MAE'] for f in factors]
    plt.plot(factors, rmse_vals, marker='o', label='RMSE', linewidth=2)
    plt.plot(factors, mae_vals, marker='s', label='MAE', linewidth=2)
    plt.xlabel('Number of Factors')
    plt.ylabel('Error')
    plt.title('SVD: Performance vs Factors', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: NMF performance vs components
    plt.subplot(3, 4, 2)
    components = list(nmf_results.keys())
    rmse_vals = [nmf_results[c]['RMSE'] for c in components]
    mae_vals = [nmf_results[c]['MAE'] for c in components]
    plt.plot(components, rmse_vals, marker='o', label='RMSE', linewidth=2)
    plt.plot(components, mae_vals, marker='s', label='MAE', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Error')
    plt.title('NMF: Performance vs Components', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: SVD vs NMF comparison
    plt.subplot(3, 4, 3)
    metrics_names = ['RMSE', 'MAE']
    svd_vals = [svd_results[50]['RMSE'], svd_results[50]['MAE']]
    nmf_vals = [nmf_results[50]['RMSE'], nmf_results[50]['MAE']]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, svd_vals, width, label='SVD', alpha=0.8)
    plt.bar(x + width/2, nmf_vals, width, label='NMF', alpha=0.8)
    plt.ylabel('Error')
    plt.title('SVD vs NMF Comparison (50 factors)', fontweight='bold')
    plt.xticks(x, metrics_names)
    plt.legend()

    # Plot 4: Ranking metrics comparison
    plt.subplot(3, 4, 4)
    ranking_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    svd_ranking = [np.mean(svd_metrics[m]) for m in ranking_names]
    nmf_ranking = [np.mean(nmf_metrics[m]) for m in ranking_names]
    x = np.arange(len(ranking_names))
    plt.bar(x - width/2, svd_ranking, width, label='SVD', alpha=0.8)
    plt.bar(x + width/2, nmf_ranking, width, label='NMF', alpha=0.8)
    plt.ylabel('Score')
    plt.title('Ranking Metrics Comparison', fontweight='bold')
    plt.xticks(x, ranking_names, rotation=15)
    plt.legend()

    # Plot 5: Rating distribution
    plt.subplot(3, 4, 5)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.title('Rating Distribution', fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Plot 6: User factors visualization (SVD)
    plt.subplot(3, 4, 6)
    plt.imshow(best_svd.U[:50, :20], cmap='viridis', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Latent Factors')
    plt.ylabel('Users (sample)')
    plt.title('SVD: User Latent Factors', fontweight='bold')

    # Plot 7: Item factors visualization (SVD)
    plt.subplot(3, 4, 7)
    plt.imshow(best_svd.Vt[:20, :50], cmap='plasma', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Items (sample)')
    plt.ylabel('Latent Factors')
    plt.title('SVD: Item Latent Factors', fontweight='bold')

    # Plot 8: Singular values
    plt.subplot(3, 4, 8)
    plt.plot(range(len(best_svd.sigma)), sorted(best_svd.sigma, reverse=True),
             marker='o', linewidth=2)
    plt.xlabel('Factor Index')
    plt.ylabel('Singular Value')
    plt.title('SVD: Singular Values', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 9: NMF User factors
    plt.subplot(3, 4, 9)
    plt.imshow(best_nmf.W[:50, :20], cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Components')
    plt.ylabel('Users (sample)')
    plt.title('NMF: User Components', fontweight='bold')

    # Plot 10: NMF Item factors
    plt.subplot(3, 4, 10)
    plt.imshow(best_nmf.H[:20, :50], cmap='GnBu', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Items (sample)')
    plt.ylabel('Components')
    plt.title('NMF: Item Components', fontweight='bold')

    # Plot 11: Prediction error distribution (SVD)
    plt.subplot(3, 4, 11)
    svd_errors = []
    for _, row in test_df.head(1000).iterrows():
        pred = best_svd.predict(row['user_id'], row['item_id'])
        svd_errors.append(pred - row['rating'])
    plt.hist(svd_errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('SVD: Prediction Error Distribution', fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # Plot 12: Prediction error distribution (NMF)
    plt.subplot(3, 4, 12)
    nmf_errors = []
    for _, row in test_df.head(1000).iterrows():
        pred = best_nmf.predict(row['user_id'], row['item_id'])
        nmf_errors.append(pred - row['rating'])
    plt.hist(nmf_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('NMF: Prediction Error Distribution', fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/13_matrix_factorization/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
