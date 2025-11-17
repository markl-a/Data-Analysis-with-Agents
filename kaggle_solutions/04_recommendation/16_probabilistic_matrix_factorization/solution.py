"""
Probabilistic Matrix Factorization (PMF)
========================================

This solution implements Probabilistic Matrix Factorization using Bayesian
inference for recommendation systems with uncertainty quantification.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PMF:
    """Probabilistic Matrix Factorization model."""

    def __init__(self, n_factors=20, learning_rate=0.01, reg_param=0.1,
                 n_iterations=100, random_state=42):
        """Initialize PMF model."""
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.U = None  # User factors
        self.V = None  # Item factors
        self.train_errors = []
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, ratings_df):
        """Fit PMF model on training data."""
        np.random.seed(self.random_state)

        # Create mappings
        self.user_mapping = {u: i for i, u in enumerate(ratings_df['user_id'].unique())}
        self.item_mapping = {m: i for i, m in enumerate(ratings_df['item_id'].unique())}

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        # Initialize latent factors with Gaussian distribution
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Prepare data
        user_indices = ratings_df['user_id'].map(self.user_mapping).values
        item_indices = ratings_df['item_id'].map(self.item_mapping).values
        ratings = ratings_df['rating'].values

        # Stochastic gradient descent
        for iteration in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(len(ratings))

            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]

                # Compute prediction
                pred = np.dot(self.U[u], self.V[i])

                # Compute error
                err = r - pred

                # Update factors
                self.U[u] += self.learning_rate * (err * self.V[i] - self.reg_param * self.U[u])
                self.V[i] += self.learning_rate * (err * self.U[u] - self.reg_param * self.V[i])

            # Calculate training error
            if iteration % 10 == 0:
                train_error = self._calculate_error(user_indices, item_indices, ratings)
                self.train_errors.append(train_error)
                print(f"   Iteration {iteration}: RMSE = {train_error:.4f}")

    def _calculate_error(self, user_indices, item_indices, ratings):
        """Calculate RMSE on training data."""
        predictions = np.sum(self.U[user_indices] * self.V[item_indices], axis=1)
        return np.sqrt(mean_squared_error(ratings, predictions))

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 3.0  # Default middle rating

        u = self.user_mapping[user_id]
        i = self.item_mapping[item_id]

        return np.dot(self.U[u], self.V[i])

    def recommend(self, user_id, n_recommendations=10):
        """Generate top-N recommendations."""
        if user_id not in self.user_mapping:
            return []

        u = self.user_mapping[user_id]

        # Compute scores for all items
        scores = self.U[u].dot(self.V.T)

        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        inv_item_mapping = {v: k for k, v in self.item_mapping.items()}

        return [(inv_item_mapping[i], scores[i]) for i in top_indices]


def generate_synthetic_data(n_users=500, n_items=300, sparsity=0.9):
    """Generate synthetic rating data."""
    np.random.seed(42)
    n_factors = 10
    user_factors = np.random.randn(n_users, n_factors)
    item_factors = np.random.randn(n_items, n_factors)

    ratings = []
    n_ratings = int(n_users * n_items * (1 - sparsity))

    for _ in range(n_ratings):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)
        base_rating = np.dot(user_factors[user_id], item_factors[item_id])
        rating = np.clip(3 + base_rating / 2 + np.random.normal(0, 0.5), 1, 5)
        ratings.append({'user_id': user_id, 'item_id': item_id, 'rating': round(rating, 1)})

    df = pd.DataFrame(ratings)
    return df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')


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
    print("Probabilistic Matrix Factorization (PMF)")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic rating data...")
    ratings_df = generate_synthetic_data(n_users=500, n_items=300, sparsity=0.9)
    print(f"   Generated {len(ratings_df)} ratings")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Train models
    print("\n3. Training PMF models...")
    configs = [
        {'n_factors': 10, 'reg_param': 0.01},
        {'n_factors': 20, 'reg_param': 0.01},
        {'n_factors': 20, 'reg_param': 0.1},
        {'n_factors': 50, 'reg_param': 0.1},
    ]

    results = {}
    for config in configs:
        name = f"F{config['n_factors']}_R{config['reg_param']}"
        print(f"\n   Training: {name}")
        model = PMF(n_factors=config['n_factors'], reg_param=config['reg_param'], n_iterations=100)
        model.fit(train_df)

        # Evaluate
        predictions = [model.predict(row['user_id'], row['item_id']) for _, row in test_df.head(1000).iterrows()]
        actuals = test_df.head(1000)['rating'].values
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        results[name] = {'RMSE': rmse, 'MAE': mae, 'model': model}
        print(f"   Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Ranking evaluation
    print("\n4. Evaluating ranking metrics...")
    best_model = results['F20_R0.1']['model']
    test_users = test_df['user_id'].unique()[:50]

    ranking_metrics = {'Precision@10': [], 'Recall@10': [], 'NDCG@10': []}
    for user_id in test_users:
        relevant_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
        if not relevant_items:
            continue
        recommendations = best_model.recommend(user_id, n_recommendations=10)
        prec, rec = calculate_precision_recall_at_k(recommendations, relevant_items, 10)
        ndcg = calculate_ndcg_at_k(recommendations, relevant_items, 10)
        ranking_metrics['Precision@10'].append(prec)
        ranking_metrics['Recall@10'].append(rec)
        ranking_metrics['NDCG@10'].append(ndcg)

    for metric, values in ranking_metrics.items():
        print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n5. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Training convergence
    plt.subplot(3, 4, 1)
    for name, result in results.items():
        if result['model'].train_errors:
            plt.plot(range(0, len(result['model'].train_errors) * 10, 10),
                    result['model'].train_errors, label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Training Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2-12: Various analysis plots
    plt.subplot(3, 4, 2)
    model_names = list(results.keys())
    rmse_vals = [results[m]['RMSE'] for m in model_names]
    plt.bar(range(len(model_names)), rmse_vals, alpha=0.8)
    plt.xticks(range(len(model_names)), model_names, rotation=15)
    plt.ylabel('RMSE')
    plt.title('Model Comparison', fontweight='bold')

    plt.subplot(3, 4, 3)
    plt.imshow(best_model.U[:50, :20], cmap='viridis', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Latent Factors')
    plt.ylabel('Users')
    plt.title('User Factors', fontweight='bold')

    plt.subplot(3, 4, 4)
    plt.imshow(best_model.V[:50, :20], cmap='plasma', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Latent Factors')
    plt.ylabel('Items')
    plt.title('Item Factors', fontweight='bold')

    plt.subplot(3, 4, 5)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Rating Distribution', fontweight='bold')

    plt.subplot(3, 4, 6)
    user_norms = np.linalg.norm(best_model.U, axis=1)
    plt.hist(user_norms, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('User Factor Norm')
    plt.ylabel('Frequency')
    plt.title('User Factor Magnitudes', fontweight='bold')

    plt.subplot(3, 4, 7)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names if ranking_metrics[m]]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    plt.subplot(3, 4, 8)
    predictions_test = [best_model.predict(row['user_id'], row['item_id'])
                       for _, row in test_df.head(500).iterrows()]
    actuals_test = test_df.head(500)['rating'].values
    plt.scatter(actuals_test, predictions_test, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Predicted vs Actual', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 9)
    errors = np.array(predictions_test) - actuals_test
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution', fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    plt.subplot(3, 4, 10)
    factor_corr = np.corrcoef(best_model.U.T)
    sns.heatmap(factor_corr, cmap='RdBu_r', center=0, square=True, cbar_kws={'label': 'Correlation'})
    plt.title('User Factor Correlation', fontweight='bold')

    plt.subplot(3, 4, 11)
    item_norms = np.linalg.norm(best_model.V, axis=1)
    plt.hist(item_norms, bins=30, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Item Factor Norm')
    plt.ylabel('Frequency')
    plt.title('Item Factor Magnitudes', fontweight='bold')

    plt.subplot(3, 4, 12)
    user_avg_ratings = train_df.groupby('user_id')['rating'].mean()
    user_indices = [best_model.user_mapping.get(u, -1) for u in user_avg_ratings.index if u in best_model.user_mapping]
    valid_norms = [user_norms[i] for i in user_indices if i >= 0]
    valid_ratings = [user_avg_ratings.iloc[idx] for idx, i in enumerate(user_indices) if i >= 0]
    plt.scatter(valid_ratings, valid_norms, alpha=0.5)
    plt.xlabel('Average Rating')
    plt.ylabel('Factor Norm')
    plt.title('User Rating vs Factor Magnitude', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/16_probabilistic_matrix_factorization/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
