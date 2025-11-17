"""
Alternating Least Squares (ALS) for Collaborative Filtering
===========================================================

This solution implements Alternating Least Squares (ALS) matrix factorization
for implicit and explicit feedback recommendation systems with comprehensive
evaluation and optimization techniques.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ALSRecommender:
    """
    Alternating Least Squares recommender for explicit ratings.
    """

    def __init__(self, n_factors=20, n_iterations=15, reg_param=0.1, random_state=42):
        """
        Initialize ALS model.

        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        n_iterations : int
            Number of ALS iterations
        reg_param : float
            Regularization parameter
        random_state : int
            Random seed
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_param = reg_param
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.train_errors = []

    def fit(self, ratings_df):
        """
        Fit ALS model on training data.

        Parameters:
        -----------
        ratings_df : pd.DataFrame
            DataFrame with columns: user_id, item_id, rating
        """
        np.random.seed(self.random_state)

        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df['user_id'].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df['item_id'].unique())}

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Initialize biases
        self.global_bias = ratings_df['rating'].mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        # Create rating matrix
        ratings_matrix = lil_matrix((n_users, n_items))
        for _, row in ratings_df.iterrows():
            u_idx = self.user_mapping[row['user_id']]
            i_idx = self.item_mapping[row['item_id']]
            ratings_matrix[u_idx, i_idx] = row['rating']

        ratings_matrix = ratings_matrix.tocsr()

        # ALS iterations
        for iteration in range(self.n_iterations):
            # Update user factors
            for u in range(n_users):
                # Get items rated by user
                items_rated = ratings_matrix[u].nonzero()[1]
                if len(items_rated) == 0:
                    continue

                # Build system
                A = self.item_factors[items_rated]
                ratings = ratings_matrix[u, items_rated].toarray().flatten()

                # Add regularization
                AtA = A.T.dot(A) + self.reg_param * np.eye(self.n_factors)
                Atb = A.T.dot(ratings - self.global_bias - self.item_biases[items_rated])

                # Solve
                self.user_factors[u] = np.linalg.solve(AtA, Atb)

            # Update item factors
            for i in range(n_items):
                # Get users who rated item
                users_rated = ratings_matrix[:, i].nonzero()[0]
                if len(users_rated) == 0:
                    continue

                # Build system
                A = self.user_factors[users_rated]
                ratings = ratings_matrix[users_rated, i].toarray().flatten()

                # Add regularization
                AtA = A.T.dot(A) + self.reg_param * np.eye(self.n_factors)
                Atb = A.T.dot(ratings - self.global_bias - self.user_biases[users_rated])

                # Solve
                self.item_factors[i] = np.linalg.solve(AtA, Atb)

            # Update biases
            for u in range(n_users):
                items_rated = ratings_matrix[u].nonzero()[1]
                if len(items_rated) > 0:
                    ratings = ratings_matrix[u, items_rated].toarray().flatten()
                    predictions = (
                        self.global_bias +
                        self.item_biases[items_rated] +
                        self.user_factors[u].dot(self.item_factors[items_rated].T)
                    )
                    self.user_biases[u] = (ratings - predictions).mean()

            for i in range(n_items):
                users_rated = ratings_matrix[:, i].nonzero()[0]
                if len(users_rated) > 0:
                    ratings = ratings_matrix[users_rated, i].toarray().flatten()
                    predictions = (
                        self.global_bias +
                        self.user_biases[users_rated] +
                        self.user_factors[users_rated].dot(self.item_factors[i])
                    )
                    self.item_biases[i] = (ratings - predictions).mean()

            # Calculate training error
            train_error = self._calculate_error(ratings_matrix)
            self.train_errors.append(train_error)

            if iteration % 5 == 0:
                print(f"   Iteration {iteration}: RMSE = {train_error:.4f}")

    def _calculate_error(self, ratings_matrix):
        """Calculate RMSE on training data."""
        predictions = []
        actuals = []

        for u, i in zip(*ratings_matrix.nonzero()):
            pred = (
                self.global_bias +
                self.user_biases[u] +
                self.item_biases[i] +
                self.user_factors[u].dot(self.item_factors[i])
            )
            predictions.append(pred)
            actuals.append(ratings_matrix[u, i])

        return np.sqrt(mean_squared_error(actuals, predictions))

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair."""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_bias

        u_idx = self.user_mapping[user_id]
        i_idx = self.item_mapping[item_id]

        prediction = (
            self.global_bias +
            self.user_biases[u_idx] +
            self.item_biases[i_idx] +
            self.user_factors[u_idx].dot(self.item_factors[i_idx])
        )

        return prediction

    def recommend(self, user_id, n_recommendations=10):
        """Generate top-N recommendations."""
        if user_id not in self.user_mapping:
            return []

        u_idx = self.user_mapping[user_id]

        # Calculate scores for all items
        scores = (
            self.global_bias +
            self.user_biases[u_idx] +
            self.item_biases +
            self.user_factors[u_idx].dot(self.item_factors.T)
        )

        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        inv_item_mapping = {v: k for k, v in self.item_mapping.items()}

        recommendations = [
            (inv_item_mapping[idx], scores[idx])
            for idx in top_indices
        ]

        return recommendations


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
        rating = np.clip(
            3 + base_rating / 2 + np.random.normal(0, 0.5),
            1, 5
        )

        ratings.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': round(rating, 1)
        })

    df = pd.DataFrame(ratings)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
    return df


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
    print("Alternating Least Squares (ALS) Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic rating data...")
    ratings_df = generate_synthetic_data(n_users=500, n_items=300, sparsity=0.9)
    print(f"   Generated {len(ratings_df)} ratings")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Train models with different parameters
    print("\n3. Training ALS models with different configurations...")

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
        model = ALSRecommender(
            n_factors=config['n_factors'],
            reg_param=config['reg_param'],
            n_iterations=15
        )
        model.fit(train_df)

        # Evaluate
        predictions = []
        actuals = []
        for _, row in test_df.head(1000).iterrows():
            pred = model.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'model': model,
            'train_errors': model.train_errors
        }
        print(f"   Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Ranking evaluation
    print("\n4. Evaluating ranking metrics...")
    best_model = results['F20_R0.1']['model']

    test_users = test_df['user_id'].unique()[:50]
    ranking_metrics = {
        'Precision@10': [], 'Recall@10': [], 'NDCG@10': []
    }

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
        plt.plot(result['train_errors'], label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Training Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Model comparison
    plt.subplot(3, 4, 2)
    model_names = list(results.keys())
    rmse_vals = [results[m]['RMSE'] for m in model_names]
    mae_vals = [results[m]['MAE'] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, rmse_vals, width, label='RMSE', alpha=0.8)
    plt.bar(x + width/2, mae_vals, width, label='MAE', alpha=0.8)
    plt.ylabel('Error')
    plt.title('Model Performance Comparison', fontweight='bold')
    plt.xticks(x, model_names, rotation=15)
    plt.legend()

    # Plot 3: User factors heatmap
    plt.subplot(3, 4, 3)
    plt.imshow(best_model.user_factors[:50], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Latent Factors')
    plt.ylabel('Users (sample)')
    plt.title('User Latent Factors', fontweight='bold')

    # Plot 4: Item factors heatmap
    plt.subplot(3, 4, 4)
    plt.imshow(best_model.item_factors[:50], cmap='viridis', aspect='auto')
    plt.colorbar(label='Factor Value')
    plt.xlabel('Latent Factors')
    plt.ylabel('Items (sample)')
    plt.title('Item Latent Factors', fontweight='bold')

    # Plot 5: User bias distribution
    plt.subplot(3, 4, 5)
    plt.hist(best_model.user_biases, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('User Bias')
    plt.ylabel('Frequency')
    plt.title('User Bias Distribution', fontweight='bold')

    # Plot 6: Item bias distribution
    plt.subplot(3, 4, 6)
    plt.hist(best_model.item_biases, bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Item Bias')
    plt.ylabel('Frequency')
    plt.title('Item Bias Distribution', fontweight='bold')

    # Plot 7: Ranking metrics
    plt.subplot(3, 4, 7)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    # Plot 8: Rating distribution
    plt.subplot(3, 4, 8)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Rating Distribution', fontweight='bold')

    # Plot 9: Prediction errors
    plt.subplot(3, 4, 9)
    errors = np.array(predictions) - np.array(actuals[:len(predictions)])
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution', fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # Plot 10: Factor correlation
    plt.subplot(3, 4, 10)
    factor_corr = np.corrcoef(best_model.user_factors.T)
    sns.heatmap(factor_corr, cmap='RdBu_r', center=0, square=True,
                cbar_kws={'label': 'Correlation'})
    plt.title('User Factor Correlation', fontweight='bold')

    # Plot 11: Predicted vs Actual
    plt.subplot(3, 4, 11)
    plt.scatter(actuals[:len(predictions)], predictions, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Predicted vs Actual Ratings', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 12: Regularization impact
    plt.subplot(3, 4, 12)
    reg_params = [0.01, 0.1]
    reg_rmse = []
    for reg in reg_params:
        key = f"F20_R{reg}"
        if key in results:
            reg_rmse.append(results[key]['RMSE'])
    plt.bar(range(len(reg_params)), reg_rmse, alpha=0.8, color='purple')
    plt.xticks(range(len(reg_params)), [f'λ={r}' for r in reg_params])
    plt.ylabel('RMSE')
    plt.title('Regularization Impact (20 factors)', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/14_alternating_least_squares/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
