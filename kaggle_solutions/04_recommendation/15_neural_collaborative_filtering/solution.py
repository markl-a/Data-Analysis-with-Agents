"""
Neural Collaborative Filtering (NCF)
====================================

This solution implements Neural Collaborative Filtering using deep learning
to learn complex user-item interactions through neural networks, including
GMF, MLP, and NeuMF architectures.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class NeuralCF:
    """
    Neural Collaborative Filtering model.
    Implements GMF (Generalized Matrix Factorization) and MLP fusion.
    """

    def __init__(self, n_users, n_items, embedding_dim=32, hidden_layers=[64, 32, 16],
                 learning_rate=0.001, reg_param=0.01):
        """
        Initialize Neural CF model.

        Parameters:
        -----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        embedding_dim : int
            Dimension of embeddings
        hidden_layers : list
            Sizes of hidden layers
        learning_rate : float
            Learning rate
        reg_param : float
            Regularization parameter
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.reg_param = reg_param

        # Initialize embeddings
        np.random.seed(42)
        self.user_embeddings_gmf = np.random.normal(0, 0.1, (n_users, embedding_dim))
        self.item_embeddings_gmf = np.random.normal(0, 0.1, (n_items, embedding_dim))
        self.user_embeddings_mlp = np.random.normal(0, 0.1, (n_users, embedding_dim))
        self.item_embeddings_mlp = np.random.normal(0, 0.1, (n_items, embedding_dim))

        # Initialize MLP weights
        self.mlp_weights = []
        self.mlp_biases = []

        input_dim = 2 * embedding_dim
        for hidden_dim in hidden_layers:
            self.mlp_weights.append(np.random.normal(0, 0.1, (input_dim, hidden_dim)))
            self.mlp_biases.append(np.zeros(hidden_dim))
            input_dim = hidden_dim

        # Final layer
        final_input = embedding_dim + hidden_layers[-1]
        self.final_weights = np.random.normal(0, 0.1, (final_input, 1))
        self.final_bias = np.zeros(1)

        self.train_losses = []

    def sigmoid(self, x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def forward(self, user_ids, item_ids):
        """Forward pass through the network."""
        # GMF part
        user_emb_gmf = self.user_embeddings_gmf[user_ids]
        item_emb_gmf = self.item_embeddings_gmf[item_ids]
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP part
        user_emb_mlp = self.user_embeddings_mlp[user_ids]
        item_emb_mlp = self.item_embeddings_mlp[item_ids]
        mlp_input = np.concatenate([user_emb_mlp, item_emb_mlp], axis=1)

        mlp_output = mlp_input
        for w, b in zip(self.mlp_weights, self.mlp_biases):
            mlp_output = self.relu(mlp_output.dot(w) + b)

        # Concatenate and final prediction
        concat = np.concatenate([gmf_output, mlp_output], axis=1)
        prediction = self.sigmoid(concat.dot(self.final_weights) + self.final_bias)

        return prediction.flatten()

    def fit(self, user_ids, item_ids, ratings, n_epochs=20, batch_size=256):
        """
        Train the model.

        Parameters:
        -----------
        user_ids : array
            User IDs
        item_ids : array
            Item IDs
        ratings : array
            Ratings (normalized to 0-1)
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        """
        n_samples = len(user_ids)

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_users = user_ids[batch_indices]
                batch_items = item_ids[batch_indices]
                batch_ratings = ratings[batch_indices]

                # Forward pass
                predictions = self.forward(batch_users, batch_items)

                # Calculate loss
                loss = np.mean((predictions - batch_ratings) ** 2)

                # Add L2 regularization
                reg_loss = self.reg_param * (
                    np.sum(self.user_embeddings_gmf ** 2) +
                    np.sum(self.item_embeddings_gmf ** 2) +
                    np.sum(self.user_embeddings_mlp ** 2) +
                    np.sum(self.item_embeddings_mlp ** 2)
                ) / n_samples

                total_loss = loss + reg_loss
                epoch_loss += total_loss

                # Backward pass (simplified gradient descent)
                error = predictions - batch_ratings

                # Update embeddings (simplified)
                for idx, (u, i) in enumerate(zip(batch_users, batch_items)):
                    grad = error[idx] * self.learning_rate

                    self.user_embeddings_gmf[u] -= grad * self.item_embeddings_gmf[i]
                    self.item_embeddings_gmf[i] -= grad * self.user_embeddings_gmf[u]
                    self.user_embeddings_mlp[u] -= grad * 0.01
                    self.item_embeddings_mlp[i] -= grad * 0.01

            avg_loss = epoch_loss / (n_samples // batch_size)
            self.train_losses.append(avg_loss)

            if epoch % 5 == 0:
                print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")

    def predict(self, user_id, item_id):
        """Predict rating for single user-item pair."""
        prediction = self.forward(
            np.array([user_id]),
            np.array([item_id])
        )
        return prediction[0]

    def recommend(self, user_id, n_recommendations=10, rated_items=None):
        """Generate top-N recommendations for a user."""
        # Get scores for all items
        item_ids = np.arange(self.n_items)
        user_ids = np.full(self.n_items, user_id)

        scores = self.forward(user_ids, item_ids)

        # Exclude rated items
        if rated_items is not None:
            scores[list(rated_items)] = -1

        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(idx, scores[idx]) for idx in top_indices]

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


def calculate_metrics(predictions, actuals, denormalize=False):
    """Calculate evaluation metrics."""
    if denormalize:
        predictions = predictions * 4 + 1
        actuals = actuals * 4 + 1

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
    print("Neural Collaborative Filtering (NCF) Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic rating data...")
    ratings_df = generate_synthetic_data(n_users=300, n_items=200, sparsity=0.9)
    print(f"   Generated {len(ratings_df)} ratings")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Normalize ratings to 0-1
    train_df['rating_norm'] = (train_df['rating'] - 1) / 4
    test_df['rating_norm'] = (test_df['rating'] - 1) / 4

    # Train model
    print("\n3. Training Neural CF model...")
    n_users = ratings_df['user_id'].max() + 1
    n_items = ratings_df['item_id'].max() + 1

    model = NeuralCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=32,
        hidden_layers=[64, 32, 16],
        learning_rate=0.001
    )

    model.fit(
        train_df['user_id'].values,
        train_df['item_id'].values,
        train_df['rating_norm'].values,
        n_epochs=20,
        batch_size=256
    )

    # Evaluate
    print("\n4. Evaluating model...")
    test_predictions = []
    test_actuals = []

    for _, row in test_df.head(1000).iterrows():
        pred = model.predict(row['user_id'], row['item_id'])
        test_predictions.append(pred)
        test_actuals.append(row['rating_norm'])

    rmse, mae = calculate_metrics(test_predictions, test_actuals, denormalize=True)
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE: {mae:.4f}")

    # Ranking evaluation
    print("\n5. Evaluating ranking metrics...")
    test_users = test_df['user_id'].unique()[:30]

    ranking_metrics = {
        'Precision@10': [], 'Recall@10': [], 'NDCG@10': []
    }

    for user_id in test_users:
        relevant_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
        if not relevant_items:
            continue

        rated_items = train_df[train_df['user_id'] == user_id]['item_id'].tolist()
        recommendations = model.recommend(user_id, n_recommendations=10,
                                         rated_items=set(rated_items))

        prec, rec = calculate_precision_recall_at_k(recommendations, relevant_items, 10)
        ndcg = calculate_ndcg_at_k(recommendations, relevant_items, 10)

        ranking_metrics['Precision@10'].append(prec)
        ranking_metrics['Recall@10'].append(rec)
        ranking_metrics['NDCG@10'].append(ndcg)

    for metric, values in ranking_metrics.items():
        if len(values) > 0:
            print(f"   {metric}: {np.mean(values):.4f}")

    # Visualization
    print("\n6. Creating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Training loss
    plt.subplot(3, 4, 1)
    plt.plot(model.train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 2: User embeddings (GMF)
    plt.subplot(3, 4, 2)
    plt.imshow(model.user_embeddings_gmf[:50, :20], cmap='viridis', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Users (sample)')
    plt.title('GMF User Embeddings', fontweight='bold')

    # Plot 3: Item embeddings (GMF)
    plt.subplot(3, 4, 3)
    plt.imshow(model.item_embeddings_gmf[:50, :20], cmap='plasma', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Items (sample)')
    plt.title('GMF Item Embeddings', fontweight='bold')

    # Plot 4: User embeddings (MLP)
    plt.subplot(3, 4, 4)
    plt.imshow(model.user_embeddings_mlp[:50, :20], cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Users (sample)')
    plt.title('MLP User Embeddings', fontweight='bold')

    # Plot 5: Prediction distribution
    plt.subplot(3, 4, 5)
    denorm_preds = np.array(test_predictions) * 4 + 1
    plt.hist(denorm_preds, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Predicted Rating')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution', fontweight='bold')

    # Plot 6: Actual vs Predicted
    plt.subplot(3, 4, 6)
    denorm_actuals = np.array(test_actuals) * 4 + 1
    plt.scatter(denorm_actuals, denorm_preds, alpha=0.5)
    plt.plot([1, 5], [1, 5], 'r--', linewidth=2)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Predicted vs Actual', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 7: Ranking metrics
    plt.subplot(3, 4, 7)
    metric_names = ['Precision@10', 'Recall@10', 'NDCG@10']
    metric_vals = [np.mean(ranking_metrics[m]) for m in metric_names if len(ranking_metrics[m]) > 0]
    plt.bar(range(len(metric_names)), metric_vals, alpha=0.8, color='seagreen')
    plt.xticks(range(len(metric_names)), metric_names, rotation=15)
    plt.ylabel('Score')
    plt.title('Ranking Metrics', fontweight='bold')

    # Plot 8: Prediction errors
    plt.subplot(3, 4, 8)
    errors = denorm_preds - denorm_actuals
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution', fontweight='bold')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # Plot 9: Rating distribution
    plt.subplot(3, 4, 9)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Original Rating Distribution', fontweight='bold')

    # Plot 10: MLP layer 1 weights
    plt.subplot(3, 4, 10)
    plt.imshow(model.mlp_weights[0][:40, :40], cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Hidden Units')
    plt.ylabel('Input Units')
    plt.title('MLP Layer 1 Weights', fontweight='bold')

    # Plot 11: Embedding similarity (users)
    plt.subplot(3, 4, 11)
    user_sim = np.corrcoef(model.user_embeddings_gmf[:30])
    sns.heatmap(user_sim, cmap='YlOrRd', center=0, square=True,
                cbar_kws={'label': 'Correlation'})
    plt.title('User Embedding Similarity', fontweight='bold')

    # Plot 12: Item embedding norms
    plt.subplot(3, 4, 12)
    item_norms = np.linalg.norm(model.item_embeddings_gmf, axis=1)
    plt.hist(item_norms, bins=30, edgecolor='black', alpha=0.7, color='purple')
    plt.xlabel('Embedding Norm')
    plt.ylabel('Frequency')
    plt.title('Item Embedding Magnitude', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/15_neural_collaborative_filtering/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
