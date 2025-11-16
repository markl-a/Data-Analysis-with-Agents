"""
Music Recommendation System using Matrix Factorization (SVD)
=============================================================
This solution demonstrates a music recommendation system using
Singular Value Decomposition (SVD) for collaborative filtering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class MusicRecommender:
    """Music recommendation system using SVD-based matrix factorization."""

    def __init__(self, n_factors=50):
        """
        Initialize the recommender.

        Args:
            n_factors: Number of latent factors for matrix factorization
        """
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.predictions = None
        self.U = None
        self.sigma = None
        self.Vt = None

    def generate_data(self, n_users=500, n_songs=300, n_interactions=8000):
        """
        Generate synthetic music listening data.

        Args:
            n_users: Number of users
            n_songs: Number of songs
            n_interactions: Number of user-song interactions

        Returns:
            DataFrame with user_id, song_id, play_count, rating
        """
        print("Generating synthetic music data...")

        # Generate song metadata
        genres = ['Rock', 'Pop', 'Jazz', 'Electronic', 'Hip-Hop', 'Classical', 'Country', 'R&B']
        song_data = []
        for song_id in range(n_songs):
            song_data.append({
                'song_id': song_id,
                'genre': np.random.choice(genres),
                'tempo': np.random.randint(60, 180),
                'energy': np.random.uniform(0, 1)
            })
        self.songs_df = pd.DataFrame(song_data)

        # Generate user preferences (some users prefer certain genres)
        user_genre_pref = {}
        for user_id in range(n_users):
            preferred_genres = np.random.choice(genres, size=np.random.randint(1, 4), replace=False)
            user_genre_pref[user_id] = preferred_genres

        # Generate interactions based on preferences
        interactions = []
        for _ in range(n_interactions):
            user_id = np.random.randint(0, n_users)

            # 70% chance of picking from preferred genre
            if np.random.random() < 0.7 and user_id in user_genre_pref:
                preferred_songs = self.songs_df[
                    self.songs_df['genre'].isin(user_genre_pref[user_id])
                ]['song_id'].values
                if len(preferred_songs) > 0:
                    song_id = np.random.choice(preferred_songs)
                else:
                    song_id = np.random.randint(0, n_songs)
            else:
                song_id = np.random.randint(0, n_songs)

            # Higher play count and rating for preferred genres
            is_preferred = (user_id in user_genre_pref and
                          self.songs_df.iloc[song_id]['genre'] in user_genre_pref[user_id])

            if is_preferred:
                play_count = np.random.randint(5, 50)
                rating = np.random.randint(3, 6)
            else:
                play_count = np.random.randint(1, 15)
                rating = np.random.randint(1, 5)

            interactions.append({
                'user_id': user_id,
                'song_id': song_id,
                'play_count': play_count,
                'rating': rating
            })

        df = pd.DataFrame(interactions)
        # Remove duplicates, keep the one with highest play count
        df = df.sort_values('play_count', ascending=False).drop_duplicates(['user_id', 'song_id'])

        print(f"Generated {len(df)} interactions for {df['user_id'].nunique()} users and {df['song_id'].nunique()} songs")
        return df

    def prepare_matrix(self, df):
        """Create user-item rating matrix."""
        self.user_item_matrix = df.pivot_table(
            index='user_id',
            columns='song_id',
            values='rating',
            fill_value=0
        )
        print(f"Matrix shape: {self.user_item_matrix.shape}")
        sparsity = 1 - (df.shape[0] / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]))
        print(f"Matrix sparsity: {sparsity:.2%}")

    def train(self):
        """Train SVD model."""
        print(f"\nTraining SVD with {self.n_factors} factors...")

        # Perform SVD
        matrix = self.user_item_matrix.values
        user_ratings_mean = np.mean(matrix, axis=1)
        matrix_normalized = matrix - user_ratings_mean.reshape(-1, 1)

        # SVD
        U, sigma, Vt = svds(matrix_normalized, k=self.n_factors)

        # Store components
        self.U = U
        self.sigma = sigma
        self.Vt = Vt
        self.user_ratings_mean = user_ratings_mean

        # Make predictions
        sigma_diag = np.diag(sigma)
        self.predictions = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
        self.predictions_df = pd.DataFrame(
            self.predictions,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

        print("Training complete!")

    def recommend_songs(self, user_id, n=10, exclude_known=True):
        """
        Recommend songs for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_known: Whether to exclude already rated songs

        Returns:
            List of recommended song IDs with predicted ratings
        """
        if user_id not in self.predictions_df.index:
            print(f"User {user_id} not found. Recommending popular songs.")
            return self._recommend_popular(n)

        user_predictions = self.predictions_df.loc[user_id].sort_values(ascending=False)

        if exclude_known:
            known_songs = self.user_item_matrix.loc[user_id]
            known_songs = known_songs[known_songs > 0].index
            user_predictions = user_predictions.drop(known_songs, errors='ignore')

        recommendations = user_predictions.head(n)
        return [(song_id, score) for song_id, score in recommendations.items()]

    def _recommend_popular(self, n=10):
        """Recommend popular songs for cold start users."""
        popularity = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
        return [(song_id, score) for song_id, score in popularity.head(n).items()]

    def evaluate(self, test_df):
        """
        Evaluate model performance.

        Args:
            test_df: Test dataframe with user_id, song_id, rating

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")

        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            user_id = row['user_id']
            song_id = row['song_id']

            if user_id in self.predictions_df.index and song_id in self.predictions_df.columns:
                pred = self.predictions_df.loc[user_id, song_id]
                predictions.append(pred)
                actuals.append(row['rating'])

        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        else:
            rmse = mae = float('inf')

        # Calculate precision@k and recall@k
        k = 10
        precision_scores = []
        recall_scores = []

        for user_id in test_df['user_id'].unique():
            if user_id not in self.predictions_df.index:
                continue

            # Get top k recommendations
            recs = self.recommend_songs(user_id, n=k, exclude_known=True)
            rec_songs = [song_id for song_id, _ in recs]

            # Get actual liked songs (rating >= 4)
            actual_songs = test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4)]['song_id'].values

            if len(actual_songs) > 0:
                hits = len(set(rec_songs) & set(actual_songs))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_songs))

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0
        }

        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")

        return metrics

    def visualize_results(self):
        """Create visualizations of the recommendation system."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Rating distribution
        ax = axes[0, 0]
        ratings_flat = self.user_item_matrix.values.flatten()
        ratings_flat = ratings_flat[ratings_flat > 0]
        ax.hist(ratings_flat, bins=5, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Ratings')
        ax.grid(True, alpha=0.3)

        # 2. User activity distribution
        ax = axes[0, 1]
        user_activity = (self.user_item_matrix > 0).sum(axis=1)
        ax.hist(user_activity, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Number of Rated Songs')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Activity Distribution')
        ax.grid(True, alpha=0.3)

        # 3. Song popularity distribution
        ax = axes[1, 0]
        song_popularity = (self.user_item_matrix > 0).sum(axis=0)
        ax.hist(song_popularity, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Number of Songs')
        ax.set_title('Song Popularity Distribution')
        ax.grid(True, alpha=0.3)

        # 4. Explained variance by factors
        ax = axes[1, 1]
        explained_var = (self.sigma ** 2) / np.sum(self.sigma ** 2)
        ax.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'o-')
        ax.set_xlabel('Number of Factors')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('SVD Explained Variance')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/music_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/music_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Music Recommendation System using Matrix Factorization (SVD)")
    print("=" * 70)

    # Initialize recommender
    recommender = MusicRecommender(n_factors=50)

    # Generate data
    df = recommender.generate_data(n_users=500, n_songs=300, n_interactions=8000)

    # Split into train/test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print(f"\nTrain set: {len(train_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")

    # Prepare matrix and train
    recommender.prepare_matrix(train_df)
    recommender.train()

    # Evaluate
    metrics = recommender.evaluate(test_df)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_user = train_df['user_id'].value_counts().head(1).index[0]
    print(f"\nRecommendations for User {test_user}:")
    print(f"User has rated {(recommender.user_item_matrix.loc[test_user] > 0).sum()} songs")

    recommendations = recommender.recommend_songs(test_user, n=10)
    for i, (song_id, score) in enumerate(recommendations, 1):
        genre = recommender.songs_df[recommender.songs_df['song_id'] == song_id]['genre'].values
        genre_str = genre[0] if len(genre) > 0 else 'Unknown'
        print(f"{i}. Song {song_id} ({genre_str}) - Predicted Score: {score:.2f}")

    # Visualize
    recommender.visualize_results()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
