"""
Video Content Recommendation System using Embedding-Based Methods
==================================================================
This solution demonstrates a video recommendation system using
neural embedding-based collaborative filtering and content features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class VideoRecommender:
    """Video recommendation system using embedding-based collaborative filtering."""

    def __init__(self, embedding_dim=32, learning_rate=0.01, epochs=50):
        """
        Initialize recommender.

        Args:
            embedding_dim: Dimension of user/video embeddings
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.user_embeddings = None
        self.video_embeddings = None
        self.user_bias = None
        self.video_bias = None
        self.global_mean = 0

    def generate_data(self, n_videos=400, n_users=600):
        """
        Generate synthetic video viewing data.

        Args:
            n_videos: Number of videos
            n_users: Number of users

        Returns:
            videos_df, interactions_df
        """
        print("Generating synthetic video data...")

        # Video attributes
        categories = ['Education', 'Entertainment', 'Music', 'Gaming', 'News',
                     'Sports', 'Technology', 'Cooking', 'Travel', 'Fitness']
        durations = [5, 10, 15, 20, 30, 45, 60]  # minutes
        upload_years = [2020, 2021, 2022, 2023, 2024]

        # Generate videos
        videos = []
        for video_id in range(n_videos):
            category = np.random.choice(categories)
            duration = np.random.choice(durations)

            # Video quality (affects engagement)
            quality = np.random.uniform(0.3, 1.0)

            videos.append({
                'video_id': video_id,
                'title': f'Video_{video_id}_{category}',
                'category': category,
                'duration': duration,
                'upload_year': np.random.choice(upload_years),
                'quality': quality,
                'views': 0,
                'avg_watch_time': 0
            })

        self.videos_df = pd.DataFrame(videos)

        # Generate user preferences
        user_category_pref = {}
        user_duration_pref = {}

        for user_id in range(n_users):
            # Users prefer 1-3 categories
            n_pref = np.random.randint(1, 4)
            user_category_pref[user_id] = np.random.choice(categories, size=n_pref, replace=False)

            # Duration preference (short/medium/long)
            user_duration_pref[user_id] = np.random.choice(['short', 'medium', 'long'])

        # Generate interactions
        interactions = []
        for user_id in range(n_users):
            # Each user watches 5-25 videos
            n_watches = np.random.randint(5, 26)

            for _ in range(n_watches):
                # 70% chance of watching preferred category
                if np.random.random() < 0.7:
                    pref_videos = self.videos_df[
                        self.videos_df['category'].isin(user_category_pref[user_id])
                    ]
                    if len(pref_videos) > 0:
                        video = pref_videos.sample(1).iloc[0]
                    else:
                        video = self.videos_df.sample(1).iloc[0]
                else:
                    video = self.videos_df.sample(1).iloc[0]

                # Watch time based on preferences and video quality
                is_preferred_category = video['category'] in user_category_pref[user_id]

                duration_pref = user_duration_pref[user_id]
                if duration_pref == 'short' and video['duration'] <= 15:
                    is_preferred_duration = True
                elif duration_pref == 'medium' and 15 < video['duration'] <= 30:
                    is_preferred_duration = True
                elif duration_pref == 'long' and video['duration'] > 30:
                    is_preferred_duration = True
                else:
                    is_preferred_duration = False

                # Calculate watch time percentage
                base_watch = 0.3
                if is_preferred_category:
                    base_watch += 0.3
                if is_preferred_duration:
                    base_watch += 0.2
                base_watch += video['quality'] * 0.2

                watch_percentage = min(base_watch + np.random.normal(0, 0.1), 1.0)
                watch_percentage = max(watch_percentage, 0.05)

                # Engagement score (like/dislike)
                if watch_percentage > 0.7:
                    engagement = np.random.choice([1, 0.5, 0], p=[0.7, 0.2, 0.1])
                elif watch_percentage > 0.4:
                    engagement = np.random.choice([1, 0.5, 0, -0.5], p=[0.3, 0.4, 0.2, 0.1])
                else:
                    engagement = np.random.choice([0, -0.5, -1], p=[0.4, 0.4, 0.2])

                # Implicit rating (combine watch time and engagement)
                implicit_rating = (watch_percentage * 3 + (engagement + 1) * 2) / 2
                implicit_rating = min(max(implicit_rating, 0), 5)

                interactions.append({
                    'user_id': user_id,
                    'video_id': video['video_id'],
                    'watch_percentage': watch_percentage,
                    'engagement': engagement,
                    'implicit_rating': implicit_rating,
                    'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
                })

        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.drop_duplicates(['user_id', 'video_id'])

        # Update video statistics
        video_stats = interactions_df.groupby('video_id').agg({
            'watch_percentage': ['count', 'mean']
        }).reset_index()
        video_stats.columns = ['video_id', 'views', 'avg_watch_time']

        self.videos_df = self.videos_df.merge(video_stats, on='video_id', how='left')
        self.videos_df['views'] = self.videos_df['views'].fillna(0)
        self.videos_df['avg_watch_time'] = self.videos_df['avg_watch_time'].fillna(0)

        print(f"Generated {len(self.videos_df)} videos and {len(interactions_df)} interactions")
        print(f"Category distribution: {self.videos_df['category'].value_counts().to_dict()}")

        return self.videos_df, interactions_df

    def train_embeddings(self, interactions_df, n_users, n_videos):
        """
        Train user and video embeddings using matrix factorization.

        Uses gradient descent to minimize prediction error.

        Args:
            interactions_df: User-video interaction data
            n_users: Number of users
            n_videos: Number of videos
        """
        print(f"\nTraining embeddings ({self.embedding_dim}D) for {self.epochs} epochs...")

        # Initialize embeddings randomly
        self.user_embeddings = np.random.normal(0, 0.1, (n_users, self.embedding_dim))
        self.video_embeddings = np.random.normal(0, 0.1, (n_videos, self.embedding_dim))

        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.video_bias = np.zeros(n_videos)

        # Global mean rating
        self.global_mean = interactions_df['implicit_rating'].mean()

        # Prepare training data
        user_ids = interactions_df['user_id'].values
        video_ids = interactions_df['video_id'].values
        ratings = interactions_df['implicit_rating'].values

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(len(interactions_df))

            total_loss = 0
            for idx in indices:
                user_id = user_ids[idx]
                video_id = video_ids[idx]
                rating = ratings[idx]

                # Predict
                prediction = (self.global_mean +
                            self.user_bias[user_id] +
                            self.video_bias[video_id] +
                            np.dot(self.user_embeddings[user_id],
                                  self.video_embeddings[video_id]))

                # Error
                error = rating - prediction
                total_loss += error ** 2

                # Gradient descent updates
                # Update biases
                self.user_bias[user_id] += self.learning_rate * error
                self.video_bias[video_id] += self.learning_rate * error

                # Update embeddings
                user_emb_grad = error * self.video_embeddings[video_id]
                video_emb_grad = error * self.user_embeddings[user_id]

                self.user_embeddings[user_id] += self.learning_rate * user_emb_grad
                self.video_embeddings[video_id] += self.learning_rate * video_emb_grad

            # Print progress
            if (epoch + 1) % 10 == 0:
                rmse = np.sqrt(total_loss / len(interactions_df))
                print(f"Epoch {epoch + 1}/{self.epochs}, RMSE: {rmse:.4f}")

        print("Training complete!")

    def predict_rating(self, user_id, video_id):
        """
        Predict rating for user-video pair.

        Args:
            user_id: User ID
            video_id: Video ID

        Returns:
            Predicted rating
        """
        if user_id >= len(self.user_embeddings) or video_id >= len(self.video_embeddings):
            return self.global_mean

        prediction = (self.global_mean +
                     self.user_bias[user_id] +
                     self.video_bias[video_id] +
                     np.dot(self.user_embeddings[user_id],
                           self.video_embeddings[video_id]))

        return prediction

    def recommend_videos(self, user_id, n=10, exclude_watched=True, interactions_df=None):
        """
        Recommend videos for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_watched: Whether to exclude already watched videos
            interactions_df: User interaction history

        Returns:
            List of (video_id, predicted_rating) tuples
        """
        if user_id >= len(self.user_embeddings):
            print(f"User {user_id} not found. Recommending popular videos.")
            return self._recommend_popular(n)

        # Predict ratings for all videos
        predictions = []
        for video_id in range(len(self.video_embeddings)):
            # Skip watched videos if requested
            if exclude_watched and interactions_df is not None:
                if video_id in interactions_df[interactions_df['user_id'] == user_id]['video_id'].values:
                    continue

            pred_rating = self.predict_rating(user_id, video_id)
            predictions.append((video_id, pred_rating))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]

    def find_similar_videos(self, video_id, n=5):
        """
        Find videos similar to a given video based on embeddings.

        Args:
            video_id: Video ID
            n: Number of similar videos

        Returns:
            List of (video_id, similarity) tuples
        """
        if video_id >= len(self.video_embeddings):
            return []

        video_emb = self.video_embeddings[video_id]

        # Calculate cosine similarity with all other videos
        similarities = []
        for other_id in range(len(self.video_embeddings)):
            if other_id == video_id:
                continue

            other_emb = self.video_embeddings[other_id]

            # Cosine similarity
            similarity = np.dot(video_emb, other_emb) / (
                np.linalg.norm(video_emb) * np.linalg.norm(other_emb) + 1e-8
            )

            similarities.append((other_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:n]

    def _recommend_popular(self, n=10):
        """Recommend popular videos for cold start."""
        popular = self.videos_df.nlargest(n, 'views')
        return [(row['video_id'], 5.0) for _, row in popular.iterrows()]

    def evaluate(self, train_df, test_df):
        """Evaluate recommendation quality."""
        print("\nEvaluating model...")

        mae_scores = []
        rmse_scores = []
        precision_scores = []
        recall_scores = []

        k = 10

        for user_id in test_df['user_id'].unique():
            if user_id >= len(self.user_embeddings):
                continue

            # Get recommendations
            recs = self.recommend_videos(user_id, n=k, exclude_watched=True, interactions_df=train_df)
            rec_videos = [vid for vid, _ in recs]

            # Get actual high-engagement videos (implicit_rating >= 3.5)
            actual_videos = test_df[
                (test_df['user_id'] == user_id) &
                (test_df['implicit_rating'] >= 3.5)
            ]['video_id'].values

            if len(actual_videos) > 0:
                hits = len(set(rec_videos) & set(actual_videos))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_videos))

            # Rating prediction accuracy
            user_test = test_df[test_df['user_id'] == user_id]
            for _, row in user_test.iterrows():
                video_id = row['video_id']
                actual_rating = row['implicit_rating']
                predicted_rating = self.predict_rating(user_id, video_id)

                mae_scores.append(abs(actual_rating - predicted_rating))
                rmse_scores.append((actual_rating - predicted_rating) ** 2)

        metrics = {
            'mae': np.mean(mae_scores) if mae_scores else float('inf'),
            'rmse': np.sqrt(np.mean(rmse_scores)) if rmse_scores else float('inf'),
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0
        }

        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")

        return metrics

    def visualize_results(self, interactions_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Category distribution
        ax = axes[0, 0]
        category_counts = self.videos_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of Videos')
        ax.set_title('Video Distribution by Category')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. Watch percentage distribution
        ax = axes[0, 1]
        ax.hist(interactions_df['watch_percentage'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Watch Percentage')
        ax.set_ylabel('Frequency')
        ax.set_title('Video Watch Percentage Distribution')
        ax.grid(True, alpha=0.3)

        # 3. Engagement distribution
        ax = axes[1, 0]
        engagement_counts = interactions_df['engagement'].value_counts().sort_index()
        engagement_counts.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
        ax.set_xlabel('Engagement Score')
        ax.set_ylabel('Frequency')
        ax.set_title('User Engagement Distribution')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)

        # 4. Implicit rating distribution
        ax = axes[1, 1]
        ax.hist(interactions_df['implicit_rating'], bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Implicit Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Implicit Rating Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/video_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/video_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Video Content Recommendation System using Embedding-Based Methods")
    print("=" * 70)

    # Initialize recommender
    recommender = VideoRecommender(embedding_dim=32, learning_rate=0.01, epochs=50)

    # Generate data
    videos_df, interactions_df = recommender.generate_data(n_videos=400, n_users=600)

    # Split data
    train_interactions = interactions_df.sample(frac=0.8, random_state=42)
    test_interactions = interactions_df.drop(train_interactions.index)

    print(f"\nTrain interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    # Train model
    n_users = interactions_df['user_id'].nunique()
    n_videos = interactions_df['video_id'].nunique()
    recommender.train_embeddings(train_interactions, n_users, n_videos)

    # Evaluate
    metrics = recommender.evaluate(train_interactions, test_interactions)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_user = train_interactions['user_id'].value_counts().head(1).index[0]
    print(f"\nRecommendations for User {test_user}:")

    user_watches = train_interactions[train_interactions['user_id'] == test_user]
    print(f"User has watched {len(user_watches)} videos")

    # Show user's top-rated videos
    top_watched = user_watches.nlargest(3, 'implicit_rating')
    print("\nUser's favorite videos:")
    for _, interaction in top_watched.iterrows():
        video = videos_df[videos_df['video_id'] == interaction['video_id']].iloc[0]
        print(f"  - {video['title'][:40]} ({video['category']}, {video['duration']}min)")
        print(f"    Watch: {interaction['watch_percentage']:.0%}, Rating: {interaction['implicit_rating']:.2f}")

    # Get recommendations
    recommendations = recommender.recommend_videos(test_user, n=10, exclude_watched=True, interactions_df=train_interactions)
    print("\nTop 10 Recommended Videos:")
    for i, (video_id, predicted_rating) in enumerate(recommendations, 1):
        video = videos_df[videos_df['video_id'] == video_id].iloc[0]
        print(f"{i}. {video['title'][:45]}")
        print(f"   Category: {video['category']} | Duration: {video['duration']}min | Predicted Rating: {predicted_rating:.2f}")

    # Similar videos
    print("\n" + "=" * 70)
    sample_video = videos_df[videos_df['views'] > videos_df['views'].median()].sample(1).iloc[0]
    print(f"Videos similar to: {sample_video['title']} ({sample_video['category']})")

    similar = recommender.find_similar_videos(sample_video['video_id'], n=5)
    for i, (video_id, similarity) in enumerate(similar, 1):
        video = videos_df[videos_df['video_id'] == video_id].iloc[0]
        print(f"{i}. {video['title'][:50]} ({video['category']}) - Similarity: {similarity:.4f}")

    # Visualize
    recommender.visualize_results(interactions_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
