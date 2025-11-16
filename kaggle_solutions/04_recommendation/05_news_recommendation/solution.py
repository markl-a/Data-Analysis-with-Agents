"""
News Article Recommendation System using Content-Based Filtering
================================================================
This solution demonstrates a news recommendation system using TF-IDF
and cosine similarity for content-based filtering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class NewsRecommender:
    """News recommendation system using content-based filtering."""

    def __init__(self, max_features=500):
        """
        Initialize the recommender.

        Args:
            max_features: Maximum number of TF-IDF features
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_matrix = None
        self.article_similarity = None
        self.articles_df = None
        self.user_profiles = {}

    def generate_data(self, n_articles=500, n_users=300):
        """
        Generate synthetic news article data.

        Args:
            n_articles: Number of articles
            n_users: Number of users

        Returns:
            articles_df, interactions_df
        """
        print("Generating synthetic news data...")

        # Article categories and keywords
        categories = {
            'Politics': ['election', 'government', 'policy', 'congress', 'senate', 'president', 'vote', 'law'],
            'Technology': ['software', 'hardware', 'AI', 'machine learning', 'startup', 'innovation', 'app', 'digital'],
            'Sports': ['football', 'basketball', 'soccer', 'tennis', 'championship', 'tournament', 'team', 'player'],
            'Business': ['market', 'stock', 'economy', 'company', 'trade', 'finance', 'investment', 'revenue'],
            'Health': ['medicine', 'doctor', 'hospital', 'disease', 'treatment', 'wellness', 'nutrition', 'fitness'],
            'Entertainment': ['movie', 'music', 'celebrity', 'film', 'show', 'actor', 'artist', 'concert'],
            'Science': ['research', 'study', 'scientist', 'discovery', 'experiment', 'theory', 'data', 'climate'],
            'World': ['international', 'global', 'country', 'nation', 'conflict', 'diplomacy', 'crisis', 'treaty']
        }

        # Generate articles
        articles = []
        for article_id in range(n_articles):
            category = np.random.choice(list(categories.keys()))
            keywords = categories[category]

            # Generate article content
            num_keywords = np.random.randint(3, 8)
            selected_keywords = np.random.choice(keywords, size=num_keywords, replace=True)

            # Add some random common words
            common_words = ['news', 'report', 'today', 'new', 'latest', 'important', 'major', 'recent']
            num_common = np.random.randint(2, 5)
            selected_common = np.random.choice(common_words, size=num_common, replace=True)

            content = ' '.join(list(selected_keywords) + list(selected_common))

            articles.append({
                'article_id': article_id,
                'category': category,
                'content': content,
                'publish_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })

        self.articles_df = pd.DataFrame(articles)

        # Generate user reading history
        user_category_pref = {}
        for user_id in range(n_users):
            # Each user prefers 1-3 categories
            n_pref = np.random.randint(1, 4)
            user_category_pref[user_id] = np.random.choice(list(categories.keys()), size=n_pref, replace=False)

        # Generate interactions
        interactions = []
        for user_id in range(n_users):
            # Each user reads 5-30 articles
            n_reads = np.random.randint(5, 31)

            for _ in range(n_reads):
                # 80% chance of reading preferred category
                if np.random.random() < 0.8:
                    pref_articles = self.articles_df[
                        self.articles_df['category'].isin(user_category_pref[user_id])
                    ]
                    if len(pref_articles) > 0:
                        article = pref_articles.sample(1).iloc[0]
                    else:
                        article = self.articles_df.sample(1).iloc[0]
                else:
                    article = self.articles_df.sample(1).iloc[0]

                # Reading time and engagement
                is_preferred = article['category'] in user_category_pref[user_id]
                if is_preferred:
                    read_time = np.random.randint(60, 300)  # seconds
                    engagement = np.random.uniform(0.6, 1.0)
                else:
                    read_time = np.random.randint(10, 100)
                    engagement = np.random.uniform(0.1, 0.5)

                interactions.append({
                    'user_id': user_id,
                    'article_id': article['article_id'],
                    'read_time': read_time,
                    'engagement': engagement,
                    'timestamp': article['publish_date'] + pd.Timedelta(days=np.random.randint(0, 30))
                })

        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.drop_duplicates(['user_id', 'article_id'])

        print(f"Generated {len(self.articles_df)} articles and {len(interactions_df)} interactions")
        print(f"Categories: {self.articles_df['category'].value_counts().to_dict()}")

        return self.articles_df, interactions_df

    def build_content_model(self):
        """Build TF-IDF model and compute article similarities."""
        print("\nBuilding content-based model...")

        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.articles_df['content'])

        # Compute article similarity matrix
        self.article_similarity = cosine_similarity(self.tfidf_matrix)

        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def build_user_profiles(self, interactions_df):
        """Build user profiles based on reading history."""
        print("Building user profiles...")

        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]

            # Weight by engagement
            weighted_vectors = []
            weights = []

            for _, row in user_interactions.iterrows():
                article_idx = row['article_id']
                if article_idx < len(self.articles_df):
                    article_vector = self.tfidf_matrix[article_idx].toarray()[0]
                    weighted_vectors.append(article_vector * row['engagement'])
                    weights.append(row['engagement'])

            if weighted_vectors:
                # User profile is weighted average of read articles
                user_profile = np.sum(weighted_vectors, axis=0) / (np.sum(weights) + 1e-8)
                self.user_profiles[user_id] = user_profile

        print(f"Built profiles for {len(self.user_profiles)} users")

    def recommend_for_user(self, user_id, n=10, exclude_read=True, interactions_df=None):
        """
        Recommend articles for a user based on their profile.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_read: Whether to exclude already read articles
            interactions_df: User interaction data

        Returns:
            List of (article_id, similarity_score) tuples
        """
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found. Recommending trending articles.")
            return self._recommend_trending(n)

        user_profile = self.user_profiles[user_id]

        # Compute similarity between user profile and all articles
        article_scores = cosine_similarity([user_profile], self.tfidf_matrix)[0]

        # Create recommendations
        article_ids = np.arange(len(self.articles_df))
        recommendations = list(zip(article_ids, article_scores))

        # Exclude already read articles
        if exclude_read and interactions_df is not None:
            read_articles = set(interactions_df[interactions_df['user_id'] == user_id]['article_id'].values)
            recommendations = [(aid, score) for aid, score in recommendations if aid not in read_articles]

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:n]

    def recommend_similar_articles(self, article_id, n=5):
        """
        Recommend articles similar to a given article.

        Args:
            article_id: Article ID
            n: Number of recommendations

        Returns:
            List of (article_id, similarity_score) tuples
        """
        if article_id >= len(self.article_similarity):
            return []

        similarities = self.article_similarity[article_id]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]  # Exclude self

        return [(idx, similarities[idx]) for idx in similar_indices]

    def _recommend_trending(self, n=10):
        """Recommend trending articles for cold start."""
        # For simplicity, recommend most recent articles
        trending = self.articles_df.nlargest(n, 'publish_date')
        return [(row['article_id'], 1.0) for _, row in trending.iterrows()]

    def evaluate(self, interactions_df, test_interactions_df):
        """
        Evaluate recommendation quality.

        Args:
            interactions_df: Training interactions
            test_interactions_df: Test interactions

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")

        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        k = 10

        for user_id in test_interactions_df['user_id'].unique():
            if user_id not in self.user_profiles:
                continue

            # Get recommendations
            recs = self.recommend_for_user(user_id, n=k, exclude_read=True, interactions_df=interactions_df)
            rec_articles = [article_id for article_id, _ in recs]

            # Get actual engaged articles (engagement > 0.5)
            actual_articles = test_interactions_df[
                (test_interactions_df['user_id'] == user_id) &
                (test_interactions_df['engagement'] > 0.5)
            ]['article_id'].values

            if len(actual_articles) > 0:
                hits = len(set(rec_articles) & set(actual_articles))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_articles))

                # NDCG calculation
                dcg = sum([1.0 / np.log2(i + 2) for i, aid in enumerate(rec_articles) if aid in actual_articles])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_articles), k))])
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        metrics = {
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0,
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0
        }

        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")

        return metrics

    def visualize_results(self, interactions_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Category distribution
        ax = axes[0, 0]
        category_counts = self.articles_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of Articles')
        ax.set_title('Article Distribution by Category')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. Reading time distribution
        ax = axes[0, 1]
        ax.hist(interactions_df['read_time'], bins=50, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Read Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Reading Time Distribution')
        ax.grid(True, alpha=0.3)

        # 3. Engagement distribution
        ax = axes[1, 0]
        ax.hist(interactions_df['engagement'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Engagement Score')
        ax.set_ylabel('Frequency')
        ax.set_title('User Engagement Distribution')
        ax.grid(True, alpha=0.3)

        # 4. User activity
        ax = axes[1, 1]
        user_activity = interactions_df['user_id'].value_counts()
        ax.hist(user_activity, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Number of Articles Read')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Reading Activity Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/news_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/news_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("News Article Recommendation System using Content-Based Filtering")
    print("=" * 70)

    # Initialize recommender
    recommender = NewsRecommender(max_features=500)

    # Generate data
    articles_df, interactions_df = recommender.generate_data(n_articles=500, n_users=300)

    # Split interactions
    train_interactions = interactions_df.sample(frac=0.8, random_state=42)
    test_interactions = interactions_df.drop(train_interactions.index)

    print(f"\nTrain interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")

    # Build model
    recommender.build_content_model()
    recommender.build_user_profiles(train_interactions)

    # Evaluate
    metrics = recommender.evaluate(train_interactions, test_interactions)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_user = train_interactions['user_id'].value_counts().head(1).index[0]
    print(f"\nRecommendations for User {test_user}:")

    user_reads = train_interactions[train_interactions['user_id'] == test_user]
    print(f"User has read {len(user_reads)} articles")
    print(f"Preferred categories: {articles_df[articles_df['article_id'].isin(user_reads['article_id'])]['category'].value_counts().head(3).to_dict()}")

    recommendations = recommender.recommend_for_user(test_user, n=10, exclude_read=True, interactions_df=train_interactions)
    print("\nTop 10 Recommended Articles:")
    for i, (article_id, score) in enumerate(recommendations, 1):
        article = articles_df[articles_df['article_id'] == article_id].iloc[0]
        print(f"{i}. Article {article_id} ({article['category']}) - Score: {score:.4f}")

    # Similar articles
    print("\n" + "=" * 70)
    sample_article = articles_df.sample(1).iloc[0]
    print(f"Articles similar to: Article {sample_article['article_id']} ({sample_article['category']})")
    similar = recommender.recommend_similar_articles(sample_article['article_id'], n=5)
    for i, (article_id, score) in enumerate(similar, 1):
        article = articles_df[articles_df['article_id'] == article_id].iloc[0]
        print(f"{i}. Article {article_id} ({article['category']}) - Similarity: {score:.4f}")

    # Visualize
    recommender.visualize_results(interactions_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
