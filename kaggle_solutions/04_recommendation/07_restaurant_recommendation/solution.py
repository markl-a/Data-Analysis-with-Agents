"""
Restaurant Recommendation System using Item-Based Collaborative Filtering
==========================================================================
This solution demonstrates a restaurant recommendation system using
item-based collaborative filtering with location-aware features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class RestaurantRecommender:
    """Restaurant recommendation system using item-based collaborative filtering."""

    def __init__(self):
        """Initialize the recommender."""
        self.user_item_matrix = None
        self.item_similarity = None
        self.restaurants_df = None

    def generate_data(self, n_restaurants=350, n_users=400):
        """
        Generate synthetic restaurant and user data.

        Args:
            n_restaurants: Number of restaurants
            n_users: Number of users

        Returns:
            restaurants_df, reviews_df
        """
        print("Generating synthetic restaurant data...")

        # Restaurant attributes
        cuisines = ['Italian', 'Chinese', 'Japanese', 'Mexican', 'Indian',
                   'Thai', 'French', 'American', 'Mediterranean', 'Korean']
        price_ranges = ['$', '$$', '$$$', '$$$$']
        neighborhoods = ['Downtown', 'Uptown', 'West Side', 'East Side',
                        'Midtown', 'Suburb A', 'Suburb B', 'Waterfront']

        # Generate restaurants
        restaurants = []
        for restaurant_id in range(n_restaurants):
            cuisine = np.random.choice(cuisines)

            # Location (latitude, longitude)
            lat = 40.7 + np.random.uniform(-0.1, 0.1)  # Around NYC
            lon = -74.0 + np.random.uniform(-0.1, 0.1)

            restaurants.append({
                'restaurant_id': restaurant_id,
                'name': f'Restaurant_{restaurant_id}',
                'cuisine': cuisine,
                'price_range': np.random.choice(price_ranges),
                'neighborhood': np.random.choice(neighborhoods),
                'latitude': lat,
                'longitude': lon,
                'avg_rating': 0,  # Will be calculated from reviews
                'num_reviews': 0
            })

        self.restaurants_df = pd.DataFrame(restaurants)

        # Generate user preferences
        user_cuisine_pref = {}
        user_locations = {}

        for user_id in range(n_users):
            # Users prefer 1-3 cuisines
            n_pref = np.random.randint(1, 4)
            user_cuisine_pref[user_id] = np.random.choice(cuisines, size=n_pref, replace=False)

            # User home location
            user_locations[user_id] = {
                'lat': 40.7 + np.random.uniform(-0.15, 0.15),
                'lon': -74.0 + np.random.uniform(-0.15, 0.15)
            }

        # Generate reviews
        reviews = []
        for user_id in range(n_users):
            # Each user reviews 3-15 restaurants
            n_reviews = np.random.randint(3, 16)

            for _ in range(n_reviews):
                # 70% chance of choosing preferred cuisine
                if np.random.random() < 0.7:
                    pref_restaurants = self.restaurants_df[
                        self.restaurants_df['cuisine'].isin(user_cuisine_pref[user_id])
                    ]
                    if len(pref_restaurants) > 0:
                        restaurant = pref_restaurants.sample(1).iloc[0]
                    else:
                        restaurant = self.restaurants_df.sample(1).iloc[0]
                else:
                    restaurant = self.restaurants_df.sample(1).iloc[0]

                # Calculate distance
                user_loc = user_locations[user_id]
                distance = np.sqrt((user_loc['lat'] - restaurant['latitude'])**2 +
                                 (user_loc['lon'] - restaurant['longitude'])**2) * 111  # km

                # Rating influenced by preference and distance
                is_preferred = restaurant['cuisine'] in user_cuisine_pref[user_id]
                is_close = distance < 5  # Within 5 km

                if is_preferred and is_close:
                    rating = np.random.randint(4, 6)
                elif is_preferred or is_close:
                    rating = np.random.randint(3, 5)
                else:
                    rating = np.random.randint(1, 5)

                reviews.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant['restaurant_id'],
                    'rating': rating,
                    'distance': distance,
                    'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
                })

        reviews_df = pd.DataFrame(reviews)
        reviews_df = reviews_df.drop_duplicates(['user_id', 'restaurant_id'])

        # Update restaurant statistics
        restaurant_stats = reviews_df.groupby('restaurant_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        restaurant_stats.columns = ['restaurant_id', 'avg_rating', 'num_reviews']

        self.restaurants_df = self.restaurants_df.merge(restaurant_stats, on='restaurant_id', how='left')
        self.restaurants_df['avg_rating'] = self.restaurants_df['avg_rating'].fillna(3.0)
        self.restaurants_df['num_reviews'] = self.restaurants_df['num_reviews'].fillna(0)

        print(f"Generated {len(self.restaurants_df)} restaurants and {len(reviews_df)} reviews")
        print(f"Cuisine distribution: {self.restaurants_df['cuisine'].value_counts().head().to_dict()}")

        return self.restaurants_df, reviews_df

    def build_model(self, reviews_df):
        """Build item-based collaborative filtering model."""
        print("\nBuilding item-based collaborative filtering model...")

        # Create user-item rating matrix
        self.user_item_matrix = reviews_df.pivot_table(
            index='user_id',
            columns='restaurant_id',
            values='rating',
            fill_value=0
        )

        print(f"User-item matrix shape: {self.user_item_matrix.shape}")

        # Calculate sparsity
        total_elements = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        non_zero_elements = (self.user_item_matrix != 0).sum().sum()
        sparsity = 1 - (non_zero_elements / total_elements)
        print(f"Matrix sparsity: {sparsity:.2%}")

        # Compute item-item similarity matrix
        # Transpose so items are rows
        item_matrix = self.user_item_matrix.T

        # Calculate cosine similarity between items
        self.item_similarity = cosine_similarity(item_matrix)

        print(f"Item similarity matrix shape: {self.item_similarity.shape}")

    def recommend_for_user(self, user_id, n=10, exclude_visited=True):
        """
        Recommend restaurants for a user using item-based CF.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_visited: Whether to exclude already visited restaurants

        Returns:
            List of (restaurant_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found. Recommending popular restaurants.")
            return self._recommend_popular(n)

        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]

        # Predict ratings for all items
        predictions = {}

        for item_id in range(len(self.restaurants_df)):
            if item_id not in self.user_item_matrix.columns:
                continue

            # Skip if already rated and we want to exclude
            if exclude_visited and user_ratings[item_id] > 0:
                continue

            # Get similarity scores to this item
            if item_id >= len(self.item_similarity):
                continue

            similarities = self.item_similarity[item_id]

            # Find items the user has rated
            rated_items = user_ratings[user_ratings > 0].index.values

            # Calculate predicted rating
            numerator = 0
            denominator = 0

            for rated_item in rated_items:
                if rated_item < len(similarities):
                    sim = similarities[rated_item]
                    if sim > 0:  # Only consider positive similarities
                        numerator += sim * user_ratings[rated_item]
                        denominator += abs(sim)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions[item_id] = predicted_rating

        # Sort by predicted rating
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return recommendations[:n]

    def recommend_similar_restaurants(self, restaurant_id, n=5):
        """
        Find restaurants similar to a given restaurant.

        Args:
            restaurant_id: Restaurant ID
            n: Number of similar restaurants

        Returns:
            List of (restaurant_id, similarity_score) tuples
        """
        if restaurant_id >= len(self.item_similarity):
            return []

        similarities = self.item_similarity[restaurant_id]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]  # Exclude self

        return [(idx, similarities[idx]) for idx in similar_indices]

    def _recommend_popular(self, n=10):
        """Recommend popular restaurants for cold start."""
        popular = self.restaurants_df.nlargest(n, 'num_reviews')
        return [(row['restaurant_id'], row['avg_rating']) for _, row in popular.iterrows()]

    def evaluate(self, train_df, test_df):
        """
        Evaluate recommendation quality.

        Args:
            train_df: Training reviews
            test_df: Test reviews

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")

        mae_scores = []
        rmse_scores = []
        precision_scores = []
        recall_scores = []

        k = 10

        for user_id in test_df['user_id'].unique():
            if user_id not in self.user_item_matrix.index:
                continue

            # Get recommendations
            recs = self.recommend_for_user(user_id, n=k, exclude_visited=True)
            rec_restaurants = [rid for rid, _ in recs]

            # Get actual high-rated restaurants (rating >= 4)
            actual_restaurants = test_df[
                (test_df['user_id'] == user_id) &
                (test_df['rating'] >= 4)
            ]['restaurant_id'].values

            if len(actual_restaurants) > 0:
                hits = len(set(rec_restaurants) & set(actual_restaurants))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_restaurants))

            # Rating prediction accuracy
            user_test = test_df[test_df['user_id'] == user_id]
            for _, row in user_test.iterrows():
                restaurant_id = row['restaurant_id']
                actual_rating = row['rating']

                # Predict rating
                all_recs = self.recommend_for_user(user_id, n=len(self.restaurants_df), exclude_visited=False)
                pred_dict = dict(all_recs)

                if restaurant_id in pred_dict:
                    predicted_rating = pred_dict[restaurant_id]
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

    def visualize_results(self, reviews_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Cuisine distribution
        ax = axes[0, 0]
        cuisine_counts = self.restaurants_df['cuisine'].value_counts()
        cuisine_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Cuisine')
        ax.set_ylabel('Number of Restaurants')
        ax.set_title('Restaurant Distribution by Cuisine')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. Rating distribution
        ax = axes[0, 1]
        ax.hist(reviews_df['rating'], bins=5, edgecolor='black', alpha=0.7, color='green', range=(0.5, 5.5))
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Rating Distribution')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)

        # 3. Price range distribution
        ax = axes[1, 0]
        price_counts = self.restaurants_df['price_range'].value_counts()
        price_order = ['$', '$$', '$$$', '$$$$']
        price_counts = price_counts.reindex(price_order, fill_value=0)
        price_counts.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
        ax.set_xlabel('Price Range')
        ax.set_ylabel('Number of Restaurants')
        ax.set_title('Restaurant Distribution by Price')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)

        # 4. Reviews per user
        ax = axes[1, 1]
        user_activity = reviews_df['user_id'].value_counts()
        ax.hist(user_activity, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Number of Reviews')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Review Activity Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/restaurant_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/restaurant_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Restaurant Recommendation System using Item-Based Collaborative Filtering")
    print("=" * 70)

    # Initialize recommender
    recommender = RestaurantRecommender()

    # Generate data
    restaurants_df, reviews_df = recommender.generate_data(n_restaurants=350, n_users=400)

    # Split data
    train_reviews = reviews_df.sample(frac=0.8, random_state=42)
    test_reviews = reviews_df.drop(train_reviews.index)

    print(f"\nTrain reviews: {len(train_reviews)}")
    print(f"Test reviews: {len(test_reviews)}")

    # Build model
    recommender.build_model(train_reviews)

    # Evaluate
    metrics = recommender.evaluate(train_reviews, test_reviews)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_user = train_reviews['user_id'].value_counts().head(1).index[0]
    print(f"\nRecommendations for User {test_user}:")

    user_reviews = train_reviews[train_reviews['user_id'] == test_user]
    print(f"User has reviewed {len(user_reviews)} restaurants")

    # Show user's top-rated restaurants
    top_rated = user_reviews.nlargest(3, 'rating')
    print("\nUser's favorite restaurants:")
    for _, review in top_rated.iterrows():
        restaurant = restaurants_df[restaurants_df['restaurant_id'] == review['restaurant_id']].iloc[0]
        print(f"  - {restaurant['name']} ({restaurant['cuisine']}, {restaurant['price_range']}) - Rated: {review['rating']}")

    # Get recommendations
    recommendations = recommender.recommend_for_user(test_user, n=10, exclude_visited=True)
    print("\nTop 10 Recommended Restaurants:")
    for i, (restaurant_id, predicted_rating) in enumerate(recommendations, 1):
        restaurant = restaurants_df[restaurants_df['restaurant_id'] == restaurant_id].iloc[0]
        print(f"{i}. {restaurant['name']} ({restaurant['cuisine']}, {restaurant['price_range']}, {restaurant['neighborhood']})")
        print(f"   Predicted Rating: {predicted_rating:.2f} | Avg Rating: {restaurant['avg_rating']:.2f}")

    # Similar restaurants
    print("\n" + "=" * 70)
    sample_restaurant = restaurants_df[restaurants_df['num_reviews'] > 5].sample(1).iloc[0]
    print(f"Restaurants similar to: {sample_restaurant['name']} ({sample_restaurant['cuisine']})")

    similar = recommender.recommend_similar_restaurants(sample_restaurant['restaurant_id'], n=5)
    for i, (restaurant_id, similarity) in enumerate(similar, 1):
        restaurant = restaurants_df[restaurants_df['restaurant_id'] == restaurant_id].iloc[0]
        print(f"{i}. {restaurant['name']} ({restaurant['cuisine']}, {restaurant['price_range']}) - Similarity: {similarity:.4f}")

    # Visualize
    recommender.visualize_results(reviews_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
