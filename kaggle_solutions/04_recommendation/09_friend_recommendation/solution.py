"""
Social Network Friend Recommendation System using Graph-Based Methods
=====================================================================
This solution demonstrates a friend recommendation system using
graph-based algorithms including common neighbors, Jaccard similarity,
and Adamic-Adar index.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class FriendRecommender:
    """Friend recommendation system using graph-based methods."""

    def __init__(self):
        """Initialize the recommender."""
        self.graph = defaultdict(set)  # Adjacency list
        self.users_df = None

    def generate_data(self, n_users=500, avg_friends=20):
        """
        Generate synthetic social network data.

        Args:
            n_users: Number of users
            avg_friends: Average number of friends per user

        Returns:
            users_df, friendships_df
        """
        print("Generating synthetic social network data...")

        # User attributes
        interests = ['Sports', 'Music', 'Technology', 'Travel', 'Food', 'Art',
                    'Gaming', 'Reading', 'Movies', 'Fitness', 'Photography', 'Cooking']
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                    'San Francisco', 'Seattle', 'Boston', 'Austin', 'Denver']
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']

        # Generate users
        users = []
        for user_id in range(n_users):
            # Each user has 2-4 interests
            n_interests = np.random.randint(2, 5)
            user_interests = list(np.random.choice(interests, size=n_interests, replace=False))

            users.append({
                'user_id': user_id,
                'interests': user_interests,
                'location': np.random.choice(locations),
                'age_group': np.random.choice(age_groups),
                'join_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 1460))
            })

        self.users_df = pd.DataFrame(users)

        # Generate friendships using preferential attachment and homophily
        friendships = []
        user_friend_counts = defaultdict(int)

        for user_id in range(n_users):
            user = self.users_df.iloc[user_id]

            # Number of friends for this user (follows power law-ish distribution)
            n_friends = int(np.random.gamma(shape=2, scale=avg_friends/2))
            n_friends = min(n_friends, n_users - 1)  # Can't have more friends than users

            # Already existing friends
            current_friends = len(self.graph[user_id])
            n_new_friends = max(0, n_friends - current_friends)

            for _ in range(n_new_friends):
                # Select friend with probability based on:
                # 1. Common interests (homophily)
                # 2. Same location (proximity)
                # 3. Already popular users (preferential attachment)

                # Calculate scores for all potential friends
                candidates = []
                for candidate_id in range(n_users):
                    if candidate_id == user_id or candidate_id in self.graph[user_id]:
                        continue  # Skip self and existing friends

                    candidate = self.users_df.iloc[candidate_id]

                    # Homophily score
                    common_interests = len(set(user['interests']) & set(candidate['interests']))
                    same_location = int(user['location'] == candidate['location'])
                    same_age = int(user['age_group'] == candidate['age_group'])

                    # Preferential attachment (popular users more likely to be chosen)
                    popularity = user_friend_counts[candidate_id] + 1

                    # Combined score
                    score = (common_interests * 3 +
                           same_location * 2 +
                           same_age * 1 +
                           np.log(popularity))

                    candidates.append((candidate_id, score))

                if not candidates:
                    break

                # Sample friend based on scores
                candidate_ids, scores = zip(*candidates)
                scores = np.array(scores)
                scores = np.exp(scores)  # Exponentiate for stronger preferences
                probabilities = scores / scores.sum()

                friend_id = np.random.choice(candidate_ids, p=probabilities)

                # Add bidirectional friendship
                self.graph[user_id].add(friend_id)
                self.graph[friend_id].add(user_id)
                user_friend_counts[user_id] += 1
                user_friend_counts[friend_id] += 1

                friendships.append({
                    'user_id': min(user_id, friend_id),
                    'friend_id': max(user_id, friend_id),
                    'timestamp': user['join_date'] + pd.Timedelta(days=np.random.randint(0, 365))
                })

        friendships_df = pd.DataFrame(friendships).drop_duplicates()

        print(f"Generated {len(self.users_df)} users and {len(friendships_df)} friendships")
        print(f"Average friends per user: {len(friendships_df) * 2 / len(self.users_df):.1f}")
        print(f"Network density: {len(friendships_df) / (n_users * (n_users - 1) / 2):.4f}")

        return self.users_df, friendships_df

    def common_neighbors(self, user_id, candidate_id):
        """
        Calculate number of common neighbors.

        Args:
            user_id: User ID
            candidate_id: Candidate friend ID

        Returns:
            Number of common neighbors
        """
        if user_id not in self.graph or candidate_id not in self.graph:
            return 0

        common = self.graph[user_id] & self.graph[candidate_id]
        return len(common)

    def jaccard_similarity(self, user_id, candidate_id):
        """
        Calculate Jaccard similarity of friend networks.

        Jaccard = |A ∩ B| / |A ∪ B|

        Args:
            user_id: User ID
            candidate_id: Candidate friend ID

        Returns:
            Jaccard similarity score
        """
        if user_id not in self.graph or candidate_id not in self.graph:
            return 0.0

        intersection = self.graph[user_id] & self.graph[candidate_id]
        union = self.graph[user_id] | self.graph[candidate_id]

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def adamic_adar(self, user_id, candidate_id):
        """
        Calculate Adamic-Adar index.

        AA = Σ 1/log(|neighbors(z)|) for z in common_neighbors

        Gives more weight to common neighbors with fewer friends.

        Args:
            user_id: User ID
            candidate_id: Candidate friend ID

        Returns:
            Adamic-Adar score
        """
        if user_id not in self.graph or candidate_id not in self.graph:
            return 0.0

        common = self.graph[user_id] & self.graph[candidate_id]

        score = 0.0
        for neighbor in common:
            neighbor_degree = len(self.graph[neighbor])
            if neighbor_degree > 1:
                score += 1.0 / np.log(neighbor_degree)

        return score

    def content_similarity(self, user_id, candidate_id):
        """
        Calculate content-based similarity (interests, location, age).

        Args:
            user_id: User ID
            candidate_id: Candidate friend ID

        Returns:
            Content similarity score
        """
        if user_id >= len(self.users_df) or candidate_id >= len(self.users_df):
            return 0.0

        user = self.users_df.iloc[user_id]
        candidate = self.users_df.iloc[candidate_id]

        # Interest similarity
        common_interests = len(set(user['interests']) & set(candidate['interests']))
        total_interests = len(set(user['interests']) | set(candidate['interests']))
        interest_sim = common_interests / total_interests if total_interests > 0 else 0

        # Location similarity
        location_sim = 1.0 if user['location'] == candidate['location'] else 0.0

        # Age group similarity
        age_sim = 1.0 if user['age_group'] == candidate['age_group'] else 0.0

        # Weighted combination
        return 0.5 * interest_sim + 0.3 * location_sim + 0.2 * age_sim

    def hybrid_score(self, user_id, candidate_id):
        """
        Calculate hybrid recommendation score.

        Combines graph-based and content-based signals.

        Args:
            user_id: User ID
            candidate_id: Candidate friend ID

        Returns:
            Hybrid score
        """
        # Graph-based scores
        cn = self.common_neighbors(user_id, candidate_id)
        jaccard = self.jaccard_similarity(user_id, candidate_id)
        aa = self.adamic_adar(user_id, candidate_id)

        # Content-based score
        content = self.content_similarity(user_id, candidate_id)

        # Normalize and combine
        # Adamic-Adar typically ranges 0-10, common neighbors 0-50
        normalized_aa = min(aa / 10.0, 1.0)
        normalized_cn = min(cn / 20.0, 1.0)

        # Weighted combination
        score = (0.3 * normalized_cn +
                0.3 * normalized_aa +
                0.2 * jaccard +
                0.2 * content)

        return score

    def recommend_friends(self, user_id, n=10, method='hybrid'):
        """
        Recommend friends for a user.

        Args:
            user_id: User ID
            n: Number of recommendations
            method: 'common_neighbors', 'jaccard', 'adamic_adar', 'content', or 'hybrid'

        Returns:
            List of (candidate_id, score) tuples
        """
        if user_id not in self.graph:
            return []

        # Get current friends
        current_friends = self.graph[user_id]

        # Score all non-friend candidates
        candidates = []
        for candidate_id in range(len(self.users_df)):
            if candidate_id == user_id or candidate_id in current_friends:
                continue

            if method == 'common_neighbors':
                score = self.common_neighbors(user_id, candidate_id)
            elif method == 'jaccard':
                score = self.jaccard_similarity(user_id, candidate_id)
            elif method == 'adamic_adar':
                score = self.adamic_adar(user_id, candidate_id)
            elif method == 'content':
                score = self.content_similarity(user_id, candidate_id)
            else:  # hybrid
                score = self.hybrid_score(user_id, candidate_id)

            if score > 0:
                candidates.append((candidate_id, score))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:n]

    def evaluate(self, friendships_df, test_friendships_df):
        """
        Evaluate recommendation quality.

        Args:
            friendships_df: Training friendships
            test_friendships_df: Test friendships (held-out)

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")

        precision_scores = []
        recall_scores = []

        k = 10

        # For each user with new friendships in test set
        test_users = set(test_friendships_df['user_id'].unique()) | set(test_friendships_df['friend_id'].unique())

        for user_id in test_users:
            if user_id not in self.graph:
                continue

            # Get recommendations
            recs = self.recommend_friends(user_id, n=k, method='hybrid')
            rec_users = [uid for uid, _ in recs]

            # Get actual new friends from test set
            actual_friends = set()
            for _, row in test_friendships_df.iterrows():
                if row['user_id'] == user_id:
                    actual_friends.add(row['friend_id'])
                elif row['friend_id'] == user_id:
                    actual_friends.add(row['user_id'])

            # Remove existing friends
            actual_friends = actual_friends - self.graph[user_id]

            if len(actual_friends) > 0:
                hits = len(set(rec_users) & actual_friends)
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_friends))

        metrics = {
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0
        }

        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")

        return metrics

    def visualize_results(self, friendships_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Friend count distribution
        ax = axes[0, 0]
        friend_counts = [len(friends) for friends in self.graph.values()]
        ax.hist(friend_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of Friends')
        ax.set_ylabel('Number of Users')
        ax.set_title('Friend Count Distribution (Power Law)')
        ax.grid(True, alpha=0.3)

        # 2. Interest distribution
        ax = axes[0, 1]
        all_interests = [interest for interests in self.users_df['interests'] for interest in interests]
        interest_counts = pd.Series(all_interests).value_counts()
        interest_counts.plot(kind='bar', ax=ax, color='green', edgecolor='black')
        ax.set_xlabel('Interest')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Interest Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 3. Location distribution
        ax = axes[1, 0]
        location_counts = self.users_df['location'].value_counts()
        location_counts.plot(kind='bar', ax=ax, color='orange', edgecolor='black')
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Location Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 4. Age group distribution
        ax = axes[1, 1]
        age_counts = self.users_df['age_group'].value_counts()
        age_order = ['18-25', '26-35', '36-45', '46-55', '56+']
        age_counts = age_counts.reindex(age_order, fill_value=0)
        age_counts.plot(kind='bar', ax=ax, color='purple', edgecolor='black')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Age Distribution')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/friend_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/friend_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Social Network Friend Recommendation System using Graph-Based Methods")
    print("=" * 70)

    # Initialize recommender
    recommender = FriendRecommender()

    # Generate data
    users_df, friendships_df = recommender.generate_data(n_users=500, avg_friends=20)

    # Split friendships (simulate future friendships)
    train_friendships = friendships_df.sample(frac=0.9, random_state=42)
    test_friendships = friendships_df.drop(train_friendships.index)

    # Rebuild graph with only training friendships
    recommender.graph = defaultdict(set)
    for _, row in train_friendships.iterrows():
        recommender.graph[row['user_id']].add(row['friend_id'])
        recommender.graph[row['friend_id']].add(row['user_id'])

    print(f"\nTrain friendships: {len(train_friendships)}")
    print(f"Test friendships: {len(test_friendships)}")

    # Evaluate
    metrics = recommender.evaluate(train_friendships, test_friendships)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Friend Recommendations")
    print("=" * 70)

    test_user = np.random.choice([uid for uid in range(len(users_df)) if len(recommender.graph[uid]) > 5])
    user = users_df.iloc[test_user]

    print(f"\nRecommendations for User {test_user}:")
    print(f"Interests: {', '.join(user['interests'])}")
    print(f"Location: {user['location']}")
    print(f"Age Group: {user['age_group']}")
    print(f"Current friends: {len(recommender.graph[test_user])}")

    # Show current friends' interests
    friend_interests = []
    for friend_id in list(recommender.graph[test_user])[:3]:
        friend = users_df.iloc[friend_id]
        friend_interests.extend(friend['interests'])

    print(f"Friends' popular interests: {', '.join([f'{k}({v})' for k, v in Counter(friend_interests).most_common(3)])}")

    # Get recommendations with different methods
    print("\nTop 10 Friend Recommendations (Hybrid Method):")
    recommendations = recommender.recommend_friends(test_user, n=10, method='hybrid')

    for i, (candidate_id, score) in enumerate(recommendations, 1):
        candidate = users_df.iloc[candidate_id]
        cn = recommender.common_neighbors(test_user, candidate_id)
        common_ints = set(user['interests']) & set(candidate['interests'])

        print(f"{i}. User {candidate_id} (Score: {score:.4f})")
        print(f"   Interests: {', '.join(candidate['interests'])}")
        print(f"   Common neighbors: {cn}, Common interests: {', '.join(common_ints) if common_ints else 'None'}")

    # Compare methods
    print("\n" + "=" * 70)
    print("Comparison of Different Methods")
    print("=" * 70)

    for method in ['common_neighbors', 'jaccard', 'adamic_adar', 'content', 'hybrid']:
        recs = recommender.recommend_friends(test_user, n=5, method=method)
        print(f"\n{method.upper()}:")
        for i, (candidate_id, score) in enumerate(recs[:3], 1):
            print(f"  {i}. User {candidate_id} - Score: {score:.4f}")

    # Visualize
    recommender.visualize_results(friendships_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
