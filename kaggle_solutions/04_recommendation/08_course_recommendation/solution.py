"""
Online Course Recommendation System using User-Based Collaborative Filtering
============================================================================
This solution demonstrates a course recommendation system using
user-based collaborative filtering and skill progression modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class CourseRecommender:
    """Course recommendation system using user-based collaborative filtering."""

    def __init__(self, n_neighbors=20):
        """
        Initialize recommender.

        Args:
            n_neighbors: Number of similar users to consider
        """
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        self.courses_df = None

    def generate_data(self, n_courses=300, n_users=500):
        """
        Generate synthetic course and enrollment data.

        Args:
            n_courses: Number of courses
            n_users: Number of users

        Returns:
            courses_df, enrollments_df
        """
        print("Generating synthetic course data...")

        # Course categories and levels
        categories = ['Programming', 'Data Science', 'Web Development', 'Mobile Development',
                     'Business', 'Marketing', 'Design', 'Personal Development']
        levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        durations = [2, 4, 6, 8, 10, 12]  # hours

        # Generate courses
        courses = []
        for course_id in range(n_courses):
            category = np.random.choice(categories)
            level = np.random.choice(levels)

            # Create learning path dependencies
            if level == 'Beginner':
                prerequisites = []
            elif level == 'Intermediate':
                # Might have 0-1 prerequisite
                if np.random.random() < 0.6:
                    beginner_courses = [c['course_id'] for c in courses if c['level'] == 'Beginner' and c['category'] == category]
                    prerequisites = [np.random.choice(beginner_courses)] if beginner_courses else []
                else:
                    prerequisites = []
            else:
                # Advanced/Expert more likely to have prerequisites
                if np.random.random() < 0.8:
                    lower_courses = [c['course_id'] for c in courses if c['category'] == category and c['level'] in ['Beginner', 'Intermediate']]
                    prerequisites = list(np.random.choice(lower_courses, size=min(2, len(lower_courses)), replace=False)) if lower_courses else []
                else:
                    prerequisites = []

            courses.append({
                'course_id': course_id,
                'title': f'Course_{course_id}_{category}_{level}',
                'category': category,
                'level': level,
                'duration': np.random.choice(durations),
                'prerequisites': prerequisites,
                'avg_rating': 0,
                'num_enrollments': 0
            })

        self.courses_df = pd.DataFrame(courses)

        # Generate user learning paths
        enrollments = []
        for user_id in range(n_users):
            # User's focus areas (1-2 categories)
            n_interests = np.random.randint(1, 3)
            user_interests = np.random.choice(categories, size=n_interests, replace=False)

            # User's current skill level
            user_level_progress = {cat: np.random.choice(['Beginner', 'Intermediate']) for cat in user_interests}

            # Each user enrolls in 3-12 courses
            n_enrollments = np.random.randint(3, 13)

            for _ in range(n_enrollments):
                # 80% chance of taking course in interest area
                if np.random.random() < 0.8 and len(user_interests) > 0:
                    category = np.random.choice(user_interests)
                    current_level = user_level_progress.get(category, 'Beginner')

                    # Select appropriate level courses
                    level_order = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
                    max_level_idx = min(level_order.index(current_level) + 1, len(level_order) - 1)
                    suitable_levels = level_order[:max_level_idx + 1]

                    suitable_courses = self.courses_df[
                        (self.courses_df['category'] == category) &
                        (self.courses_df['level'].isin(suitable_levels))
                    ]
                else:
                    suitable_courses = self.courses_df

                if len(suitable_courses) == 0:
                    continue

                course = suitable_courses.sample(1).iloc[0]

                # Completion and rating based on level match
                is_interested = course['category'] in user_interests
                is_right_level = course['level'] in ['Beginner', 'Intermediate']

                if is_interested and is_right_level:
                    completion = np.random.uniform(0.7, 1.0)
                    rating = np.random.randint(4, 6)
                elif is_interested or is_right_level:
                    completion = np.random.uniform(0.4, 0.9)
                    rating = np.random.randint(3, 5)
                else:
                    completion = np.random.uniform(0.1, 0.6)
                    rating = np.random.randint(2, 5)

                enrollments.append({
                    'user_id': user_id,
                    'course_id': course['course_id'],
                    'rating': rating,
                    'completion': completion,
                    'time_spent': course['duration'] * completion,
                    'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
                })

                # Update user progress
                if completion > 0.8 and course['category'] in user_level_progress:
                    current = user_level_progress[course['category']]
                    level_order = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
                    if current != 'Expert':
                        next_idx = level_order.index(current) + 1
                        user_level_progress[course['category']] = level_order[next_idx]

        enrollments_df = pd.DataFrame(enrollments)
        enrollments_df = enrollments_df.drop_duplicates(['user_id', 'course_id'])

        # Update course statistics
        course_stats = enrollments_df.groupby('course_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        course_stats.columns = ['course_id', 'avg_rating', 'num_enrollments']

        self.courses_df = self.courses_df.merge(course_stats, on='course_id', how='left')
        self.courses_df['avg_rating'] = self.courses_df['avg_rating'].fillna(3.5)
        self.courses_df['num_enrollments'] = self.courses_df['num_enrollments'].fillna(0)

        print(f"Generated {len(self.courses_df)} courses and {len(enrollments_df)} enrollments")
        print(f"Category distribution: {self.courses_df['category'].value_counts().to_dict()}")

        return self.courses_df, enrollments_df

    def build_model(self, enrollments_df):
        """Build user-based collaborative filtering model."""
        print("\nBuilding user-based collaborative filtering model...")

        # Create user-item rating matrix
        self.user_item_matrix = enrollments_df.pivot_table(
            index='user_id',
            columns='course_id',
            values='rating',
            fill_value=0
        )

        print(f"User-item matrix shape: {self.user_item_matrix.shape}")

        # Calculate sparsity
        total_elements = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        non_zero_elements = (self.user_item_matrix != 0).sum().sum()
        sparsity = 1 - (non_zero_elements / total_elements)
        print(f"Matrix sparsity: {sparsity:.2%}")

        # Compute user-user similarity matrix
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        print(f"User similarity matrix shape: {self.user_similarity.shape}")

    def recommend_for_user(self, user_id, n=10, exclude_taken=True):
        """
        Recommend courses for a user using user-based CF.

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_taken: Whether to exclude already taken courses

        Returns:
            List of (course_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found. Recommending popular courses.")
            return self._recommend_popular(n)

        # Find similar users
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarities = self.user_similarity[user_idx]

        # Get top-k similar users (excluding self)
        similar_user_indices = np.argsort(similarities)[::-1][1:self.n_neighbors + 1]

        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]

        # Predict ratings for all courses
        predictions = {}

        for course_id in self.user_item_matrix.columns:
            # Skip if already taken and we want to exclude
            if exclude_taken and user_ratings[course_id] > 0:
                continue

            # Calculate weighted average of similar users' ratings
            numerator = 0
            denominator = 0

            for sim_user_idx in similar_user_indices:
                sim_user_id = self.user_item_matrix.index[sim_user_idx]
                sim_user_rating = self.user_item_matrix.loc[sim_user_id, course_id]

                if sim_user_rating > 0:
                    sim_score = similarities[sim_user_idx]
                    numerator += sim_score * sim_user_rating
                    denominator += abs(sim_score)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions[course_id] = predicted_rating

        # Sort by predicted rating
        recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return recommendations[:n]

    def _recommend_popular(self, n=10):
        """Recommend popular courses for cold start."""
        popular = self.courses_df.nlargest(n, 'num_enrollments')
        return [(row['course_id'], row['avg_rating']) for _, row in popular.iterrows()]

    def evaluate(self, train_df, test_df):
        """Evaluate recommendation quality."""
        print("\nEvaluating model...")

        mae_scores = []
        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        k = 10

        for user_id in test_df['user_id'].unique():
            if user_id not in self.user_item_matrix.index:
                continue

            # Get recommendations
            recs = self.recommend_for_user(user_id, n=k, exclude_taken=True)
            rec_courses = [cid for cid, _ in recs]

            # Get actual high-rated courses (rating >= 4)
            actual_courses = test_df[
                (test_df['user_id'] == user_id) &
                (test_df['rating'] >= 4)
            ]['course_id'].values

            if len(actual_courses) > 0:
                hits = len(set(rec_courses) & set(actual_courses))
                precision_scores.append(hits / k if k > 0 else 0)
                recall_scores.append(hits / len(actual_courses))

                # NDCG
                dcg = sum([1.0 / np.log2(i + 2) for i, cid in enumerate(rec_courses) if cid in actual_courses])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_courses), k))])
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

            # MAE for rating prediction
            user_test = test_df[test_df['user_id'] == user_id]
            for _, row in user_test.iterrows():
                course_id = row['course_id']
                actual_rating = row['rating']

                all_recs = self.recommend_for_user(user_id, n=len(self.courses_df), exclude_taken=False)
                pred_dict = dict(all_recs)

                if course_id in pred_dict:
                    predicted_rating = pred_dict[course_id]
                    mae_scores.append(abs(actual_rating - predicted_rating))

        metrics = {
            'mae': np.mean(mae_scores) if mae_scores else float('inf'),
            'precision@10': np.mean(precision_scores) if precision_scores else 0,
            'recall@10': np.mean(recall_scores) if recall_scores else 0,
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0
        }

        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Precision@10: {metrics['precision@10']:.4f}")
        print(f"Recall@10: {metrics['recall@10']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")

        return metrics

    def visualize_results(self, enrollments_df):
        """Create visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Category distribution
        ax = axes[0, 0]
        category_counts = self.courses_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of Courses')
        ax.set_title('Course Distribution by Category')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 2. Completion rate distribution
        ax = axes[0, 1]
        ax.hist(enrollments_df['completion'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Completion Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Course Completion Distribution')
        ax.grid(True, alpha=0.3)

        # 3. Rating distribution
        ax = axes[1, 0]
        ax.hist(enrollments_df['rating'], bins=5, edgecolor='black', alpha=0.7, color='orange', range=(0.5, 5.5))
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Course Rating Distribution')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)

        # 4. Level distribution
        ax = axes[1, 1]
        level_counts = self.courses_df['level'].value_counts()
        level_order = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
        level_counts = level_counts.reindex(level_order, fill_value=0)
        level_counts.plot(kind='bar', ax=ax, color='purple', edgecolor='black')
        ax.set_xlabel('Level')
        ax.set_ylabel('Number of Courses')
        ax.set_title('Course Distribution by Level')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/course_recommendation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to /tmp/course_recommendation_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Online Course Recommendation System using User-Based Collaborative Filtering")
    print("=" * 70)

    # Initialize recommender
    recommender = CourseRecommender(n_neighbors=20)

    # Generate data
    courses_df, enrollments_df = recommender.generate_data(n_courses=300, n_users=500)

    # Split data
    train_enrollments = enrollments_df.sample(frac=0.8, random_state=42)
    test_enrollments = enrollments_df.drop(train_enrollments.index)

    print(f"\nTrain enrollments: {len(train_enrollments)}")
    print(f"Test enrollments: {len(test_enrollments)}")

    # Build model
    recommender.build_model(train_enrollments)

    # Evaluate
    metrics = recommender.evaluate(train_enrollments, test_enrollments)

    # Example recommendations
    print("\n" + "=" * 70)
    print("Example Recommendations")
    print("=" * 70)

    test_user = train_enrollments['user_id'].value_counts().head(1).index[0]
    print(f"\nRecommendations for User {test_user}:")

    user_enrollments = train_enrollments[train_enrollments['user_id'] == test_user]
    print(f"User has enrolled in {len(user_enrollments)} courses")

    # Show user's completed courses
    completed = user_enrollments[user_enrollments['completion'] > 0.8].nlargest(3, 'rating')
    print("\nUser's top completed courses:")
    for _, enrollment in completed.iterrows():
        course = courses_df[courses_df['course_id'] == enrollment['course_id']].iloc[0]
        print(f"  - {course['title'][:40]} ({course['level']}) - Rating: {enrollment['rating']}, Completion: {enrollment['completion']:.0%}")

    # Get recommendations
    recommendations = recommender.recommend_for_user(test_user, n=10, exclude_taken=True)
    print("\nTop 10 Recommended Courses:")
    for i, (course_id, predicted_rating) in enumerate(recommendations, 1):
        course = courses_df[courses_df['course_id'] == course_id].iloc[0]
        print(f"{i}. {course['title'][:50]}")
        print(f"   Level: {course['level']} | Category: {course['category']} | Predicted Rating: {predicted_rating:.2f}")

    # Visualize
    recommender.visualize_results(enrollments_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
