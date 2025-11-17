"""
Knowledge-Based Recommendation System
=====================================

This solution implements knowledge-based recommendations using rule-based
constraints, user preferences, and domain knowledge.

Author: Kaggle Solutions
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class KnowledgeBasedRecommender:
    """Knowledge-based recommender using rules and constraints."""

    def __init__(self):
        """Initialize knowledge-based recommender."""
        self.items_df = None
        self.rules = []
        self.scaler = StandardScaler()

    def fit(self, items_df):
        """Fit the recommender with item catalog."""
        self.items_df = items_df.copy()

        # Normalize numerical features
        numerical_cols = items_df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'item_id']

        if len(numerical_cols) > 0:
            self.items_df[numerical_cols] = self.scaler.fit_transform(
                items_df[numerical_cols]
            )

    def add_rule(self, rule_func, weight=1.0, name="Custom Rule"):
        """Add a recommendation rule."""
        self.rules.append({'func': rule_func, 'weight': weight, 'name': name})

    def recommend(self, user_preferences, n_recommendations=10):
        """Generate recommendations based on user preferences and rules."""
        if self.items_df is None:
            return []

        scores = np.zeros(len(self.items_df))

        # Apply each rule
        for rule in self.rules:
            rule_scores = rule['func'](self.items_df, user_preferences)
            scores += rule['weight'] * rule_scores

        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]

        recommendations = [
            (self.items_df.iloc[idx]['item_id'], scores[idx])
            for idx in top_indices
        ]

        return recommendations

    def explain_recommendation(self, item_id, user_preferences):
        """Explain why an item was recommended."""
        item = self.items_df[self.items_df['item_id'] == item_id].iloc[0]

        explanations = []
        for rule in self.rules:
            rule_scores = rule['func'](self.items_df, user_preferences)
            item_idx = self.items_df[self.items_df['item_id'] == item_id].index[0]
            score = rule_scores[item_idx]

            if score > 0:
                explanations.append({
                    'rule': rule['name'],
                    'score': score,
                    'weight': rule['weight']
                })

        return explanations


def generate_synthetic_data(n_items=300):
    """Generate synthetic item catalog."""
    np.random.seed(42)

    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    items = []
    for i in range(n_items):
        items.append({
            'item_id': i,
            'category': categories[i % len(categories)],
            'brand': brands[i % len(brands)],
            'price': np.random.uniform(10, 500),
            'rating': np.random.uniform(3, 5),
            'num_reviews': np.random.randint(0, 1000),
            'popularity': np.random.uniform(0, 100),
            'year': 2020 + (i % 5)
        })

    return pd.DataFrame(items)


def category_match_rule(items_df, user_preferences):
    """Rule: Match user preferred category."""
    scores = np.zeros(len(items_df))

    if 'preferred_category' in user_preferences:
        preferred = user_preferences['preferred_category']
        scores = (items_df['category'] == preferred).astype(float)

    return scores


def price_range_rule(items_df, user_preferences):
    """Rule: Match user price range."""
    scores = np.zeros(len(items_df))

    if 'min_price' in user_preferences and 'max_price' in user_preferences:
        min_price = user_preferences['min_price']
        max_price = user_preferences['max_price']

        in_range = (items_df['price'] >= min_price) & (items_df['price'] <= max_price)
        scores = in_range.astype(float)

    return scores


def rating_threshold_rule(items_df, user_preferences):
    """Rule: Minimum rating threshold."""
    scores = np.zeros(len(items_df))

    if 'min_rating' in user_preferences:
        min_rating = user_preferences['min_rating']
        scores = (items_df['rating'] >= min_rating).astype(float)

    return scores


def brand_preference_rule(items_df, user_preferences):
    """Rule: Preferred brands."""
    scores = np.zeros(len(items_df))

    if 'preferred_brands' in user_preferences:
        preferred_brands = user_preferences['preferred_brands']
        scores = items_df['brand'].isin(preferred_brands).astype(float)

    return scores


def popularity_rule(items_df, user_preferences):
    """Rule: Popularity score."""
    # Normalize popularity
    max_pop = items_df['popularity'].max()
    return items_df['popularity'] / max_pop if max_pop > 0 else np.zeros(len(items_df))


def main():
    """Main execution function."""
    print("=" * 80)
    print("Knowledge-Based Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic item catalog...")
    items_df = generate_synthetic_data(n_items=300)
    print(f"   Generated {len(items_df)} items")
    print(f"   Categories: {items_df['category'].unique()}")

    # Train recommender
    print("\n2. Initializing knowledge-based recommender...")
    recommender = KnowledgeBasedRecommender()
    recommender.fit(items_df)

    # Add rules
    recommender.add_rule(category_match_rule, weight=3.0, name="Category Match")
    recommender.add_rule(price_range_rule, weight=2.0, name="Price Range")
    recommender.add_rule(rating_threshold_rule, weight=2.5, name="Rating Threshold")
    recommender.add_rule(brand_preference_rule, weight=1.5, name="Brand Preference")
    recommender.add_rule(popularity_rule, weight=1.0, name="Popularity")
    print(f"   Added {len(recommender.rules)} recommendation rules")

    # Test recommendations
    print("\n3. Generating recommendations for sample users...")

    test_preferences = [
        {
            'preferred_category': 'Electronics',
            'min_price': 100,
            'max_price': 300,
            'min_rating': 4.0,
            'preferred_brands': ['BrandA', 'BrandB']
        },
        {
            'preferred_category': 'Books',
            'min_price': 10,
            'max_price': 50,
            'min_rating': 3.5,
            'preferred_brands': ['BrandC']
        },
        {
            'preferred_category': 'Clothing',
            'min_price': 20,
            'max_price': 100,
            'min_rating': 4.2,
            'preferred_brands': ['BrandD', 'BrandE']
        }
    ]

    all_recommendations = []
    for i, prefs in enumerate(test_preferences):
        recommendations = recommender.recommend(prefs, n_recommendations=10)
        all_recommendations.append(recommendations)
        print(f"\n   User {i+1} preferences: {prefs['preferred_category']}, "
              f"${prefs['min_price']}-${prefs['max_price']}")
        print(f"   Top 5 recommendations:")
        for j, (item_id, score) in enumerate(recommendations[:5], 1):
            item = items_df[items_df['item_id'] == item_id].iloc[0]
            print(f"      {j}. Item {item_id} ({item['category']}, "
                  f"${item['price']:.2f}, rating: {item['rating']:.2f}) - Score: {score:.2f}")

    # Visualization
    print("\n4. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Category distribution
    plt.subplot(3, 4, 1)
    items_df['category'].value_counts().plot(kind='bar', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Item Category Distribution', fontweight='bold')
    plt.xticks(rotation=45)

    # Plot 2: Price distribution
    plt.subplot(3, 4, 2)
    plt.hist(items_df['price'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution', fontweight='bold')

    # Plot 3: Rating distribution
    plt.subplot(3, 4, 3)
    plt.hist(items_df['rating'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Rating Distribution', fontweight='bold')

    # Plot 4: Brand distribution
    plt.subplot(3, 4, 4)
    items_df['brand'].value_counts().plot(kind='bar', alpha=0.7, color='green')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.title('Brand Distribution', fontweight='bold')

    # Plot 5: Rule weights
    plt.subplot(3, 4, 5)
    rule_names = [r['name'] for r in recommender.rules]
    rule_weights = [r['weight'] for r in recommender.rules]
    plt.barh(range(len(rule_names)), rule_weights, alpha=0.8, color='purple')
    plt.yticks(range(len(rule_names)), rule_names)
    plt.xlabel('Weight')
    plt.title('Rule Weights', fontweight='bold')

    # Plot 6: Popularity vs Rating
    plt.subplot(3, 4, 6)
    plt.scatter(items_df['popularity'], items_df['rating'], alpha=0.5)
    plt.xlabel('Popularity')
    plt.ylabel('Rating')
    plt.title('Popularity vs Rating', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 7: Price by category
    plt.subplot(3, 4, 7)
    items_df.boxplot(column='price', by='category', ax=plt.gca())
    plt.xlabel('Category')
    plt.ylabel('Price ($)')
    plt.title('Price by Category', fontweight='bold')
    plt.suptitle('')

    # Plot 8: Recommendations per category
    plt.subplot(3, 4, 8)
    rec_categories = []
    for recs in all_recommendations:
        for item_id, _ in recs:
            item = items_df[items_df['item_id'] == item_id].iloc[0]
            rec_categories.append(item['category'])

    pd.Series(rec_categories).value_counts().plot(kind='bar', alpha=0.7, color='orange')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Recommended Items by Category', fontweight='bold')
    plt.xticks(rotation=45)

    # Plot 9: Year distribution
    plt.subplot(3, 4, 9)
    items_df['year'].value_counts().sort_index().plot(kind='bar', alpha=0.7, color='teal')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Items by Year', fontweight='bold')

    # Plot 10: Num reviews distribution
    plt.subplot(3, 4, 10)
    plt.hist(items_df['num_reviews'], bins=30, edgecolor='black', alpha=0.7, color='pink')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Frequency')
    plt.title('Review Count Distribution', fontweight='bold')

    # Plot 11: Recommendation scores
    plt.subplot(3, 4, 11)
    for i, recs in enumerate(all_recommendations[:3]):
        scores = [score for _, score in recs]
        plt.plot(range(1, len(scores)+1), scores, marker='o', label=f'User {i+1}', linewidth=2)

    plt.xlabel('Rank')
    plt.ylabel('Score')
    plt.title('Recommendation Scores by Rank', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 12: Price range of recommendations
    plt.subplot(3, 4, 12)
    rec_prices = []
    for recs in all_recommendations:
        for item_id, _ in recs:
            item = items_df[items_df['item_id'] == item_id].iloc[0]
            rec_prices.append(item['price'])

    plt.hist(rec_prices, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.title('Recommended Items Price Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/04_recommendation/20_knowledge_based_recommendations/analysis_plots.png',
                dpi=300, bbox_inches='tight')
    print("   Saved visualization")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
