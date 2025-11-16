"""
Social Media Hashtag Prediction
Predict hashtags from social media post content

Dataset: Synthetic social media posts
Difficulty: ⭐⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import re
import warnings
warnings.filterwarnings('ignore')


class HashtagPredictor:
    """Social media hashtag prediction system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            lowercase=True
        )
        self.mlb = MultiLabelBinarizer()
        self.model = None
        self.hashtags = ['#tech', '#business', '#lifestyle', '#fitness', '#food',
                        '#travel', '#photography', '#fashion', '#motivation', '#news']

    def create_sample_data(self, n_samples=1000):
        """Create synthetic social media posts"""
        np.random.seed(42)

        # Topic-specific vocabularies
        topic_vocab = {
            '#tech': [
                'new smartphone launched', 'AI breakthrough', 'coding tips',
                'software update', 'tech innovation', 'gadget review',
                'app development', 'cloud computing', 'cybersecurity news',
                'programming tutorial', 'latest technology', 'digital transformation'
            ],
            '#business': [
                'startup success', 'entrepreneurship tips', 'market analysis',
                'business growth', 'investment strategy', 'leadership advice',
                'company news', 'industry trends', 'sales techniques',
                'networking event', 'business development', 'finance insights'
            ],
            '#lifestyle': [
                'morning routine', 'productivity hacks', 'work life balance',
                'home decor ideas', 'sustainable living', 'minimalist lifestyle',
                'self care tips', 'daily habits', 'life goals',
                'personal growth', 'wellness journey', 'mindful living'
            ],
            '#fitness': [
                'workout routine', 'gym session', 'healthy eating',
                'fitness goals', 'training tips', 'muscle building',
                'cardio workout', 'yoga practice', 'weight loss journey',
                'nutrition advice', 'fitness motivation', 'running challenge'
            ],
            '#food': [
                'delicious recipe', 'cooking at home', 'restaurant review',
                'food photography', 'baking cookies', 'healthy meals',
                'foodie life', 'dinner ideas', 'breakfast inspiration',
                'dessert recipe', 'meal prep', 'culinary adventures'
            ],
            '#travel': [
                'vacation vibes', 'exploring cities', 'beach holiday',
                'travel photography', 'adventure time', 'wanderlust',
                'travel tips', 'destination guide', 'road trip',
                'travel inspiration', 'bucket list', 'cultural experience'
            ],
            '#photography': [
                'photo of the day', 'landscape photography', 'portrait session',
                'camera settings', 'photography tips', 'editing tutorial',
                'golden hour', 'street photography', 'nature shots',
                'photography gear', 'composition techniques', 'visual storytelling'
            ],
            '#fashion': [
                'outfit of the day', 'style inspiration', 'fashion trends',
                'seasonal collection', 'designer fashion', 'street style',
                'wardrobe essentials', 'fashion week', 'accessory love',
                'sustainable fashion', 'beauty trends', 'makeup tutorial'
            ],
            '#motivation': [
                'never give up', 'success mindset', 'daily inspiration',
                'positive vibes', 'believe in yourself', 'dream big',
                'stay focused', 'motivational quote', 'achieve goals',
                'inspiration daily', 'success story', 'keep pushing'
            ],
            '#news': [
                'breaking news', 'latest updates', 'current events',
                'news alert', 'top stories', 'headline news',
                'world news', 'local news', 'news analysis',
                'developing story', 'news report', 'media coverage'
            ]
        }

        posts = []
        post_hashtags = []

        for _ in range(n_samples):
            # Select 1-3 hashtags per post
            num_tags = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            selected_tags = np.random.choice(self.hashtags, num_tags, replace=False)

            # Generate post content
            post_parts = []
            for tag in selected_tags:
                content = np.random.choice(topic_vocab[tag])
                post_parts.append(content)

            # Add connecting words
            connectors = ['and', 'with', 'featuring', 'plus', 'including']
            if len(post_parts) > 1:
                post_text = f"{post_parts[0]} {np.random.choice(connectors)} {post_parts[1]}"
            else:
                post_text = post_parts[0]

            # Add emojis occasionally
            if np.random.random() > 0.6:
                emojis = ['!', '!!!', '...', '?']
                post_text += np.random.choice(emojis)

            posts.append(post_text)
            post_hashtags.append(list(selected_tags))

        return pd.DataFrame({
            'post': posts,
            'hashtags': post_hashtags
        })

    def extract_features(self, df):
        """Extract post features"""
        df = df.copy()

        # Text features
        df['post_length'] = df['post'].apply(len)
        df['word_count'] = df['post'].apply(lambda x: len(x.split()))
        df['num_hashtags'] = df['hashtags'].apply(len)

        # Content features
        df['has_exclamation'] = df['post'].str.contains('!').astype(int)
        df['has_question'] = df['post'].str.contains(r'\?').astype(int)
        df['avg_word_length'] = df['post'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        return df

    def train(self, X_train, y_train):
        """Train multi-label classifier"""
        # Use OneVsRest strategy for multi-label classification
        base_classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.model = OneVsRestClassifier(base_classifier)

        self.model.fit(X_train, y_train)
        print("Model training completed")

    def evaluate(self, X_test, y_test, df_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)

        # Convert back to labels for display
        pred_labels = self.mlb.inverse_transform(predictions)
        true_labels = self.mlb.inverse_transform(y_test)

        print("\n=== Model Evaluation ===")
        print(f"Hamming Loss: {hamming_loss(y_test, predictions):.4f}")
        print(f"Subset Accuracy (Exact Match): {accuracy_score(y_test, predictions):.4f}")

        # Per-hashtag statistics
        print("\n=== Per-Hashtag Performance ===")
        for i, tag in enumerate(self.mlb.classes_):
            y_true = y_test[:, i]
            y_pred = predictions[:, i]
            accuracy = accuracy_score(y_true, y_pred)
            print(f"{tag}: {accuracy:.4f}")

        # Create visualization
        self.visualize_predictions(true_labels, pred_labels, df_test)

        return predictions

    def visualize_predictions(self, true_labels, pred_labels, df_test):
        """Visualize prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Hashtag frequency in predictions
        all_pred_tags = [tag for tags in pred_labels for tag in tags]
        pred_counts = pd.Series(all_pred_tags).value_counts()
        pred_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Predicted Hashtag Frequency', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Hashtag')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # True hashtag frequency
        all_true_tags = [tag for tags in true_labels for tag in tags]
        true_counts = pd.Series(all_true_tags).value_counts()
        true_counts.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Actual Hashtag Frequency', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Hashtag')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Hashtags per post distribution
        num_tags_pred = [len(tags) for tags in pred_labels]
        num_tags_true = [len(tags) for tags in true_labels]
        axes[1, 0].hist([num_tags_true, num_tags_pred], bins=5, label=['Actual', 'Predicted'],
                       color=['lightcoral', 'skyblue'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Number of Hashtags per Post', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Number of Hashtags')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()

        # Post length distribution
        axes[1, 1].hist(df_test['post_length'], bins=30, color='seagreen',
                       edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Post Length Distribution', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Character Count')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('hashtag_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'hashtag_analysis.png'")

    def visualize_data(self, df):
        """Visualize training data distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Hashtags per post
        df['num_hashtags'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], color='steelblue'
        )
        axes[0, 0].set_title('Number of Hashtags per Post', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Number of Hashtags')
        axes[0, 0].set_ylabel('Count')

        # Word count distribution
        axes[0, 1].hist(df['word_count'], bins=25, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Word Count Distribution', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')

        # Most common hashtags
        all_tags = [tag for tags in df['hashtags'] for tag in tags]
        tag_counts = pd.Series(all_tags).value_counts()
        tag_counts.plot(kind='barh', ax=axes[1, 0], color='seagreen')
        axes[1, 0].set_title('Hashtag Frequency in Training Data', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Count')

        # Average post length by number of hashtags
        df.groupby('num_hashtags')['post_length'].mean().plot(
            kind='bar', ax=axes[1, 1], color='orchid'
        )
        axes[1, 1].set_title('Avg Post Length by Number of Hashtags', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Number of Hashtags')
        axes[1, 1].set_ylabel('Average Character Count')

        plt.tight_layout()
        plt.savefig('hashtag_data_analysis.png', dpi=300, bbox_inches='tight')
        print("Data visualization saved as 'hashtag_data_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Social Media Hashtag Prediction")
    print("=" * 70)

    # Initialize predictor
    predictor = HashtagPredictor()

    # Create sample data
    print("\nCreating synthetic social media posts...")
    df = predictor.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample post:\n{df['post'].iloc[0]}")
    print(f"Hashtags: {df['hashtags'].iloc[0]}")

    # Extract features
    print("\nExtracting features...")
    df = predictor.extract_features(df)

    # Data exploration
    print(f"\n=== Data Statistics ===")
    print(f"Average hashtags per post: {df['num_hashtags'].mean():.2f}")
    print(f"Average word count: {df['word_count'].mean():.2f}")

    # Visualize training data
    print("\nGenerating data visualizations...")
    predictor.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for multi-label classification...")
    X = predictor.vectorizer.fit_transform(df['post'])
    y = predictor.mlb.fit_transform(df['hashtags'])

    # Split data
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Number of unique hashtags: {len(predictor.mlb.classes_)}")

    # Train model
    print("\nTraining multi-label classifier...")
    predictor.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    predictions = predictor.evaluate(X_test, y_test, df_test)

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(5):
        sample_text = df_test['post'].iloc[i]
        sample_vectorized = predictor.vectorizer.transform([sample_text])
        prediction = predictor.model.predict(sample_vectorized)
        pred_tags = predictor.mlb.inverse_transform(prediction)[0]
        actual_tags = df_test['hashtags'].iloc[i]

        print(f"\nPost: {sample_text}")
        print(f"Predicted: {list(pred_tags)}")
        print(f"Actual: {actual_tags}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
