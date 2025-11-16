"""
Product Review Rating Prediction
Predict star ratings (1-5) from product review text

Dataset: Synthetic product reviews
Difficulty: ⭐⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ReviewRatingPredictor:
    """Product review rating prediction system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.9
        )
        self.model = None
        self.ratings = [1, 2, 3, 4, 5]

    def create_sample_data(self, n_samples=1200):
        """Create synthetic product review data"""
        np.random.seed(42)

        # Rating-specific vocabularies
        rating_vocab = {
            1: {  # Very negative
                'adjectives': ['terrible', 'awful', 'horrible', 'worst', 'useless', 'broken',
                             'disappointing', 'defective', 'poor quality', 'waste of money'],
                'verbs': ['hate', 'regret', 'disappointed', 'failed', 'broke', 'stopped working'],
                'phrases': ['do not buy', 'complete waste', 'totally disappointed',
                          'worst purchase ever', 'save your money', 'extremely dissatisfied']
            },
            2: {  # Negative
                'adjectives': ['bad', 'poor', 'subpar', 'mediocre', 'cheap', 'flimsy',
                             'unreliable', 'frustrating', 'not great', 'below average'],
                'verbs': ['dislike', 'struggle', 'issues with', 'problems with', 'not satisfied'],
                'phrases': ['not worth it', 'expected better', 'many problems',
                          'would not recommend', 'not as advertised', 'could be better']
            },
            3: {  # Neutral
                'adjectives': ['okay', 'decent', 'acceptable', 'average', 'fair', 'reasonable',
                             'adequate', 'satisfactory', 'basic', 'standard'],
                'verbs': ['works', 'functions', 'does the job', 'gets by', 'serves purpose'],
                'phrases': ['its okay', 'nothing special', 'meets expectations',
                          'average product', 'does what it says', 'fairly good']
            },
            4: {  # Positive
                'adjectives': ['good', 'nice', 'solid', 'reliable', 'quality', 'well-made',
                             'impressive', 'satisfied', 'happy', 'pleased'],
                'verbs': ['like', 'enjoy', 'recommend', 'works well', 'performs great'],
                'phrases': ['very satisfied', 'great value', 'works perfectly',
                          'highly recommend', 'exceeded expectations', 'love it']
            },
            5: {  # Very positive
                'adjectives': ['excellent', 'amazing', 'outstanding', 'perfect', 'incredible',
                             'fantastic', 'superb', 'exceptional', 'phenomenal', 'best'],
                'verbs': ['love', 'adore', 'absolutely recommend', 'cant live without'],
                'phrases': ['absolutely perfect', 'best purchase ever', 'exceeded all expectations',
                          '10 out of 10', 'highly highly recommend', 'worth every penny']
            }
        }

        reviews = []
        ratings = []

        for _ in range(n_samples):
            # Select rating with realistic distribution
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.10, 0.15, 0.35, 0.30])
            vocab = rating_vocab[rating]

            # Generate review
            review_parts = []

            # Opening statement
            adj = np.random.choice(vocab['adjectives'])
            verb = np.random.choice(vocab['verbs'])
            review_parts.append(f"I {verb} this product.")

            # Main content
            phrase = np.random.choice(vocab['phrases'])
            review_parts.append(f"The product is {adj}. {phrase.capitalize()}.")

            # Additional details
            if np.random.random() > 0.5:
                adj2 = np.random.choice(vocab['adjectives'])
                review_parts.append(f"Build quality is {adj2}.")

            # Closing statement
            if rating >= 4:
                review_parts.append("Would buy again.")
            elif rating <= 2:
                review_parts.append("Would not buy again.")

            review_text = ' '.join(review_parts)
            reviews.append(review_text)
            ratings.append(rating)

        return pd.DataFrame({
            'review': reviews,
            'rating': ratings
        })

    def extract_features(self, df):
        """Extract review features"""
        df = df.copy()

        # Text statistics
        df['review_length'] = df['review'].apply(len)
        df['word_count'] = df['review'].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['review'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        # Sentiment indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'fantastic']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']

        df['positive_word_count'] = df['review'].apply(
            lambda x: sum([1 for word in positive_words if word in x.lower()])
        )
        df['negative_word_count'] = df['review'].apply(
            lambda x: sum([1 for word in negative_words if word in x.lower()])
        )
        df['exclamation_count'] = df['review'].str.count('!')
        df['question_count'] = df['review'].str.count(r'\?')

        # Sentiment score
        df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']

        return df

    def train(self, X_train, y_train):
        """Train rating prediction model"""
        # Use Gradient Boosting for ordinal classification
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.model.fit(X_train, y_train)
        print("Model training completed")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions,
                                   target_names=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.ratings)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                   xticklabels=['1★', '2★', '3★', '4★', '5★'],
                   yticklabels=['1★', '2★', '3★', '4★', '5★'],
                   cbar_kws={'label': 'Number of Reviews'})
        plt.title('Review Rating Prediction - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Rating')
        plt.xlabel('Predicted Rating')
        plt.tight_layout()
        plt.savefig('review_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'review_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize review data distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Rating distribution
        df['rating'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], color='steelblue'
        )
        axes[0, 0].set_title('Rating Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Star Rating')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticklabels(['1★', '2★', '3★', '4★', '5★'], rotation=0)

        # Word count by rating
        df.boxplot(column='word_count', by='rating', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count by Rating', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Star Rating')
        axes[0, 1].set_ylabel('Word Count')
        plt.sca(axes[0, 1])
        plt.xticks(range(1, 6), ['1★', '2★', '3★', '4★', '5★'])

        # Sentiment score by rating
        df.groupby('rating')['sentiment_score'].mean().plot(
            kind='bar', ax=axes[0, 2], color='coral'
        )
        axes[0, 2].set_title('Average Sentiment Score by Rating', fontsize=12, pad=10)
        axes[0, 2].set_xlabel('Star Rating')
        axes[0, 2].set_ylabel('Sentiment Score')
        axes[0, 2].set_xticklabels(['1★', '2★', '3★', '4★', '5★'], rotation=0)

        # Positive word count by rating
        df.groupby('rating')['positive_word_count'].mean().plot(
            kind='bar', ax=axes[1, 0], color='seagreen'
        )
        axes[1, 0].set_title('Avg Positive Words by Rating', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Star Rating')
        axes[1, 0].set_ylabel('Positive Word Count')
        axes[1, 0].set_xticklabels(['1★', '2★', '3★', '4★', '5★'], rotation=0)

        # Negative word count by rating
        df.groupby('rating')['negative_word_count'].mean().plot(
            kind='bar', ax=axes[1, 1], color='crimson'
        )
        axes[1, 1].set_title('Avg Negative Words by Rating', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Star Rating')
        axes[1, 1].set_ylabel('Negative Word Count')
        axes[1, 1].set_xticklabels(['1★', '2★', '3★', '4★', '5★'], rotation=0)

        # Review length distribution
        axes[1, 2].hist(df['review_length'], bins=30, color='orchid',
                       edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Review Length Distribution', fontsize=12, pad=10)
        axes[1, 2].set_xlabel('Character Count')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('review_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'review_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Product Review Rating Prediction")
    print("=" * 70)

    # Initialize predictor
    predictor = ReviewRatingPredictor()

    # Create sample data
    print("\nCreating synthetic product review data...")
    df = predictor.create_sample_data(n_samples=1200)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample review:\n{df['review'].iloc[0]}")
    print(f"Rating: {df['rating'].iloc[0]} stars")

    # Data exploration
    print(f"\n=== Rating Distribution ===")
    print(df['rating'].value_counts().sort_index())
    print(f"\nAverage rating: {df['rating'].mean():.2f}")

    # Extract features
    print("\nExtracting features...")
    df = predictor.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    predictor.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = predictor.vectorizer.fit_transform(df['review'])
    y = df['rating']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train model
    print("\nTraining Gradient Boosting classifier...")
    predictor.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    predictions = predictor.evaluate(X_test, y_test)

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(5):
        sample_text = df['review'].iloc[i]
        sample_vectorized = predictor.vectorizer.transform([sample_text])
        prediction = predictor.model.predict(sample_vectorized)[0]
        actual = df['rating'].iloc[i]

        print(f"\nReview: {sample_text}")
        print(f"Predicted: {prediction} stars")
        print(f"Actual: {actual} stars")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
