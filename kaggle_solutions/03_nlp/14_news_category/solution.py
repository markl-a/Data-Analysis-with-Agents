"""
News Article Category Classification
Classify news articles into categories (Politics, Sports, Technology, Business, Entertainment)

Dataset: Synthetic news articles
Difficulty: ⭐⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class NewsClassifier:
    """News article category classification system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.model = None
        self.categories = ['Politics', 'Sports', 'Technology', 'Business', 'Entertainment']

    def create_sample_data(self, n_samples=1200):
        """Create synthetic news article data"""
        np.random.seed(42)

        # Domain-specific vocabularies
        category_vocab = {
            'Politics': {
                'subjects': ['president', 'senator', 'congress', 'parliament', 'government',
                           'election', 'vote', 'policy', 'law', 'minister', 'legislation'],
                'verbs': ['announced', 'proposed', 'voted', 'debated', 'enacted',
                         'approved', 'vetoed', 'campaigned', 'addressed'],
                'adjectives': ['democratic', 'republican', 'bipartisan', 'controversial',
                             'political', 'federal', 'legislative', 'constitutional'],
                'contexts': ['in Washington', 'at the Capitol', 'in a speech', 'during the session',
                           'at a rally', 'in the Senate', 'to Congress']
            },
            'Sports': {
                'subjects': ['team', 'player', 'coach', 'athlete', 'championship',
                           'tournament', 'league', 'stadium', 'game', 'match', 'season'],
                'verbs': ['scored', 'won', 'defeated', 'competed', 'trained',
                         'played', 'finished', 'dominated', 'advanced'],
                'adjectives': ['championship', 'winning', 'athletic', 'competitive',
                             'professional', 'record-breaking', 'spectacular', 'dominant'],
                'contexts': ['in the finals', 'at the stadium', 'during the game', 'this season',
                           'in overtime', 'on the field', 'in the tournament']
            },
            'Technology': {
                'subjects': ['company', 'startup', 'developer', 'software', 'AI',
                           'algorithm', 'platform', 'app', 'technology', 'innovation', 'system'],
                'verbs': ['launched', 'developed', 'released', 'innovated', 'created',
                         'announced', 'upgraded', 'integrated', 'optimized'],
                'adjectives': ['innovative', 'advanced', 'cutting-edge', 'digital',
                             'artificial', 'cloud-based', 'automated', 'intelligent'],
                'contexts': ['in Silicon Valley', 'at the conference', 'in latest update',
                           'in the market', 'for users', 'in beta', 'worldwide']
            },
            'Business': {
                'subjects': ['company', 'corporation', 'CEO', 'market', 'stock',
                           'investor', 'revenue', 'profit', 'economy', 'industry', 'earnings'],
                'verbs': ['reported', 'increased', 'declined', 'invested', 'acquired',
                         'merged', 'announced', 'forecast', 'grew'],
                'adjectives': ['financial', 'economic', 'profitable', 'quarterly',
                             'corporate', 'commercial', 'fiscal', 'market-leading'],
                'contexts': ['on Wall Street', 'in Q4', 'this quarter', 'year-over-year',
                           'in the market', 'according to analysts', 'in trading']
            },
            'Entertainment': {
                'subjects': ['actor', 'movie', 'film', 'show', 'series', 'director',
                           'celebrity', 'award', 'performance', 'premiere', 'album'],
                'verbs': ['starred', 'premiered', 'released', 'performed', 'won',
                         'nominated', 'directed', 'produced', 'debuted'],
                'adjectives': ['blockbuster', 'award-winning', 'popular', 'critically acclaimed',
                             'Hollywood', 'dramatic', 'entertaining', 'successful'],
                'contexts': ['in theaters', 'at the premiere', 'on streaming', 'at the awards',
                           'this weekend', 'in Hollywood', 'worldwide']
            }
        }

        articles = []
        labels = []

        for _ in range(n_samples):
            category = np.random.choice(self.categories)
            vocab = category_vocab[category]

            # Generate article headline and body
            num_sentences = np.random.randint(3, 7)
            sentences = []

            for _ in range(num_sentences):
                subject = np.random.choice(vocab['subjects'])
                verb = np.random.choice(vocab['verbs'])
                adjective = np.random.choice(vocab['adjectives'])
                context = np.random.choice(vocab['contexts'])

                # Generate varied sentence structures
                templates = [
                    f"The {adjective} {subject} {verb} {context}",
                    f"{subject.capitalize()} {verb} as {adjective} developments unfold {context}",
                    f"Reports indicate the {subject} {verb} {context}",
                    f"A {adjective} {subject} recently {verb} {context}",
                    f"Officials confirmed the {subject} {verb} {context}"
                ]
                sentence = np.random.choice(templates)
                sentences.append(sentence)

            article_text = '. '.join(sentences) + '.'
            articles.append(article_text)
            labels.append(category)

        return pd.DataFrame({
            'article': articles,
            'category': labels
        })

    def extract_features(self, df):
        """Extract additional features from articles"""
        df = df.copy()

        # Text statistics
        df['article_length'] = df['article'].apply(len)
        df['word_count'] = df['article'].apply(lambda x: len(x.split()))
        df['sentence_count'] = df['article'].apply(lambda x: x.count('.'))
        df['avg_word_length'] = df['article'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        # Named entity indicators (simple keyword counting)
        df['has_numbers'] = df['article'].str.contains(r'\d+').astype(int)
        df['has_quotes'] = df['article'].str.contains(r'["\']').astype(int)
        df['exclamation_count'] = df['article'].str.count('!')

        return df

    def train_multiple_models(self, X_train, y_train):
        """Train and compare multiple models"""
        models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            results[name] = {'model': model, 'train_score': train_score}
            print(f"{name} - Training Accuracy: {train_score:.4f}")

        # Select best model (Logistic Regression typically performs well)
        self.model = results['Logistic Regression']['model']
        return results

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=self.categories))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.categories)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.title('News Category Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('news_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'news_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize news article distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Category distribution
        category_counts = df['category'].value_counts()
        category_counts.plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('News Category Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Word count by category
        df.boxplot(column='word_count', by='category', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count Distribution by Category', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Word Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45, ha='right')

        # Average word length by category
        df.groupby('category')['avg_word_length'].mean().plot(
            kind='barh', ax=axes[1, 0], color='coral'
        )
        axes[1, 0].set_title('Average Word Length by Category', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Average Word Length')

        # Sentence count distribution
        axes[1, 1].hist(df['sentence_count'], bins=20, color='seagreen', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Sentence Count Distribution', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Number of Sentences')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('news_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'news_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("News Article Category Classification")
    print("=" * 70)

    # Initialize classifier
    classifier = NewsClassifier()

    # Create sample data
    print("\nCreating synthetic news article data...")
    df = classifier.create_sample_data(n_samples=1200)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample article:\n{df['article'].iloc[0]}")
    print(f"Category: {df['category'].iloc[0]}")

    # Data exploration
    print(f"\n=== Category Distribution ===")
    print(df['category'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = classifier.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    classifier.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = classifier.vectorizer.fit_transform(df['article'])
    y = df['category']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train multiple models
    print("\nTraining multiple models...")
    results = classifier.train_multiple_models(X_train, y_train)

    # Evaluate best model
    print("\nEvaluating best model...")
    predictions = classifier.evaluate(X_test, y_test)

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(3):
        sample_text = df['article'].iloc[i]
        sample_vectorized = classifier.vectorizer.transform([sample_text])
        prediction = classifier.model.predict(sample_vectorized)[0]
        actual = df['category'].iloc[i]

        print(f"\nArticle {i+1}: {sample_text[:100]}...")
        print(f"Predicted: {prediction}")
        print(f"Actual: {actual}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
