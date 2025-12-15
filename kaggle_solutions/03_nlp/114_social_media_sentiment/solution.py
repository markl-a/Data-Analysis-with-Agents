"""
Social Media Sentiment Analysis

Analyze sentiment in social media posts (Twitter) for brand monitoring,
market research and public opinion analysis.

Dataset: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
Difficulty: ⭐⭐ Intermediate Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import re
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class SentimentAnalyzer:
    """Social Media Sentiment Analysis Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.best_model = None

        # Sentiment lexicons
        self.positive_words = {
            'love', 'great', 'awesome', 'amazing', 'excellent', 'wonderful',
            'fantastic', 'good', 'nice', 'happy', 'best', 'beautiful',
            'perfect', 'brilliant', 'superb', 'thanks', 'thank', 'glad',
            'excited', 'enjoy', 'favorite', 'recommend', 'impressive'
        }
        self.negative_words = {
            'hate', 'bad', 'terrible', 'awful', 'horrible', 'worst',
            'poor', 'disappointing', 'disappointed', 'sad', 'angry',
            'annoying', 'boring', 'waste', 'sucks', 'stupid', 'fail',
            'failed', 'useless', 'ugly', 'broken', 'wrong', 'problem'
        }

    def create_sample_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """Create synthetic Twitter sentiment dataset."""
        np.random.seed(42)

        entities = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Tesla',
                   'Netflix', 'Spotify', 'Twitter', 'Facebook', 'Samsung']

        # Templates for different sentiments
        positive_templates = [
            "I absolutely love {entity}! Best product ever! 😍",
            "Just got the new {entity} and it's amazing! Highly recommend!",
            "{entity} customer service is fantastic. Thanks for the help!",
            "Can't believe how great {entity} is. Game changer! 🔥",
            "Finally switched to {entity}. Should have done it years ago!",
            "{entity} just keeps getting better and better! 👏",
            "So happy with my {entity} purchase. Worth every penny!",
            "The new {entity} update is awesome! Love the new features!",
        ]

        negative_templates = [
            "Terrible experience with {entity}. Never again! 😤",
            "{entity} is the worst. Total waste of money.",
            "Can't believe how bad {entity} has become. So disappointed.",
            "{entity} customer support is horrible. Been waiting for hours!",
            "Just lost all my data because of {entity}. Unbelievable! 💔",
            "The new {entity} is such a downgrade. What were they thinking?",
            "Avoid {entity} at all costs. Complete disaster.",
            "{entity} used to be good but now it's just awful.",
        ]

        neutral_templates = [
            "Just saw the new {entity} announcement. Interesting.",
            "Thinking about trying {entity}. Anyone have experience?",
            "{entity} released their quarterly report today.",
            "Comparing {entity} with competitors. Hard to decide.",
            "The {entity} event is tomorrow. Will be watching.",
            "{entity} is making some changes apparently.",
            "Has anyone used {entity} recently? Thoughts?",
            "Reading about {entity}'s new strategy.",
        ]

        data = []
        sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
        sentiment_probs = [0.35, 0.25, 0.30, 0.10]

        for _ in range(n_samples):
            entity = np.random.choice(entities)
            sentiment = np.random.choice(sentiments, p=sentiment_probs)

            if sentiment == 'Positive':
                template = np.random.choice(positive_templates)
                text = template.format(entity=entity)
            elif sentiment == 'Negative':
                template = np.random.choice(negative_templates)
                text = template.format(entity=entity)
            elif sentiment == 'Neutral':
                template = np.random.choice(neutral_templates)
                text = template.format(entity=entity)
            else:  # Irrelevant
                irrelevant = [
                    "What's for lunch today? 🍕",
                    "The weather is nice today.",
                    "Anyone watching the game tonight?",
                    "Just finished my morning run!",
                    "Can't wait for the weekend!",
                ]
                text = np.random.choice(irrelevant)

            # Add some variations
            if np.random.random() > 0.7:
                text = text.upper() if np.random.random() > 0.5 else text.lower()
            if np.random.random() > 0.8:
                text += " " + np.random.choice(['lol', 'haha', 'omg', 'tbh', 'imo'])

            data.append({
                'text': text,
                'entity': entity if sentiment != 'Irrelevant' else 'None',
                'sentiment': sentiment
            })

        return pd.DataFrame(data)

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove special characters but keep emojis info
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract additional features from text."""
        words = text.lower().split()

        features = {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'positive_word_count': sum(1 for w in words if w in self.positive_words),
            'negative_word_count': sum(1 for w in words if w in self.negative_words),
        }

        return features

    def analyze_data(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Social Media Sentiment Analysis', fontsize=16)

        # Sentiment distribution
        df['sentiment'].value_counts().plot(kind='bar', ax=axes[0, 0],
                                            color=['green', 'red', 'gray', 'blue'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Entity distribution
        entity_counts = df[df['entity'] != 'None']['entity'].value_counts()
        entity_counts.plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('Entity Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Sentiment by entity
        sentiment_entity = pd.crosstab(df['entity'], df['sentiment'], normalize='index')
        sentiment_entity = sentiment_entity.drop('None', errors='ignore')
        sentiment_entity.plot(kind='bar', stacked=True, ax=axes[0, 2],
                             color=['blue', 'red', 'green', 'gray'])
        axes[0, 2].set_title('Sentiment by Entity')
        axes[0, 2].legend(title='Sentiment')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Text length distribution
        df['text_length'] = df['text'].str.len()
        for sent in ['Positive', 'Negative', 'Neutral']:
            subset = df[df['sentiment'] == sent]['text_length']
            axes[1, 0].hist(subset, bins=30, alpha=0.5, label=sent)
        axes[1, 0].set_title('Text Length by Sentiment')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].legend()

        # Word count distribution
        df['word_count'] = df['text'].str.split().str.len()
        df.boxplot(column='word_count', by='sentiment', ax=axes[1, 1])
        axes[1, 1].set_title('Word Count by Sentiment')
        plt.suptitle('')

        # Top words (all)
        all_words = ' '.join(df['text'].apply(self.preprocess_text)).split()
        word_freq = Counter(all_words).most_common(15)
        words, counts = zip(*word_freq)
        axes[1, 2].barh(words, counts, color='steelblue')
        axes[1, 2].set_title('Top 15 Words')
        axes[1, 2].invert_yaxis()

        # Positive vs Negative word counts
        df['pos_words'] = df['text'].apply(
            lambda x: sum(1 for w in x.lower().split() if w in self.positive_words))
        df['neg_words'] = df['text'].apply(
            lambda x: sum(1 for w in x.lower().split() if w in self.negative_words))

        sentiment_words = df.groupby('sentiment')[['pos_words', 'neg_words']].mean()
        sentiment_words.plot(kind='bar', ax=axes[2, 0])
        axes[2, 0].set_title('Avg Sentiment Words by Label')
        axes[2, 0].tick_params(axis='x', rotation=45)

        # Time simulation (hour of day)
        df['hour'] = np.random.randint(0, 24, len(df))
        hourly = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
        hourly.plot(ax=axes[2, 1])
        axes[2, 1].set_title('Simulated Hourly Sentiment')
        axes[2, 1].set_xlabel('Hour of Day')

        # Exclamation usage
        df['exclamations'] = df['text'].str.count('!')
        df.boxplot(column='exclamations', by='sentiment', ax=axes[2, 2])
        axes[2, 2].set_title('Exclamation Usage by Sentiment')
        plt.suptitle('')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/sentiment_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for modeling."""
        # Preprocess texts
        processed_texts = df['text'].apply(self.preprocess_text)

        # TF-IDF features
        if fit:
            X_tfidf = self.tfidf.fit_transform(processed_texts).toarray()
        else:
            X_tfidf = self.tfidf.transform(processed_texts).toarray()

        # Encode labels
        if fit:
            y = self.label_encoder.fit_transform(df['sentiment'])
        else:
            y = self.label_encoder.transform(df['sentiment'])

        return X_tfidf, y

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train classification models."""
        print("\nTraining models...")

        self.models['Naive Bayes'] = MultinomialNB()
        self.models['Naive Bayes'].fit(X_train, y_train)

        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['Logistic Regression'].fit(X_train, y_train)

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)

        self.models['Linear SVM'] = LinearSVC(max_iter=1000, random_state=42)
        self.models['Linear SVM'].fit(X_train, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1 Macro': f1_score(y_test, y_pred, average='macro'),
                'F1 Weighted': f1_score(y_test, y_pred, average='weighted')
            })

        results_df = pd.DataFrame(results).sort_values('F1 Macro', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_results(self, results: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sentiment Classification Results', fontsize=16)

        # Model comparison
        results.set_index('Model')[['Accuracy', 'F1 Macro', 'F1 Weighted']].plot(
            kind='bar', ax=axes[0, 0]
        )
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(loc='lower right')

        # Confusion matrix
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        labels = self.label_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=labels, yticklabels=labels)
        axes[0, 1].set_title('Confusion Matrix (Best Model)')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Per-class F1 scores
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        class_f1 = {label: report[label]['f1-score'] for label in labels}
        axes[1, 0].bar(class_f1.keys(), class_f1.values(), color='steelblue')
        axes[1, 0].set_title('F1 Score per Sentiment Class')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)

        # Feature importance (top words for positive/negative)
        if hasattr(self.best_model, 'coef_'):
            feature_names = self.tfidf.get_feature_names_out()

            # Get positive class index
            pos_idx = list(labels).index('Positive')
            neg_idx = list(labels).index('Negative')

            if len(self.best_model.coef_.shape) > 1:
                top_pos = np.argsort(self.best_model.coef_[pos_idx])[-10:]
                top_neg = np.argsort(self.best_model.coef_[neg_idx])[-10:]

                pos_words = [feature_names[i] for i in top_pos]
                neg_words = [feature_names[i] for i in top_neg]

                axes[1, 1].barh(range(10), [self.best_model.coef_[pos_idx][i] for i in top_pos],
                               alpha=0.7, label='Positive')
                axes[1, 1].set_yticks(range(10))
                axes[1, 1].set_yticklabels(pos_words)
                axes[1, 1].set_title('Top Features for Positive Sentiment')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available',
                           ha='center', va='center')
            axes[1, 1].set_title('Feature Importance')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/sentiment_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("SOCIAL MEDIA SENTIMENT ANALYSIS")
    print("=" * 70)

    analyzer = SentimentAnalyzer()

    # Create data
    print("\nCreating synthetic dataset...")
    df = analyzer.create_sample_data(n_samples=3000)
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    # Analysis
    analyzer.analyze_data(df)

    # Prepare features
    X, y = analyzer.prepare_features(df, fit=True)
    print(f"\nFeature matrix shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate
    analyzer.train_models(X_train, y_train)
    results = analyzer.evaluate_models(X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    # Visualize
    analyzer.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    best = results.iloc[0]
    print(f"Best Model: {best['Model']}")
    print(f"F1 Macro: {best['F1 Macro']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
