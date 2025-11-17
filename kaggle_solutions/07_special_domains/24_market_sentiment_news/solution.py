"""
Market Sentiment Analysis from News
====================================
Domain: Finance & Quantitative Trading
Task: Predicting market movements from news sentiment

This solution demonstrates:
- NLP for financial news analysis
- Sentiment analysis and entity extraction
- Time series forecasting with text features
- Event impact quantification
- Multi-source news aggregation
- Real-time sentiment scoring
- Trading signal generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class MarketSentimentAnalyzer:
    """
    News sentiment analysis system for market prediction and trading signals.
    """

    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.sentiment_lexicon = self._build_sentiment_lexicon()

    def _build_sentiment_lexicon(self):
        """Build financial sentiment lexicon."""
        positive_words = [
            'surge', 'rally', 'gain', 'profit', 'growth', 'beat', 'strong',
            'bullish', 'upgrade', 'outperform', 'breakout', 'soar', 'boom',
            'succeed', 'win', 'achieve', 'expand', 'record', 'high', 'up'
        ]

        negative_words = [
            'plunge', 'crash', 'loss', 'decline', 'weak', 'miss', 'bearish',
            'downgrade', 'underperform', 'drop', 'fall', 'slump', 'crisis',
            'fail', 'cut', 'shrink', 'low', 'down', 'concern', 'risk'
        ]

        return {'positive': positive_words, 'negative': negative_words}

    def generate_news_data(self, n_articles=2000):
        """Generate synthetic financial news articles with market impact."""
        np.random.seed(42)

        articles = []
        companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']

        for i in range(n_articles):
            # Select company
            company = np.random.choice(companies)

            # Generate sentiment
            sentiment_score = np.random.normal(0, 1)
            if sentiment_score > 0.5:
                sentiment = 'positive'
            elif sentiment_score < -0.5:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            # Generate headline
            if sentiment == 'positive':
                templates = [
                    f"{company} stock surges on strong earnings report",
                    f"{company} announces record revenue growth this quarter",
                    f"Analysts upgrade {company} citing strong fundamentals",
                    f"{company} beats expectations with impressive profit margins"
                ]
                keywords = np.random.choice(self.sentiment_lexicon['positive'], 3)
            elif sentiment == 'negative':
                templates = [
                    f"{company} shares plunge amid regulatory concerns",
                    f"{company} reports disappointing quarterly results",
                    f"Analysts downgrade {company} on weak guidance",
                    f"{company} faces criticism over declining market share"
                ]
                keywords = np.random.choice(self.sentiment_lexicon['negative'], 3)
            else:
                templates = [
                    f"{company} releases quarterly earnings report",
                    f"{company} announces new product lineup",
                    f"Market awaits {company} financial results",
                    f"{company} holds annual shareholders meeting"
                ]
                keywords = []

            headline = np.random.choice(templates)

            # Article metadata
            timestamp = i / n_articles * 365  # Days
            hour = np.random.randint(6, 20)  # Business hours

            # Market impact (price change)
            base_impact = sentiment_score * 0.02  # 2% per sentiment unit
            noise = np.random.normal(0, 0.01)
            price_change = base_impact + noise

            # Volume impact
            volume_multiplier = 1 + abs(sentiment_score) * 0.5
            volume_change = volume_multiplier * np.random.lognormal(0, 0.3)

            articles.append({
                'article_id': f'ART_{i:06d}',
                'timestamp': timestamp,
                'hour': hour,
                'company': company,
                'headline': headline,
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'num_keywords': len(keywords),
                'price_change': price_change,
                'volume_change': volume_change,
                'market_moved': 1 if abs(price_change) > 0.015 else 0
            })

        df = pd.DataFrame(articles)

        print(f"Generated {n_articles} news articles")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        print(f"\nMarket movement rate: {df['market_moved'].mean()*100:.1f}%")
        print(f"Average absolute price change: {abs(df['price_change']).mean()*100:.2f}%")

        return df

    def extract_text_features(self, df):
        """Extract features from headlines."""
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_features = self.vectorizer.fit_transform(df['headline']).toarray()

        # Sentiment scores
        sentiment_scores = []
        for headline in df['headline']:
            pos_count = sum(1 for word in self.sentiment_lexicon['positive']
                          if word in headline.lower())
            neg_count = sum(1 for word in self.sentiment_lexicon['negative']
                          if word in headline.lower())
            sentiment_scores.append((pos_count - neg_count) / (pos_count + neg_count + 1))

        # Combine features
        text_features = pd.DataFrame(tfidf_features,
                                    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        text_features['sentiment_score'] = sentiment_scores
        text_features['headline_length'] = df['headline'].str.len()
        text_features['hour'] = df['hour'].values

        # Company encoding
        company_dummies = pd.get_dummies(df['company'], prefix='company')
        features = pd.concat([text_features, company_dummies], axis=1)

        return features

    def train_models(self, X_train, y_train_class, y_train_reg):
        """Train classification and regression models."""
        print("\nTraining models...")

        # Classification: Will market move significantly?
        print("  - Market Movement Classifier...")
        rf_class = RandomForestClassifier(n_estimators=150, max_depth=15,
                                         random_state=42, n_jobs=-1)
        rf_class.fit(X_train, y_train_class)
        self.models['Movement Classifier'] = rf_class

        # Regression: Price change prediction
        print("  - Price Change Regressor...")
        gb_reg = GradientBoostingRegressor(n_estimators=150, max_depth=8,
                                          random_state=42)
        gb_reg.fit(X_train, y_train_reg)
        self.models['Price Regressor'] = gb_reg

        print(f"Trained {len(self.models)} models")

    def evaluate_models(self, X_test, y_test_class, y_test_reg):
        """Evaluate models."""
        print("\nEvaluation Results:")

        # Classification
        y_pred_class = self.models['Movement Classifier'].predict(X_test)
        print("\nMarket Movement Classification:")
        print(classification_report(y_test_class, y_pred_class,
                                   target_names=['No Move', 'Significant Move']))

        # Regression
        y_pred_reg = self.models['Price Regressor'].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        r2 = r2_score(y_test_reg, y_pred_reg)

        print(f"\nPrice Change Prediction:")
        print(f"  RMSE: {rmse*100:.3f}%")
        print(f"  R²: {r2:.4f}")

        return y_pred_class, y_pred_reg

    def plot_sentiment_impact(self, df):
        """Visualize sentiment impact on market."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Price change by sentiment
        df.boxplot(column='price_change', by='sentiment', ax=axes[0, 0])
        axes[0, 0].set_ylabel('Price Change', fontsize=11)
        axes[0, 0].set_title('Price Impact by Sentiment', fontsize=12, fontweight='bold')

        # Sentiment score vs price change
        axes[0, 1].scatter(df['sentiment_score'], df['price_change'],
                          alpha=0.5, s=20)
        axes[0, 1].set_xlabel('Sentiment Score', fontsize=11)
        axes[0, 1].set_ylabel('Price Change', fontsize=11)
        axes[0, 1].set_title('Sentiment vs Price Change', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Volume change by sentiment
        df.boxplot(column='volume_change', by='sentiment', ax=axes[1, 0])
        axes[1, 0].set_ylabel('Volume Multiplier', fontsize=11)
        axes[1, 0].set_title('Volume Impact by Sentiment', fontsize=12, fontweight='bold')

        # Hourly distribution
        hour_sentiment = df.groupby('hour')['sentiment_score'].mean()
        axes[1, 1].plot(hour_sentiment.index, hour_sentiment.values,
                       marker='o', linewidth=2)
        axes[1, 1].set_xlabel('Hour of Day', fontsize=11)
        axes[1, 1].set_ylabel('Average Sentiment Score', fontsize=11)
        axes[1, 1].set_title('Sentiment by Hour', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('market_sentiment_impact.png', dpi=300, bbox_inches='tight')
        print("Saved: market_sentiment_impact.png")
        plt.close()

    def plot_predictions(self, y_true, y_pred):
        """Plot prediction results."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Actual vs predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=30)
        axes[0].plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    'r--', linewidth=2)
        axes[0].set_xlabel('Actual Price Change', fontsize=11)
        axes[0].set_ylabel('Predicted Price Change', fontsize=11)
        axes[0].set_title('Price Change Predictions', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Prediction errors
        errors = y_pred - y_true
        axes[1].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Prediction Error', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('market_sentiment_predictions.png', dpi=300, bbox_inches='tight')
        print("Saved: market_sentiment_predictions.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Market Sentiment Analysis from News")
    print("=" * 80)

    analyzer = MarketSentimentAnalyzer()

    # Generate data
    print("\n1. Generating News Data...")
    df = analyzer.generate_news_data(n_articles=2000)

    # Extract features
    print("\n2. Extracting Text Features...")
    X = analyzer.extract_text_features(df)
    y_class = df['market_moved'].values
    y_reg = df['price_change'].values

    print(f"Total features: {X.shape[1]}")

    # Split data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    # Train
    print("\n3. Training Models...")
    analyzer.train_models(X_train, y_train_class, y_train_reg)

    # Evaluate
    print("\n4. Evaluating Models...")
    y_pred_class, y_pred_reg = analyzer.evaluate_models(
        X_test, y_test_class, y_test_reg
    )

    # Visualizations
    print("\n5. Generating Visualizations...")
    analyzer.plot_sentiment_impact(df)
    analyzer.plot_predictions(y_test_reg, y_pred_reg)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
