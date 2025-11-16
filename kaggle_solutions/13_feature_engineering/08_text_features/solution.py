"""
Kaggle Solution: Text Feature Extraction
=========================================
Demonstrates multiple text feature engineering techniques including
TF-IDF, count vectorization, character n-grams, and basic text statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_product_reviews(n_samples=2000):
    """Generate synthetic product reviews with sentiment."""
    positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'love', 'perfect',
                     'best', 'awesome', 'brilliant', 'outstanding', 'superb', 'terrific']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'disappointing', 'poor', 'bad',
                     'useless', 'waste', 'broken', 'defective', 'junk', 'garbage']
    neutral_words = ['product', 'item', 'bought', 'received', 'ordered', 'package', 'delivery',
                    'price', 'quality', 'works', 'ok', 'average', 'decent']

    data = []
    for _ in range(n_samples):
        # Determine sentiment
        sentiment = np.random.choice(['positive', 'negative'], p=[0.6, 0.4])

        # Build review
        if sentiment == 'positive':
            words = (np.random.choice(positive_words, size=np.random.randint(3, 8)).tolist() +
                    np.random.choice(neutral_words, size=np.random.randint(2, 5)).tolist())
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
        else:
            words = (np.random.choice(negative_words, size=np.random.randint(3, 8)).tolist() +
                    np.random.choice(neutral_words, size=np.random.randint(2, 5)).tolist())
            rating = np.random.choice([1, 2], p=[0.6, 0.4])

        # Shuffle and join
        np.random.shuffle(words)
        review = ' '.join(words)

        # Add some noise
        if np.random.random() < 0.2:
            review += ' ' + np.random.choice(neutral_words, size=3).tolist().__str__()

        # Add punctuation
        review += np.random.choice(['!', '.', '...'], p=[0.2, 0.6, 0.2])

        # Random capitalization
        if np.random.random() < 0.3:
            review = review.upper()

        data.append({
            'review': review,
            'rating': rating,
            'sentiment': 1 if sentiment == 'positive' else 0
        })

    return pd.DataFrame(data)


def extract_basic_text_features(df, text_col='review'):
    """Extract basic statistical features from text."""
    df = df.copy()

    df['text_length'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    df['avg_word_length'] = df[text_col].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['char_count'] = df[text_col].str.len()
    df['uppercase_ratio'] = df[text_col].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    df['digit_count'] = df[text_col].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['exclamation_count'] = df[text_col].str.count('!')
    df['question_count'] = df[text_col].str.count(r'\?')
    df['punctuation_count'] = df[text_col].apply(lambda x: sum(1 for c in str(x) if c in '.,!?;:'))

    return df


def extract_lexical_features(df, text_col='review'):
    """Extract lexical diversity features."""
    df = df.copy()

    def lexical_diversity(text):
        words = str(text).lower().split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    df['lexical_diversity'] = df[text_col].apply(lexical_diversity)
    df['unique_word_count'] = df[text_col].apply(lambda x: len(set(str(x).lower().split())))

    return df


def create_sentiment_features(df, text_col='review'):
    """Create sentiment-based features."""
    df = df.copy()

    positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'fantastic', 'love', 'perfect',
                     'best', 'awesome', 'brilliant', 'outstanding', 'superb', 'terrific']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'disappointing', 'poor', 'bad',
                     'useless', 'waste', 'broken', 'defective', 'junk', 'garbage']

    def count_words(text, word_list):
        text_lower = str(text).lower()
        return sum(1 for word in word_list if word in text_lower)

    df['positive_word_count'] = df[text_col].apply(lambda x: count_words(x, positive_words))
    df['negative_word_count'] = df[text_col].apply(lambda x: count_words(x, negative_words))
    df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']

    return df


def evaluate_feature_set(X_train, X_test, y_train, y_test, feature_set_name, use_logistic=False):
    """Train and evaluate model with specific feature set."""
    if use_logistic:
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'feature_set': feature_set_name,
        'n_features': X_train.shape[1],
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'model': model
    }


def plot_results(results, df):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. F1 Score comparison
    ax1 = fig.add_subplot(gs[0, :2])
    feature_sets = [r['feature_set'] for r in results]
    f1_scores = [r['f1'] for r in results]
    colors = plt.cm.RdYlGn(np.array(f1_scores) / max(f1_scores))
    bars = ax1.barh(range(len(feature_sets)), f1_scores, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(feature_sets)))
    ax1.set_yticklabels(feature_sets)
    ax1.set_xlabel('F1 Score', fontsize=12)
    ax1.set_title('Performance by Feature Set', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax1.text(score, i, f' {score:.4f}', va='center')

    # 2. Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    accs = [r['accuracy'] for r in results]
    ax2.barh(range(len(feature_sets)), accs, alpha=0.7, color='steelblue')
    ax2.set_yticks(range(len(feature_sets)))
    ax2.set_yticklabels(feature_sets)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Text length distribution by sentiment
    ax3 = fig.add_subplot(gs[1, 0])
    df_with_features = extract_basic_text_features(df)
    ax3.hist(df_with_features[df_with_features['sentiment'] == 1]['text_length'],
            bins=30, alpha=0.5, label='Positive', edgecolor='black')
    ax3.hist(df_with_features[df_with_features['sentiment'] == 0]['text_length'],
            bins=30, alpha=0.5, label='Negative', edgecolor='black')
    ax3.set_xlabel('Text Length', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Text Length by Sentiment', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Word count distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df_with_features[df_with_features['sentiment'] == 1]['word_count'],
            bins=20, alpha=0.5, label='Positive', edgecolor='black', color='green')
    ax4.hist(df_with_features[df_with_features['sentiment'] == 0]['word_count'],
            bins=20, alpha=0.5, label='Negative', edgecolor='black', color='red')
    ax4.set_xlabel('Word Count', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Word Count by Sentiment', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Sentiment word counts
    ax5 = fig.add_subplot(gs[1, 2])
    df_sentiment = create_sentiment_features(df_with_features)
    avg_positive = df_sentiment.groupby('sentiment')['positive_word_count'].mean()
    avg_negative = df_sentiment.groupby('sentiment')['negative_word_count'].mean()
    x = np.arange(2)
    width = 0.35
    ax5.bar(x - width/2, avg_positive, width, label='Positive Words', alpha=0.8, color='green')
    ax5.bar(x + width/2, avg_negative, width, label='Negative Words', alpha=0.8, color='red')
    ax5.set_xlabel('Actual Sentiment', fontsize=10)
    ax5.set_ylabel('Average Word Count', fontsize=10)
    ax5.set_title('Sentiment Word Usage', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Negative', 'Positive'])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Feature count vs performance
    ax6 = fig.add_subplot(gs[2, 0])
    n_features = [r['n_features'] for r in results]
    ax6.scatter(n_features, f1_scores, s=200, alpha=0.6, c=range(len(results)), cmap='viridis')
    for i, r in enumerate(results):
        ax6.annotate(r['feature_set'].split()[0], (r['n_features'], r['f1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax6.set_xlabel('Number of Features', fontsize=12)
    ax6.set_ylabel('F1 Score', fontsize=12)
    ax6.set_title('Features vs Performance', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)

    # 7. Confusion matrix (best model)
    ax7 = fig.add_subplot(gs[2, 1])
    best_result = max(results, key=lambda x: x['f1'])
    cm = confusion_matrix(df.iloc[len(df)//5:]['sentiment'], best_result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7, cbar=False)
    ax7.set_xlabel('Predicted', fontsize=10)
    ax7.set_ylabel('Actual', fontsize=10)
    ax7.set_title(f'Confusion Matrix: {best_result["feature_set"]}', fontsize=12, fontweight='bold')
    ax7.set_xticklabels(['Negative', 'Positive'])
    ax7.set_yticklabels(['Negative', 'Positive'])

    # 8. Rating distribution
    ax8 = fig.add_subplot(gs[2, 2])
    rating_counts = df['rating'].value_counts().sort_index()
    colors_rating = ['red', 'orange', 'gray', 'lightgreen', 'green']
    ax8.bar(rating_counts.index, rating_counts.values, alpha=0.7,
           color=[colors_rating[i-1] for i in rating_counts.index])
    ax8.set_xlabel('Rating', fontsize=12)
    ax8.set_ylabel('Count', fontsize=12)
    ax8.set_title('Rating Distribution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/08_text_features/text_features_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'text_features_analysis.png'")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Text Feature Extraction Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic product reviews...")
    df = generate_product_reviews(n_samples=2000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Positive reviews: {(df['sentiment'] == 1).sum()} ({(df['sentiment'] == 1).mean():.1%})")
    print(f"   Negative reviews: {(df['sentiment'] == 0).sum()} ({(df['sentiment'] == 0).mean():.1%})")

    # Sample reviews
    print("\n2. Sample Reviews:")
    print("   Positive:", df[df['sentiment'] == 1]['review'].iloc[0])
    print("   Negative:", df[df['sentiment'] == 0]['review'].iloc[0])

    # Split data
    print("\n3. Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

    results = []

    # Feature Set 1: Basic text statistics only
    print("\n4. Feature Set 1: Basic Text Statistics...")
    train_basic = extract_basic_text_features(train_df)
    test_basic = extract_basic_text_features(test_df)
    basic_features = ['text_length', 'word_count', 'avg_word_length', 'uppercase_ratio',
                     'exclamation_count', 'punctuation_count', 'rating']

    result = evaluate_feature_set(train_basic[basic_features], test_basic[basic_features],
                                  train_df['sentiment'], test_df['sentiment'],
                                  "Basic Text Stats")
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Feature Set 2: With sentiment features
    print("\n5. Feature Set 2: With Sentiment Features...")
    train_sentiment = create_sentiment_features(train_basic)
    test_sentiment = create_sentiment_features(test_basic)
    sentiment_features = basic_features + ['positive_word_count', 'negative_word_count', 'sentiment_score']

    result = evaluate_feature_set(train_sentiment[sentiment_features], test_sentiment[sentiment_features],
                                  train_df['sentiment'], test_df['sentiment'],
                                  "With Sentiment Features")
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Feature Set 3: TF-IDF (word level)
    print("\n6. Feature Set 3: TF-IDF (word-level)...")
    tfidf_word = TfidfVectorizer(max_features=100, ngram_range=(1, 1), min_df=2)
    X_train_tfidf = tfidf_word.fit_transform(train_df['review'])
    X_test_tfidf = tfidf_word.transform(test_df['review'])

    result = evaluate_feature_set(X_train_tfidf, X_test_tfidf,
                                  train_df['sentiment'], test_df['sentiment'],
                                  "TF-IDF (words)", use_logistic=True)
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Feature Set 4: TF-IDF (with bigrams)
    print("\n7. Feature Set 4: TF-IDF (with bigrams)...")
    tfidf_bigram = TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=2)
    X_train_bigram = tfidf_bigram.fit_transform(train_df['review'])
    X_test_bigram = tfidf_bigram.transform(test_df['review'])

    result = evaluate_feature_set(X_train_bigram, X_test_bigram,
                                  train_df['sentiment'], test_df['sentiment'],
                                  "TF-IDF (bigrams)", use_logistic=True)
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Feature Set 5: Count Vectorizer
    print("\n8. Feature Set 5: Count Vectorizer...")
    count_vec = CountVectorizer(max_features=100, ngram_range=(1, 1), min_df=2)
    X_train_count = count_vec.fit_transform(train_df['review'])
    X_test_count = count_vec.transform(test_df['review'])

    result = evaluate_feature_set(X_train_count, X_test_count,
                                  train_df['sentiment'], test_df['sentiment'],
                                  "Count Vectorizer", use_logistic=True)
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Feature Set 6: Character n-grams
    print("\n9. Feature Set 6: Character N-Grams...")
    char_ngram = TfidfVectorizer(max_features=100, analyzer='char', ngram_range=(2, 4), min_df=2)
    X_train_char = char_ngram.fit_transform(train_df['review'])
    X_test_char = char_ngram.transform(test_df['review'])

    result = evaluate_feature_set(X_train_char, X_test_char,
                                  train_df['sentiment'], test_df['sentiment'],
                                  "Character N-Grams", use_logistic=True)
    results.append(result)
    print(f"   F1: {result['f1']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Summary
    print("\n10. Results Summary:")
    print("-" * 80)
    print(f"{'Feature Set':<30} {'Features':<12} {'F1 Score':<12} {'Accuracy':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['feature_set']:<30} {r['n_features']:<12} {r['f1']:<12.4f} {r['accuracy']:<12.4f}")

    # Best feature set
    print("\n11. Best Feature Set:")
    best_result = max(results, key=lambda x: x['f1'])
    baseline_result = results[0]
    print(f"    Feature Set: {best_result['feature_set']}")
    print(f"    F1 Score: {best_result['f1']:.4f}")
    print(f"    Accuracy: {best_result['accuracy']:.4f}")
    print(f"    Features: {best_result['n_features']}")
    print(f"\n    Improvement over baseline:")
    print(f"    F1 improvement: {((best_result['f1'] - baseline_result['f1']) / baseline_result['f1'] * 100):.2f}%")

    # Top TF-IDF features
    print("\n12. Top 15 TF-IDF Features (bigrams):")
    feature_names = tfidf_bigram.get_feature_names_out()
    tfidf_scores = X_train_bigram.toarray().mean(axis=0)
    top_indices = tfidf_scores.argsort()[-15:][::-1]
    for idx in top_indices:
        print(f"    {feature_names[idx]:20s}: {tfidf_scores[idx]:.4f}")

    # Visualizations
    print("\n13. Creating visualizations...")
    plot_results(results, df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
