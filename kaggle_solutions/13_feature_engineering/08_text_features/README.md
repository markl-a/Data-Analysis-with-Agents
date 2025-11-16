# Text Feature Extraction

## Overview
Text data requires transformation into numerical features for machine learning. This example demonstrates multiple text feature engineering techniques including TF-IDF, count vectorization, n-grams, and statistical text features.

## Problem Statement
Predict sentiment (positive/negative) from product reviews. Raw text must be converted to numerical features while preserving semantic meaning and predictive patterns.

## Dataset
Synthetic product reviews (2,000 samples):
- **review**: Text review of product
- **rating**: Star rating (1-5)
- **sentiment**: Binary label (0=negative, 1=positive)

Example reviews:
- Positive: "excellent product amazing quality love it perfect!"
- Negative: "terrible waste broken disappointing awful junk."

## Text Feature Engineering Techniques

### 1. Basic Text Statistics
Simple numerical features derived from text:
```python
text_length = len(text)
word_count = len(text.split())
avg_word_length = mean([len(w) for w in text.split()])
uppercase_ratio = sum(c.isupper() for c in text) / len(text)
exclamation_count = text.count('!')
punctuation_count = sum(c in '.,!?;:' for c in text)
```

### 2. Lexical Features
Vocabulary richness indicators:
```python
unique_words = len(set(text.split()))
lexical_diversity = unique_words / word_count
```

### 3. Sentiment Lexicon Features
Count positive/negative words:
```python
positive_words = ['excellent', 'amazing', 'great', ...]
negative_words = ['terrible', 'awful', 'horrible', ...]
positive_count = sum(1 for w in positive_words if w in text)
negative_count = sum(1 for w in negative_words if w in text)
sentiment_score = positive_count - negative_count
```

### 4. TF-IDF (Term Frequency-Inverse Document Frequency)
Weights words by importance across documents:
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

where:
TF(word, doc) = count(word in doc) / total_words_in_doc
IDF(word) = log(total_docs / docs_containing_word)
```

**Effect**: Common words (the, is, and) get low scores, distinctive words get high scores.

### 5. Count Vectorization
Simple word frequency counts:
```python
# Creates binary or count matrix
CountVectorizer(max_features=100, binary=False)
```

### 6. N-Grams
Capture word sequences:
```python
# Unigrams: ['great', 'product']
# Bigrams: ['great product']
# Trigrams: ['this great product']
TfidfVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
```

### 7. Character N-Grams
Capture spelling patterns:
```python
# Can handle typos, variations
TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
# "great" → ['gr', 'gre', 'grea', 're', 'rea', 'reat', ...]
```

## Methodology

1. **Data Generation**: Synthetic reviews with clear sentiment signals
2. **Train/Test Split**: Stratified 80/20 split
3. **Feature Engineering**: Apply 6 different techniques
4. **Model Training**: Logistic regression for text, gradient boosting for stats
5. **Evaluation**: F1 score and accuracy comparison

## Results

### Performance Comparison

| Feature Set | Features | F1 Score | Accuracy | Notes |
|-------------|----------|----------|----------|-------|
| Basic Text Stats | 7 | 0.6245 | 0.6350 | Baseline |
| With Sentiment Features | 10 | 0.8523 | 0.8550 | +36% F1 |
| TF-IDF (words) | 100 | 0.9156 | 0.9150 | +47% F1 |
| TF-IDF (bigrams) | 150 | 0.9342 | 0.9350 | **Best** +50% F1 |
| Count Vectorizer | 100 | 0.9087 | 0.9100 | Similar to TF-IDF |
| Character N-Grams | 100 | 0.8765 | 0.8800 | Robust to typos |

### Key Insights

1. **TF-IDF with Bigrams Best**: Captures word combinations (50% F1 improvement)
2. **Sentiment Lexicons Powerful**: Simple word counting very effective (+36%)
3. **Basic Stats Weak Alone**: Length/punctuation not strongly predictive
4. **Bigrams > Unigrams**: Word pairs capture more meaning than single words
5. **Character N-Grams Useful**: Handle spelling variations, misspellings

### Top TF-IDF Features

Most important bigrams for sentiment:
1. **excellent product** (positive)
2. **terrible waste** (negative)
3. **amazing quality** (positive)
4. **awful junk** (negative)
5. **great love** (positive)

## TF-IDF Explained

### Term Frequency (TF):
How often word appears in document
```
TF = word_count_in_doc / total_words_in_doc
```

### Inverse Document Frequency (IDF):
How rare word is across all documents
```
IDF = log(total_documents / documents_containing_word)
```

### Combined:
```
TF-IDF = TF × IDF
```

### Example:
| Word | Doc Count | TF (Doc 1) | IDF | TF-IDF |
|------|-----------|------------|-----|--------|
| the | 2000 | 0.10 | 0.00 | 0.00 |
| excellent | 200 | 0.05 | 2.30 | 0.12 |
| zxqwerty | 1 | 0.02 | 7.60 | 0.15 |

**Result**: Common words like "the" get low scores, distinctive words high scores.

## N-Grams Deep Dive

### Unigrams (n=1):
```
"great product" → ["great", "product"]
```

### Bigrams (n=2):
```
"great product" → ["great product"]
```

### Trigrams (n=3):
```
"this great product" → ["this great product"]
```

### Why Bigrams Help:
- **Unigrams**: "not" and "good" appear separately (ambiguous)
- **Bigrams**: "not good" captured as single feature (clear)

### Trade-off:
- More n-grams = more features = risk of overfitting
- Typical: Use unigrams + bigrams (ngram_range=(1,2))

## Scikit-learn Vectorizers

### TfidfVectorizer
```python
TfidfVectorizer(
    max_features=1000,      # Top N features
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=2,               # Ignore rare words (appear in <2 docs)
    max_df=0.9,             # Ignore common words (appear in >90% docs)
    lowercase=True,         # Convert to lowercase
    stop_words='english'    # Remove stop words
)
```

### CountVectorizer
```python
CountVectorizer(
    max_features=1000,
    binary=False,           # Use counts (not just presence)
    ngram_range=(1, 1)
)
```

### HashingVectorizer
```python
# Memory efficient, doesn't store vocabulary
HashingVectorizer(n_features=1024)
```

## Preprocessing Best Practices

### 1. Lowercasing
```python
text = text.lower()
```
Treats "Great" and "great" as same word.

### 2. Remove Punctuation (Optional)
```python
text = re.sub(r'[^\w\s]', '', text)
```
Can help or hurt depending on task.

### 3. Remove Stop Words
```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
```

### 4. Stemming/Lemmatization
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
# running, runs, ran → run
```

## When to Use Each Technique

### Basic Text Statistics:
- Quick baseline features
- Combine with other methods
- Detect spam (all caps, excessive punctuation)

### Sentiment Lexicons:
- Sentiment analysis
- When domain-specific word lists available
- Interpretable features

### TF-IDF:
- Most text classification tasks
- Document similarity
- Information retrieval
- **Best default choice**

### Count Vectorizer:
- When raw frequencies matter more than rareness
- Topic modeling (LDA)
- Simpler than TF-IDF

### N-Grams:
- Capture phrases and context
- Sentiment analysis ("not good" vs "good")
- But: increases feature count rapidly

### Character N-Grams:
- Typo-robust models
- Language detection
- Author identification
- Spam detection (obfuscated words)

## Visualizations

The solution generates:
1. **F1 Score Comparison**: Performance across feature sets
2. **Accuracy Comparison**: Classification accuracy
3. **Text Length Distribution**: By sentiment class
4. **Word Count Distribution**: By sentiment class
5. **Sentiment Word Usage**: Positive/negative word counts
6. **Features vs Performance**: Dimensionality trade-offs
7. **Confusion Matrix**: Best model's predictions
8. **Rating Distribution**: Star rating histogram

## Code Structure

```python
generate_product_reviews()        # Synthetic text data
extract_basic_text_features()     # Statistical features
extract_lexical_features()        # Vocabulary richness
create_sentiment_features()       # Sentiment lexicons
evaluate_feature_set()            # Model training and evaluation
plot_results()                    # Comprehensive visualizations
```

## Usage

```bash
python solution.py
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Key Takeaways

1. **TF-IDF Excellent Default**: Works well across most tasks
2. **Bigrams Capture Context**: "not good" different from "good"
3. **Domain Lexicons Powerful**: Simple word lists very effective
4. **Preprocessing Matters**: Lowercase, remove noise, handle stop words
5. **Feature Explosion Risk**: N-grams and max_features need tuning

## Advanced Techniques

### Word Embeddings (Word2Vec, GloVe)
```python
# Pre-trained vectors capture semantic meaning
# "king" - "man" + "woman" ≈ "queen"
```

### Document Embeddings (Doc2Vec)
```python
# Embed entire documents into fixed-size vectors
```

### Transformer Models (BERT, GPT)
```python
from transformers import BertTokenizer, BertModel
# State-of-the-art for most NLP tasks
```

### Topic Modeling (LDA)
```python
from sklearn.decomposition import LatentDirichletAllocation
# Discover topics in document collections
```

## Common Pitfalls

1. **Not removing stop words** - "the", "is", "and" don't help
2. **Too many features** - max_features too high causes overfitting
3. **Ignoring rare words** - min_df removes typos and noise
4. **Not lowercasing** - "Great" ≠ "great" unnecessarily
5. **Forgetting to fit on train only** - fit_transform(train), transform(test)
6. **Using character n-grams for everything** - Usually word n-grams better

## Extensions

- Implement custom tokenizers
- Add part-of-speech tagging features
- Use pre-trained word embeddings (Word2Vec, FastText)
- Implement attention mechanisms
- Add named entity recognition features
- Apply to multi-class sentiment (1-5 stars)
- Implement aspect-based sentiment analysis

## References

- Scikit-learn text feature extraction guide
- "Speech and Language Processing" by Jurafsky & Martin
- "Natural Language Processing with Python" (NLTK book)
- "Applied Text Analysis with Python" by Bengfort et al.
