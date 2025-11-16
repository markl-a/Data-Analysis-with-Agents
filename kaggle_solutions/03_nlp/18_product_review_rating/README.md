# 18. Product Review Rating Prediction

## Overview
Predict star ratings (1-5) from product review text using NLP and machine learning. This ordinal classification problem combines sentiment analysis with rating prediction.

**Difficulty**: ⭐⭐⭐ Intermediate
**Domain**: E-commerce, Customer Feedback, Sentiment Analysis
**Techniques**: TF-IDF, Gradient Boosting, Ordinal Classification

## Problem Statement
Given a product review text, predict the star rating (1-5 stars):
- **1 Star** - Very negative, terrible experience
- **2 Stars** - Negative, below expectations
- **3 Stars** - Neutral, acceptable
- **4 Stars** - Positive, satisfied
- **5 Stars** - Very positive, excellent

## Dataset
- **Size**: 1,200 synthetic product reviews
- **Features**: Review text with rating-specific vocabulary
- **Labels**: Star ratings (1-5)
- **Distribution**: Realistic (more 4-5 star reviews, fewer 1-2 stars)

## Methodology

### 1. Data Generation
- Rating-specific vocabularies:
  - Adjectives matching sentiment intensity
  - Action verbs indicating satisfaction level
  - Common review phrases
- Realistic review structure:
  - Opening statement
  - Product assessment
  - Quality evaluation
  - Purchase recommendation
- Realistic rating distribution (skewed toward positive)

### 2. Feature Engineering
- **TF-IDF vectorization** (1-3 grams, 3000 features)
- Review length and word count
- Average word length
- Positive word count
- Negative word count
- Exclamation mark frequency
- Question mark frequency
- Sentiment score (positive - negative words)

### 3. Model
- **Gradient Boosting Classifier**
  - 100 estimators
  - Learning rate: 0.1
  - Max depth: 5
  - Treats as ordinal classification problem

### 4. Evaluation Metrics
- **Accuracy** - Exact rating match
- **Mean Absolute Error (MAE)** - Average rating distance
- Per-rating precision, recall, F1-score
- Confusion matrix showing prediction patterns

## Key Features
- Ordinal classification approach
- Sentiment-aware feature engineering
- Rating-specific vocabulary generation
- Realistic review patterns
- Comprehensive evaluation metrics

## Results
- **Expected Accuracy**: ~75-85%
- **Expected MAE**: ~0.3-0.5 stars
- **Insights**: Sentiment words are strong predictors; adjacent ratings often confused

## Visualizations
1. **Rating distribution** - Class balance in dataset
2. **Word count by rating** - Review length patterns
3. **Sentiment score** - Positive/negative word balance
4. **Positive words by rating** - Clear upward trend
5. **Negative words by rating** - Clear downward trend
6. **Confusion matrix** - Adjacent rating confusion

## Use Cases
- Automated review analysis
- Product quality monitoring
- Customer satisfaction tracking
- Review filtering and sorting
- Fake review detection
- Sentiment trend analysis
- Competitive analysis
- Product improvement insights

## Running the Code
```bash
python solution.py
```

## Output Files
- `review_analysis.png` - Data distribution and feature visualizations
- `review_confusion_matrix.png` - Model performance

## Key Learnings
1. Star ratings correlate strongly with sentiment word usage
2. Positive reviews tend to be shorter and more enthusiastic
3. Negative reviews often provide more detailed complaints
4. 3-star reviews are hardest to predict (neutral sentiment)
5. Adjacent ratings (4 vs 5, 1 vs 2) are frequently confused
6. TF-IDF with trigrams captures rating-specific phrases
7. MAE is important for ordinal classification evaluation

## Ordinal Classification Challenges
- Treating ordered categories (1<2<3<4<5)
- Adjacent class confusion
- Imbalanced distribution (more positive reviews)
- Neutral sentiment ambiguity
- Context-dependent ratings

## Extensions
- Extract aspect-based sentiment (price, quality, shipping)
- Detect fake or spam reviews
- Predict review helpfulness
- Multi-product category analysis
- Temporal rating trends
- Reviewer credibility scoring
- Review summarization
- Compare ratings across products
- Detect review manipulation
- Sentiment intensity scoring
- Add image-based rating prediction
- Cross-platform review analysis
