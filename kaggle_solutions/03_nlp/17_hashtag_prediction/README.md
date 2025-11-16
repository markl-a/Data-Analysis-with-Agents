# 17. Social Media Hashtag Prediction

## Overview
Predict relevant hashtags for social media posts using multi-label text classification. This system analyzes post content and recommends appropriate hashtags.

**Difficulty**: ⭐⭐⭐ Intermediate
**Domain**: Social Media, Content Marketing, Digital Marketing
**Techniques**: TF-IDF, Multi-label Classification, One-vs-Rest

## Problem Statement
Given a social media post text, predict one or more relevant hashtags from a set of popular hashtags:
- #tech
- #business
- #lifestyle
- #fitness
- #food
- #travel
- #photography
- #fashion
- #motivation
- #news

## Dataset
- **Size**: 1,000 synthetic social media posts
- **Features**: Post text with topic-specific content
- **Labels**: Multi-label (1-3 hashtags per post)
- **Distribution**: Varying hashtag combinations

## Methodology

### 1. Data Generation
- Topic-specific content vocabularies
- Realistic social media post patterns
- Multi-label assignments (1-3 hashtags per post)
- Natural language variations
- Emoji and punctuation patterns

### 2. Feature Engineering
- **TF-IDF vectorization** (1-2 grams, 2000 features)
- Post length and word count
- Number of hashtags per post
- Exclamation mark presence
- Question mark presence
- Average word length

### 3. Model
- **Multi-label Classification**
  - OneVsRestClassifier wrapper
  - Logistic Regression base classifier
  - Independent binary classifier per hashtag
  - Probability threshold for predictions

### 4. Evaluation Metrics
- **Hamming Loss** - Average label-wise error
- **Subset Accuracy** - Exact match of all labels
- Per-hashtag accuracy
- Label frequency analysis

## Key Features
- Multi-label classification approach
- Topic-specific vocabulary generation
- Content-hashtag correlation modeling
- Flexible label assignment (1-3 tags)
- Social media text patterns

## Results
- **Expected Hamming Loss**: ~0.05-0.15
- **Subset Accuracy**: ~60-75%
- **Per-hashtag Accuracy**: ~85-95%
- **Insights**: Content keywords strongly predict hashtags

## Visualizations
1. **Predicted hashtag frequency** - Distribution of recommendations
2. **Actual hashtag frequency** - True label distribution
3. **Hashtags per post** - Multi-label count distribution
4. **Post length distribution** - Text characteristics
5. **Training data analysis** - Hashtag patterns

## Use Cases
- Automated hashtag suggestions
- Social media content optimization
- Marketing campaign tagging
- Content categorization
- Trend analysis
- Social media scheduling tools
- Influencer marketing
- Brand monitoring

## Running the Code
```bash
python solution.py
```

## Output Files
- `hashtag_data_analysis.png` - Training data visualizations
- `hashtag_analysis.png` - Prediction results

## Key Learnings
1. Multi-label classification requires different evaluation metrics
2. OneVsRestClassifier enables independent hashtag prediction
3. Content keywords are strong predictors of relevant hashtags
4. Posts can have multiple relevant hashtags
5. TF-IDF captures topic-specific terms effectively
6. Hamming loss better reflects partial correctness

## Multi-label Challenges
- Imbalanced label combinations
- Label correlation and dependencies
- Threshold selection for predictions
- Evaluation complexity
- Overfitting to popular hashtags

## Extensions
- Add more hashtags (50-100 common tags)
- Implement hashtag embeddings
- Detect trending hashtags
- Personalized hashtag recommendations
- Hashtag performance prediction
- Multi-language hashtag support
- Image-based hashtag prediction
- Temporal hashtag trends
- Hashtag reach estimation
- Competitive hashtag analysis
