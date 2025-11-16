# News Article Recommendation System using Content-Based Filtering

## Overview
This solution implements a news article recommendation system using content-based filtering with TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. The system recommends articles based on content similarity to what users have previously read.

## Problem Statement
Given user reading history and article content, build a recommendation system that can:
- Recommend personalized news articles to users
- Find similar articles to a given article
- Handle new users and new articles (cold start problem)
- Account for user engagement and reading behavior

## Approach

### 1. Content-Based Filtering
Unlike collaborative filtering, content-based filtering recommends items similar to what a user has liked in the past, based on item features (content).

### 2. TF-IDF Vectorization
Converts article text into numerical feature vectors:

**Term Frequency (TF)**: How often a term appears in a document
```
TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

**Inverse Document Frequency (IDF)**: How rare/common a term is across all documents
```
IDF(t) = log(Total number of documents / Number of documents containing term t)
```

**TF-IDF Score**:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### 3. User Profile Construction
Each user's profile is a weighted average of the TF-IDF vectors of articles they've read:
```
User_Profile = Σ(engagement_i × article_vector_i) / Σ(engagement_i)
```

### 4. Recommendation Generation
Recommendations are generated using cosine similarity:
```
similarity(user_profile, article) = (user_profile · article) / (||user_profile|| × ||article||)
```

## Key Features

1. **Personalized Recommendations**: Based on user's reading history
2. **Similar Article Discovery**: Find articles related to current reading
3. **Engagement Weighting**: More engaged reads have higher influence
4. **Cold Start Handling**: Trending articles for new users
5. **Content Analysis**: Understanding article topics and categories

## Evaluation Metrics

1. **Precision@10**
   - Proportion of top-10 recommendations that user engaged with
   - Measures recommendation accuracy

2. **Recall@10**
   - Proportion of user's engaged articles found in top-10
   - Measures recommendation coverage

3. **NDCG@10 (Normalized Discounted Cumulative Gain)**
   - Considers ranking position in evaluation
   - Higher-ranked relevant items contribute more

## Implementation Details

### Algorithm Workflow
1. Generate synthetic news articles with categories and content
2. Create user reading history with engagement metrics
3. Build TF-IDF matrix for all articles
4. Compute article-article similarity matrix
5. Build user profiles from reading history
6. Generate recommendations using cosine similarity
7. Evaluate using precision, recall, and NDCG

### Data Generation
- **Articles**: 500 articles across 8 categories
- **Users**: 300 users with category preferences
- **Interactions**: 5-30 articles per user with engagement scores
- **Categories**: Politics, Technology, Sports, Business, Health, Entertainment, Science, World

### Cold Start Solutions

**New Users**:
- Recommend trending/popular articles
- Can ask for category preferences
- Can use demographic information

**New Articles**:
- Content-based filtering handles this naturally
- New article's TF-IDF vector can be computed immediately

## Results

Typical performance metrics:
- **Precision@10**: ~0.20-0.35
- **Recall@10**: ~0.25-0.40
- **NDCG@10**: ~0.30-0.50

## Visualizations

The solution generates four visualizations:

1. **Article Distribution by Category**: Shows category balance
2. **Reading Time Distribution**: User engagement patterns
3. **Engagement Score Distribution**: Quality of interactions
4. **User Activity Distribution**: Number of articles read per user

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Advantages of Content-Based Filtering

1. **No Cold Start for Items**: New articles can be recommended immediately
2. **Transparency**: Easy to explain why an article was recommended
3. **User Independence**: Doesn't need data from other users
4. **Niche Interests**: Can recommend unpopular articles that match user interests

## Limitations

1. **Over-specialization**: Tends to recommend similar content
2. **Limited Discovery**: Hard to recommend outside user's normal interests
3. **Feature Engineering**: Requires good content representation
4. **Content Analysis**: Needs actual article text/metadata

## Improvements and Extensions

1. **Hybrid Approach**: Combine with collaborative filtering
2. **Deep Learning**: Use BERT or transformers for better text representation
3. **Temporal Decay**: Weight recent reads more than older ones
4. **Diversity**: Ensure recommendations cover multiple topics
5. **Click-Through Prediction**: Predict probability of user clicking
6. **Real-time Updates**: Update user profiles as they read
7. **Multi-modal**: Include images, videos, author information

## Business Applications

- **News Websites**: NYTimes, BBC, CNN recommendation engines
- **Content Platforms**: Medium, Substack article recommendations
- **RSS Readers**: Personalized feed curation
- **Email Newsletters**: Personalized content selection
- **Mobile Apps**: Push notification prioritization

## Technical Considerations

1. **Scalability**: TF-IDF scales well to millions of articles
2. **Real-time**: User profiles can be updated incrementally
3. **Storage**: Sparse matrix storage for efficiency
4. **Computation**: Pre-compute article similarities for speed
5. **Freshness**: Balance between relevance and recency

## References

- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information processing & management, 24(5), 513-523.
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. Recommender systems handbook, 73-105.
