# Restaurant Recommendation System using Item-Based Collaborative Filtering

## Overview
This solution implements a restaurant recommendation system using item-based collaborative filtering with location-aware features. The system recommends restaurants based on similarity to places users have previously enjoyed.

## Problem Statement
Given user restaurant reviews and ratings, build a recommendation system that can:
- Recommend personalized restaurant suggestions
- Find restaurants similar to user's favorites
- Consider location, cuisine, and price preferences
- Handle sparse user-restaurant interaction data

## Approach

### Item-Based Collaborative Filtering

Unlike user-based CF (which finds similar users), item-based CF finds similar items (restaurants):

1. **Build Item-Item Similarity Matrix**: Calculate similarity between all restaurant pairs based on user ratings
2. **Generate Predictions**: For each user, predict ratings based on their previous ratings and item similarities

**Key Advantage**: Item similarities are more stable than user similarities (restaurants don't change behavior like users do)

### Similarity Calculation

For two restaurants i and j:
```
similarity(i,j) = cosine_similarity(ratings_i, ratings_j)
```

Where ratings_i is the vector of all user ratings for restaurant i.

### Rating Prediction

To predict user u's rating for restaurant i:
```
pred(u,i) = Σ(similarity(i,j) × rating(u,j)) / Σ|similarity(i,j)|
```

Where j ranges over all restaurants that user u has rated.

## Key Features

1. **Item-Based CF**: More stable and scalable than user-based
2. **Location Awareness**: Incorporates geographic proximity
3. **Cuisine Preferences**: Models user taste patterns
4. **Price Sensitivity**: Accounts for budget preferences
5. **Cold Start Handling**: Popular restaurants for new users
6. **Similar Restaurant Discovery**: "If you liked X, try Y"

## Data Generation

### Restaurant Attributes
- **Cuisines**: 10 types (Italian, Chinese, Japanese, Mexican, etc.)
- **Price Ranges**: $, $$, $$$, $$$$
- **Neighborhoods**: 8 different areas
- **Locations**: Latitude/longitude coordinates
- **Statistics**: Average rating, number of reviews

### User Behavior
- Users have cuisine preferences (1-3 favorite cuisines)
- Users have home locations
- 70% of visits to preferred cuisines
- Distance affects ratings (closer restaurants rated higher)
- Each user reviews 3-15 restaurants

## Evaluation Metrics

1. **MAE (Mean Absolute Error)**
   - Average difference between predicted and actual ratings
   - Measures prediction accuracy

2. **RMSE (Root Mean Squared Error)**
   - Penalizes larger errors more
   - Standard metric for rating prediction

3. **Precision@10**
   - Of top-10 recommendations, how many were highly rated?
   - Measures recommendation quality

4. **Recall@10**
   - Of all highly-rated restaurants, how many in top-10?
   - Measures recommendation coverage

## Implementation Details

### Algorithm Workflow
1. Generate synthetic restaurant and review data
2. Create user-item rating matrix (users × restaurants)
3. Calculate item-item similarity matrix using cosine similarity
4. For each user, predict ratings for unvisited restaurants
5. Rank restaurants by predicted rating
6. Evaluate using multiple metrics

### Matrix Sparsity
Restaurant data is typically very sparse (95-98%):
- Most users visit small fraction of restaurants
- Most restaurants visited by small fraction of users
- Item-based CF handles sparsity well

## Results

Typical performance metrics:
- **MAE**: ~0.6-0.9 (on 1-5 scale)
- **RMSE**: ~0.8-1.2
- **Precision@10**: ~0.20-0.35
- **Recall@10**: ~0.25-0.40

## Visualizations

The solution generates four visualizations:

1. **Restaurant Distribution by Cuisine**: Category balance
2. **Rating Distribution**: User rating patterns
3. **Price Range Distribution**: Restaurant price levels
4. **User Activity Distribution**: Reviews per user

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
- scipy

## Advantages of Item-Based CF

1. **Stability**: Restaurant characteristics don't change frequently
2. **Scalability**: Can pre-compute item similarities
3. **Explainability**: "You liked A, which is similar to B"
4. **Quality**: Often outperforms user-based for sparse data
5. **Serendipity**: Can recommend diverse options

## Limitations

1. **Cold Start Items**: New restaurants have no similarity data
2. **Popularity Bias**: Tends to recommend popular restaurants
3. **Computational Cost**: O(n²) similarity calculations
4. **Filter Bubble**: May not recommend outside user's normal range

## Improvements and Extensions

1. **Location Filtering**: Only show restaurants within reasonable distance
2. **Time-Aware**: Consider day of week, meal time, season
3. **Context-Aware**: Occasion (date night, family dinner, business lunch)
4. **Group Recommendations**: Find restaurants all friends will enjoy
5. **Review Text Analysis**: Use NLP on reviews for better understanding
6. **Image Analysis**: Use food photos for visual similarity
7. **Hybrid with Content**: Combine with cuisine/attribute matching
8. **Real-Time Updates**: Update recommendations as user browses

## Business Applications

### For Platforms
- **Yelp**: Restaurant recommendations on home page
- **Google Maps**: "Places you might like"
- **OpenTable**: Smart reservation suggestions
- **Uber Eats/DoorDash**: Delivery recommendations

### For Restaurants
- **Cross-Promotion**: Partner with similar restaurants
- **Market Analysis**: Understand competitive landscape
- **Menu Optimization**: Learn from similar successful restaurants
- **Pricing Strategy**: Compare with similar establishments

### For Users
- **Discovery**: Find new favorite restaurants
- **Planning**: Choose restaurants for special occasions
- **Exploration**: Venture outside usual preferences
- **Confidence**: Make better dining decisions

## Advanced Features

1. **Temporal Patterns**: Lunch vs. dinner preferences
2. **Social Features**: Friends' recommendations weighted higher
3. **Reservation Integration**: Show availability in recommendations
4. **Menu Matching**: Recommend based on specific dishes
5. **Dietary Restrictions**: Filter for vegetarian, gluten-free, etc.
6. **Ambiance Matching**: Quiet vs. lively atmosphere
7. **Wait Time Prediction**: Show estimated wait times
8. **Special Events**: Recommend for birthdays, anniversaries

## Technical Considerations

1. **Pre-computation**: Calculate similarities offline for speed
2. **Incremental Updates**: Update similarities as new reviews arrive
3. **Top-K Similarity**: Store only most similar items
4. **Distributed Computing**: Use Spark for large-scale computation
5. **Caching**: Cache recommendations for frequent users
6. **A/B Testing**: Test different similarity metrics

## References

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. WWW '01.
- Linden, G., Smith, B., & York, J. (2003). Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet computing, 7(1), 76-80.
