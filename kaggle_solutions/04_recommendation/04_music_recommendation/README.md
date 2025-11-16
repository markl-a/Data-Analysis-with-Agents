# Music Recommendation System using Matrix Factorization (SVD)

## Overview
This solution implements a music recommendation system using Singular Value Decomposition (SVD) for collaborative filtering. The system analyzes user listening patterns to recommend songs that users are likely to enjoy.

## Problem Statement
Given user-song interaction data (play counts and ratings), build a recommendation system that can:
- Predict ratings for unheard songs
- Recommend personalized song lists for users
- Handle sparse user-item matrices efficiently
- Provide recommendations for cold-start users

## Approach

### 1. Data Generation
- Creates synthetic music listening data with 500 users and 300 songs
- Generates user preferences for specific music genres
- Simulates realistic listening patterns (users prefer certain genres)
- Includes play counts and ratings (1-5 scale)

### 2. Matrix Factorization (SVD)
The solution uses Singular Value Decomposition to decompose the user-item rating matrix:

```
R ≈ U × Σ × V^T
```

Where:
- R is the user-item rating matrix (n_users × n_songs)
- U contains user latent factors (n_users × k)
- Σ is the diagonal matrix of singular values (k × k)
- V^T contains song latent factors (k × n_songs)
- k is the number of latent factors (50 in this implementation)

### 3. Key Features
- **Dimensionality Reduction**: Reduces the rating matrix to 50 latent factors
- **Mean Normalization**: Centers ratings by user mean to handle rating bias
- **Cold Start Handling**: Falls back to popularity-based recommendations
- **Evaluation Metrics**: RMSE, MAE, Precision@10, Recall@10

## Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**
   - Measures prediction accuracy for ratings
   - Lower is better

2. **MAE (Mean Absolute Error)**
   - Average absolute difference between predicted and actual ratings
   - More interpretable than RMSE

3. **Precision@10**
   - Proportion of recommended songs that user actually likes
   - Measures recommendation relevance

4. **Recall@10**
   - Proportion of liked songs that were recommended
   - Measures recommendation coverage

## Implementation Details

### Algorithm Workflow
1. Generate synthetic user-song interaction data
2. Create user-item rating matrix (sparse)
3. Normalize ratings by subtracting user mean
4. Apply SVD to decompose the matrix
5. Reconstruct predictions by multiplying factors
6. Generate top-N recommendations for each user
7. Evaluate using multiple metrics

### Cold Start Problem
For users with no history:
- Recommend most popular songs across all users
- Can be enhanced with demographic or content-based filtering

## Results

Typical performance metrics:
- **RMSE**: ~0.8-1.2 (on 1-5 rating scale)
- **MAE**: ~0.6-0.9
- **Precision@10**: ~0.15-0.25
- **Recall@10**: ~0.20-0.35

## Visualizations

The solution generates four visualizations:

1. **Rating Distribution**: Shows how users rate songs
2. **User Activity Distribution**: Number of songs rated per user
3. **Song Popularity Distribution**: Number of ratings per song
4. **Explained Variance**: How much variance is captured by latent factors

## Usage

```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

## Key Insights

1. **Matrix Sparsity**: Music rating matrices are typically 95-99% sparse
2. **Long Tail**: Most songs have few ratings, few songs have many ratings
3. **Latent Factors**: 50 factors capture ~60-80% of variance in user preferences
4. **Genre Preference**: Users consistently prefer certain genres, making collaborative filtering effective

## Improvements and Extensions

1. **Hybrid Approach**: Combine with content-based features (genre, tempo, artist)
2. **Implicit Feedback**: Use play counts and skips instead of explicit ratings
3. **Temporal Dynamics**: Account for changing user preferences over time
4. **Context-Aware**: Consider time of day, mood, activity when recommending
5. **Deep Learning**: Use neural collaborative filtering or autoencoders

## Business Applications

- **Streaming Services**: Spotify, Apple Music, YouTube Music
- **Playlist Generation**: Automatic playlist creation based on user taste
- **Discovery**: Help users find new artists and songs
- **User Retention**: Personalized recommendations increase engagement
- **A/B Testing**: Compare different recommendation strategies

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems handbook. Springer.
