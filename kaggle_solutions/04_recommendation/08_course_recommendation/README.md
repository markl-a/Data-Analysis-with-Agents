# Online Course Recommendation System using User-Based Collaborative Filtering

## Overview
This solution implements an online course recommendation system using user-based collaborative filtering. The system recommends courses based on the learning patterns of similar users, making it ideal for personalized learning path suggestions.

## Problem Statement
Given user course enrollment and completion data, build a recommendation system that can:
- Recommend personalized course suggestions
- Consider skill level progression
- Respect prerequisite requirements
- Handle diverse learning goals and paces
- Balance exploration and exploitation

## Approach

### User-Based Collaborative Filtering

The core idea: "Users who agreed in the past tend to agree in the future"

1. **Find Similar Users**: Identify users with similar course-taking patterns
2. **Aggregate Ratings**: Use similar users' ratings to predict unknown ratings
3. **Generate Recommendations**: Recommend highly-predicted courses

**Prediction Formula**:
```
pred(u,i) = Σ(similarity(u,v) × rating(v,i)) / Σ|similarity(u,v)|
```

Where:
- u is the target user
- v ranges over k most similar users
- i is the course to predict

### User Similarity

Using cosine similarity between user rating vectors:
```
similarity(u,v) = (ratings_u · ratings_v) / (||ratings_u|| × ||ratings_v||)
```

## Key Features

1. **User-Based CF**: Leverages "wisdom of the crowd"
2. **Skill Progression**: Models beginner → intermediate → advanced paths
3. **Prerequisites**: Tracks course dependencies
4. **Completion Tracking**: Uses completion rates in addition to ratings
5. **Category Specialization**: Users focus on 1-2 categories
6. **Cold Start Handling**: Popular courses for new users

## Data Generation

### Course Attributes
- **Categories**: Programming, Data Science, Web Dev, Mobile Dev, Business, Marketing, Design, Personal Development
- **Levels**: Beginner, Intermediate, Advanced, Expert
- **Duration**: 2-12 hours
- **Prerequisites**: Courses may require prior courses

### User Behavior Modeling
- Users have 1-2 focus areas (categories)
- Users progress through levels in their focus areas
- 80% of enrollments in interest areas
- Higher completion for appropriate-level courses
- Rating correlates with interest and level match

### Learning Path Progression
- Beginners start with beginner courses
- Successful completion unlocks next level
- 80%+ completion required to progress
- Users naturally advance through levels

## Evaluation Metrics

1. **MAE (Mean Absolute Error)**
   - Average error in rating predictions
   - Measures prediction accuracy

2. **Precision@10**
   - Proportion of recommended courses user would rate highly
   - Measures recommendation relevance

3. **Recall@10**
   - Proportion of all good courses found in recommendations
   - Measures recommendation coverage

4. **NDCG@10**
   - Normalized Discounted Cumulative Gain
   - Accounts for ranking position

## Implementation Details

### Algorithm Workflow
1. Generate synthetic course catalog and enrollments
2. Create user-item rating matrix
3. Calculate user-user similarity matrix using cosine similarity
4. For each user, find k most similar users (k=20)
5. Predict ratings as weighted average of similar users
6. Rank courses by predicted rating
7. Evaluate using multiple metrics

### Matrix Construction
- Rows: Users (learners)
- Columns: Courses
- Values: Ratings (1-5 scale)
- Sparsity: 95-98% (most user-course pairs unobserved)

### Similarity Computation
- Cosine similarity between user rating vectors
- Only considers courses both users have rated
- Handles different rating scales naturally

## Results

Typical performance metrics:
- **MAE**: ~0.7-1.0 (on 1-5 scale)
- **Precision@10**: ~0.25-0.40
- **Recall@10**: ~0.30-0.45
- **NDCG@10**: ~0.35-0.50

## Visualizations

The solution generates four visualizations:

1. **Course Distribution by Category**: Balance across subjects
2. **Completion Rate Distribution**: Student engagement levels
3. **Rating Distribution**: Quality perception patterns
4. **Level Distribution**: Course difficulty balance

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

## Advantages of User-Based CF

1. **Serendipity**: Can recommend unexpected but relevant courses
2. **No Domain Knowledge**: Doesn't need course metadata
3. **Captures Trends**: Learns from collective behavior
4. **Personalization**: Different users get different recommendations
5. **Social Proof**: Recommendations backed by similar learners

## Limitations

1. **Scalability**: O(n²) user comparisons for n users
2. **Sparsity**: Many users have few ratings
3. **Cold Start Users**: New users have no similarity data
4. **Popularity Bias**: May over-recommend popular courses
5. **Shilling Attacks**: Vulnerable to fake ratings

## Improvements and Extensions

1. **Hybrid Approach**: Combine with content-based (skill matching)
2. **Sequential Patterns**: Model learning path sequences
3. **Time Decay**: Weight recent enrollments higher
4. **Skill Graphs**: Build comprehensive skill dependency graphs
5. **Learning Objectives**: Explicitly model career goals
6. **Peer Groups**: Find study buddies with similar interests
7. **Adaptive Difficulty**: Recommend courses matching current skill
8. **Multi-Armed Bandits**: Balance exploration vs. exploitation

## Business Applications

### For Learning Platforms
- **Coursera**: "Courses you might like"
- **Udemy**: Personalized course discovery
- **LinkedIn Learning**: Career path recommendations
- **Khan Academy**: Personalized learning paths
- **edX**: Professional certificate programs

### For Learners
- **Career Development**: Courses for career transitions
- **Skill Gaps**: Identify and fill knowledge gaps
- **Learning Paths**: Structured progression through topics
- **Peer Learning**: Connect with similar learners

### For Course Creators
- **Market Analysis**: Understand learner demand
- **Course Sequencing**: Design better learning paths
- **Content Gaps**: Identify missing courses
- **Pricing Strategy**: Compare with similar courses

## Advanced Features

1. **Learning Style Matching**: Visual vs. reading-based
2. **Time Commitment**: Match available time to course duration
3. **Certification Paths**: Recommend certificate programs
4. **Job Market Alignment**: Skills demanded by employers
5. **Real-Time Progress**: Update recommendations as user learns
6. **Collaborative Projects**: Suggest team learning opportunities
7. **Mentor Matching**: Connect learners with mentors
8. **Knowledge Testing**: Adaptive skill assessments

## Educational Impact

1. **Personalized Learning**: Each student's unique path
2. **Reduced Dropouts**: Better course-learner matching
3. **Skill Development**: Systematic capability building
4. **Career Mobility**: Enable career transitions
5. **Lifelong Learning**: Continuous education support
6. **Accessibility**: Match learning needs and abilities

## Technical Considerations

1. **Incremental Updates**: Update similarities as users learn
2. **Cold Start**: Content-based for new users/courses
3. **Scalability**: Use approximate nearest neighbors
4. **Real-Time**: Pre-compute user similarities
5. **A/B Testing**: Test different k values and similarity metrics
6. **Privacy**: Protect student learning data

## References

- Drachsler, H., Verbert, K., Santos, O. C., & Manouselis, N. (2015). Panorama of recommender systems to support learning. Handbook on learning analytics, 421-451.
- Bobadilla, J., Ortega, F., Hernando, A., & Gutiérrez, A. (2013). Recommender systems survey. Knowledge-based systems, 46, 109-132.
