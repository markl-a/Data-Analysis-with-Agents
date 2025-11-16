# 14. News Article Category Classification

## Overview
Automatically classify news articles into categories (Politics, Sports, Technology, Business, Entertainment) using natural language processing and machine learning techniques.

**Difficulty**: ⭐⭐⭐ Intermediate
**Domain**: Media, Journalism, Content Management
**Techniques**: TF-IDF, Naive Bayes, Logistic Regression, Multi-class Classification

## Problem Statement
Given a news article text, automatically classify it into one of five categories:
- Politics
- Sports
- Technology
- Business
- Entertainment

## Dataset
- **Size**: 1,200 synthetic news articles
- **Features**: Article text with category-specific vocabulary
- **Categories**: 5 balanced categories
- **Distribution**: ~240 articles per category

## Methodology

### 1. Data Generation
- Domain-specific vocabularies for each category
- Realistic sentence structures
- Category-appropriate subjects, verbs, adjectives, and contexts
- Varied article lengths (3-7 sentences)

### 2. Feature Engineering
- **TF-IDF vectorization** (1-2 grams, 3000 features)
- Article length and word count
- Sentence count
- Average word length
- Presence of numbers and quotes
- Punctuation patterns

### 3. Models Compared
- **Naive Bayes** (MultinomialNB with alpha=0.1)
- **Logistic Regression** (C=1.0, max_iter=1000) - Best performer
- **Random Forest** (100 estimators, max_depth=10)

### 4. Evaluation Metrics
- Classification accuracy
- Per-category precision, recall, F1-score
- Confusion matrix
- Cross-category misclassification analysis

## Key Features
- Multi-model comparison framework
- Domain-specific vocabulary generation
- Comprehensive text feature extraction
- Category-specific analysis and visualization
- Realistic article simulation

## Results
- **Expected Accuracy**: ~90-95%
- **Best Model**: Logistic Regression
- **Insights**: Politics and Business may have some overlap; Sports and Entertainment are highly distinct

## Visualizations
1. **Category distribution** - Balance across news types
2. **Word count by category** - Article length patterns
3. **Average word length** - Vocabulary complexity
4. **Sentence count distribution** - Article structure
5. **Confusion matrix** - Classification performance

## Use Cases
- Automated news categorization
- Content recommendation systems
- News aggregation and filtering
- Media monitoring and analysis
- Content management systems
- RSS feed categorization

## Running the Code
```bash
python solution.py
```

## Output Files
- `news_analysis.png` - Data distribution visualizations
- `news_confusion_matrix.png` - Model performance

## Key Learnings
1. Domain-specific vocabularies create clear category boundaries
2. TF-IDF with bigrams captures category-specific phrases
3. Logistic Regression performs well for multi-class text classification
4. Business and Politics articles may share economic terminology
5. Entertainment and Sports have distinct vocabularies

## Extensions
- Add more categories (Health, Science, World News)
- Implement hierarchical classification
- Extract named entities (people, organizations, locations)
- Detect article sentiment within categories
- Implement multilingual news classification
- Add article summarization
- Detect trending topics within categories
