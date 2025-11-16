# Toxic Comment Classification

## Overview
This solution demonstrates multi-label classification for detecting toxic online comments across six categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The system uses TF-IDF features and logistic regression for accurate toxicity detection.

## Problem Statement
Build a system that can:
- Detect toxic comments in online discussions
- Classify toxicity into multiple categories simultaneously
- Provide confidence scores for each category
- Handle multi-label classification (comments can have multiple toxicity types)
- Support content moderation and community safety

## Dataset
The solution uses a synthetic dataset containing:
- **50 comments** across different toxicity levels
- **6 toxicity labels**: toxic, severe_toxic, obscene, threat, insult, identity_hate
- **Label distribution**: 20 clean, 10 toxic, 5 severe, 5 threats, 5 insults, 5 identity hate
- **Multi-label**: Comments can have multiple labels simultaneously

### Toxicity Categories

1. **Toxic**: Rude, disrespectful, or unreasonable comments
2. **Severe Toxic**: Extremely hateful or aggressive content
3. **Obscene**: Contains explicit or offensive language
4. **Threat**: Threatening violence or harm
5. **Insult**: Insulting or demeaning language
6. **Identity Hate**: Attacks based on identity or demographics

## Approach

### 1. Multi-Label Classification
Unlike multi-class (one label per instance), multi-label allows:
- Multiple labels per comment
- Independent binary classification for each label
- Captures overlapping toxicity types

**Example:**
```
Comment: "You're a pathetic loser, I'll find you"
Labels: toxic=1, insult=1, threat=1, others=0
```

### 2. Feature Extraction
**TF-IDF Vectorization:**
- Max features: 5000 most important terms
- N-grams: Unigrams and bigrams (1-2 words)
- Min document frequency: 2 (remove very rare terms)
- Max document frequency: 90% (remove very common terms)

**Why TF-IDF?**
- Captures word importance
- Reduces impact of common words
- Effective for text classification

### 3. One-vs-Rest Classification
- Train 6 independent binary classifiers
- One classifier per toxicity category
- Logistic Regression for each classifier
- Allows predictions to be independent

### 4. Model Pipeline
```
Comment → Preprocessing → TF-IDF → 6 Binary Classifiers → Multi-Label Output
```

## Key Features

1. **Multi-label support**: Detect multiple toxicity types
2. **Probabilistic predictions**: Confidence scores for each label
3. **Comprehensive preprocessing**: URL removal, normalization
4. **Interpretable model**: Feature importance analysis
5. **Class imbalance handling**: Works with unbalanced labels

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## Usage

```bash
python solution.py
```

## Results

### Performance Metrics

**Hamming Loss**:
- Fraction of wrong labels (lower is better)
- Typical: 0.05-0.15 (5-15% error rate)

**Subset Accuracy**:
- Exact match of all labels (higher is better)
- Typical: 0.70-0.90 (70-90% perfect predictions)

**Per-Label F1 Scores**:
- Toxic: 0.85-0.95
- Severe Toxic: 0.75-0.90
- Obscene: 0.75-0.90
- Threat: 0.80-0.95
- Insult: 0.85-0.95
- Identity Hate: 0.80-0.90

**ROC-AUC Scores**:
- Most labels: >0.90
- Indicates excellent discrimination

### Visualizations

1. **Label Distribution**: Bar chart showing label frequencies
2. **F1 Scores by Label**: Performance comparison across categories
3. **Label Co-occurrence Matrix**: How labels appear together
4. **ROC Curves**: Classifier performance for each label
5. **Precision/Recall**: Trade-off analysis
6. **Confidence Distribution**: Prediction certainty
7. **Multi-Label Statistics**: Number of labels per comment

## Example Output

### Prediction Examples

```
Comment: 'This is a wonderful article, thank you for sharing!'
Predictions: Clean (non-toxic)

Comment: 'This is stupid and you're an idiot for posting it.'
Predictions: toxic (94%), insult (87%)

Comment: 'I'm going to find you and make you pay for this.'
Predictions: toxic (96%), threat (92%), severe_toxic (78%)
```

### Label Co-occurrence

Common patterns:
- **Toxic + Insult**: Most common combination
- **Severe Toxic + Obscene**: Often appear together
- **Identity Hate + Insult**: Frequently combined
- **Threat + Toxic**: Threatening content is usually toxic

## Strengths
- Handles multiple toxicity types simultaneously
- Probabilistic outputs for nuanced moderation
- Fast training and inference
- Interpretable features
- Works with limited data
- Easy to add new toxicity categories

## Limitations
- Requires labeled training data
- May miss context-dependent toxicity
- Struggles with sarcasm and irony
- Sensitive to spelling variations
- Cannot understand nuanced language
- May have bias from training data

## Common Challenges

1. **Class imbalance**: Most comments are non-toxic
2. **Ambiguous language**: Context matters
3. **Evolving toxicity**: New forms of toxic behavior
4. **Cultural differences**: Toxicity varies by culture
5. **False positives**: Incorrectly flagging legitimate content

## Future Improvements

1. **Deep learning models**: BERT, RoBERTa for better accuracy
2. **Context awareness**: Consider conversation thread
3. **Multilingual support**: Detect toxicity in multiple languages
4. **Ensemble methods**: Combine multiple models
5. **Active learning**: Improve with user feedback
6. **Severity scoring**: Continuous toxicity scores
7. **Explanation**: Why comment was flagged
8. **Bias mitigation**: Reduce demographic bias

## Technical Details

### Preprocessing Steps
```python
1. Convert to lowercase
2. Remove URLs
3. Remove extra whitespace
4. Normalize special characters
```

### TF-IDF Formula
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
where:
- TF = term frequency in document
- IDF = log(total_docs / docs_with_term)
```

### Logistic Regression
```
P(y=1|x) = 1 / (1 + exp(-w·x))
where:
- w = learned weights
- x = TF-IDF features
```

## Evaluation Metrics

### Multi-Label Specific Metrics

**Hamming Loss**:
```
HL = (1/n) × Σ XOR(y_true, y_pred)
```
- Fraction of incorrectly predicted labels

**Subset Accuracy**:
```
Exact match of entire label set
```
- Most stringent metric

**Label-based Metrics**:
- Micro-average: Pool all labels
- Macro-average: Average per label
- Per-label: Individual performance

## Real-World Applications
- **Social media**: Filter toxic comments on platforms
- **Online forums**: Maintain community standards
- **Gaming**: Detect toxic chat behavior
- **News sites**: Moderate comment sections
- **Content moderation**: Automated first-pass filtering
- **Research**: Study online toxicity patterns
- **Education**: Monitor cyberbullying

## Model Deployment Considerations

### Production Requirements
1. **Speed**: Process comments in real-time (<100ms)
2. **Accuracy**: Balance precision and recall
3. **False positives**: Minimize incorrect flagging
4. **False negatives**: Don't miss truly toxic content
5. **Explainability**: Show why comment was flagged

### Moderation Workflow
```
Comment → Classifier → Confidence Threshold
                              ↓
                      High confidence → Auto-action
                      Medium → Human review
                      Low → Allow
```

## Alternative Approaches

### Traditional Methods
1. **Keyword matching**: Simple but limited
2. **Regular expressions**: Pattern-based detection
3. **Naive Bayes**: Fast probabilistic classifier
4. **SVM**: Support Vector Machines

### Deep Learning
1. **CNN**: Convolutional Neural Networks for text
2. **LSTM/GRU**: Recurrent networks for sequences
3. **BERT**: Pretrained transformers
4. **RoBERTa**: Optimized BERT variant
5. **DistilBERT**: Faster, smaller BERT

### Ensemble Methods
1. **Voting**: Combine multiple classifiers
2. **Stacking**: Train meta-classifier
3. **Boosting**: Sequential error correction

## Best Practices

### Data Collection
1. **Diverse examples**: Cover all toxicity types
2. **Balanced dataset**: Ensure representation
3. **Quality labels**: Use multiple annotators
4. **Regular updates**: Add new toxicity patterns

### Model Training
1. **Cross-validation**: Ensure robust performance
2. **Hyperparameter tuning**: Optimize performance
3. **Class weights**: Handle imbalance
4. **Feature selection**: Remove noisy features

### Deployment
1. **A/B testing**: Compare model versions
2. **Monitoring**: Track performance metrics
3. **Feedback loop**: Learn from corrections
4. **Transparency**: Explain decisions to users

## Ethical Considerations

### Bias and Fairness
- **Demographic bias**: Avoid targeting specific groups
- **False positives**: Impact on free speech
- **Cultural sensitivity**: Context matters
- **Human oversight**: Don't fully automate

### Transparency
- Explain why content was flagged
- Allow appeals and corrections
- Document model limitations
- Regular bias audits

## Performance Optimization

1. **Feature pruning**: Remove low-importance features
2. **Vocabulary limiting**: Cap feature count
3. **Sparse matrices**: Efficient storage
4. **Batch prediction**: Process multiple comments
5. **Model caching**: Reuse vectorizer

## References
- Kaggle Toxic Comment Classification Challenge
- One-vs-Rest multi-label classification
- TF-IDF for text representation
- ROC-AUC for imbalanced classification
- Perspective API (Google Jigsaw)
- Content moderation best practices
