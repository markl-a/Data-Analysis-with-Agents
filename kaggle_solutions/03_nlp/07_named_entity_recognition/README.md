# Named Entity Recognition (NER)

## Overview
This solution demonstrates Named Entity Recognition (NER) for extracting and classifying named entities such as Person, Organization, Location, and Date from unstructured text using a feature-based machine learning approach.

## Problem Statement
Build a system that can:
- Identify named entities in text
- Classify entities into predefined categories (PER, ORG, LOC, DATE)
- Extract entity boundaries accurately
- Evaluate performance using standard NER metrics

## Dataset
The solution uses a synthetic dataset containing:
- **20 training sentences** with manually labeled entities
- **4 test sentences** for evaluation
- Entity types: Person (PER), Organization (ORG), Location (LOC), Date (DATE)
- BIO tagging scheme (B=Beginning, I=Inside, O=Outside)

### Example Annotations
```
Sentence: Apple Inc was founded by Steve Jobs in California
Labels:   B-ORG I-ORG O    O       O  B-PER I-PER O  B-LOC
```

## Approach

### 1. Feature Engineering
For each word, extract:
- **Word features**: lowercase form, capitalization, length
- **Prefix/Suffix**: First/last 1-3 characters
- **Character patterns**: All caps, all lowercase, numeric
- **Position features**: First/last word in sentence
- **Context features**: Previous and next word information
- **Orthographic features**: Hyphens, special characters

### 2. BIO Tagging Scheme
- **B-XXX**: Beginning of entity type XXX
- **I-XXX**: Inside (continuation) of entity type XXX
- **O**: Outside any entity

### 3. Model Architecture
- **Feature extraction**: Convert words to feature dictionaries
- **Vectorization**: DictVectorizer for sparse feature matrix
- **Classification**: Logistic Regression for multi-class prediction
- **Post-processing**: Combine B-I tags to extract complete entities

### 4. Evaluation Metrics
- **Precision**: How many extracted entities are correct
- **Recall**: How many actual entities were found
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of true instances per class

## Key Features

1. **Feature-based approach**: Doesn't require large pretrained models
2. **Context awareness**: Uses surrounding words for better predictions
3. **Multiple entity types**: Handles PER, ORG, LOC, DATE
4. **BIO tagging**: Proper entity boundary detection
5. **Extensible**: Easy to add new entity types or features

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
```

## Usage

```bash
python solution.py
```

## Results

### Performance Metrics
- **Overall F1 Score**: 0.7-0.85 (varies by entity type)
- **Person (PER)**: Highest accuracy due to capitalization patterns
- **Location (LOC)**: Good performance with geographic context
- **Organization (ORG)**: Moderate accuracy, can confuse with locations
- **Date (DATE)**: Good performance with temporal keywords

### Visualizations

1. **F1 Scores by Entity Type**: Bar chart showing performance per category
2. **Entity Distribution**: Pie chart of entity type frequencies
3. **Precision/Recall/F1 Comparison**: Multi-metric comparison across types
4. **Most Common Entities**: Top 10 extracted entities

## Example Output

```
Sentence: Tim Cook leads Apple from Cupertino California
Entities:
  - Tim Cook (PER)
  - Apple (ORG)
  - Cupertino (LOC)
  - California (LOC)

Sentence: Jeff Bezos started Amazon in Seattle in July 1994
Entities:
  - Jeff Bezos (PER)
  - Amazon (ORG)
  - Seattle (LOC)
  - July 1994 (DATE)
```

## Feature Importance

Most useful features for NER:
1. **Capitalization**: Strong indicator for proper nouns
2. **Previous word**: Context helps disambiguate
3. **Word position**: First word often capitalized regardless
4. **Prefix patterns**: Common name/location prefixes
5. **Numeric patterns**: Helps identify dates

## Strengths
- Fast training and inference
- Interpretable features
- Works with small datasets
- No GPU required
- Easy to debug and improve

## Limitations
- Requires feature engineering
- May miss context-dependent entities
- Struggles with rare entity types
- Sensitive to capitalization errors
- Limited semantic understanding

## Common Errors

1. **Organization vs Location**: "Paris" could be city or company name
2. **Person vs Organization**: "Jobs" could be person or occupation
3. **Nested entities**: "Bank of America" contains location
4. **Abbreviations**: "UN" vs "United Nations"
5. **Multi-word entities**: Boundary detection challenges

## Future Improvements

1. **Deep learning models**: Use BiLSTM-CRF or BERT for better accuracy
2. **External knowledge**: Integrate gazetteers (lists of known entities)
3. **Word embeddings**: Use Word2Vec or GloVe for semantic features
4. **Active learning**: Iteratively improve with user feedback
5. **Domain adaptation**: Fine-tune for specific domains (medical, legal)
6. **Entity linking**: Connect entities to knowledge bases
7. **Relation extraction**: Identify relationships between entities

## Technical Details

### Feature Vector Example
```python
{
    'word': 'apple',
    'is_capitalized': True,
    'prefix-2': 'Ap',
    'suffix-2': 'le',
    'prev_word': 'founded',
    'next_word': 'inc'
}
```

### Model Training
- Classifier: Logistic Regression
- Max iterations: 1000
- Regularization: Default L2
- Multi-class strategy: One-vs-Rest

## Real-World Applications
- Information extraction from documents
- Resume parsing (extract skills, companies, dates)
- News article analysis (identify people, places, organizations)
- Medical records (extract drug names, diseases, dates)
- Legal document processing
- Social media monitoring
- Customer support ticket analysis

## Alternative Approaches

1. **Rule-based**: Regular expressions and pattern matching
2. **CRF (Conditional Random Fields)**: Traditional sequence labeling
3. **BiLSTM-CRF**: Deep learning with CRF layer
4. **BERT-based**: Pretrained transformer models
5. **spaCy**: Production-ready NER library
6. **Stanford NER**: Java-based NER toolkit

## References
- BIO tagging scheme for sequence labeling
- Logistic Regression for multi-class classification
- Feature engineering for NLP tasks
- NLTK for text processing
- CoNLL-2003 NER shared task
