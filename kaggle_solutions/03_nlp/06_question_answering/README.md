# Question Answering System

## Overview
This solution demonstrates a simple extractive question answering (QA) system that retrieves relevant answers from given contexts using TF-IDF vectorization and cosine similarity.

## Problem Statement
Build a system that can:
- Accept questions in natural language
- Search through a knowledge base of contexts
- Return the most relevant answer with confidence scores
- Evaluate answer quality and accuracy

## Dataset
The solution uses a synthetic dataset containing:
- **8 knowledge contexts** covering various topics (Technology, Nature, History, Science)
- **16 questions** with expected answers
- Topics include Python programming, Machine Learning, Amazon rainforest, Climate change, Internet history, and more

## Approach

### 1. Text Preprocessing
- Convert text to lowercase
- Remove special characters while preserving sentence structure
- Tokenize contexts into sentences for better granularity

### 2. Feature Extraction
- **TF-IDF Vectorization**: Convert text to numerical features
- **N-grams**: Use both unigrams and bigrams (1-2 word phrases)
- **Stop words removal**: Filter common English words
- **Feature limit**: 500 most important features

### 3. Answer Retrieval
- Convert question to TF-IDF vector
- Calculate cosine similarity with all context sentences
- Return top-k most similar sentences as answers
- Include confidence scores for each answer

### 4. Evaluation Metrics
- **Accuracy**: Percentage of questions with correct answers
- **Confidence scores**: Cosine similarity values
- **Score distribution**: Analyze correct vs incorrect answers

## Model Architecture

```
Question → Preprocessing → TF-IDF Vectorization
                                  ↓
Context Sentences ← TF-IDF Vectorization
                                  ↓
                    Cosine Similarity Calculation
                                  ↓
                    Top-K Answer Retrieval
```

## Key Features

1. **Extractive QA**: Returns actual sentences from contexts
2. **Sentence-level retrieval**: Splits contexts into sentences for precision
3. **Multi-answer support**: Can return multiple candidate answers
4. **Confidence scoring**: Provides similarity scores for each answer
5. **Interactive demo**: Test with custom questions

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
- **Accuracy**: ~70-80% (varies based on question complexity)
- **Average Confidence**: 0.3-0.5 (cosine similarity range: 0-1)
- **Processing**: Fast retrieval (<1 second per question)

### Visualizations

1. **Answer Accuracy Bar Chart**: Shows correct vs incorrect predictions
2. **Confidence Score Distribution**: Histogram of similarity scores
3. **Score Comparison Box Plot**: Correct vs incorrect answer confidence
4. **Top Questions Chart**: Highest confidence predictions

## Example Output

```
Q: Who created Python programming language?
A: Python is a high-level programming language created by Guido van Rossum.
Expected: Guido van Rossum
Score: 0.523

Q: When was the World Wide Web invented?
A: Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.
Expected: 1989
Score: 0.612
```

## Strengths
- Simple and interpretable approach
- Fast inference time
- No need for large pretrained models
- Works well for factual questions
- Easily adaptable to new domains

## Limitations
- Cannot generate new answers (extractive only)
- Struggles with complex reasoning questions
- Dependent on exact matches in contexts
- Limited semantic understanding
- Performance drops with paraphrased questions

## Future Improvements

1. **Advanced models**: Use BERT or other transformers for better semantic understanding
2. **Named Entity Recognition**: Extract specific entities from contexts
3. **Answer generation**: Implement abstractive QA instead of extractive
4. **Question classification**: Route different question types to specialized handlers
5. **Context ranking**: Implement BM25 or other ranking algorithms
6. **Multi-hop reasoning**: Handle questions requiring multiple context pieces

## Technical Details

### TF-IDF Parameters
- Max features: 500
- N-gram range: (1, 2)
- Stop words: English
- Tokenization: Word-based

### Similarity Metric
- Cosine similarity between question and sentence vectors
- Range: 0 (no similarity) to 1 (identical)
- Threshold: No fixed threshold, uses top-k retrieval

## Real-World Applications
- Customer support chatbots
- FAQ systems
- Document search engines
- Knowledge base queries
- Educational Q&A platforms

## References
- TF-IDF: Term Frequency-Inverse Document Frequency
- Cosine Similarity for text similarity
- NLTK for text preprocessing
- Extractive vs Abstractive QA approaches
