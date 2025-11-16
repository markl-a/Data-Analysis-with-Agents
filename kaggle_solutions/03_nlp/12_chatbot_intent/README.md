# Chatbot Intent Classification

## Overview
This solution demonstrates intent classification for chatbots, enabling conversational AI systems to understand user intentions and route requests to appropriate handlers. The system classifies user queries into 10 different intents using TF-IDF features and multiple classification algorithms.

## Problem Statement
Build a system that can:
- Automatically detect user intent from text queries
- Support multiple intent categories
- Provide confidence scores for predictions
- Compare different classification algorithms
- Handle variations in user expressions
- Enable intelligent chatbot routing

## Dataset
The solution uses a synthetic chatbot dataset containing:
- **100 user queries** across 10 intent categories
- **10 samples per intent** for balanced training
- **Real-world conversational patterns**

### Intent Categories

1. **greeting**: Hello, hi, good morning, etc.
2. **goodbye**: Bye, see you later, farewell, etc.
3. **booking**: Make reservations, book appointments
4. **cancel**: Cancel bookings or subscriptions
5. **price_inquiry**: Ask about costs and pricing
6. **product_inquiry**: Ask about product details
7. **complaint**: Express dissatisfaction
8. **thanks**: Show gratitude and appreciation
9. **help**: Request assistance
10. **hours**: Ask about business hours

## Approach

### 1. Intent Classification Pipeline
```
User Query → Preprocessing → TF-IDF → Classifier → Intent + Confidence
```

### 2. Feature Extraction
**TF-IDF Vectorization:**
- Max features: 1000 most important terms
- N-grams: Unigrams and bigrams (1-2 words)
- Stop words: Remove common English words
- Lowercase normalization

**Why TF-IDF for Intent?**
- Captures keyword importance
- Works well with limited training data
- Fast inference for real-time chatbots
- Interpretable features

### 3. Classification Algorithms

**Logistic Regression** (Primary):
- Fast and accurate
- Probabilistic outputs
- Feature importance analysis
- Good for multi-class problems

**Naive Bayes**:
- Probabilistic approach
- Works well with text
- Fast training
- Good baseline

**SVM (Support Vector Machine)**:
- Strong performance
- Handles high-dimensional data
- Good generalization

**Random Forest**:
- Ensemble method
- Robust to noise
- Feature importance

### 4. Training Process
1. Preprocess queries (lowercase, remove special chars)
2. Extract TF-IDF features
3. Train classifier on intent labels
4. Evaluate with cross-validation
5. Compare multiple algorithms

## Key Features

1. **Multi-class classification**: Single intent per query
2. **Confidence scoring**: Probability for each intent
3. **Algorithm comparison**: Test multiple classifiers
4. **Feature analysis**: Top keywords per intent
5. **Robust preprocessing**: Handle varied user inputs
6. **Cross-validation**: Ensure generalization

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

**Overall Accuracy**: 90-100% (on test set)
- Varies by classifier
- Best: Logistic Regression and SVM

**Cross-Validation**: 85-95% (5-fold CV)
- Ensures robust performance
- Detects overfitting

**Per-Intent Performance**:
- Most intents: F1 > 0.90
- Challenging: Similar intents may confuse

### Classifier Comparison

| Classifier | Accuracy | Speed | Interpretability |
|------------|----------|-------|------------------|
| Logistic Regression | 95-100% | Fast | High |
| Naive Bayes | 90-95% | Fastest | Medium |
| SVM | 95-100% | Medium | Low |
| Random Forest | 90-95% | Slow | Medium |

### Visualizations

1. **Confusion Matrix**: Shows prediction accuracy and confusions
2. **Intent Distribution**: Pie chart of dataset balance
3. **Per-Intent Accuracy**: Performance by category
4. **Classifier Comparison**: Algorithm performance
5. **Confidence Distribution**: Prediction certainty
6. **F1 Scores**: Harmonic mean of precision and recall

## Example Output

### Intent Detection Examples

```
Query: 'Hello, how can I help you?'
Intent: greeting (confidence: 98.5%)
Top 3: greeting (98.5%), help (1.2%), thanks (0.3%)

Query: 'I want to book a table for two'
Intent: booking (confidence: 96.7%)
Top 3: booking (96.7%), cancel (2.1%), help (1.2%)

Query: 'How much does this cost?'
Intent: price_inquiry (confidence: 99.2%)
Top 3: price_inquiry (99.2%), product_inquiry (0.5%), help (0.3%)
```

### Top Keywords per Intent

- **greeting**: hello, hi, morning, hey, greetings
- **booking**: book, reserve, reservation, appointment, table
- **price_inquiry**: much, cost, price, expensive, rates
- **complaint**: unhappy, terrible, awful, complaint, manager
- **thanks**: thank, thanks, appreciate, grateful, appreciated

## Strengths
- High accuracy with small datasets
- Fast training and inference (<100ms)
- Interpretable predictions
- Easy to add new intents
- Works offline (no API calls)
- Provides confidence scores
- Multiple algorithm options

## Limitations
- Requires labeled training data
- May struggle with ambiguous queries
- Sensitive to typos and slang
- Cannot handle out-of-domain queries
- Limited context awareness
- No dialogue history consideration

## Common Challenges

1. **Overlapping intents**: Similar expressions for different intents
2. **Ambiguous queries**: Could belong to multiple intents
3. **Rare intents**: Limited training examples
4. **Typos and errors**: User input variations
5. **New intents**: Need retraining for new categories

## Future Improvements

1. **Deep learning**: Use BERT or RoBERTa for better accuracy
2. **Context awareness**: Consider conversation history
3. **Entity extraction**: Extract parameters from queries
4. **Multilingual support**: Handle multiple languages
5. **Active learning**: Improve with user corrections
6. **Confidence thresholds**: Fallback for uncertain predictions
7. **Intent hierarchies**: Multi-level intent structure
8. **Slot filling**: Extract values for intent parameters

## Technical Details

### Preprocessing Steps
```python
1. Convert to lowercase
2. Remove special characters (keep ? and !)
3. Remove extra whitespace
4. Normalize punctuation
```

### TF-IDF Formula
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)
where:
- TF = frequency of term in document
- IDF = log(total_docs / docs_containing_term)
```

### Logistic Regression for Multi-class
```
P(class_i|x) = exp(w_i·x) / Σ exp(w_j·x)
- Softmax function for probabilities
- One-vs-rest or multinomial approach
```

## Real-World Applications

### Customer Service Chatbots
- Route queries to appropriate departments
- Provide automated responses
- Escalate to human agents when needed

### Virtual Assistants
- Understand user commands
- Execute actions (set reminders, search, etc.)
- Provide contextual help

### E-commerce Bots
- Handle product inquiries
- Process orders and bookings
- Manage cancellations and complaints

### Support Systems
- Classify support tickets
- Auto-respond to common questions
- Prioritize urgent issues

## Deployment Architecture

```
User Input → Intent Classifier → Intent + Confidence
                                       ↓
                              Confidence Check
                                       ↓
                    High → Execute Handler
                    Medium → Ask Clarification
                    Low → Fallback Response
```

### Production Considerations

1. **Response Time**: <100ms for real-time chat
2. **Confidence Threshold**: Set minimum confidence (e.g., 0.7)
3. **Fallback Handling**: What to do when uncertain
4. **Logging**: Track predictions for improvement
5. **A/B Testing**: Compare model versions

## Best Practices

### Data Collection
1. **Diverse examples**: Cover all intent variations
2. **Real user queries**: Use actual chat logs
3. **Balanced dataset**: Equal examples per intent
4. **Quality over quantity**: Clean, accurate labels

### Model Training
1. **Cross-validation**: Ensure robust performance
2. **Hyperparameter tuning**: Optimize settings
3. **Regular updates**: Retrain with new data
4. **Version control**: Track model changes

### Deployment
1. **Monitor accuracy**: Track production metrics
2. **Collect feedback**: Learn from errors
3. **Gradual rollout**: Test before full deployment
4. **Fallback mechanism**: Handle uncertain cases

## Error Handling

### Low Confidence Predictions
```python
if confidence < 0.7:
    return "I'm not sure I understand. Could you rephrase?"
```

### Out-of-Domain Queries
```python
if all_probabilities < 0.5:
    return "I can help with: booking, pricing, support..."
```

### Ambiguous Queries
```python
if top_2_probabilities_close:
    return "Did you mean: [intent1] or [intent2]?"
```

## Evaluation Metrics

### Standard Metrics
- **Accuracy**: Overall correctness
- **Precision**: Correct positives / all positives
- **Recall**: Correct positives / all actual
- **F1 Score**: Harmonic mean of precision and recall

### Chatbot-Specific Metrics
- **Intent detection rate**: Successfully classified queries
- **Fallback rate**: Queries with low confidence
- **Task completion**: User achieves goal
- **User satisfaction**: Explicit feedback

## Alternative Approaches

### Rule-Based Systems
- **Pros**: No training data needed, interpretable
- **Cons**: Hard to maintain, limited coverage

### Keyword Matching
- **Pros**: Simple, fast
- **Cons**: Brittle, misses context

### Embeddings (Word2Vec, GloVe)
- **Pros**: Semantic understanding
- **Cons**: Requires more data

### Transformers (BERT, GPT)
- **Pros**: State-of-the-art accuracy
- **Cons**: Slower, resource-intensive

### Intent + Entity Models
- **Pros**: Extract both intent and parameters
- **Cons**: More complex training

## Integration Example

```python
# In chatbot main loop
user_message = get_user_input()
intent, confidence = classifier.predict(user_message)

if confidence > 0.8:
    handler = intent_handlers[intent]
    response = handler.execute(user_message)
elif confidence > 0.5:
    response = f"Did you mean {intent}?"
else:
    response = "I'm sorry, I didn't understand that."

send_response(response)
```

## Performance Optimization

1. **Model caching**: Load once, reuse
2. **Feature pruning**: Remove low-importance features
3. **Batch prediction**: Process multiple queries
4. **Model compression**: Reduce model size
5. **Quantization**: Use smaller data types

## References
- TF-IDF for text classification
- Logistic Regression for multi-class problems
- Scikit-learn text classification
- Chatbot intent detection techniques
- Natural Language Understanding (NLU)
- Rasa NLU framework
- Dialogflow intent matching
