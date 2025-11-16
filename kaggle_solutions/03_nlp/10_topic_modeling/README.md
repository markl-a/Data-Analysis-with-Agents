# Topic Modeling with LDA

## Overview
This solution demonstrates unsupervised topic modeling using Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) to automatically discover hidden thematic structures in document collections.

## Problem Statement
Build a system that can:
- Automatically discover topics in a document collection
- Assign documents to topics
- Extract representative keywords for each topic
- Visualize topic distributions
- Compare different topic modeling algorithms

## Dataset
The solution uses a synthetic document collection containing:
- **18 documents** across 6 thematic areas
- **Topics**: Technology, Health & Medicine, Environment, Finance & Business, Education, Sports
- **~150-200 words** per document
- Diverse vocabulary and subject matter

## Approach

### 1. Latent Dirichlet Allocation (LDA)
LDA is a probabilistic generative model that assumes:
- Each document is a mixture of topics
- Each topic is a mixture of words
- Documents sharing words are likely to share topics

**Mathematical Foundation:**
```
Document → Topic Distribution
Topic → Word Distribution
```

### 2. Non-negative Matrix Factorization (NMF)
NMF factorizes the document-term matrix into:
- Document-topic matrix
- Topic-word matrix

**Advantages:**
- Interpretable parts-based representation
- Often produces clearer topics than LDA
- Faster convergence

### 3. Feature Extraction
**CountVectorizer Parameters:**
- Max features: 1000 most common words
- Stop words: Remove English stop words
- Max document frequency: 80% (remove very common words)
- Min document frequency: 2 (remove very rare words)

### 4. Model Pipeline
```
Documents → Preprocessing → Vectorization → Topic Model → Topic Assignment
```

## Key Features

1. **Automatic topic discovery**: No manual labeling required
2. **Soft clustering**: Documents can belong to multiple topics
3. **Topic interpretation**: Top words reveal topic meaning
4. **Comparative analysis**: LDA vs NMF comparison
5. **Word clouds**: Visual topic representation
6. **Interactive exploration**: Query topic distribution for new documents

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
wordcloud
scipy
```

## Usage

```bash
python solution.py
```

## Results

### Performance Metrics

**Perplexity**:
- Measures how well model predicts held-out documents
- Lower is better
- Typical range: 500-2000 for our dataset

**Topic Coherence**:
- Measures semantic similarity of top words
- Higher is better
- Evaluated through topic separation

**Document Entropy**:
- Measures topic diversity per document
- Low entropy: Document focused on one topic
- High entropy: Document covers multiple topics

### Discovered Topics (Example)

**Topic 1: Technology**
- Keywords: artificial, intelligence, machine, learning, data, neural, networks, algorithm
- Documents: 3 (17%)

**Topic 2: Health & Medicine**
- Keywords: health, medical, disease, treatment, therapy, exercise, mental, wellness
- Documents: 3 (17%)

**Topic 3: Environment**
- Keywords: climate, energy, environmental, renewable, solar, carbon, wildlife, conservation
- Documents: 3 (17%)

**Topic 4: Finance & Business**
- Keywords: market, investment, business, financial, trading, cryptocurrency, startup
- Documents: 3 (17%)

**Topic 5: Education & Sports**
- Keywords: education, learning, students, training, athletes, sports, skills, development
- Documents: 6 (33%)

### Visualizations

1. **Topic Word Clouds**: Visual representation of top words per topic
2. **Document-Topic Heatmap**: Shows topic distribution across documents
3. **Documents per Topic**: Bar chart of topic sizes
4. **Topic Prevalence**: Average probability of each topic
5. **Topic Entropy Distribution**: How focused vs diverse documents are

## Example Output

### Document Analysis
```
Document 1:
  Text: Artificial intelligence and machine learning are revolutionizing...
  Main Topic: 1 (87%)
  Distribution: T1:87%, T2:3%, T3:5%, T4:3%, T5:2%

Document 2:
  Text: Medical research focuses on developing new treatments for diseases...
  Main Topic: 2 (91%)
  Distribution: T1:2%, T2:91%, T3:3%, T4:2%, T5:2%
```

### Topic Words
```
Topic 1: artificial(0.245), intelligence(0.198), machine(0.187), learning(0.165)
Topic 2: health(0.267), medical(0.223), disease(0.189), treatment(0.156)
```

## LDA vs NMF Comparison

| Aspect | LDA | NMF |
|--------|-----|-----|
| Type | Probabilistic | Algebraic |
| Interpretation | Clearer | Sometimes sharper |
| Speed | Slower | Faster |
| Sparsity | Natural | Enforced |
| Best For | General text | Short documents |

## Strengths
- Unsupervised learning (no labels needed)
- Discovers hidden patterns automatically
- Handles large document collections
- Provides interpretable results
- Flexible number of topics
- Works across domains

## Limitations
- Number of topics must be pre-specified
- Results can vary across runs (LDA)
- Difficult to evaluate objectively
- May produce overlapping topics
- Sensitive to preprocessing choices
- Requires sufficient documents

## Common Challenges

1. **Optimal number of topics**: No definitive answer
2. **Topic interpretation**: Sometimes unclear meaning
3. **Topic stability**: Results vary with random initialization
4. **Short documents**: Less context for accurate assignment
5. **Evolving topics**: Cannot adapt to temporal changes

## Future Improvements

1. **Optimal topic selection**: Grid search over topic numbers
2. **Topic coherence metrics**: Automated quality measurement
3. **Hierarchical topics**: Multi-level topic structure
4. **Dynamic topics**: Track topic evolution over time
5. **Supervised LDA**: Incorporate document metadata
6. **Neural topic models**: Deep learning approaches (VAE, ETM)
7. **Topic labeling**: Automatic topic name generation
8. **Cross-lingual topics**: Multi-language topic modeling

## Technical Details

### LDA Generative Process
```
For each document d:
  1. Draw topic distribution θ_d ~ Dirichlet(α)
  2. For each word position n:
     a. Draw topic z_dn ~ Multinomial(θ_d)
     b. Draw word w_dn ~ Multinomial(φ_{z_dn})
```

### Hyperparameters
- **α (alpha)**: Document-topic density (default: 1/n_topics)
- **β (beta)**: Topic-word density (default: 1/n_topics)
- **Iterations**: Number of training iterations (default: 20)

### NMF Formulation
```
V ≈ W × H
where:
- V: Document-term matrix (n_docs × n_words)
- W: Document-topic matrix (n_docs × n_topics)
- H: Topic-word matrix (n_topics × n_words)
```

## Model Selection

### Choosing Number of Topics

**Methods:**
1. **Perplexity**: Train models with different k, choose lowest perplexity
2. **Coherence score**: Measure topic coherence, choose highest
3. **Manual inspection**: Evaluate interpretability
4. **Domain knowledge**: Use expected number of themes

**Rules of Thumb:**
- Small corpus (<100 docs): 5-10 topics
- Medium corpus (100-1000 docs): 10-50 topics
- Large corpus (>1000 docs): 50-200 topics

## Real-World Applications
- News article categorization
- Customer review analysis
- Scientific paper organization
- Social media trend detection
- Email categorization
- Content recommendation
- Survey response analysis
- Legal document clustering

## Alternative Approaches

### Traditional Methods
1. **LSA (Latent Semantic Analysis)**: SVD-based
2. **pLSA (Probabilistic LSA)**: Probabilistic variant
3. **HDP (Hierarchical Dirichlet Process)**: Automatic topic number

### Modern Methods
1. **BERTopic**: BERT embeddings + clustering
2. **Top2Vec**: Doc2Vec + UMAP + HDBSCAN
3. **CTM (Contextualized Topic Models)**: Neural networks
4. **ETM (Embedded Topic Model)**: Word embeddings
5. **LDA2Vec**: Combines LDA with Word2Vec

## Performance Optimization

1. **Vocabulary pruning**: Remove rare and common words
2. **Batch processing**: Process documents in batches
3. **Online learning**: Incremental updates for new documents
4. **Parallel processing**: Utilize multiple CPU cores
5. **Feature selection**: Limit vocabulary size

## Evaluation Metrics

### Intrinsic Metrics
- **Perplexity**: Predictive power
- **Topic coherence**: Semantic quality
- **Topic diversity**: Uniqueness of topics

### Extrinsic Metrics
- **Classification accuracy**: If labels available
- **Information retrieval**: Document search quality
- **Human evaluation**: Expert judgment

## Best Practices

1. **Preprocessing**: Remove noise, normalize text
2. **Stopwords**: Use domain-specific stopword lists
3. **N-grams**: Include bigrams for better phrases
4. **Multiple runs**: Average results across runs
5. **Validation**: Check topics make sense
6. **Documentation**: Record preprocessing decisions

## References
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by NMF
- Topic coherence measures
- scikit-learn LDA documentation
- Gensim topic modeling library
