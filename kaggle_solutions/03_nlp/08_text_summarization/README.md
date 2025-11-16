# Extractive Text Summarization

## Overview
This solution demonstrates extractive text summarization techniques that automatically generate concise summaries by selecting the most important sentences from documents. Three different algorithms are implemented and compared: TF-IDF, TextRank, and Frequency-based scoring.

## Problem Statement
Build a system that can:
- Automatically generate summaries from long documents
- Extract the most informative sentences
- Maintain coherence and readability
- Achieve good compression while preserving key information
- Compare different summarization approaches

## Dataset
The solution uses synthetic documents covering:
- **Artificial Intelligence Revolution** - Technology and AI advances
- **Climate Change Impact** - Environmental challenges
- **Space Exploration Advances** - Recent space discoveries

Each document contains 8-10 sentences with diverse topics and complexity.

## Approach

### 1. TF-IDF Based Summarization
- **Calculate TF-IDF scores** for all words in sentences
- **Score sentences** by sum of TF-IDF values
- **Rank and select** top-scoring sentences
- **Advantages**: Fast, effective for keyword-rich documents
- **Disadvantages**: May miss contextual importance

### 2. TextRank Algorithm
- **Build similarity graph** between sentences
- **Apply PageRank** algorithm to find central sentences
- **Extract top-ranked** sentences
- **Advantages**: Considers sentence relationships
- **Disadvantages**: Computationally more expensive

### 3. Frequency-Based Scoring
- **Calculate word frequencies** in document
- **Score sentences** by sum of word frequencies
- **Normalize by length** to avoid bias toward long sentences
- **Advantages**: Simple and interpretable
- **Disadvantages**: May favor common words

## Algorithm Comparison

| Method | Speed | Quality | Complexity | Best For |
|--------|-------|---------|------------|----------|
| TF-IDF | Fast | Good | Low | Technical documents |
| TextRank | Medium | Best | Medium | General text |
| Frequency | Fastest | Fair | Very Low | Simple summaries |

## Key Features

1. **Multiple algorithms**: Compare three different approaches
2. **Configurable length**: Adjust number of sentences in summary
3. **Quality metrics**: Evaluate compression ratio and coverage
4. **Sentence ordering**: Maintains original document order
5. **Preprocessing**: Text cleaning and normalization

## Requirements

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
networkx
```

## Usage

```bash
python solution.py
```

## Results

### Performance Metrics

**Compression Ratio**: Ratio of summary length to original length
- Target: 30-40% of original
- Actual: Varies by document (typically 25-35%)

**Coverage Score**: Percentage of original vocabulary in summary
- Target: 50-70%
- Actual: Usually 40-60%

**Sentence Ratio**: Number of sentences selected
- Default: 3 sentences per document
- Adjustable based on needs

### Visualizations

1. **Compression Ratios Bar Chart**: Shows compression for each document
2. **Coverage Scores**: Word coverage in summaries
3. **Method Comparison**: TF-IDF vs TextRank vs Frequency
4. **Length Comparison**: Original vs summary lengths

## Example Output

### Original Text (890 chars):
```
Artificial intelligence is transforming the world at an unprecedented pace.
Machine learning algorithms are now capable of performing tasks that were once
thought to be exclusively human. Deep learning, a subset of machine learning,
uses neural networks with multiple layers to analyze complex patterns...
[truncated]
```

### TF-IDF Summary (245 chars):
```
Artificial intelligence is transforming the world at an unprecedented pace.
Companies across industries are investing billions in AI research and
development. The future of AI promises both opportunities and challenges
for society.
```

### TextRank Summary (268 chars):
```
Machine learning algorithms are now capable of performing tasks that were
once thought to be exclusively human. Companies across industries are
investing billions in AI research and development. Researchers emphasize
the importance of developing responsible AI systems.
```

## Evaluation Metrics

### Automatic Metrics
- **Compression Ratio**: How much text is reduced
- **Word Coverage**: Vocabulary preservation
- **Sentence Selection**: Diversity of selected sentences

### Quality Indicators
- **Coherence**: Do sentences flow logically?
- **Informativeness**: Are key points captured?
- **Readability**: Is summary easy to understand?

## Strengths
- Works without training data
- Fast and efficient processing
- Maintains original sentence structure
- Easy to understand and implement
- No pretrained models required

## Limitations
- Cannot generate new sentences (extractive only)
- May include redundant information
- Might miss implicit connections
- Depends on sentence quality in original
- No semantic understanding

## Common Challenges

1. **Sentence selection bias**: First/last sentences often selected
2. **Redundancy**: Similar sentences may be selected
3. **Coherence**: Selected sentences may not flow well
4. **Length vs Quality**: Trade-off between brevity and completeness
5. **Domain sensitivity**: Performance varies by text type

## Future Improvements

1. **Abstractive summarization**: Generate new sentences
2. **Multi-document summarization**: Combine multiple sources
3. **Neural approaches**: Use BERT or T5 models
4. **Sentence fusion**: Combine information from multiple sentences
5. **Query-focused**: Generate summaries for specific questions
6. **Hierarchical summarization**: Multi-level summaries
7. **Entity-aware**: Preserve important named entities
8. **Coreference resolution**: Handle pronouns correctly

## Technical Details

### TF-IDF Formula
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
where:
- TF(t, d) = frequency of term t in document d
- IDF(t) = log(N / df(t))
- N = total documents
- df(t) = documents containing term t
```

### TextRank Formula
```
Score(Si) = (1-d) + d × Σ(wji × Score(Sj) / Σ wjk)
where:
- d = damping factor (0.85)
- wji = similarity between sentences i and j
- Sum over all sentences j linking to i
```

### Cosine Similarity
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

## Real-World Applications
- News article summarization
- Research paper abstracts
- Email thread summaries
- Meeting notes condensation
- Legal document summaries
- Medical record summaries
- Product review aggregation
- Social media content curation

## Alternative Approaches

### Extractive Methods
1. **LexRank**: Graph-based like TextRank
2. **LSA**: Latent Semantic Analysis
3. **SumBasic**: Probability-based selection
4. **LUHN**: Early heuristic method

### Abstractive Methods
1. **Sequence-to-Sequence**: LSTM/GRU based
2. **Transformer models**: BERT, GPT, T5
3. **BART**: Denoising autoencoder
4. **PEGASUS**: Pretrained for summarization

## Performance Tips

1. **Preprocessing**: Remove noise and special characters
2. **Stop words**: Filter common words for better scoring
3. **Sentence length**: Consider minimum/maximum lengths
4. **Position bias**: Weight first/last sentences differently
5. **Diversity**: Avoid selecting similar sentences

## References
- TF-IDF: Term Frequency-Inverse Document Frequency
- TextRank: Mihalcea and Tarau (2004)
- PageRank: Page et al. (1999)
- ROUGE metrics for summarization evaluation
- Extractive vs Abstractive summarization
