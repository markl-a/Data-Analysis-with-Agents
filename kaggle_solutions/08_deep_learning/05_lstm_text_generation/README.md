# LSTM for Text Generation

Generate text character-by-character using Long Short-Term Memory (LSTM) networks.

## Problem Description

Text generation is the task of automatically creating coherent text sequences. This implementation uses LSTM networks to learn patterns in text at the character level and generate new text that mimics the style and structure of the training data.

## Approach

### Architecture

```
Input Sequence ──> Embedding ──> LSTM ──> LSTM ──> Dense ──> Softmax ──> Next Character
  (40 chars)        (64-dim)    (128)     (128)    (128)    (vocab_size)
```

### Model Architecture

```
Input (sequence_length,)
    ↓
Embedding (sequence_length, 64)
    ↓
LSTM (sequence_length, 128) [return_sequences=True]
    ↓
Dropout (0.2)
    ↓
LSTM (128)
    ↓
Dropout (0.2)
    ↓
Dense (128) + ReLU
    ↓
Dense (vocab_size) + Softmax
    ↓
Output (vocab_size)
```

### Training Process

1. **Prepare Data**:
   - Extract all unique characters from text
   - Create character-to-index and index-to-character mappings
   - Generate sequences of fixed length (e.g., 40 characters)
   - Each sequence's target is the next character

2. **Training**:
   - Feed sequences to LSTM
   - Predict probability distribution over next character
   - Minimize cross-entropy loss
   - Update weights via backpropagation

3. **Generation**:
   - Start with seed text
   - Predict next character probabilistically
   - Append to sequence and repeat
   - Use temperature to control randomness

### Text Generation with Temperature

Temperature controls randomness in sampling:

```python
predictions = log(predictions) / temperature
predictions = exp(predictions) / sum(exp(predictions))
```

- **Low Temperature (0.5)**: More conservative, repeats patterns
- **Medium Temperature (1.0)**: Balanced creativity and coherence
- **High Temperature (1.5)**: More random, creative but less coherent

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Sequence Length**: 40 characters
- **Embedding Dimension**: 64
- **LSTM Units**: 128 (per layer)
- **Number of LSTM Layers**: 2
- **Dropout Rate**: 0.2
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Cross-Entropy
- **Epochs**: 50
- **Batch Size**: 128

## Features

1. Character-level text modeling
2. Multi-layer LSTM architecture
3. Dropout for regularization
4. Temperature-based sampling
5. Multiple seed text generation
6. Training metrics visualization

## Usage

```bash
python solution.py
```

## Output

The script generates:
1. Training and validation loss/accuracy curves
2. Generated text samples with different temperatures
3. Text file with all generated samples
4. Performance metrics

## Results

Expected outputs:
- Convergence to ~0.5-1.0 validation loss
- 70-80% character prediction accuracy
- Coherent text generation at low temperatures
- Creative text at high temperatures
- Learning of common patterns and word structures

## Text Generation Examples

**Low Temperature (0.5):**
- More predictable
- Repeats common patterns
- Grammatically correct but repetitive

**Medium Temperature (1.0):**
- Balanced creativity
- Mix of learned patterns and variations
- Generally coherent

**High Temperature (1.5):**
- More random and creative
- May generate unusual combinations
- Less coherent but more diverse

## Parameters

Key hyperparameters you can tune:

```python
seq_length = 40          # Input sequence length
embedding_dim = 64       # Character embedding dimension
lstm_units = 128         # LSTM hidden units
epochs = 50              # Training epochs
batch_size = 128         # Batch size
dropout_rate = 0.2       # Dropout rate
temperature = 1.0        # Sampling temperature
```

## Technical Notes

1. **Sequence Length**: Longer sequences capture more context but slower training
2. **LSTM vs GRU**: LSTM has more parameters, GRU is faster
3. **Embedding**: Learns dense representations of characters
4. **Stateful LSTM**: Can maintain state across batches for longer dependencies
5. **Beam Search**: Alternative to temperature sampling for better quality

## LSTM Architecture Benefits

**Why LSTM for Text Generation?**

1. **Long-term Dependencies**: Can remember patterns over many characters
2. **Sequential Nature**: Processes text sequentially, character by character
3. **Gating Mechanism**: Learns what to remember and forget
4. **Gradient Flow**: Avoids vanishing gradient problem

**LSTM Cell:**
```
Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell State:  C_t = f_t * C_{t-1} + i_t * tanh(W_C · [h_{t-1}, x_t] + b_C)
Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden:      h_t = o_t * tanh(C_t)
```

## Character-level vs Word-level

**Character-level (This Implementation):**
- Smaller vocabulary
- Can generate any word (including typos)
- Learns spelling and morphology
- Slower generation

**Word-level:**
- Larger vocabulary
- Only generates known words
- Faster training and generation
- Needs larger corpus

## Comparison with Other Approaches

| Method | Pros | Cons |
|--------|------|------|
| N-grams | Fast, simple | Limited context |
| RNN | Sequential processing | Vanishing gradients |
| LSTM | Long dependencies | Slower training |
| GRU | Faster than LSTM | Slightly worse performance |
| Transformer | Best quality | Requires more data |

## Applications

1. **Creative Writing**: Story generation, poetry
2. **Code Generation**: Auto-complete for programming
3. **Music Generation**: Generate musical notation
4. **Chatbots**: Response generation
5. **Data Augmentation**: Generate synthetic training data
6. **Autocomplete**: Predictive text input

## Extensions

Potential improvements:

1. **Attention Mechanism**: Focus on relevant parts of input
2. **Transformer Models**: Use self-attention (GPT-style)
3. **Beam Search**: Generate multiple candidates, pick best
4. **Hierarchical Models**: Character + word level
5. **Conditional Generation**: Control style/topic
6. **Bidirectional LSTM**: Context from both directions (for understanding, not generation)
7. **Pre-trained Embeddings**: Use word2vec or GloVe

## Common Issues

**Repetitive Output:**
- Increase temperature
- Add more training data
- Use beam search with diversity

**Incoherent Text:**
- Decrease temperature
- Train longer
- Increase sequence length

**Mode Collapse:**
- Model generates same patterns
- Solution: Increase diversity penalty, use different seeds

**Overfitting:**
- Increase dropout
- Add more data
- Use regularization

## Training Tips

1. **Start Small**: Test with small text corpus first
2. **Monitor Validation**: Stop if validation loss increases
3. **Learning Rate**: Use learning rate scheduling
4. **Batch Size**: Larger batches for stability
5. **Data Quality**: Clean and preprocess text
6. **Sequence Length**: Experiment with different lengths

## References

- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Karpathy (2015): "The Unreasonable Effectiveness of Recurrent Neural Networks"
- Graves (2013): "Generating Sequences With Recurrent Neural Networks"
- Sutskever et al. (2011): "Generating Text with Recurrent Neural Networks"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
```
