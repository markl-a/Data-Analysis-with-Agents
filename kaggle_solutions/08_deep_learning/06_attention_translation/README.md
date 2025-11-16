# Attention Mechanism for Translation

Implement sequence-to-sequence translation with Bahdanau attention mechanism.

## Problem Description

Neural machine translation converts text from one language to another using neural networks. This implementation uses an encoder-decoder architecture with attention mechanism, allowing the model to focus on relevant parts of the input when generating each output token.

## Approach

### Architecture

```
Input Sequence ──> Encoder ──> Context ──┐
                    (GRU)                 │
                                          ├──> Attention ──> Decoder ──> Output
Previous Output ──────────────────────────┘     Weights       (GRU)     Sequence
```

### Encoder-Decoder with Attention

```
┌─────────────────────────────────────────────────────────┐
│ ENCODER                                                 │
│                                                         │
│ Input → Embedding → GRU → Encoder Outputs              │
│                           Encoder State                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ ATTENTION MECHANISM                                     │
│                                                         │
│ Query (Decoder State) ─┐                               │
│                        ├─→ Attention Scores → Softmax  │
│ Values (Encoder Outs) ─┘        ↓                      │
│                           Context Vector                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ DECODER                                                 │
│                                                         │
│ Previous Token → Embedding ─┬─→ GRU → Output           │
│                             │                           │
│ Context Vector ─────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### Bahdanau Attention Mechanism

**Attention Score Calculation:**
```
score(h_t, h_s) = v^T * tanh(W1 * h_t + W2 * h_s)
```

**Attention Weights:**
```
α_ts = softmax(score(h_t, h_s))
```

**Context Vector:**
```
c_t = Σ α_ts * h_s
```

Where:
- h_t: Decoder hidden state at time t
- h_s: Encoder hidden state at position s
- α_ts: Attention weight for source position s at target time t
- c_t: Context vector at time t

### Training Process

1. **Encoder Phase**:
   - Process entire input sequence
   - Generate encoder outputs and final state
   - Outputs used as attention "values"

2. **Decoder Phase** (with Teacher Forcing):
   - Initialize decoder state with encoder state
   - For each target position:
     - Compute attention over encoder outputs
     - Get context vector (weighted sum)
     - Concatenate context with embedded input
     - Generate prediction
     - Use actual target as next input

3. **Inference**:
   - Same as training but use predicted tokens
   - Generate until `<end>` token or max length

## Implementation Details

- **Framework**: TensorFlow/Keras
- **Encoder**: Single-layer GRU
- **Decoder**: Single-layer GRU with attention
- **Attention**: Bahdanau (additive) attention
- **Embedding Dimension**: 64
- **Hidden Units**: 128
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Cross-Entropy
- **Epochs**: 20
- **Batch Size**: 64

## Features

1. Bahdanau attention implementation
2. Encoder-decoder architecture
3. Teacher forcing during training
4. Character-level tokenization
5. Synthetic number-to-word translation task
6. Attention weight visualization capability

## Usage

```bash
python solution.py
```

## Output

The script generates:
1. Training loss curve
2. Translation examples
3. Comparison with ground truth
4. Performance metrics

## Results

Expected outputs:
- Successful learning of number-to-word mapping
- Low final training loss (~0.1-0.5)
- Correct translations for test samples
- Smooth convergence

## Attention Mechanism Benefits

**Why Attention?**

1. **Information Bottleneck**: Fixed-size context vector is limiting
2. **Long Sequences**: Earlier tokens forgotten in long sequences
3. **Alignment**: Different word orders in source/target
4. **Focus**: Model learns what's important for each output

**Before Attention:**
```
Encoder → [Fixed Context Vector] → Decoder
```

**With Attention:**
```
Encoder → [Dynamic Context] → Decoder
         ↑              ↓
         └─ Attention ──┘
```

## Types of Attention

**Bahdanau (Additive) Attention:**
```
score = v^T * tanh(W1*h_t + W2*h_s)
```

**Luong (Multiplicative) Attention:**
```
score = h_t^T * W * h_s
```

**Scaled Dot-Product Attention:**
```
score = (Q * K^T) / sqrt(d_k)
```

**Self-Attention:**
```
Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
```

## Parameters

Key hyperparameters you can tune:

```python
embedding_dim = 64       # Embedding dimension
units = 128              # GRU hidden units
batch_size = 64          # Batch size
epochs = 20              # Training epochs
max_length = 10          # Maximum sequence length
```

## Technical Notes

1. **Teacher Forcing**: Use ground truth as decoder input during training
2. **Inference**: Use predicted token as next input
3. **Masking**: Ignore padding tokens in attention
4. **Gradient Clipping**: Prevents exploding gradients
5. **Beam Search**: Alternative to greedy decoding for better quality

## Comparison: With vs Without Attention

| Aspect | Without Attention | With Attention |
|--------|------------------|----------------|
| Context | Fixed vector | Dynamic per timestep |
| Long sequences | Performance degrades | Handles better |
| Interpretability | Black box | Can visualize alignment |
| Complexity | Simpler | More parameters |
| Quality | Lower | Higher |

## Applications

1. **Machine Translation**: Language-to-language translation
2. **Text Summarization**: Long text to short summary
3. **Image Captioning**: Image to text description
4. **Speech Recognition**: Audio to text
5. **Question Answering**: Question + context to answer
6. **Dialogue Systems**: Context to response

## Extensions

Potential improvements:

1. **Multi-Head Attention**: Multiple attention mechanisms in parallel
2. **Transformer Architecture**: Self-attention only, no RNN
3. **Beam Search**: Generate multiple candidates
4. **Byte-Pair Encoding**: Better tokenization
5. **Pre-training**: Use pretrained embeddings
6. **Bidirectional Encoder**: Process input in both directions
7. **Coverage Mechanism**: Prevent repetition
8. **Copy Mechanism**: Copy from source when needed

## Common Issues

**Attention Doesn't Work:**
- Check attention weight distribution
- Ensure proper masking of padding
- Verify gradient flow

**Poor Translation Quality:**
- Increase model capacity
- More training data
- Longer training
- Tune learning rate

**Slow Training:**
- Reduce sequence length
- Decrease batch size
- Use GRU instead of LSTM

**Repetitive Output:**
- Add coverage penalty
- Use diverse beam search
- Increase temperature

## Attention Visualization

Attention weights show alignment between source and target:

```
Source:  1 2 3 4
Target:  one  two  three four
Weights: High along diagonal (correct alignment)
```

This helps debug and understand model behavior.

## Historical Context

**Evolution of Seq2Seq:**

1. **2014**: Basic Encoder-Decoder (Sutskever et al.)
2. **2015**: Attention Mechanism (Bahdanau et al.)
3. **2016**: Google Neural Machine Translation
4. **2017**: Transformer (Vaswani et al.)
5. **2018+**: BERT, GPT, T5 (attention-based models)

## References

- Bahdanau et al. (2014): "Neural Machine Translation by Jointly Learning to Align and Translate"
- Luong et al. (2015): "Effective Approaches to Attention-based Neural Machine Translation"
- Vaswani et al. (2017): "Attention Is All You Need"
- Sutskever et al. (2014): "Sequence to Sequence Learning with Neural Networks"

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```
