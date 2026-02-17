# Build Your Own GPT

A complete GPT implementation in a single file with zero dependencies. Trains a character-level language model on a dataset of names, then generates new ones.

Based on [@karpathy's](https://github.com/karpathy) pure-Python GPT. Ported to TypeScript with detailed comments.

## Quick Start

```bash
npx tsx ml/gpt/gpt.ts
```

On first run it downloads a names dataset (~2MB), trains for 1000 steps (~minutes on CPU), then generates 20 new names.

## How It Works

This implementation builds a GPT from absolute scratch — no PyTorch, no TensorFlow, not even a matrix library. Every computation happens at the scalar level with automatic gradient tracking. This makes it incredibly slow but maximally educational.

### The Five Pieces

A GPT is built from five core ideas stacked together:

```
1. Tokenizer       — text → numbers
2. Autograd        — automatic calculus
3. Transformer     — the neural network architecture
4. Training loop   — learning from data
5. Inference       — generating new text
```

---

### 1. Tokenizer

Neural networks operate on numbers, not text. A tokenizer converts between the two.

This implementation uses the simplest possible tokenizer: **character-level encoding**. Each unique character in the dataset gets an integer ID:

```
Dataset characters: a, b, c, ..., z
Token IDs:          0, 1, 2, ..., 25
BOS (special):      26
```

The BOS (Beginning of Sequence) token marks where sequences start and end. A name like `"ada"` becomes `[26, 0, 3, 0, 26]` — BOS, a, d, a, BOS.

Production GPTs (like GPT-4) use **subword tokenization** (BPE) which encodes common character sequences as single tokens (`"the"` → one token instead of three). This is more efficient but the principle is identical.

---

### 2. Autograd Engine

Training a neural network requires computing **gradients** — how much each parameter contributed to the error. The autograd engine does this automatically.

#### The Value Class

Every number in the computation flows through `Value` nodes that remember how they were computed:

```typescript
class Value {
  data: number;       // the actual number (forward pass result)
  grad: number;       // dLoss/dThis (filled in during backward pass)
  children: Value[];  // inputs that produced this value
  localGrads: number[]; // dThis/dChild for each child
}
```

When you write `a.mul(b)`, it creates a new Value whose:
- `data` = a.data * b.data
- `children` = [a, b]
- `localGrads` = [b.data, a.data] (because d(a*b)/da = b and d(a*b)/db = a)

#### Backpropagation

Calling `loss.backward()` walks the computation graph in reverse and applies the **chain rule**:

```
child.grad += localGrad * parent.grad
```

This propagates the gradient from the loss all the way back to every parameter, telling us exactly how to adjust each one to reduce the loss.

The algorithm:
1. Topological sort of the computation graph
2. Set loss.grad = 1 (dLoss/dLoss = 1)
3. Walk in reverse, accumulating gradients via the chain rule

---

### 3. Transformer Architecture

The transformer is the neural network that learns patterns in the data. Here's what happens when a token enters the model:

```
token_id ─→ [Token Embedding] ─→ ┐
                                  ├─→ [Add] ─→ [RMSNorm] ─→ [Transformer Layer] ─→ [LM Head] ─→ logits
pos_id ──→ [Position Embedding] ─→ ┘
```

#### Embeddings

Each token ID indexes into a learned embedding table, producing a vector of `N_EMBD` (16) numbers. Position embeddings encode *where* the token appears in the sequence. These are added together so the model knows both *what* and *where*.

#### Transformer Layer

Each layer has two blocks with residual connections:

```
         ┌──────────────────────────────┐
  x ─────┤                              │
         │  [RMSNorm] → [Attention] ────┤──→ [+] ─────┐
         └──────────────────────────────┘    │         │
                                    x ───────┘         │
         ┌─────────────────────────────────────────────┤
         │  [RMSNorm] → [MLP] ─────────────────────────┤──→ [+] → output
         └─────────────────────────────────────────────┘    │
                                                   x ──────┘
```

**Multi-Head Self-Attention** — the core innovation. Each position can attend to all previous positions to gather context:

1. Project input into **Queries** (Q), **Keys** (K), and **Values** (V)
2. Split Q, K, V into `N_HEAD` (4) independent heads
3. Per head: compute attention weights = softmax(Q · K^T / √d)
4. Per head: output = weighted sum of V using those weights
5. Concatenate heads and project back

The Q/K dot product measures "how relevant is position t to the current position?" and the softmax converts these scores to weights that sum to 1.

**KV Cache**: During generation, previously computed keys and values are cached so each new token only needs to compute its own Q, K, V rather than reprocessing the entire sequence.

**Feed-Forward MLP** — a two-layer network that processes each position independently:

```
x → Linear(16 → 64) → ReLU → Linear(64 → 16)
```

The 4x expansion (16 → 64) gives the model more capacity to learn complex features.

**Residual connections** (`x + layer(x)`) allow gradients to flow directly through the network, making training stable even with many layers.

**RMSNorm** normalizes vectors by their root-mean-square, preventing activations from growing too large or small. Simpler than LayerNorm (no mean subtraction or learned parameters).

#### LM Head

A final linear projection maps the transformer output back to vocabulary-sized logits — one score per possible next token.

---

### 4. Training Loop

The training loop teaches the model to predict the next character:

```
For each step:
  1. Pick a document (name) from the dataset
  2. Tokenize it: [BOS, c, h, a, r, s, BOS]
  3. For each position, predict the next token:
       Input: [BOS]  → predict 'c'
       Input: [c]    → predict 'h'
       Input: [h]    → predict 'a'
       ...
  4. Compute cross-entropy loss: -log(P(correct token))
  5. Backpropagate to get gradients
  6. Update parameters with Adam optimizer
```

#### Cross-Entropy Loss

The loss for each position is `-log(P(correct next token))`. If the model assigns probability 0.9 to the right answer, the loss is -log(0.9) = 0.105 (low). If it assigns 0.01, the loss is -log(0.01) = 4.6 (high). The model learns by minimizing this.

#### Adam Optimizer

Adam maintains two running averages per parameter:
- **m**: mean of recent gradients (momentum — keeps moving in a consistent direction)
- **v**: mean of recent squared gradients (adaptive learning rate — bigger updates for rarely-changing parameters)

Update rule:
```
m = β₁ * m + (1 - β₁) * grad          // momentum
v = β₂ * v + (1 - β₂) * grad²         // adaptive scaling
param -= lr * (m / √v)                 // actual update
```

Bias correction compensates for the zero-initialization of m and v in early steps.

A linear learning rate decay (`lr * (1 - step/total_steps)`) gradually reduces the step size for fine-grained convergence.

---

### 5. Inference

After training, the model generates new names by sampling:

```
1. Start with BOS token
2. Forward through model → probability distribution over next token
3. Apply temperature scaling (divide logits by temperature before softmax)
4. Sample from the distribution
5. If sampled token is BOS → sequence complete
6. Otherwise append to output, go to step 2
```

**Temperature** controls randomness:
- `temperature → 0`: always picks the highest-probability token (deterministic)
- `temperature = 1`: samples from the raw learned distribution
- `temperature > 1`: flattens the distribution (more random/creative)

---

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_LAYER` | 1 | Transformer layers (depth) |
| `N_EMBD` | 16 | Embedding dimension (width) |
| `BLOCK_SIZE` | 16 | Max sequence length |
| `N_HEAD` | 4 | Attention heads |
| `LEARNING_RATE` | 0.01 | Initial learning rate |
| `BETA1` | 0.85 | Adam momentum decay |
| `BETA2` | 0.99 | Adam variance decay |
| `NUM_STEPS` | 1000 | Training iterations |
| `TEMPERATURE` | 0.5 | Sampling temperature |

These are intentionally tiny for CPU training. Real GPTs scale each dimension by 10-1000x and train on GPUs.

## What This Teaches vs. Production GPTs

This implementation is **algorithmically identical** to GPT-2/3/4 at the conceptual level. The differences are purely engineering:

| This Implementation | Production GPT |
|---------------------|----------------|
| Scalar autograd (Value class) | Tensor libraries (PyTorch/JAX) |
| Character tokenizer | BPE subword tokenizer |
| 1 layer, 16-dim | 96 layers, 12288-dim |
| ~5K parameters | ~175B parameters |
| CPU, single-threaded | GPU clusters, parallelized |
| Names dataset | Internet-scale text |
| ReLU activation | GeLU activation |
| RMSNorm | LayerNorm (GPT-2) / RMSNorm (LLaMA) |

The core ideas — attention, residual connections, learned embeddings, autoregressive training — are exactly the same.

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — original pure-Python implementation
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer paper
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — the architecture this follows
