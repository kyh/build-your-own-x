/**
 * The most atomic way to train and run inference for a GPT in pure,
 * dependency-free TypeScript. This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Based on @karpathy's pure-Python GPT implementation.
 * Follows GPT-2 architecture with minor simplifications:
 *   - RMSNorm instead of LayerNorm
 *   - No biases
 *   - ReLU instead of GeLU
 */

import { readFileSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";

// ─────────────────────────────────────────────────────────────────────────────
// Seeded PRNG (Mulberry32) — deterministic random number generation
// ─────────────────────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);

/** Gaussian random via Box-Muller transform */
function gaussRandom(mean: number, std: number): number {
  const u1 = rand();
  const u2 = rand();
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Shuffle array in-place using Fisher-Yates */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Weighted random selection — returns the chosen index */
function weightedChoice(weights: number[]): number {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rand() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset — load a list of names (one per line) as our training corpus
// ─────────────────────────────────────────────────────────────────────────────

const INPUT_PATH = new URL("input.txt", import.meta.url).pathname;

if (!existsSync(INPUT_PATH)) {
  console.log("Downloading dataset...");
  const url =
    "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
  execSync(`curl -sL "${url}" -o "${INPUT_PATH}"`);
}

const docs = shuffle(
  readFileSync(INPUT_PATH, "utf-8")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean),
);
console.log(`num docs: ${docs.length}`);

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer — character-level encoding/decoding
//
// Each unique character in the dataset gets a token id (0..n-1).
// A special BOS (Beginning of Sequence) token marks sequence boundaries.
// ─────────────────────────────────────────────────────────────────────────────

const uchars = [...new Set(docs.join(""))].sort();
const BOS = uchars.length; // special Beginning-of-Sequence token id
const vocabSize = uchars.length + 1; // +1 for BOS
console.log(`vocab size: ${vocabSize}`);

// ─────────────────────────────────────────────────────────────────────────────
// Autograd engine — scalar-level automatic differentiation
//
// Each Value node stores:
//   - data:        the scalar result of the forward pass
//   - grad:        dLoss/dThis, accumulated during backward pass
//   - children:    input nodes in the computation graph
//   - localGrads:  dThis/dChild for each child (chain rule factors)
//
// Calling .backward() on the loss node propagates gradients to all
// parameters via reverse-mode autodiff (backpropagation).
// ─────────────────────────────────────────────────────────────────────────────

class Value {
  data: number;
  grad: number = 0;
  private children: Value[];
  private localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.children = children;
    this.localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(n: number): Value {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  /**
   * Backpropagation — reverse-mode automatic differentiation.
   *
   * 1. Topological sort of the computation graph (post-order DFS)
   * 2. Walk nodes in reverse order, applying the chain rule:
   *    child.grad += localGrad * parent.grad
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v.children) buildTopo(child);
      topo.push(v);
    };

    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v.children.length; j++) {
        v.children[j].grad += v.localGrads[j] * v.grad;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model hyperparameters
// ─────────────────────────────────────────────────────────────────────────────

const N_LAYER = 1; // number of transformer layers (depth)
const N_EMBD = 16; // embedding dimension (width)
const BLOCK_SIZE = 16; // max sequence length (context window)
const N_HEAD = 4; // number of attention heads
const HEAD_DIM = N_EMBD / N_HEAD; // dimension per head (derived)

// ─────────────────────────────────────────────────────────────────────────────
// Parameter initialization
//
// All parameters are stored as 2D arrays of Value nodes (matrices).
// Small random Gaussian initialization (std=0.08) prevents symmetry breaking.
// ─────────────────────────────────────────────────────────────────────────────

type Matrix = Value[][];

function matrix(nout: number, nin: number, std = 0.08): Matrix {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(gaussRandom(0, std))),
  );
}

const stateDict: Record<string, Matrix> = {
  wte: matrix(vocabSize, N_EMBD), // token embedding table
  wpe: matrix(BLOCK_SIZE, N_EMBD), // positional embedding table
  lm_head: matrix(vocabSize, N_EMBD), // final projection to vocab logits
};

for (let i = 0; i < N_LAYER; i++) {
  stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD); // query projection
  stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD); // key projection
  stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD); // value projection
  stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD); // output projection
  stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD); // MLP up-project
  stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD); // MLP down-project
}

// Flatten all parameters into a single list for the optimizer
const params: Value[] = Object.values(stateDict).flatMap((mat) =>
  mat.flatMap((row) => row),
);
console.log(`num params: ${params.length}`);

// ─────────────────────────────────────────────────────────────────────────────
// Model building blocks
// ─────────────────────────────────────────────────────────────────────────────

/** Matrix-vector multiply: y = W @ x (each row of W dotted with x) */
function linear(x: Value[], w: Matrix): Value[] {
  return w.map((wRow) =>
    wRow.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), new Value(0)),
  );
}

/**
 * Softmax — converts raw logits to a probability distribution.
 * Subtracts max for numerical stability (prevents overflow in exp).
 */
function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = exps.reduce((sum, e) => sum.add(e), new Value(0));
  return exps.map((e) => e.div(total));
}

/**
 * RMSNorm (Root Mean Square Normalization)
 * Simpler alternative to LayerNorm — normalizes by RMS of the vector.
 * No learnable scale/bias parameters in this minimal implementation.
 */
function rmsnorm(x: Value[]): Value[] {
  const ms = x
    .reduce((sum, xi) => sum.add(xi.mul(xi)), new Value(0))
    .div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT forward pass
//
// Processes one token at a time (for simplicity). Uses a KV cache so that
// previously computed keys/values are reused across positions.
//
// Architecture per layer:
//   1. Multi-head self-attention (with causal masking via KV cache)
//   2. Feed-forward MLP (expand 4x, ReLU, project back)
//   Both blocks use residual connections and pre-norm (RMSNorm).
// ─────────────────────────────────────────────────────────────────────────────

function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
): Value[] {
  // Look up token and position embeddings, combine them
  const tokEmb = stateDict["wte"][tokenId];
  const posEmb = stateDict["wpe"][posId];
  let x = tokEmb.map((t, i) => t.add(posEmb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < N_LAYER; li++) {
    // ── 1) Multi-head self-attention ──
    const xResidual1 = x;
    x = rmsnorm(x);

    // Project input into queries, keys, values
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);

    // Append to KV cache (enables autoregressive generation)
    keys[li].push(k);
    values[li].push(v);

    // Compute attention independently per head
    const xAttn: Value[] = [];
    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;

      // Slice this head's portion of Q, K, V
      const qH = q.slice(hs, hs + HEAD_DIM);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + HEAD_DIM));
      const vH = values[li].map((vi) => vi.slice(hs, hs + HEAD_DIM));

      // Scaled dot-product attention: softmax(Q·K^T / √d) · V
      const attnLogits = kH.map((kt) =>
        qH
          .reduce((sum, qj, j) => sum.add(qj.mul(kt[j])), new Value(0))
          .div(Math.sqrt(HEAD_DIM)),
      );
      const attnWeights = softmax(attnLogits);

      // Weighted sum of values
      for (let j = 0; j < HEAD_DIM; j++) {
        xAttn.push(
          attnWeights.reduce(
            (sum, w, t) => sum.add(w.mul(vH[t][j])),
            new Value(0),
          ),
        );
      }
    }

    // Project concatenated heads back to embedding dimension
    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual1[i])); // residual connection

    // ── 2) Feed-forward MLP ──
    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]); // expand to 4x
    x = x.map((xi) => xi.relu()); // non-linearity
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]); // project back
    x = x.map((a, i) => a.add(xResidual2[i])); // residual connection
  }

  // Final projection to vocabulary logits
  return linear(x, stateDict["lm_head"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop — Adam optimizer with linear learning rate decay
//
// For each training step:
//   1. Tokenize a document (wrap with BOS tokens)
//   2. Forward each token through the model, collecting per-position losses
//   3. Backpropagate the average loss to compute gradients
//   4. Update parameters with Adam
// ─────────────────────────────────────────────────────────────────────────────

const LEARNING_RATE = 0.01;
const BETA1 = 0.85; // exponential decay rate for first moment
const BETA2 = 0.99; // exponential decay rate for second moment
const EPS = 1e-8; // prevents division by zero

// Adam moment buffers (one per parameter)
const m = new Float64Array(params.length); // first moment (mean of gradients)
const v = new Float64Array(params.length); // second moment (mean of squared gradients)

const NUM_STEPS = 1000;

for (let step = 0; step < NUM_STEPS; step++) {
  // Tokenize: [BOS, char1, char2, ..., charN, BOS]
  const doc = docs[step % docs.length];
  const tokens = [BOS, ...doc.split("").map((ch) => uchars.indexOf(ch)), BOS];
  const n = Math.min(BLOCK_SIZE, tokens.length - 1);

  // Forward pass — build computation graph and compute loss
  const kvKeys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const kvValues: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const losses: Value[] = [];

  for (let pos = 0; pos < n; pos++) {
    const logits = gpt(tokens[pos], pos, kvKeys, kvValues);
    const probs = softmax(logits);
    // Cross-entropy loss: -log(probability of the correct next token)
    losses.push(probs[tokens[pos + 1]].log().neg());
  }

  // Average loss across positions
  const loss = losses.reduce((sum, l) => sum.add(l), new Value(0)).div(n);

  // Backward pass — compute gradients for all parameters
  loss.backward();

  // Adam optimizer step with linear LR warmdown
  const lrT = LEARNING_RATE * (1 - step / NUM_STEPS);
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad;
    v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad ** 2;
    const mHat = m[i] / (1 - BETA1 ** (step + 1)); // bias correction
    const vHat = v[i] / (1 - BETA2 ** (step + 1)); // bias correction
    p.data -= (lrT * mHat) / (Math.sqrt(vHat) + EPS);
    p.grad = 0; // zero gradients for next step
  }

  process.stdout.write(
    `\rstep ${String(step + 1).padStart(4)} / ${NUM_STEPS} | loss ${loss.data.toFixed(4)}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference — generate new names by sampling from the trained model
//
// Starting from BOS, repeatedly:
//   1. Forward the current token to get next-token probabilities
//   2. Apply temperature scaling (lower = more conservative)
//   3. Sample from the distribution
//   4. Stop when BOS is generated (end of sequence)
// ─────────────────────────────────────────────────────────────────────────────

const TEMPERATURE = 0.5; // (0, 1] — lower = more deterministic

console.log("\n--- inference (new, hallucinated names) ---");

for (let s = 0; s < 20; s++) {
  const kvKeys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const kvValues: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  let tokenId = BOS;
  const sample: string[] = [];

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const logits = gpt(tokenId, pos, kvKeys, kvValues);
    const scaled = logits.map((l) => l.div(TEMPERATURE));
    const probs = softmax(scaled);
    tokenId = weightedChoice(probs.map((p) => p.data));
    if (tokenId === BOS) break;
    sample.push(uchars[tokenId]);
  }

  console.log(`sample ${String(s + 1).padStart(2)}: ${sample.join("")}`);
}
