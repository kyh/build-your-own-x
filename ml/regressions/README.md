# Build Your Own Regression

Three regression algorithms implemented with TensorFlow.js, progressing from simple to complex: linear regression, logistic regression (binary classification), and multinomial logistic regression (multi-class classification on MNIST).

## Quick Start

```bash
# Linear regression — predict car MPG from horsepower/weight/displacement
node ml/regressions/linear-regression/index.js

# Logistic regression — predict if a car passes emissions (binary)
node ml/regressions/logistic-regression/index.js

# Multinomial logistic regression — classify handwritten digits (MNIST)
node ml/regressions/multinominal-logistic-regression/index.js
```

## How It Works

All three implementations share the same structure:

```
1. Load and preprocess data
2. Initialize weights to zero
3. Repeat for N iterations:
   a. Forward pass  — compute predictions
   b. Compute loss  — measure how wrong predictions are
   c. Gradient step — adjust weights to reduce loss
4. Evaluate on test set
```

The key difference is what function maps inputs to outputs.

---

### 1. Linear Regression

**Goal**: Predict a continuous value (miles per gallon) from numeric features (horsepower, weight, displacement).

**Model**: A straight line (hyperplane) through the data:

```
prediction = w₀ + w₁·horsepower + w₂·weight + w₃·displacement
```

In matrix form: `y = X @ W`, where X has a column of 1s prepended for the bias term.

**Loss**: Mean Squared Error (MSE)

```
MSE = (1/n) Σ (prediction - actual)²
```

**Gradient descent**: Computes how each weight contributes to the error, then nudges weights in the opposite direction:

```
slopes = Xᵀ @ (predictions - labels) / n
weights -= learningRate * slopes
```

**Evaluation**: R² (coefficient of determination) — the fraction of variance explained by the model. R² = 1 is perfect, R² = 0 means the model is no better than predicting the mean.

```
R² = 1 - Σ(actual - predicted)² / Σ(actual - mean)²
```

---

### 2. Logistic Regression (Binary Classification)

**Goal**: Predict a binary outcome (passed emissions: yes/no) from the same car features.

**Model**: Linear regression wrapped in a **sigmoid function** that squashes output to (0, 1):

```
prediction = sigmoid(X @ W) = 1 / (1 + e^(-(X @ W)))
```

The output represents the probability of the positive class. A **decision boundary** (default 0.5) converts probabilities to yes/no predictions.

**Loss**: Cross-entropy (log loss) — penalizes confident wrong predictions harshly:

```
cost = -(1/n) Σ [ y·log(p) + (1-y)·log(1-p) ]
```

If the true label is 1 and the model predicts 0.01, the loss is -log(0.01) = 4.6 (very high). If it predicts 0.99, the loss is -log(0.99) = 0.01 (very low).

**Gradient descent**: Same form as linear regression, but predictions go through sigmoid first:

```
slopes = Xᵀ @ (sigmoid(X @ W) - labels) / n
```

**Evaluation**: Accuracy — fraction of correct predictions on the test set.

---

### 3. Multinomial Logistic Regression (Multi-Class)

**Goal**: Classify handwritten digits (0-9) from 28x28 pixel images (MNIST dataset, 60K training / 10K test).

**Model**: Logistic regression generalized to multiple classes using **softmax** instead of sigmoid:

```
predictions = softmax(X @ W)
```

Where W is now a matrix [785 x 10] (784 pixels + bias, 10 classes), and softmax normalizes each row to a probability distribution over all 10 digits:

```
softmax(z)ᵢ = e^zᵢ / Σⱼ e^zⱼ
```

Labels are **one-hot encoded**: digit 3 becomes `[0,0,0,1,0,0,0,0,0,0]`.

**Prediction**: `argmax` of the softmax output — whichever digit has the highest probability.

**Memory management**: Uses `tf.tidy()` to automatically clean up intermediate tensors, critical when processing 60K images with 784 features each.

**Variance handling**: Pixels that are always 0 (corners of digit images) have zero variance, which would cause division-by-zero during standardization. The implementation adds a filler to zero-variance features.

---

### Shared Techniques

All three implementations use these techniques:

#### Feature Standardization

```
standardized = (feature - mean) / √variance
```

Centers features at 0 with unit variance so no single feature dominates the gradient just because it has larger numbers.

#### Mini-Batch Gradient Descent

Instead of computing gradients over the full dataset (slow) or one example (noisy), processes data in batches:

```
for each iteration:
  for each batch of size B:
    compute gradient on batch
    update weights
```

This balances speed and stability. Batch size is a hyperparameter (10 for cars, 500 for MNIST).

#### Adaptive Learning Rate

After each iteration, if the loss increased, the learning rate was too aggressive — halve it. If the loss decreased, cautiously increase it by 5%:

```
if cost went up:   learningRate /= 2
if cost went down: learningRate *= 1.05
```

This self-correcting mechanism prevents divergence without requiring manual tuning.

#### Bias Term

A column of 1s is prepended to the feature matrix, giving the model a learnable bias (intercept). Without it, the model would be forced to pass through the origin.

---

## Progression Summary

| | Linear | Logistic | Multinomial |
|---|--------|----------|-------------|
| **Output** | Continuous (MPG) | Binary (pass/fail) | Multi-class (digit 0-9) |
| **Activation** | None (identity) | Sigmoid | Softmax |
| **Loss** | MSE | Cross-entropy | Cross-entropy |
| **Weights shape** | [4, 1] | [4, 1] | [785, 10] |
| **Dataset** | 392 cars | 392 cars | 60K digits |
| **Metric** | R² | Accuracy | Accuracy |

## What This Teaches vs. Production ML

| This Implementation | Production Systems |
|---------------------|-------------------|
| Manual gradient computation | Automatic differentiation (autograd) |
| Full-matrix operations | GPU-accelerated sparse ops |
| Simple adaptive LR | Adam, AdaGrad, learning rate schedules |
| Linear models only | Deep neural networks |
| Standardization only | Feature engineering pipelines |
| Single train/test split | Cross-validation, early stopping |

The gradient descent loop — forward, loss, backward, update — is the same loop used to train GPT-4 and every other neural network. These implementations make that loop explicit and visible.
