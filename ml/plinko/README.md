# Build Your Own Plinko + KNN Classifier

A Plinko board simulation (using Matter.js physics) that collects ball-drop data, then uses a from-scratch K-Nearest Neighbors classifier to predict which bucket a ball will land in based on its drop position, bounciness, and size.

## Quick Start

Open `ml/plinko/index.html` in a browser. Drop balls by clicking on the board, then click "Analyze!" to run the KNN classification and see accuracy per feature in the console.

## How It Works

This project combines two ideas: a physics simulation to generate data, and a KNN classifier to learn from it.

### The Two Pieces

```
1. Plinko board   — physics simulation that generates labeled data
2. KNN classifier — predicts bucket from ball properties
```

---

### 1. Plinko Board (Data Generation)

The board is built with [Matter.js](https://brm.io/matter-js/), a 2D physics engine. It creates:
- A grid of **pegs** the ball bounces off
- **10 buckets** at the bottom that catch balls
- **Walls** to keep balls on the board

Each ball has three configurable properties:
- **Drop position** (x-coordinate where it enters)
- **Bounciness** (coefficient of restitution, 0-1)
- **Size** (radius in pixels)

When a ball settles into a bucket, the simulation records `[dropPosition, bounciness, size, bucketLabel]` as one data point. Drop enough balls and you have a labeled dataset.

Controls let you:
- Click to drop individual balls
- "Scan" to drop balls at regular intervals across the board
- "Spot" to drop many balls from one position
- Adjust bounciness and size ranges

---

### 2. KNN Classifier (Prediction)

Once data is collected, the "Analyze!" button runs a KNN classifier to answer: **given a ball's properties, which bucket will it land in?**

This is **classification** (discrete buckets 1-10), not regression (continuous values). The KNN algorithm:

```
To predict which bucket a new ball lands in:
  1. Compute distance to every ball in the training set
  2. Sort by distance
  3. Take the K closest neighbors
  4. Majority vote on their bucket labels → prediction
```

The implementation tests each feature independently to show which properties are most predictive:

```javascript
_.range(0, 3).forEach(feature => {  // 0=position, 1=bounciness, 2=size
  // test KNN using only this single feature
  // print accuracy
});
```

Typically drop position is the strongest predictor — a ball dropped on the left side usually lands in a left bucket regardless of bounciness or size.

---

### Feature Normalization (Min-Max)

Before computing distances, features are normalized to [0, 1] using min-max scaling:

```
normalized = (value - min) / (max - min)
```

Without this, drop position (~0-800 pixels) would dominate distance over bounciness (~0-1), making bounciness effectively invisible to the algorithm.

---

### Distance Function

Euclidean distance between two data points:

```
d = √( (x₁-x₂)² + (y₁-y₂)² + ... )
```

---

### Train/Test Split

The data is shuffled and split:
- **Test set**: 100 random data points (held out for evaluation)
- **Training set**: everything else

Accuracy = (correct predictions / test set size). The console reports accuracy per feature so you can see which ball properties matter most.

## What This Teaches

- **Data generation**: using simulation to create labeled training data
- **KNN classification**: majority-vote variant of KNN (vs. averaging for regression)
- **Feature importance**: testing features individually reveals which ones carry signal
- **Normalization**: why feature scaling matters for distance-based algorithms
- **Train/test split**: evaluating model accuracy on unseen data
