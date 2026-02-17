# Build Your Own K-Nearest Neighbors

A K-Nearest Neighbors (KNN) implementation for predicting house prices using TensorFlow.js. Given a home's features (location, size), it finds the K most similar homes in the dataset and averages their prices as the prediction.

## Quick Start

```bash
node ml/knn/index.js
```

Loads the King County housing dataset, splits it into training/test sets, and prints the prediction error percentage for each test point.

## How It Works

KNN is one of the simplest ML algorithms — it makes predictions based purely on similarity to known data. No training phase, no learned parameters, no neural networks. Just a distance calculation and a vote.

### The Algorithm

```
To predict the price of a new house:
  1. Compute the distance from the new house to every house in the dataset
  2. Sort by distance (closest first)
  3. Take the K closest neighbors
  4. Average their prices → that's your prediction
```

That's the entire algorithm. The intuition: houses with similar features (location, square footage) probably have similar prices.

### The Three Pieces

```
1. Data loading    — CSV → feature/label arrays
2. Standardization — normalize features to comparable scales
3. KNN function    — distance, sort, average
```

---

### 1. Data Loading

The `loadCSV` utility reads `kc_house_data.csv` (King County, WA housing sales) and extracts:
- **Features** (inputs): `lat`, `long`, `sqft_lot`, `sqft_living`
- **Labels** (output): `price`

It shuffles the data and splits it into training and test sets (`splitTest: 10` holds out 10 rows for testing).

---

### 2. Feature Standardization

Raw features have wildly different scales — latitude might be ~47.5 while square footage is ~2000. Without normalization, distance calculations would be dominated by whichever feature has the largest numbers.

**Standardization** (z-score normalization) fixes this:

```
standardized = (value - mean) / √variance
```

After standardization, every feature has mean ~0 and variance ~1, so they contribute equally to distance calculations.

This implementation uses TensorFlow.js's `tf.moments()` to compute mean and variance across the training set, then applies the same transformation to the prediction point.

---

### 3. The KNN Function

The core algorithm in one TensorFlow.js chain:

```
features                          // [N, 4] — all training houses
  .sub(mean).div(variance.pow(0.5))  // standardize
  .sub(scaledPrediction)             // difference from target
  .pow(2)                            // square differences
  .sum(1)                            // sum across features → squared distances
  .pow(0.5)                          // Euclidean distance
  .expandDims(1)                     // reshape for concat
  .concat(labels, 1)                 // attach prices to distances
  .unstack()                         // split into individual rows
  .sort(by distance)                 // closest first
  .slice(0, k)                       // take K nearest
  .reduce(average price)             // → prediction
```

**Euclidean distance** between two points:

```
d = √( (x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)² + ... )
```

Each feature dimension is a coordinate. Houses "close" in this feature space tend to have similar prices.

---

### Choosing K

K controls the bias-variance tradeoff:
- **Small K** (e.g. 1): predictions are sensitive to noise — one unusual neighbor skews everything
- **Large K** (e.g. 100): predictions are oversmoothed — distant, irrelevant houses pull the average
- **Sweet spot** (this implementation uses K=10): balanced between noise sensitivity and over-averaging

---

### Evaluating Accuracy

For each test house, the implementation computes the relative error:

```
error = (actual_price - predicted_price) / actual_price
```

A 10% error means the prediction was off by 10% of the actual sale price.

## What This Teaches vs. Production ML

| This Implementation | Production Systems |
|---------------------|-------------------|
| 4 features | Hundreds of features |
| Euclidean distance | Weighted/learned distance metrics |
| Simple average of K neighbors | Weighted average (closer = more weight) |
| Brute-force search (O(n)) | KD-trees, ball trees (O(log n)) |
| Fixed K | Cross-validated K selection |
| No feature engineering | Extensive feature engineering |

The core idea — predict by finding similar examples — is used everywhere from recommendation systems to anomaly detection.
