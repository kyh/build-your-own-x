import * as tf from "@tensorflow/tfjs";

interface LinearRegressionOptions {
  learningRate?: number;
  iterations?: number;
  batchSize?: number;
}

export default class LinearRegression {
  features: tf.Tensor2D;
  labels: tf.Tensor2D;
  weights: tf.Tensor2D;
  mseHistory: number[] = [];
  private options: Required<LinearRegressionOptions>;
  private mean: tf.Tensor | undefined;
  private variance: tf.Tensor | undefined;

  constructor(
    features: number[][],
    labels: number[][],
    options: LinearRegressionOptions = {},
  ) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels) as tf.Tensor2D;
    this.options = {
      learningRate: 0.1,
      iterations: 1000,
      batchSize: 10,
      ...options,
    };
    this.weights = tf.zeros([this.features.shape[1], 1]) as tf.Tensor2D;
  }

  gradientDescent(features: tf.Tensor2D, labels: tf.Tensor2D): void {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]) as tf.Tensor2D;

    this.weights = this.weights.sub(
      slopes.mul(this.options.learningRate),
    ) as tf.Tensor2D;
  }

  train(): void {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize,
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1],
        ) as tf.Tensor2D;
        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [batchSize, -1],
        ) as tf.Tensor2D;

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations: number[][]): tf.Tensor {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures: number[][], testLabels: number[][]): number {
    const processedFeatures = this.processFeatures(testFeatures);
    const processedLabels = tf.tensor(testLabels);

    const predictions = processedFeatures.matMul(this.weights);

    const res = processedLabels.sub(predictions).pow(2).sum().dataSync()[0];
    const tot = processedLabels
      .sub(processedLabels.mean())
      .pow(2)
      .sum()
      .dataSync()[0];

    return 1 - res / tot;
  }

  processFeatures(features: number[][]): tf.Tensor2D {
    let featuresTensor = tf.tensor(features);

    if (this.mean && this.variance) {
      featuresTensor = featuresTensor
        .sub(this.mean)
        .div(this.variance.pow(0.5));
    } else {
      featuresTensor = this.standardize(featuresTensor);
    }

    featuresTensor = tf
      .ones([featuresTensor.shape[0], 1])
      .concat(featuresTensor, 1);

    return featuresTensor as tf.Tensor2D;
  }

  private standardize(features: tf.Tensor): tf.Tensor {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
    return features.sub(mean).div(variance.pow(0.5));
  }

  private recordMSE(): void {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .dataSync()[0];

    this.mseHistory.unshift(mse);
  }

  private updateLearningRate(): void {
    if (this.mseHistory.length < 2) return;

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}
