import { createRequire } from "node:module";
import LogisticRegression from "./logistic-regression.js";

const require = createRequire(import.meta.url);
const mnist = require("mnist-data");

interface MnistSet {
  images: { values: number[][][] };
  labels: { values: number[] };
}

function loadData(): { features: number[][]; labels: number[][] } {
  const mnistData: MnistSet = mnist.training(0, 5000);

  const features = mnistData.images.values.map((image) => image.flat());
  const labels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0) as number[];
    row[label] = 1;
    return row;
  });

  return { features, labels };
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 500,
});

regression.train();

const testMnistData: MnistSet = mnist.testing(0, 1000);
const testFeatures = testMnistData.images.values.map((image) => image.flat());
const testLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0) as number[];
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testLabels);
console.log("Accuracy is", accuracy);
