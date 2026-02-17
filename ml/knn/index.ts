import * as tf from "@tensorflow/tfjs";
import loadCSV from "./load-csv.js";

function knn(
  features: tf.Tensor2D,
  labels: tf.Tensor2D,
  predictionPoint: tf.Tensor,
  k: number,
): number {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.dataSync()[0] > b.dataSync()[0] ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.dataSync()[1], 0) / k
  );
}

const { features, labels, testFeatures, testLabels } = loadCSV(
  new URL("kc_house_data.csv", import.meta.url).pathname,
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  },
) as {
  features: number[][];
  labels: number[][];
  testFeatures: number[][];
  testLabels: number[][];
};

const featuresTensor = tf.tensor(features) as tf.Tensor2D;
const labelsTensor = tf.tensor(labels) as tf.Tensor2D;

testFeatures.forEach((testPoint, i) => {
  const result = knn(featuresTensor, labelsTensor, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log("Error", err * 100);
});
