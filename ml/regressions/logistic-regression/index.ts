import loadCSV from "../load-csv.js";
import LogisticRegression from "./logistic-regression.js";

const { features, labels, testFeatures, testLabels } = loadCSV(
  new URL("../data/cars.csv", import.meta.url).pathname,
  {
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: (value: string) => (value === "TRUE" ? 1 : 0),
    },
  },
) as {
  features: number[][];
  labels: number[][];
  testFeatures: number[][];
  testLabels: number[][];
};

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

regression.train();

console.log(regression.test(testFeatures, testLabels));
