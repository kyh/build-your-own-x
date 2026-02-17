import loadCSV from "../load-csv.js";
import LinearRegression from "./linear-regression.js";

const { features, labels, testFeatures, testLabels } = loadCSV(
  new URL("../data/cars.csv", import.meta.url).pathname,
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "weight", "displacement"],
    labelColumns: ["mpg"],
  },
) as {
  features: number[][];
  labels: number[][];
  testFeatures: number[][];
  testLabels: number[][];
};

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

console.log("R2 is", r2);
regression.predict([[120, 2, 380]]).print();
