import { readFileSync } from "node:fs";

interface LoadCSVOptions {
  dataColumns?: string[];
  labelColumns?: string[];
  converters?: Record<string, (value: string) => number>;
  shuffle?: boolean;
  splitTest?: number | boolean;
}

interface SplitResult {
  features: number[][];
  labels: number[][];
  testFeatures: number[][];
  testLabels: number[][];
}

interface FullResult {
  features: number[][];
  labels: number[][];
}

function extractColumns(
  data: (string | number)[][],
  columnNames: string[],
): (string | number)[][] {
  const headers = data[0] as string[];
  const indexes = columnNames.map((col) => headers.indexOf(col));
  return data.map((row) => indexes.map((i) => row[i]));
}

export default function loadCSV(
  filename: string,
  options: LoadCSVOptions = {},
): SplitResult | FullResult {
  const {
    dataColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = false,
  } = options;

  const raw = readFileSync(filename, { encoding: "utf-8" });
  let rows: (string | number)[][] = raw
    .split("\n")
    .map((line) => line.split(","))
    .filter((row) => row.some((cell) => cell.trim() !== ""));

  const headers = rows[0] as string[];

  rows = rows.map((row, rowIdx) => {
    if (rowIdx === 0) return row;
    return row.map((cell, colIdx) => {
      const colName = headers[colIdx];
      if (converters[colName]) {
        const converted = converters[colName](cell as string);
        return isNaN(converted) ? cell : converted;
      }
      const result = parseFloat((cell as string).replace('"', ""));
      return isNaN(result) ? cell : result;
    });
  });

  let labels = extractColumns(rows, labelColumns);
  let data = extractColumns(rows, dataColumns);

  data.shift();
  labels.shift();

  if (shuffle) {
    const seed = "phrase";
    let s = Array.from(seed).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
    const indices = data.map((_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      s = (s * 9301 + 49297) % 233280;
      const j = Math.floor((s / 233280) * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    data = indices.map((i) => data[i]);
    labels = indices.map((i) => labels[i]);
  }

  if (splitTest) {
    const trainSize =
      typeof splitTest === "number"
        ? splitTest
        : Math.floor(data.length / 2);

    return {
      features: data.slice(trainSize) as number[][],
      labels: labels.slice(trainSize) as number[][],
      testFeatures: data.slice(0, trainSize) as number[][],
      testLabels: labels.slice(0, trainSize) as number[][],
    };
  }

  return { features: data as number[][], labels: labels as number[][] };
}
