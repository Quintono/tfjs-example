// Load machine learning model:
const loadModel = async () => {
  const url =
    "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json";
  const model = await tf.loadLayersModel(url);
  return model;
};

// Load machine learning metadata:
// This is a JSON file with the sample data used to train the model we are loading
// inot our application. It contains about 20,000 words
const loadMetadata = async () => {
  const url =
    "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json";

  const metadata = await fetch(url);
  return metadata.json();
};

// longest piece of string in dataset used to train the model was 100 words
// we need to tranform our vector of length 7 to a vector of lenght 100
const padSequences = (sequences, metadata) => {
  return sequences.map((seq) => {
    if (seq.length > metadata.max_len) {
      seq.splice(0, seq.length - metadata.max_len);
    }
    if (seq.length < metadata.max_len) {
      const pad = [];
      for (let i = 0; i < metadata.max_len - seq.length; ++i) {
        pad.push(0);
      }
      seq = pad.concat(seq);
    }
    return seq;
  });
};

const predict = (text, model, metadata) => {
  // Turn our input text into vectors
  const trimmed = text
    .trim() // remove potential whitespace at beginning and end
    .toLowerCase() // make everything lowercase
    .replace(/(\.|\,|\!|\?)/g, "") // remove all punctuation
    .split(" "); // split into an array of substrings

  // Turn array into an array of numbers using metadata
  // This will map each word to their index in the metadata file
  const sequence = trimmed.map((word) => {
    const wordIndex = metadata.word_index[word];
    if (typeof wordIndex === "undefined") {
      return 2; // oov(out-of-vocabulary) index
    }
    return wordIndex + metadata.index_from;
  });

  const paddedSequences = padSequences([sequence], metadata);

  // now we can turn our data into a tensor
  const input = tf.tensor2d(paddedSequences, [1, metadata.max_len]);
  const prediction = model.predict(input);
  // score will be a float number between 0 and 1. The closest score to 0, the more negative
  // it is predicted to be, and the closest to 1, the more positive.
  const score = prediction.dataSync()[0];
  prediction.dispose();
  return score;
};

const getSentiment = (score) => {
  if (score > 0.66) {
    return `Score of ${score} is Positive`;
  } else if (score > 0.4) {
    return `Score of ${score} is Neutral`;
  } else {
    return `Score of ${score} is Negative`;
  }
};

const run = async (text) => {
  const model = await loadModel();
  const metadata = await loadMetadata();
  let sum = 0;
  text.forEach(function (prediction) {
    perc = predict(prediction, model, metadata);
    sum += parseFloat(perc, 10);
  });
  console.log(getSentiment(sum / text.length));
};

window.onload = () => {
  const inputText = document.getElementsByTagName("input")[0];
  const button = document.getElementsByTagName("button")[0];
  button.onclick = () => {
    run([inputText.value]);
  };
};
