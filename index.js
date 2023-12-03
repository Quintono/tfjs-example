const webcamElement = document.getElementsByClassName("webcam")[0];
const buttons = document.getElementsByTagName("button");
const predictButton = document.getElementsByClassName("predict")[0];
const classes = ["up", "down", "left", "right"];
const predictionParagraph = document.getElementsByClassName("prediction")[0];

async function app() {
  const classifier = knnClassifier.create();
  const net = await mobilenet.load();
  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async (classId) => {
    const img = await webcam.capture();
    const activiation = net.infer(img, "conv_preds");
    classifier.addExample(activiation, classId);
    img.dispose();
  };

  for (var i = 0; i < buttons.length; i++) {
    if (buttons[i] !== predictButton) {
      let index = i;
      buttons[i].onclick = () => addExample(index);
    }
  }

  predictButton.onclick = () => runPredictions();

  async function runPredictions() {
    while (true) {
      if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
        const activiation = net.infer(img, "conv_preds");
        const result = await classifier.predictClass(activiation);
        console.log("result: ", result);
        predictionParagraph.innerText = `
            prediction: ${classes[result.label]},
            probability: ${result.confidences[result.label]}`;
        img.dispose();
      }
      await tf.nextFrame();
    }
  }
}

app();
