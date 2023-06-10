const express = require("express");
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const { execSync } = require("child_process");
const fileUpload = require("express-fileupload");
const { v4: uuidv4 } = require("uuid");
const path = require("path");
const app = express();
const os = require("os");

const PORT = 3000;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const classes = ["Belly pain", "Burping", "Discomfort", "Hungry", "Tired"];

let model = null;
(async () => {
  model = await tf.loadLayersModel(
    "file://models/model.json"
  );
})()

app.use(
  fileUpload({
    useTempFiles: true,
    limits: {
      fileSize: 10 * 1024 * 1024,
    },
  })
);

app.post("/predict", async (req, res) => {
  // running python script, hasilnya gambar
  if (!req.files?.wavFile) {
    res.status(400).json({
      message: "Please upload the file",
    });
    return;
  }
  console.log("here")
  const wavPathParsed = path.parse(req.files.wavFile.tempFilePath);
  let wavPathFinal = path.join(wavPathParsed.dir, wavPathParsed.base);

  const imagePath = `./static/spectograms/${uuidv4()}.png`;
  try {
    if (os.platform() === "win32") {
      wavPathFinal = wavPathFinal.replace(/\\/g, "\\\\");
    }
    const command = `python3 ${__dirname}/convert.py ${wavPathFinal} ${imagePath}`;
    console.log("command", command);
    execSync(command, {
      timeout: 30_000,
      shell: '/bin/bash',
    });
  } catch (error) {
    res.status(500).json({
      message: `Error when converting image  : ${error}`,
    });
    return;
  }

  //console.log("loading model");

  //console.log("ini image path", imagePath);
  const image = fs.readFileSync(imagePath);
  let tensor = tf.node.decodeImage(image, 3);
  const resizedImage = tf.image.resizeBilinear(tensor, [64, 64]);
  const batchedImage = resizedImage.expandDims(0);
  var prediction = model.predict(batchedImage);
  var pIndex = tf.argMax(prediction, 1).dataSync();

  res.status(200).json({
    predictionResult: classes[pIndex],
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
