
import * as ort from "onnxruntime-web";
import "./style.css";

// ==================================================
// CONFIG
// ==================================================

const MODEL_SIZE = 640;

// Keep this low for now while debugging.
// Later you can increase to 0.25 or 0.30.
const CONFIDENCE_THRESHOLD = 0.01;

const IOU_THRESHOLD = 0.45;

const CLASS_NAMES = [
  "biodegradable",
  "cardboard",
  "glass",
  "metal",
  "paper",
  "plastic",
];

const DISPOSAL_INFO = {
  biodegradable: {
    title: "Organic / General Waste",
    emoji: "🌱",
  },

  cardboard: {
    title: "Paper / Cardboard",
    emoji: "📦",
  },

  glass: {
    title: "Glass Recycling",
    emoji: "🍾",
  },

  metal: {
    title: "Metal Recycling",
    emoji: "🥫",
  },

  paper: {
    title: "Paper Recycling",
    emoji: "📄",
  },

  plastic: {
    title: "Plastic Recycling",
    emoji: "♻️",
  },
};

// ==================================================
// GLOBALS
// ==================================================

let session = null;
let currentImage = null;
let letterboxInfo = null;

// ==================================================
// UI
// ==================================================

const app = document.querySelector("#app");

app.innerHTML = `
  <div class="app">

    <header class="header">

      <div class="brand">

        <div class="brand-icon">
          ♻️
        </div>

        <div>
          <h1>EcoScan</h1>
          <p>Smart waste detection</p>
        </div>

      </div>

      <div class="status-pill" id="statusPill">

        <span class="dot"></span>

        <span id="headerStatus">
          Loading
        </span>

      </div>

    </header>


    <main>

      <section class="hero">

        <div class="hero-eyebrow">
          ♻ AI-powered recycling
        </div>

        <h2>
          What kind of waste<br>
          <span>is this?</span>
        </h2>

        <p>
          Take or select a photo and let on-device AI
          identify the waste category instantly.
        </p>

      </section>


      <section class="scanner-card">

        <!-- ====================================== -->
        <!-- UPLOAD -->
        <!-- ====================================== -->

        <div
          class="upload-area"
          id="uploadArea"
        >

          <div class="upload-icon">
            📷
          </div>

          <h3>
            Scan your waste
          </h3>

          <p>
            Take a photo or choose one from your gallery
          </p>

          <label class="scan-button">

            <span>
              📷
            </span>

            Scan photo

            <input
              type="file"
              id="imageInput"
              accept="image/*"
              capture="environment"
              hidden
            >

          </label>

        </div>


        <!-- ====================================== -->
        <!-- LOADING -->
        <!-- ====================================== -->

        <div
          class="loading"
          id="loading"
        >

          <div class="loader-ring"></div>

          <h3>
            Analyzing image…
          </h3>

          <p>
            AI is detecting waste type
          </p>

        </div>


        <!-- ====================================== -->
        <!-- IMAGE -->
        <!-- ====================================== -->

        <div
          class="image-section"
          id="imageSection"
        >

          <div class="image-wrapper">

            <img
              id="previewImage"
              alt="Selected waste"
            >

            <canvas
              id="resultCanvas"
            ></canvas>

          </div>


          <button
            class="new-scan"
            id="newScan"
          >
            + Scan another photo
          </button>

        </div>

      </section>


      <!-- ======================================== -->
      <!-- RESULTS -->
      <!-- ======================================== -->

      <section
        class="results-section"
        id="results"
      >

        <span class="results-label">
          Detection results
        </span>

        <div id="resultContent"></div>

      </section>


      <!-- ======================================== -->
      <!-- MODEL STATUS -->
      <!-- ======================================== -->

      <div
        class="model-status"
        id="status"
      >

        <div class="status-loading"></div>

        Loading AI model…

      </div>

    </main>


    <footer>

      <span>
        EcoScan
      </span>

      <span>
        •
      </span>

      <span>
        On-device AI
      </span>

    </footer>

  </div>
`;

// ==================================================
// ELEMENTS
// ==================================================

const imageInput =
  document.querySelector("#imageInput");

const uploadArea =
  document.querySelector("#uploadArea");

const loading =
  document.querySelector("#loading");

const imageSection =
  document.querySelector("#imageSection");

const resultCanvas =
  document.querySelector("#resultCanvas");

const previewImage =
  document.querySelector("#previewImage");

const results =
  document.querySelector("#results");

const resultContent =
  document.querySelector("#resultContent");

const status =
  document.querySelector("#status");

const newScan =
  document.querySelector("#newScan");

const headerStatus =
  document.querySelector("#headerStatus");

const statusPill =
  document.querySelector("#statusPill");

// ==================================================
// LOAD MODEL
// ==================================================

async function loadModel() {

  try {

    status.innerHTML = `
      <div class="status-loading"></div>
      Loading AI model…
    `;

    headerStatus.textContent =
      "Loading";

    ort.env.wasm.numThreads =
      1;

    ort.env.wasm.simd =
      true;

    session =
      await ort.InferenceSession.create(
        "/models/best.onnx",
        {
          executionProviders: [
            "wasm",
          ],

          graphOptimizationLevel:
            "all",
        }
      );

    console.log(
      "Model loaded:",
      {
        inputs:
          session.inputNames,

        outputs:
          session.outputNames,
      }
    );

    status.className =
      "model-status ready";

    status.innerHTML =
      "✓ AI model ready";

    statusPill.className =
      "status-pill ready";

    headerStatus.textContent =
      "AI ready";

  } catch (error) {

    console.error(
      "Model loading error:",
      error
    );

    status.className =
      "model-status error";

    status.innerHTML = `
      ! Failed to load AI model:
      ${error.message}
    `;

    statusPill.className =
      "status-pill error";

    headerStatus.textContent =
      "Error";
  }
}

// ==================================================
// CREATE LETTERBOX CANVAS
// ==================================================

function createLetterboxCanvas(image) {

  const canvas =
    document.createElement("canvas");

  canvas.width =
    MODEL_SIZE;

  canvas.height =
    MODEL_SIZE;

  const ctx =
    canvas.getContext(
      "2d",
      {
        willReadFrequently: true,
      }
    );

  const imageWidth =
    image.naturalWidth ||
    image.width;

  const imageHeight =
    image.naturalHeight ||
    image.height;

  console.log(
    "createLetterboxCanvas image:",
    {
      width: imageWidth,
      height: imageHeight,
      complete: image.complete,
    }
  );

  if (
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {

    throw new Error(
      "Image has invalid dimensions."
    );
  }

  // ----------------------------------------------
  // Calculate scale
  // ----------------------------------------------

  const scale =
    Math.min(
      MODEL_SIZE / imageWidth,
      MODEL_SIZE / imageHeight
    );

  const newWidth =
    Math.round(
      imageWidth * scale
    );

  const newHeight =
    Math.round(
      imageHeight * scale
    );

  const dw =
    MODEL_SIZE - newWidth;

  const dh =
    MODEL_SIZE - newHeight;

  const left =
    Math.round(
      dw / 2
    );

  const top =
    Math.round(
      dh / 2
    );

  // ----------------------------------------------
  // Gray background
  // ----------------------------------------------

  ctx.fillStyle =
    "rgb(114, 114, 114)";

  ctx.fillRect(
    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE
  );

  // ----------------------------------------------
  // Draw resized image
  // ----------------------------------------------

  ctx.drawImage(
    image,
    left,
    top,
    newWidth,
    newHeight
  );

  // ----------------------------------------------
  // Save letterbox information
  // ----------------------------------------------

  letterboxInfo = {

    scale,

    padX:
      left,

    padY:
      top,

    originalWidth:
      imageWidth,

    originalHeight:
      imageHeight,
  };

  console.log(
    "Letterbox:",
    {
      original:
        `${imageWidth}x${imageHeight}`,

      resized:
        `${newWidth}x${newHeight}`,

      padX:
        left,

      padY:
        top,

      scale,
    }
  );

  return canvas;
}

// ==================================================
// IMAGE → TENSOR
// ==================================================

function imageToTensor(image) {

  const canvas =
    createLetterboxCanvas(
      image
    );

  const ctx =
    canvas.getContext(
      "2d",
      {
        willReadFrequently: true,
      }
    );

  const imageData =
    ctx.getImageData(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    );

  const pixels =
    imageData.data;

  // ----------------------------------------------
  // Pixel statistics
  // ----------------------------------------------

  let minPixel =
    255;

  let maxPixel =
    0;

  let sumPixel =
    0;

  let pixelCount =
    0;

  for (
    let i = 0;
    i < pixels.length;
    i += 4
  ) {

    const r =
      pixels[i];

    const g =
      pixels[i + 1];

    const b =
      pixels[i + 2];

    minPixel =
      Math.min(
        minPixel,
        r,
        g,
        b
      );

    maxPixel =
      Math.max(
        maxPixel,
        r,
        g,
        b
      );

    sumPixel +=
      r + g + b;

    pixelCount +=
      3;
  }

  console.log(
    "Input image pixel statistics:",
    {
      min:
        minPixel,

      max:
        maxPixel,

      mean:
        sumPixel /
        pixelCount,
    }
  );

  // ----------------------------------------------
  // CHW tensor
  // ----------------------------------------------

  const size =
    MODEL_SIZE *
    MODEL_SIZE;

  const input =
    new Float32Array(
      3 * size
    );

  for (
    let i = 0;
    i < size;
    i++
  ) {

    const px =
      i * 4;

    // R
    input[i] =
      pixels[px] /
      255.0;

    // G
    input[
      size + i
    ] =
      pixels[px + 1] /
      255.0;

    // B
    input[
      size * 2 + i
    ] =
      pixels[px + 2] /
      255.0;
  }

  // ----------------------------------------------
  // Tensor statistics
  // ----------------------------------------------

  let tensorMin =
    Infinity;

  let tensorMax =
    -Infinity;

  let tensorSum =
    0;

  for (
    let i = 0;
    i < input.length;
    i++
  ) {

    const value =
      input[i];

    tensorMin =
      Math.min(
        tensorMin,
        value
      );

    tensorMax =
      Math.max(
        tensorMax,
        value
      );

    tensorSum +=
      value;
  }

  console.log(
    "Tensor statistics:",
    {
      min:
        tensorMin,

      max:
        tensorMax,

      mean:
        tensorSum /
        input.length,
    }
  );

  return new ort.Tensor(
    "float32",
    input,
    [
      1,
      3,
      MODEL_SIZE,
      MODEL_SIZE,
    ]
  );
}

// ==================================================
// IOU
// ==================================================

function calculateIoU(
  boxA,
  boxB
) {

  const x1 =
    Math.max(
      boxA.x1,
      boxB.x1
    );

  const y1 =
    Math.max(
      boxA.y1,
      boxB.y1
    );

  const x2 =
    Math.min(
      boxA.x2,
      boxB.x2
    );

  const y2 =
    Math.min(
      boxA.y2,
      boxB.y2
    );

  const intersection =
    Math.max(
      0,
      x2 - x1
    ) *
    Math.max(
      0,
      y2 - y1
    );

  const areaA =
    Math.max(
      0,
      boxA.x2 - boxA.x1
    ) *
    Math.max(
      0,
      boxA.y2 - boxA.y1
    );

  const areaB =
    Math.max(
      0,
      boxB.x2 - boxB.x1
    ) *
    Math.max(
      0,
      boxB.y2 - boxB.y1
    );

  const union =
    areaA +
    areaB -
    intersection;

  if (
    union <= 0
  ) {
    return 0;
  }

  return (
    intersection /
    union
  );
}

// ==================================================
// NMS
// ==================================================

function nonMaximumSuppression(
  detections
) {

  const final =
    [];

  for (
    let classId = 0;
    classId < CLASS_NAMES.length;
    classId++
  ) {

    const cls =
      detections
        .filter(
          d =>
            d.classId ===
            classId
        )
        .sort(
          (a, b) =>
            b.confidence -
            a.confidence
        );

    while (
      cls.length > 0
    ) {

      const best =
        cls.shift();

      final.push(
        best
      );

      for (
        let i =
          cls.length - 1;
        i >= 0;
        i--
      ) {

        const iou =
          calculateIoU(
            best,
            cls[i]
          );

        if (
          iou >
          IOU_THRESHOLD
        ) {

          cls.splice(
            i,
            1
          );
        }
      }
    }
  }

  return final;
}

// ==================================================
// PARSE YOLO OUTPUT
// ==================================================

function parseYOLOOutput(
  outputTensor
) {

  const data =
    outputTensor.data;

  const dims =
    outputTensor.dims;

  console.log(
    "Parsing YOLO output:",
    dims
  );

  if (
    dims.length !== 3 ||
    dims[0] !== 1 ||
    dims[1] !== 10
  ) {

    throw new Error(
      `Unexpected YOLO output shape: ${dims.join(" × ")}`
    );
  }

  const numPredictions =
    dims[2];

  const candidates =
    [];

  // ----------------------------------------------
  // Read every YOLO prediction
  // ----------------------------------------------

  for (
    let i = 0;
    i < numPredictions;
    i++
  ) {

    let bestClassId =
      0;

    let bestClassScore =
      -Infinity;

    // --------------------------------------------
    // Find best class
    // --------------------------------------------

    for (
      let classId = 0;
      classId < CLASS_NAMES.length;
      classId++
    ) {

      const score =
        data[
          (4 + classId) *
          numPredictions +
          i
        ];

      if (
        score >
        bestClassScore
      ) {

        bestClassScore =
          score;

        bestClassId =
          classId;
      }
    }

    // --------------------------------------------
    // Bounding box
    // --------------------------------------------

    const cx =
      data[i];

    const cy =
      data[
        numPredictions + i
      ];

    const width =
      data[
        2 *
        numPredictions +
        i
      ];

    const height =
      data[
        3 *
        numPredictions +
        i
      ];

    candidates.push({

      index:
        i,

      classId:
        bestClassId,

      className:
        CLASS_NAMES[
          bestClassId
        ],

      confidence:
        bestClassScore,

      cx,
      cy,
      width,
      height,

      x1:
        cx -
        width / 2,

      y1:
        cy -
        height / 2,

      x2:
        cx +
        width / 2,

      y2:
        cy +
        height / 2,
    });
  }

  // ----------------------------------------------
  // Sort by confidence
  // ----------------------------------------------

  candidates.sort(
    (a, b) =>
      b.confidence -
      a.confidence
  );

  console.log(
    "TOP 10 RAW YOLO PREDICTIONS:"
  );

  console.table(
    candidates.slice(
      0,
      10
    )
  );

  // ----------------------------------------------
  // Confidence filtering
  // ----------------------------------------------

  const detections =
    candidates
      .filter(
        d =>
          d.confidence >=
          CONFIDENCE_THRESHOLD
      )
      .map(
        d => ({

          x1:
            d.x1,

          y1:
            d.y1,

          x2:
            d.x2,

          y2:
            d.y2,

          confidence:
            d.confidence,

          classId:
            d.classId,

          className:
            d.className,
        })
      );

  console.log(
    "Detections before NMS:",
    detections.length
  );

  // ----------------------------------------------
  // NMS
  // ----------------------------------------------

  const finalDetections =
    nonMaximumSuppression(
      detections
    );

  finalDetections.sort(
    (a, b) =>
      b.confidence -
      a.confidence
  );

  console.log(
    "Detections after NMS:",
    finalDetections.length
  );

  return finalDetections;
}

// ==================================================
// MAP 640x640 BOX → ORIGINAL IMAGE
// ==================================================

function mapBoxToOriginalImage(
  detection
) {

  if (
    !letterboxInfo
  ) {

    throw new Error(
      "Letterbox information is missing."
    );
  }

  const {
    scale,
    padX,
    padY,
    originalWidth,
    originalHeight,
  } =
    letterboxInfo;

  const clamp =
    (
      value,
      min,
      max
    ) => {

      return Math.max(
        min,
        Math.min(
          max,
          value
        )
      );
    };

  const x1 =
    (
      detection.x1 -
      padX
    ) /
    scale;

  const y1 =
    (
      detection.y1 -
      padY
    ) /
    scale;

  const x2 =
    (
      detection.x2 -
      padX
    ) /
    scale;

  const y2 =
    (
      detection.y2 -
      padY
    ) /
    scale;

  return {

    ...detection,

    x1:
      clamp(
        x1,
        0,
        originalWidth
      ),

    y1:
      clamp(
        y1,
        0,
        originalHeight
      ),

    x2:
      clamp(
        x2,
        0,
        originalWidth
      ),

    y2:
      clamp(
        y2,
        0,
        originalHeight
      ),
  };
}

// ==================================================
// DRAW DETECTIONS
// ==================================================

function drawDetections(
  detections
) {

  console.log(
    "================================"
  );

  console.log(
    "DRAW DETECTIONS"
  );

  console.log(
    "Detections:",
    detections
  );

  console.log(
    "Is array:",
    Array.isArray(
      detections
    )
  );

  console.log(
    "================================"
  );

  const previewImage =
    document.getElementById(
      "previewImage"
    );

  const canvas =
    document.getElementById(
      "resultCanvas"
    );

  if (
    !previewImage ||
    !canvas
  ) {

    console.error(
      "Preview image or result canvas not found."
    );

    return;
  }

  if (
    !Array.isArray(
      detections
    )
  ) {

    console.error(
      "drawDetections expected array:",
      detections
    );

    return;
  }

  // ----------------------------------------------
  // IMPORTANT:
  // Image must already be visible.
  // ----------------------------------------------

  const rect =
    previewImage.getBoundingClientRect();

  const displayWidth =
    Math.round(
      rect.width
    );

  const displayHeight =
    Math.round(
      rect.height
    );

  console.log(
    "Preview image rectangle:",
    {
      left:
        rect.left,

      top:
        rect.top,

      width:
        rect.width,

      height:
        rect.height,
    }
  );

  // ----------------------------------------------
  // Safety check
  // ----------------------------------------------

  if (
    displayWidth <= 0 ||
    displayHeight <= 0
  ) {

    console.error(
      "Image has zero display size.",
      {
        displayWidth,
        displayHeight,
      }
    );

    return;
  }

  // ----------------------------------------------
  // Canvas dimensions
  // ----------------------------------------------

  canvas.width =
    displayWidth;

  canvas.height =
    displayHeight;

  canvas.style.width =
    `${displayWidth}px`;

  canvas.style.height =
    `${displayHeight}px`;

  const ctx =
    canvas.getContext(
      "2d"
    );

  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  // ----------------------------------------------
  // Original image dimensions
  // ----------------------------------------------

  const originalWidth =
    previewImage.naturalWidth;

  const originalHeight =
    previewImage.naturalHeight;

  if (
    originalWidth <= 0 ||
    originalHeight <= 0
  ) {

    console.error(
      "Invalid natural image dimensions."
    );

    return;
  }

  // ----------------------------------------------
  // Scale original image → displayed image
  // ----------------------------------------------

  const scaleX =
    displayWidth /
    originalWidth;

  const scaleY =
    displayHeight /
    originalHeight;

  console.log(
    "Drawing information:",
    {
      originalWidth,
      originalHeight,

      displayWidth,
      displayHeight,

      scaleX,
      scaleY,

      detectionCount:
        detections.length,
    }
  );

  // ----------------------------------------------
  // Draw detections
  // ----------------------------------------------

  detections.forEach(
    (det, index) => {

      const x1 =
        det.x1 *
        scaleX;

      const y1 =
        det.y1 *
        scaleY;

      const x2 =
        det.x2 *
        scaleX;

      const y2 =
        det.y2 *
        scaleY;

      const width =
        x2 - x1;

      const height =
        y2 - y1;

      console.log(
        `Drawing box ${index}:`,
        {
          className:
            det.className,

          confidence:
            det.confidence,

          originalBox: {
            x1:
              det.x1,

            y1:
              det.y1,

            x2:
              det.x2,

            y2:
              det.y2,
          },

          displayBox: {
            x1,
            y1,
            x2,
            y2,
            width,
            height,
          },
        }
      );

      // ------------------------------------------
      // Ignore invalid boxes
      // ------------------------------------------

      if (
        width <= 1 ||
        height <= 1
      ) {

        console.warn(
          "Skipping invalid box:",
          det
        );

        return;
      }

      // ------------------------------------------
      // Dashed bounding box
      // ------------------------------------------

      ctx.beginPath();

      ctx.setLineDash([
        10,
        6,
      ]);

      ctx.strokeStyle =
        "#00ff00";

      ctx.lineWidth =
        4;

      ctx.strokeRect(
        x1,
        y1,
        width,
        height
      );

      // ------------------------------------------
      // Label
      // ------------------------------------------

      ctx.setLineDash([]);

      const label =
        `${det.className} ` +
        `${(
          det.confidence *
          100
        ).toFixed(1)}%`;

      ctx.font =
        "bold 18px Arial";

      const textWidth =
        ctx.measureText(
          label
        ).width;

      const labelHeight =
        26;

      const labelX =
        Math.max(
          0,
          x1
        );

      const labelY =
        Math.max(
          labelHeight,
          y1
        );

      // Label background
      ctx.fillStyle =
        "#00ff00";

      ctx.fillRect(
        labelX,
        labelY -
          labelHeight,
        textWidth + 12,
        labelHeight
      );

      // Label text
      ctx.fillStyle =
        "#000000";

      ctx.fillText(
        label,
        labelX + 6,
        labelY - 6
      );
    }
  );

  console.log(
    "Finished drawing boxes."
  );
}

// ==================================================
// RESULT UI
// ==================================================

function capitalize(
  text
) {

  if (
    !text
  ) {

    return "";
  }

  return (
    text.charAt(0).toUpperCase() +
    text.slice(1)
  );
}

// ==================================================
// SHOW RESULTS
// ==================================================

function showDetectionResults(
  detections
) {

  if (
    !Array.isArray(
      detections
    )
  ) {

    console.error(
      "showDetectionResults expected array:",
      detections
    );

    return;
  }

  if (
    detections.length === 0
  ) {

    resultContent.innerHTML = `

      <div class="error-result">

        <div class="error-icon">
          ?
        </div>

        <div>

          <strong>
            No waste detected
          </strong>

          <p>
            Try taking a clearer photo
            with the object closer to
            the camera.
          </p>

        </div>

      </div>
    `;

    return;
  }

  const primary =
    [...detections].sort(
      (a, b) =>
        b.confidence -
        a.confidence
    )[0];

  const info =
    DISPOSAL_INFO[
      primary.className
    ];

  let html = `

    <div class="success-result">

      <div class="success-icon">
        ${info?.emoji || "♻️"}
      </div>

      <div>

        <strong>
          ${capitalize(
            primary.className
          )}
        </strong>

        <p>
          Confidence:
          ${Math.round(
            primary.confidence *
            100
          )}%
        </p>

      </div>

    </div>


    <div class="recommendation">

      <span>
        Recommended disposal
      </span>

      <strong>
        ${
          info?.title ||
          "Check local recycling rules"
        }
      </strong>

    </div>


    <div class="detections-list">

      <h4>
        Detected objects
      </h4>
  `;

  [...detections]
    .sort(
      (a, b) =>
        b.confidence -
        a.confidence
    )
    .forEach(
      d => {

        html += `

          <div class="detection-item">

            <span class="detection-name">
              ${capitalize(
                d.className
              )}
            </span>

            <span class="detection-confidence">
              ${Math.round(
                d.confidence *
                100
              )}%
            </span>

          </div>
        `;
      }
    );

  html += `
    </div>
  `;

  resultContent.innerHTML =
    html;
}

// ==================================================
// IMAGE INPUT
// ==================================================

imageInput.addEventListener(
  "change",
  async event => {

    const file =
      event.target.files[0];

    if (
      !file
    ) {

      return;
    }

    if (
      !session
    ) {

      alert(
        "AI model is still loading. Please wait a moment."
      );

      return;
    }

    console.log(
      "================================"
    );

    console.log(
      "NEW IMAGE"
    );

    console.log(
      "Selected file:",
      {
        name:
          file.name,

        type:
          file.type,

        size:
          file.size,
      }
    );

    console.log(
      "================================"
    );

    // ----------------------------------------------
    // Reset UI
    // ----------------------------------------------

    uploadArea.style.display =
      "none";

    loading.style.display =
      "flex";

    imageSection.style.display =
      "none";

    results.style.display =
      "none";

    resultContent.innerHTML =
      "";

    // ----------------------------------------------
    // Create image
    // ----------------------------------------------

    const image =
      new Image();

    const imageUrl =
      URL.createObjectURL(
        file
      );

    currentImage =
      image;

    // ----------------------------------------------
    // Preview image
    // ----------------------------------------------

    previewImage.onload =
      () => {

        console.log(
          "Preview image loaded:",
          {
            width:
              previewImage.naturalWidth,

            height:
              previewImage.naturalHeight,
          }
        );
      };

    previewImage.onerror =
      error => {

        console.error(
          "Preview image loading error:",
          error
        );
      };

    previewImage.src =
      imageUrl;

    // ----------------------------------------------
    // AI image loaded
    // ----------------------------------------------

    image.onload =
      async () => {

        try {

          console.log(
            "AI image loaded:",
            {
              width:
                image.naturalWidth,

              height:
                image.naturalHeight,

              complete:
                image.complete,
            }
          );

          // ========================================
          // CREATE TENSOR
          // ========================================

          const tensor =
            imageToTensor(
              image
            );

          console.log(
            "Input tensor:",
            tensor.dims
          );

          // ========================================
          // RUN MODEL
          // ========================================

          const output =
            await session.run({
              images:
                tensor,
            });

          const outputTensor =
            output.output0;

          console.log(
            "Output shape:",
            outputTensor.dims
          );

          // ========================================
          // PARSE YOLO
          // ========================================

          const detections =
            parseYOLOOutput(
              outputTensor
            );

          console.log(
            "Parsed detections:",
            detections
          );

          // ========================================
          // MAP TO ORIGINAL IMAGE
          // ========================================

          const mappedDetections =
            detections.map(
              mapBoxToOriginalImage
            );

          console.log(
            "Mapped detections:",
            mappedDetections
          );

          // ========================================
          // IMPORTANT:
          // SHOW IMAGE FIRST
          // ========================================

          loading.style.display =
            "none";

          imageSection.style.display =
            "flex";

          results.style.display =
            "flex";

          // ========================================
          // WAIT FOR BROWSER TO RENDER IMAGE
          // ========================================

          requestAnimationFrame(
            () => {

              console.log(
                "Rendering canvas after image became visible..."
              );

              drawDetections(
                mappedDetections
              );

              showDetectionResults(
                mappedDetections
              );

              console.log(
                "Final detections:",
                mappedDetections
              );
            }
          );

          // ------------------------------------------
          // Cleanup object URL
          // ------------------------------------------

          setTimeout(
            () => {

              URL.revokeObjectURL(
                imageUrl
              );

            },
            1000
          );

        } catch (error) {

          console.error(
            "Detection error:",
            error
          );

          loading.style.display =
            "none";

          imageSection.style.display =
            "flex";

          results.style.display =
            "flex";

          resultContent.innerHTML = `

            <div class="error-result">

              <div class="error-icon">
                !
              </div>

              <div>

                <strong>
                  Detection failed
                </strong>

                <p>
                  ${error.message}
                </p>

              </div>

            </div>
          `;
        }
      };

    // ----------------------------------------------
    // AI image error
    // ----------------------------------------------

    image.onerror =
      error => {

        console.error(
          "AI image loading error:",
          error
        );

        loading.style.display =
          "none";

        alert(
          "Could not load the selected image."
        );

        URL.revokeObjectURL(
          imageUrl
        );
      };

    // ----------------------------------------------
    // IMPORTANT:
    // Set src AFTER onload
    // ----------------------------------------------

    image.src =
      imageUrl;
  }
);

// ==================================================
// NEW SCAN
// ==================================================

newScan.addEventListener(
  "click",
  () => {

    imageInput.value =
      "";

    uploadArea.style.display =
      "flex";

    imageSection.style.display =
      "none";

    loading.style.display =
      "none";

    results.style.display =
      "none";

    resultContent.innerHTML =
      "";

    // ----------------------------------------------
    // Clear canvas
    // ----------------------------------------------

    const ctx =
      resultCanvas.getContext(
        "2d"
      );

    ctx.clearRect(
      0,
      0,
      resultCanvas.width,
      resultCanvas.height
    );

    // ----------------------------------------------
    // Clear image
    // ----------------------------------------------

    previewImage.removeAttribute(
      "src"
    );

    currentImage =
      null;

    letterboxInfo =
      null;
  }
);

// ==================================================
// BOOT
// ==================================================

loadModel();

