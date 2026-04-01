const gridSize = 28;
const brushRadius = 1;

const canvas = document.getElementById("digit-canvas");
const context = canvas.getContext("2d", { willReadFrequently: false });
const statusEl = document.getElementById("status");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const recognizeButton = document.getElementById("recognize-button");
const clearButton = document.getElementById("clear-button");
const samplesStatusEl = document.getElementById("samples-status");
const samplesGridEl = document.getElementById("samples-grid");

const pixels = new Float32Array(gridSize * gridSize);
let drawing = false;

renderPixels();
setControlsEnabled(false);

canvas.addEventListener("pointerdown", (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  paintFromEvent(event);
});

canvas.addEventListener("pointermove", (event) => {
  if (drawing) {
    paintFromEvent(event);
  }
});

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

recognizeButton.addEventListener("click", () => {
  if (!window.nnlearn) {
    return;
  }

  const result = window.nnlearn.predictDigit(Array.from(pixels));
  if (!result.ok) {
    statusEl.textContent = result.error;
    return;
  }

  predictionEl.textContent = String(result.digit);
  confidenceEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
  statusEl.textContent = "Recognition complete.";
});

clearButton.addEventListener("click", () => {
  pixels.fill(0);
  renderPixels();
  predictionEl.textContent = "–";
  confidenceEl.textContent = "Confidence: –";
  statusEl.textContent = "Canvas cleared.";
});

initialize().catch((error) => {
  statusEl.textContent = error instanceof Error ? error.message : String(error);
});

async function initialize() {
  statusEl.textContent = "Starting WebAssembly runtime…";

  const go = new Go();
  const wasmModule = await loadWasmModule("./app.wasm", go.importObject);
  go.run(wasmModule.instance);
  await waitFor(() => window.nnlearn && typeof window.nnlearn.loadModel === "function");

  statusEl.textContent = "Loading model/model.json…";
  const response = await fetch("./model/model.json");
  if (!response.ok) {
    throw new Error(`failed to load model/model.json: ${response.status} ${response.statusText}`);
  }

  const modelJson = await response.text();
  const result = window.nnlearn.loadModel(modelJson);
  if (!result.ok) {
    throw new Error(result.error);
  }

  setControlsEnabled(true);
  statusEl.textContent = "Model ready. Draw a digit and click Recognize.";
  await loadTrainingSamples();
}

async function loadWasmModule(path, importObject) {
  if (WebAssembly.instantiateStreaming) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
    }
    return WebAssembly.instantiateStreaming(response, importObject);
  }

  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`failed to load ${path}: ${response.status} ${response.statusText}`);
  }

  const bytes = await response.arrayBuffer();
  return WebAssembly.instantiate(bytes, importObject);
}

async function waitFor(check, timeoutMs = 5000) {
  const startedAt = performance.now();

  while (!check()) {
    if (performance.now() - startedAt > timeoutMs) {
      throw new Error("timed out waiting for WebAssembly API");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 16));
  }
}

function stopDrawing(event) {
  if ("pointerId" in event && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  drawing = false;
}

function paintFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(gridSize - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * gridSize)));
  const y = Math.min(gridSize - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * gridSize)));

  for (let dy = -brushRadius; dy <= brushRadius; dy += 1) {
    for (let dx = -brushRadius; dx <= brushRadius; dx += 1) {
      const px = x + dx;
      const py = y + dy;
      if (px < 0 || px >= gridSize || py < 0 || py >= gridSize) {
        continue;
      }

      const distance = Math.abs(dx) + Math.abs(dy);
      const intensity = distance === 0 ? 1 : 0.55;
      const index = py * gridSize + px;
      pixels[index] = Math.max(pixels[index], intensity);
    }
  }

  renderPixels();
}

function renderPixels() {
  const image = context.createImageData(gridSize, gridSize);
  for (let i = 0; i < pixels.length; i += 1) {
    const value = Math.round(pixels[i] * 255);
    const offset = i * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }

  context.putImageData(image, 0, 0);
}

function setControlsEnabled(enabled) {
  recognizeButton.disabled = !enabled;
  clearButton.disabled = !enabled;
}

async function loadTrainingSamples() {
  samplesStatusEl.textContent = "Loading training examples…";

  const response = await fetch("./training-samples.json");
  if (!response.ok) {
    throw new Error(`failed to load training samples: ${response.status} ${response.statusText}`);
  }

  const samples = await response.json();
  renderTrainingSamples(samples);
  samplesStatusEl.textContent = `Showing ${samples.length} training examples. Click one to load it into the canvas.`;
}

function renderTrainingSamples(samples) {
  samplesGridEl.replaceChildren();

  for (const sample of samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-card";
    button.setAttribute("aria-label", `Training example ${sample.index + 1}, label ${sample.label}`);
    button.addEventListener("click", () => {
      pixels.set(sample.pixels);
      renderPixels();
      predictionEl.textContent = String(sample.label);
      confidenceEl.textContent = "Confidence: example label";
      statusEl.textContent = `Loaded training example ${sample.index + 1} (digit ${sample.label}) into the canvas.`;
    });

    const preview = document.createElement("canvas");
    preview.width = gridSize;
    preview.height = gridSize;
    drawPixels(preview.getContext("2d"), sample.pixels);

    const label = document.createElement("div");
    label.className = "sample-label";
    label.textContent = `#${sample.index + 1} → ${sample.label}`;

    button.append(preview, label);
    samplesGridEl.append(button);
  }
}

function drawPixels(targetContext, sourcePixels) {
  const image = targetContext.createImageData(gridSize, gridSize);
  for (let i = 0; i < sourcePixels.length; i += 1) {
    const value = Math.round(sourcePixels[i] * 255);
    const offset = i * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }

  targetContext.putImageData(image, 0, 0);
}
