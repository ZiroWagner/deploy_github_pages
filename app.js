// app.js — Motor de Inferencia Client-Side ONNX Runtime Web (NTP 900.058:2019)
// Soporta Clasificador (320x320), Detector YOLO26s (640x640) y Modo Híbrido Ponderado

const NTP_CLASSES = {
  plastico: {
    color: "BLANCO",
    container: "Contenedor BLANCO — Plásticos Aprovechables",
    bg: "#e9edf2",
    fg: "#16181d",
    border: "#94a3b8",
    action: "Depositar botellas PET limpias, envases HDPE/PP y bolsas plásticas limpias."
  },
  papel_carton: {
    color: "AZUL",
    container: "Contenedor AZUL — Papel y Cartón",
    bg: "#1f6feb",
    fg: "#ffffff",
    border: "#1d4ed8",
    action: "Depositar hojas de papel, cajas de cartón corrugado, periódicos y revistas secas."
  },
  metal: {
    color: "AMARILLO",
    container: "Contenedor AMARILLO — Metales",
    bg: "#f2c94c",
    fg: "#16181d",
    border: "#d97706",
    action: "Depositar latas de conserva, latas de aluminio, tapas metálicas y envases de hojalata."
  },
  vidrio: {
    color: "GRIS",
    container: "Contenedor GRIS / PLOMO — Vidrio",
    bg: "#8a94a6",
    fg: "#ffffff",
    border: "#475569",
    action: "Depositar botellas de vidrio, frascos y envases de vidrio transparente o ámbar."
  },
  organico: {
    color: "MARRÓN",
    container: "Contenedor MARRÓN — Residuos Orgánicos",
    bg: "#8b5a2b",
    fg: "#ffffff",
    border: "#78350f",
    action: "Depositar restos de alimentos, cáscaras de frutas/verduras y residuos compostables."
  },
  otros_residuos: {
    color: "NEGRO",
    container: "Contenedor NEGRO — No Aprovechables / Generales",
    bg: "#1b1f24",
    fg: "#ffffff",
    border: "#0f172a",
    action: "Depositar tecnopor (poliestireno), envolturas metalizadas, servilletas sucias y sanitarios."
  },
  peligroso: {
    color: "ROJO",
    container: "Contenedor ROJO — Residuos Peligrosos / Biocontaminados",
    bg: "#e5484d",
    fg: "#ffffff",
    border: "#b91c1c",
    action: "¡ALERTA SANITARIA! Depositar EPP clínico (mascarillas, guantes, gasas), pilas, aerosoles, jeringas y fármacos."
  }
};

// Orden de salida en ciro_cls_ntp_v3.onnx (ImageFolder alfabético)
const CLS_INDEX_TO_NAME = [
  "metal",
  "organico",
  "otros_residuos",
  "papel_carton",
  "peligroso",
  "plastico",
  "vidrio"
];

// Orden de salida en yolo26s_ntp_7c_v3.onnx (configs/taxonomy.yaml)
const DET_INDEX_TO_NAME = [
  "plastico",
  "papel_carton",
  "metal",
  "vidrio",
  "organico",
  "otros_residuos",
  "peligroso"
];

let clsSession = null;
let detSession = null;
let webcamStream = null;
let webcamLoopId = null;

const statusEl = document.getElementById("model-status");
const latencyEl = document.getElementById("inference-latency");
const modeSelect = document.getElementById("mode-select");
const canvas = document.getElementById("view-canvas");
const ctx = canvas.getContext("2d");

async function initModels() {
  try {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    statusEl.textContent = "Cargando modelo Clasificador ONNX (ciro_cls_ntp_v3.onnx)...";
    clsSession = await ort.InferenceSession.create("models/ciro_cls_ntp_v3.onnx", {
      executionProviders: ["wasm"]
    });
    statusEl.textContent = "✔ Clasificador ONNX listo (320×320 · NTP 900.058:2019)";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error al cargar ONNX: " + err.message;
  }
}

async function ensureDetModel() {
  if (detSession) return detSession;
  statusEl.textContent = "Cargando Detector Multiobjeto ONNX (yolo26s_ntp_7c_v3.onnx)...";
  detSession = await ort.InferenceSession.create("models/yolo26s_ntp_7c_v3.onnx", {
    executionProviders: ["wasm"]
  });
  statusEl.textContent = "✔ Clasificador + Detector ONNX activos (Modo Híbrido / Multiobjeto)";
  return detSession;
}

function preprocessToTensor(imgSource, targetSize) {
  const off = document.createElement("canvas");
  off.width = targetSize;
  off.height = targetSize;
  const octx = off.getContext("2d");
  octx.drawImage(imgSource, 0, 0, targetSize, targetSize);
  const imgData = octx.getImageData(0, 0, targetSize, targetSize).data;
  const area = targetSize * targetSize;
  const floatData = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    floatData[i] = imgData[i * 4] / 255.0;
    floatData[area + i] = imgData[i * 4 + 1] / 255.0;
    floatData[2 * area + i] = imgData[i * 4 + 2] / 255.0;
  }
  return new ort.Tensor("float32", floatData, [1, 3, targetSize, targetSize]);
}

function softmaxIfLogits(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  if (sum > 0.98 && sum < 1.02 && arr.every(v => v >= 0 && v <= 1)) {
    return Array.from(arr);
  }
  const maxV = Math.max(...arr);
  const exps = Array.from(arr).map(v => Math.exp(v - maxV));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
}

async function runClassifier(imgSource) {
  if (!clsSession) await initModels();
  const inputTensor = preprocessToTensor(imgSource, 320);
  const feeds = { [clsSession.inputNames[0]]: inputTensor };
  const results = await clsSession.run(feeds);
  const rawOut = results[clsSession.outputNames[0]].data;
  const probsArr = softmaxIfLogits(rawOut);
  const probMap = {};
  CLS_INDEX_TO_NAME.forEach((cname, idx) => {
    probMap[cname] = probsArr[idx] || 0;
  });
  return probMap;
}

function iouBox(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1));
  const areaB = Math.max(1, (b.x2 - b.x1) * (b.y2 - b.y1));
  return inter / (areaA + areaB - inter);
}

async function runDetector(imgSource, confThreshold = 0.22) {
  await ensureDetModel();
  const inputTensor = preprocessToTensor(imgSource, 640);
  const feeds = { [detSession.inputNames[0]]: inputTensor };
  const results = await detSession.run(feeds);
  const outTensor = results[detSession.outputNames[0]];
  const data = outTensor.data;
  const dims = outTensor.dims; // [1, 11, 8400]
  const numChannels = dims[1];
  const numAnchors = dims[2];

  const candidates = [];
  const maxPerClass = {};
  DET_INDEX_TO_NAME.forEach(c => { maxPerClass[c] = 0; });

  for (let a = 0; a < numAnchors; a++) {
    let bestClass = -1;
    let bestScore = 0;
    for (let c = 0; c < 7; c++) {
      const sc = data[(4 + c) * numAnchors + a];
      const cname = DET_INDEX_TO_NAME[c];
      if (sc > maxPerClass[cname]) maxPerClass[cname] = sc;
      if (sc > bestScore) {
        bestScore = sc;
        bestClass = c;
      }
    }
    if (bestScore >= confThreshold && bestClass >= 0) {
      const xc = data[0 * numAnchors + a];
      const yc = data[1 * numAnchors + a];
      const bw = data[2 * numAnchors + a];
      const bh = data[3 * numAnchors + a];
      candidates.push({
        className: DET_INDEX_TO_NAME[bestClass],
        score: bestScore,
        x1: (xc - bw / 2) / 640,
        y1: (yc - bh / 2) / 640,
        x2: (xc + bw / 2) / 640,
        y2: (yc + bh / 2) / 640
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const boxes = [];
  for (const cand of candidates) {
    if (boxes.length >= 12) break;
    let keep = true;
    for (const kept of boxes) {
      if (iouBox(cand, kept) > 0.48) {
        keep = false;
        break;
      }
    }
    if (keep) boxes.push(cand);
  }
  return { boxes, maxPerClass };
}

function drawSceneAndBoxes(imgSource, boxes) {
  const w = imgSource.videoWidth || imgSource.naturalWidth || imgSource.width || 640;
  const h = imgSource.videoHeight || imgSource.naturalHeight || imgSource.height || 480;
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(imgSource, 0, 0, w, h);

  for (const b of boxes) {
    const info = NTP_CLASSES[b.className] || NTP_CLASSES.otros_residuos;
    const x1 = Math.max(0, b.x1 * w);
    const y1 = Math.max(0, b.y1 * h);
    const bw = Math.min(w - x1, (b.x2 - b.x1) * w);
    const bh = Math.min(h - y1, (b.y2 - b.y1) * h);

    ctx.strokeStyle = info.bg === "#e9edf2" ? "#38bdf8" : info.bg;
    ctx.lineWidth = Math.max(3, Math.round(w / 180));
    ctx.strokeRect(x1, y1, bw, bh);

    const label = `${info.color} · ${b.className} ${(b.score * 100).toFixed(0)}%`;
    ctx.font = `bold ${Math.max(12, Math.round(w / 42))}px Inter, sans-serif`;
    const tw = ctx.measureText(label).width + 12;
    const th = Math.max(22, Math.round(w / 26));
    const ly = Math.max(0, y1 - th);
    ctx.fillStyle = info.bg;
    ctx.fillRect(x1, ly, tw, th);
    ctx.fillStyle = info.fg;
    ctx.fillText(label, x1 + 6, ly + th - 6);
  }
}

function renderResultsUI(probMap, modeLabel, boxesCount) {
  const entries = Object.entries(probMap).sort((a, b) => b[1] - a[1]);
  const [topClass, topProb] = entries[0];
  const info = NTP_CLASSES[topClass];

  const card = document.getElementById("tacho-card");
  const icon = document.getElementById("tacho-icon");
  const title = document.getElementById("tacho-title");
  const desc = document.getElementById("tacho-desc");
  const meta = document.getElementById("tacho-meta");

  icon.style.background = info.bg;
  icon.style.color = info.fg;
  icon.style.borderColor = info.border;
  icon.textContent = info.color;

  card.style.borderColor = info.border;
  title.textContent = info.container;
  desc.textContent = info.action;
  meta.textContent = `Clase canónica: ${topClass} · Confianza: ${(topProb * 100).toFixed(1)}% · Modo: ${modeLabel}` +
    (boxesCount > 0 ? ` · Cajas detectadas: ${boxesCount}` : "");

  const barsContainer = document.getElementById("prob-bars");
  barsContainer.innerHTML = "";
  for (const [cname, pval] of entries) {
    const cinfo = NTP_CLASSES[cname];
    const pct = Math.min(100, Math.max(0, pval * 100));
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div><span class="chip" style="background:${cinfo.bg};color:${cinfo.fg}">${cinfo.color}</span> <small>${cname}</small></div>
      <div class="prob-bar"><div class="prob-fill" style="width:${pct.toFixed(1)}%;background:${cinfo.bg === '#e9edf2' ? '#64748b' : cinfo.bg}"></div></div>
      <div style="text-align:right;font-weight:700">${pct.toFixed(1)}%</div>
    `;
    barsContainer.appendChild(row);
  }
}

async function runInferenceOnSource(imgSource) {
  const mode = modeSelect.value;
  const t0 = performance.now();

  if (mode === "cls") {
    const probMap = await runClassifier(imgSource);
    drawSceneAndBoxes(imgSource, []);
    const dt = (performance.now() - t0).toFixed(1);
    latencyEl.textContent = `Latencia ONNX WebAssembly: ${dt} ms`;
    renderResultsUI(probMap, "Clasificador ONNX (320px)", 0);
  } else if (mode === "det") {
    const { boxes, maxPerClass } = await runDetector(imgSource, 0.25);
    drawSceneAndBoxes(imgSource, boxes);
    const sumS = Object.values(maxPerClass).reduce((a, b) => a + b, 0) || 1;
    const normMap = {};
    Object.keys(maxPerClass).forEach(k => { normMap[k] = maxPerClass[k] / sumS; });
    const dt = (performance.now() - t0).toFixed(1);
    latencyEl.textContent = `Latencia ONNX WebAssembly: ${dt} ms`;
    renderResultsUI(normMap, "Detector YOLO26s ONNX (640px)", boxes.length);
  } else {
    // Modo Híbrido: Fusión Ponderada 0.74*Cls + 0.26*Det + Regla Sanitaria Rojo
    const probCls = await runClassifier(imgSource);
    const { boxes, maxPerClass } = await runDetector(imgSource, 0.20);
    drawSceneAndBoxes(imgSource, boxes);

    const fused = {};
    let total = 0;
    Object.keys(NTP_CLASSES).forEach(k => {
      fused[k] = 0.74 * (probCls[k] || 0) + 0.26 * (maxPerClass[k] || 0);
      total += fused[k];
    });
    // Regla de prioridad sanitaria para Rojo (peligroso)
    if (fused.peligroso >= 0.28 && (probCls.peligroso >= 0.18 || maxPerClass.peligroso >= 0.65)) {
      fused.peligroso = Math.max(fused.peligroso, 0.55);
    }
    const normTotal = Object.values(fused).reduce((a, b) => a + b, 0) || 1;
    Object.keys(fused).forEach(k => { fused[k] /= normTotal; });

    const dt = (performance.now() - t0).toFixed(1);
    latencyEl.textContent = `Latencia Híbrida ONNX WebAssembly: ${dt} ms`;
    renderResultsUI(fused, "Híbrido Ciro v3 (Fusión 0.74·Cls + 0.26·Det)", boxes.length);
  }
}

function loadSampleImage(url, elem) {
  stopWebcam();
  document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
  if (elem) elem.classList.add("active");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => runInferenceOnSource(img);
  img.src = url;
}

document.getElementById("file-input").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  stopWebcam();
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => runInferenceOnSource(img);
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

modeSelect.addEventListener("change", () => {
  const activeSample = document.querySelector(".sample-item.active img");
  if (activeSample && !webcamStream) {
    runInferenceOnSource(activeSample);
  }
});

async function toggleWebcam() {
  if (webcamStream) {
    stopWebcam();
    return;
  }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.createElement("video");
    video.srcObject = webcamStream;
    video.playsInline = true;
    await video.play();
    document.getElementById("webcam-btn").textContent = "⏹ Detener Cámara";

    async function loop() {
      if (!webcamStream) return;
      await runInferenceOnSource(video);
      webcamLoopId = requestAnimationFrame(loop);
    }
    loop();
  } catch (err) {
    alert("No se pudo acceder a la cámara web: " + err.message);
  }
}

function stopWebcam() {
  if (webcamLoopId) cancelAnimationFrame(webcamLoopId);
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const btn = document.getElementById("webcam-btn");
  if (btn) btn.textContent = "📷 Cámara en Vivo";
}

window.addEventListener("DOMContentLoaded", async () => {
  await initModels();
  const firstSample = document.querySelector(".sample-item");
  if (firstSample) {
    const imgEl = firstSample.querySelector("img");
    if (imgEl && imgEl.complete) {
      firstSample.classList.add("active");
      runInferenceOnSource(imgEl);
    } else if (imgEl) {
      imgEl.onload = () => {
        firstSample.classList.add("active");
        runInferenceOnSource(imgEl);
      };
    }
  }
});