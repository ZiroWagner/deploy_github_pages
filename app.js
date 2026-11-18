// app.js — Estación Interactiva Ciro v3 ONNX Runtime Web (NTP 900.058:2019)

const NTP_CLASSES = {
  plastico: {
    color: "BLANCO",
    short: "Plásticos",
    container: "Contenedor BLANCO — Plásticos Aprovechables",
    bg: "#e2e8f0",
    fg: "#0f172a",
    bar: "#cbd5e1",
    glow: "rgba(226, 232, 240, 0.45)",
    action: "Depositar botellas PET limpias, envases rígidos HDPE/PP y tapas plásticas."
  },
  papel_carton: {
    color: "AZUL",
    short: "Papel/Cartón",
    container: "Contenedor AZUL — Papel y Cartón",
    bg: "#2563eb",
    fg: "#ffffff",
    bar: "#3b82f6",
    glow: "rgba(37, 99, 235, 0.5)",
    action: "Depositar cajas de cartón corrugado, papel de oficina, periódicos y revistas secas."
  },
  metal: {
    color: "AMARILLO",
    short: "Metales",
    container: "Contenedor AMARILLO — Metales",
    bg: "#facc15",
    fg: "#0f172a",
    bar: "#eab308",
    glow: "rgba(250, 204, 21, 0.45)",
    action: "Depositar latas de aluminio, conservas de hojalata y envases metálicos."
  },
  vidrio: {
    color: "GRIS",
    short: "Vidrio",
    container: "Contenedor GRIS — Vidrio",
    bg: "#64748b",
    fg: "#ffffff",
    bar: "#94a3b8",
    glow: "rgba(148, 163, 184, 0.45)",
    action: "Depositar botellas y frascos de vidrio transparente, ámbar o verde sin líquidos."
  },
  organico: {
    color: "MARRÓN",
    short: "Orgánicos",
    container: "Contenedor MARRÓN — Residuos Orgánicos",
    bg: "#92400e",
    fg: "#ffffff",
    bar: "#b45309",
    glow: "rgba(180, 83, 9, 0.5)",
    action: "Depositar restos de alimentos, cáscaras de frutas/verduras y residuos compostables."
  },
  otros_residuos: {
    color: "NEGRO",
    short: "Generales",
    container: "Contenedor NEGRO — No Aprovechables",
    bg: "#1e293b",
    fg: "#ffffff",
    bar: "#475569",
    glow: "rgba(100, 116, 139, 0.45)",
    action: "Depositar tecnopor (poliestireno), envolturas multicapa, servilletas usadas y sanitarios."
  },
  peligroso: {
    color: "ROJO",
    short: "Peligrosos",
    container: "Contenedor ROJO — Peligrosos / Biocontaminados",
    bg: "#dc2626",
    fg: "#ffffff",
    bar: "#ef4444",
    glow: "rgba(239, 68, 68, 0.6)",
    action: "¡PRIORIDAD SANITARIA! EPP clínico (mascarillas, guantes, gasas), pilas, aerosoles y fármacos."
  }
};

const BIN_ORDER = [
  "plastico",
  "papel_carton",
  "metal",
  "vidrio",
  "organico",
  "otros_residuos",
  "peligroso"
];

const CLS_INDEX_TO_NAME = [
  "metal",
  "organico",
  "otros_residuos",
  "papel_carton",
  "peligroso",
  "plastico",
  "vidrio"
];

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
let currentMode = "cls";
let confThreshold = 0.25;
let currentImgSource = null;
let webcamStream = null;
let webcamLoopId = null;
const sessionCounts = {};
BIN_ORDER.forEach(k => { sessionCounts[k] = 0; });

const statusTextEl = document.getElementById("status-text");
const latencyEl = document.getElementById("latency-badge");
const canvas = document.getElementById("view-canvas");
const ctx = canvas.getContext("2d");
const viewport = document.getElementById("viewport");
const confSlider = document.getElementById("conf-slider");
const confValEl = document.getElementById("conf-val");

function initBinsDock() {
  const dock = document.getElementById("bins-dock");
  dock.innerHTML = "";
  for (const cname of BIN_ORDER) {
    const info = NTP_CLASSES[cname];
    const el = document.createElement("div");
    el.className = "ntp-bin";
    el.id = `bin-${cname}`;
    el.innerHTML = `
      <span class="bin-count" id="cnt-${cname}">0</span>
      <div class="bin-lid" style="background:${info.bg}"></div>
      <div class="bin-body" style="background:${info.bg};color:${info.fg}">${info.color}</div>
      <div class="bin-name">${info.color}</div>
      <div class="bin-sub">${info.short}</div>
    `;
    dock.appendChild(el);
  }
}

const GITHUB_PAGES_BASE = "https://zirowagner.github.io/deploy_github_pages/";
const GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ZiroWagner/deploy_github_pages/main/";

function configureOrtEnv() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  // numThreads = 1 evita requerir SharedArrayBuffer (headers COOP/COEP no presentes en GitHub Pages ni file://)
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

async function fetchModelBuffer(relPath, label) {
  const isFileProtocol = window.location.protocol === "file:";
  // Si se abre con file://, el navegador bloquea fetch("file:///...") por CORS.
  // En ese caso descargamos automáticamente el binario .onnx desde nuestro repositorio HTTPS (que tiene CORS * habilitado).
  const candidateUrls = isFileProtocol
    ? [GITHUB_PAGES_BASE + relPath, GITHUB_RAW_BASE + relPath]
    : [relPath, GITHUB_PAGES_BASE + relPath, GITHUB_RAW_BASE + relPath];

  let lastErr = null;
  for (const url of candidateUrls) {
    try {
      statusTextEl.textContent = `Cargando ${label}...`;
      const resp = await fetch(url, { mode: "cors", cache: "force-cache" });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} en ${url}`);
      }
      const arrayBuf = await resp.arrayBuffer();
      if (arrayBuf.byteLength < 100000) {
        throw new Error(`Archivo ONNX incompleto (${arrayBuf.byteLength} bytes)`);
      }
      return new Uint8Array(arrayBuf);
    } catch (err) {
      console.warn(`Intento fallido cargando ${url}:`, err);
      lastErr = err;
    }
  }
  throw lastErr || new Error(`No se pudo cargar ${relPath}`);
}

async function initModels() {
  try {
    configureOrtEnv();
    const clsBytes = await fetchModelBuffer("models/ciro_cls_ntp_v3.onnx", "Clasificador ONNX (21.3 MB)");
    clsSession = await ort.InferenceSession.create(clsBytes, {
      executionProviders: ["wasm"]
    });
    statusTextEl.textContent = "ONNX WebAssembly Activo";
  } catch (err) {
    console.error(err);
    statusTextEl.textContent = "Modo Offline: usa '📂 .ONNX Local' para seleccionar models/ciro_cls_ntp_v3.onnx";
  }
}

async function ensureDetModel() {
  if (detSession) return detSession;
  configureOrtEnv();
  const detBytes = await fetchModelBuffer("models/yolo26s_ntp_7c_v3.onnx", "Detector YOLO26s ONNX (38.1 MB)");
  detSession = await ort.InferenceSession.create(detBytes, {
    executionProviders: ["wasm"]
  });
  statusTextEl.textContent = "Clasificador + Detector ONNX Activos";
  return detSession;
}

async function loadLocalOnnxFiles(fileList) {
  configureOrtEnv();
  for (const file of fileList) {
    const fname = file.name.toLowerCase();
    statusTextEl.textContent = `Cargando ${file.name} desde disco...`;
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    if (fname.includes("det") || fname.includes("yolo26")) {
      detSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      statusTextEl.textContent = `Detector local cargado (${file.name})`;
    } else {
      clsSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      statusTextEl.textContent = `Clasificador local cargado (${file.name})`;
    }
  }
  if (currentImgSource && !webcamStream) {
    runInference(currentImgSource, false);
  }
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

async function runDetector(imgSource, confThr) {
  await ensureDetModel();
  const inputTensor = preprocessToTensor(imgSource, 640);
  const feeds = { [detSession.inputNames[0]]: inputTensor };
  const results = await detSession.run(feeds);
  const outTensor = results[detSession.outputNames[0]];
  const data = outTensor.data;
  const numAnchors = outTensor.dims[2];

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
    if (bestScore >= confThr && bestClass >= 0) {
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

function drawViewport(imgSource, boxes) {
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

    ctx.strokeStyle = info.bg === "#e2e8f0" ? "#38bdf8" : info.bg;
    ctx.lineWidth = Math.max(3, Math.round(w / 160));
    ctx.strokeRect(x1, y1, bw, bh);

    const label = `${info.color} · ${b.className} ${(b.score * 100).toFixed(0)}%`;
    ctx.font = `800 ${Math.max(12, Math.round(w / 40))}px Inter, sans-serif`;
    const tw = ctx.measureText(label).width + 14;
    const th = Math.max(24, Math.round(w / 25));
    const ly = Math.max(0, y1 - th);
    ctx.fillStyle = info.bg;
    ctx.fillRect(x1, ly, tw, th);
    ctx.fillStyle = info.fg;
    ctx.fillText(label, x1 + 7, ly + th - 6);
  }
}

function updateStationUI(probMap, modeTitle, boxesCount, incrementSession = true) {
  const entries = Object.entries(probMap).sort((a, b) => b[1] - a[1]);
  const [topClass, topProb] = entries[0];
  const info = NTP_CLASSES[topClass];

  // Iluminar contenedor ganador en la estación de 7 tachos
  BIN_ORDER.forEach(cname => {
    const binEl = document.getElementById(`bin-${cname}`);
    if (!binEl) return;
    if (cname === topClass) {
      binEl.classList.add("winner");
      binEl.style.borderColor = info.bg === "#1e293b" ? "#94a3b8" : info.bg;
      binEl.style.boxShadow = `0 8px 24px ${info.glow}`;
    } else {
      binEl.classList.remove("winner");
      binEl.style.borderColor = "transparent";
      binEl.style.boxShadow = "none";
    }
  });

  if (incrementSession && !webcamStream) {
    sessionCounts[topClass] = (sessionCounts[topClass] || 0) + 1;
    const cntEl = document.getElementById(`cnt-${topClass}`);
    if (cntEl) cntEl.textContent = sessionCounts[topClass];
  }

  // Actualizar tarjeta de decisión
  const dCard = document.getElementById("decision-card");
  const dSwatch = document.getElementById("decision-swatch");
  const dTitle = document.getElementById("decision-title");
  const dDesc = document.getElementById("decision-desc");
  const dMeta = document.getElementById("decision-meta");

  dCard.style.borderColor = info.bg === "#1e293b" ? "#64748b" : info.bg;
  dSwatch.style.background = info.bg;
  dSwatch.style.color = info.fg;
  dSwatch.innerHTML = `<span>TACHO</span><b>${info.color}</b>`;
  dTitle.textContent = info.container;
  dDesc.textContent = info.action;
  dMeta.innerHTML = `
    <span class="mtag">Confianza: ${(topProb * 100).toFixed(1)}%</span>
    <span class="mtag">Clase: ${topClass}</span>
    <span class="mtag">${modeTitle}</span>
    ${boxesCount > 0 ? `<span class="mtag">Cajas: ${boxesCount}</span>` : ""}
  `;

  // Actualizar barras de probabilidad
  const listEl = document.getElementById("prob-list");
  listEl.innerHTML = "";
  for (const [cname, pval] of entries) {
    const cinfo = NTP_CLASSES[cname];
    const pct = Math.min(100, Math.max(0, pval * 100));
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `
      <div class="prob-label">
        <i class="swatch-dot" style="background:${cinfo.bg}"></i>
        <span>${cinfo.color} <small style="color:var(--mut);font-weight:500">(${cinfo.short})</small></span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:${pct.toFixed(1)}%;background:${cinfo.bar}"></div>
      </div>
      <div class="prob-pct">${pct.toFixed(1)}%</div>
    `;
    listEl.appendChild(item);
  }
}

async function runInference(imgSource, incrementSession = true) {
  if (!imgSource) return;
  currentImgSource = imgSource;
  const t0 = performance.now();

  if (currentMode === "cls") {
    const probMap = await runClassifier(imgSource);
    drawViewport(imgSource, []);
    latencyEl.textContent = `⚡ ${(performance.now() - t0).toFixed(1)} ms`;
    updateStationUI(probMap, "Clasificador ONNX 320px", 0, incrementSession);
  } else if (currentMode === "det") {
    const { boxes, maxPerClass } = await runDetector(imgSource, confThreshold);
    drawViewport(imgSource, boxes);
    const sumS = Object.values(maxPerClass).reduce((a, b) => a + b, 0) || 1;
    const normMap = {};
    Object.keys(maxPerClass).forEach(k => { normMap[k] = maxPerClass[k] / sumS; });
    latencyEl.textContent = `⚡ ${(performance.now() - t0).toFixed(1)} ms`;
    updateStationUI(normMap, `Detector YOLO26s (conf≥${Math.round(confThreshold * 100)}%)`, boxes.length, incrementSession);
  } else {
    const probCls = await runClassifier(imgSource);
    const { boxes, maxPerClass } = await runDetector(imgSource, confThreshold);
    drawViewport(imgSource, boxes);

    const fused = {};
    Object.keys(NTP_CLASSES).forEach(k => {
      fused[k] = 0.74 * (probCls[k] || 0) + 0.26 * (maxPerClass[k] || 0);
    });
    if (fused.peligroso >= 0.28 && (probCls.peligroso >= 0.18 || maxPerClass.peligroso >= 0.65)) {
      fused.peligroso = Math.max(fused.peligroso, 0.55);
    }
    const normTotal = Object.values(fused).reduce((a, b) => a + b, 0) || 1;
    Object.keys(fused).forEach(k => { fused[k] /= normTotal; });

    latencyEl.textContent = `⚡ ${(performance.now() - t0).toFixed(1)} ms`;
    updateStationUI(fused, "Híbrido Ciro v3 (0.74·Cls + 0.26·Det)", boxes.length, incrementSession);
  }
}

function setMode(mode, btnEl) {
  currentMode = mode;
  document.querySelectorAll(".mode-pill").forEach(b => b.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");
  if (currentImgSource && !webcamStream) {
    runInference(currentImgSource, false);
  }
}

confSlider.addEventListener("input", e => {
  confThreshold = parseInt(e.target.value, 10) / 100;
  confValEl.textContent = `${e.target.value}%`;
  if (currentImgSource && !webcamStream && currentMode !== "cls") {
    runInference(currentImgSource, false);
  }
});

function loadSample(cardEl) {
  stopWebcam();
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  if (!cardEl) return;
  cardEl.classList.add("active");
  const thumbImg = cardEl.querySelector("img");
  if (!thumbImg) return;
  // Las miniaturas están embebidas como data:image/jpeg;base64,... para evitar cualquier bloqueo CORS en file://
  const img = new Image();
  img.onload = () => runInference(img, true);
  img.src = thumbImg.src;
}

function pickRandomSample() {
  const visibleCards = Array.from(document.querySelectorAll(".sample-card")).filter(
    c => c.style.display !== "none"
  );
  if (!visibleCards.length) return;
  const pick = visibleCards[Math.floor(Math.random() * visibleCards.length)];
  pick.click();
}

function filterSamples(colorFilter, chipEl) {
  document.querySelectorAll(".fchip").forEach(c => c.classList.remove("active"));
  if (chipEl) chipEl.classList.add("active");
  document.querySelectorAll(".sample-card").forEach(card => {
    const c = card.getAttribute("data-color");
    card.style.display = (colorFilter === "ALL" || c === colorFilter) ? "block" : "none";
  });
}

function resetSessionCounts() {
  BIN_ORDER.forEach(k => {
    sessionCounts[k] = 0;
    const el = document.getElementById(`cnt-${k}`);
    if (el) el.textContent = "0";
  });
}

// Drag & Drop en el Viewport
viewport.addEventListener("dragover", e => {
  e.preventDefault();
  viewport.classList.add("dragover");
});
viewport.addEventListener("dragleave", () => viewport.classList.remove("dragover"));
viewport.addEventListener("drop", e => {
  e.preventDefault();
  viewport.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleImageFile(file);
});

document.getElementById("file-input").addEventListener("change", e => {
  if (e.target.files[0]) handleImageFile(e.target.files[0]);
});

const onnxInputEl = document.getElementById("onnx-input");
if (onnxInputEl) {
  onnxInputEl.addEventListener("change", e => {
    if (e.target.files && e.target.files.length) {
      loadLocalOnnxFiles(e.target.files);
    }
  });
}

function handleImageFile(file) {
  stopWebcam();
  document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => runInference(img, true);
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

async function toggleWebcam() {
  if (webcamStream) {
    stopWebcam();
    return;
  }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    const video = document.createElement("video");
    video.srcObject = webcamStream;
    video.playsInline = true;
    await video.play();
    const btn = document.getElementById("webcam-btn");
    btn.textContent = "⏹ Detener Cámara";
    btn.classList.add("live-active");

    async function loop() {
      if (!webcamStream) return;
      await runInference(video, false);
      webcamLoopId = requestAnimationFrame(loop);
    }
    loop();
  } catch (err) {
    alert("No se pudo iniciar la cámara web: " + err.message);
  }
}

function stopWebcam() {
  if (webcamLoopId) cancelAnimationFrame(webcamLoopId);
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const btn = document.getElementById("webcam-btn");
  if (btn) {
    btn.textContent = "📷 Cámara en Vivo";
    btn.classList.remove("live-active");
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  initBinsDock();
  await initModels();
  const firstCard = document.querySelector(".sample-card");
  if (firstCard) firstCard.click();
});