"use strict";

const retryDelayMilliseconds = globalThis.retryDelayMilliseconds;
const $ = (id) => document.getElementById(id);
const input = $("image-input");
const dropZone = $("drop-zone");
let currentFile = null;
let previewUrl = null;
let controller = null;
let revision = 0;
let modelLoaded = false;
let modelPending = false;
let modelBusy = false;
let modelBusyMessage = "正在处理模型";
let modelName = null;
let pendingModelName = null;

function modelReady() {
  return modelLoaded && !modelPending;
}

function updateModelStatus(state) {
  modelLoaded = state.loaded === true;
  modelPending = state.pending === true;
  modelName = modelLoaded ? (state.model_name || modelName) : null;
  pendingModelName = modelPending ? (state.pending_name || pendingModelName) : null;
  const status = $("model-status");

  if (modelBusy) {
    status.textContent = `${modelBusyMessage} ${pendingModelName || modelName || "模型"}…`;
    status.dataset.state = "loading";
    $("model-help").textContent = "模型文件正在处理，请稍候。";
  } else if (modelPending) {
    status.textContent = `${pendingModelName || "自定义模型"} · 待加载`;
    status.dataset.state = "pending";
    $("model-help").textContent = "自定义模型已上传。点击“加载模型”后开始使用，加载期间请勿上传图片识别。";
  } else if (modelLoaded) {
    status.textContent = `${modelName || "模型"} · 已加载`;
    status.dataset.state = "loaded";
    $("model-help").textContent = `当前已加载 ${modelName || "模型"}。选择图片后将自动识别。`;
  } else {
    status.textContent = "模型未加载";
    status.dataset.state = "idle";
    $("model-help").textContent = "未上传自定义模型时，点击“加载模型”使用默认模型。模型加载完成后才能识别图片。";
  }
  $("load-model-button").textContent = modelBusy ? "正在处理模型…" : "加载模型";
  $("load-model-button").disabled = modelBusy;
  $("model-pt-button").disabled = modelBusy;
  $("model-onnx-button").disabled = modelBusy;
}

function clearResult() {
  $("result-image").hidden = true;
  $("result-image").removeAttribute("src");
  $("result-placeholder").hidden = false;
  $("details-panel").hidden = true;
  $("detection-rows").replaceChildren();
  $("result-summary").textContent = "等待上传图片";
  $("result-time").textContent = "";
  $("result-status").textContent = "等待上传";
  $("result-status").dataset.state = "idle";
  for (let id = 0; id < 4; id += 1) $("count-" + id).textContent = "—";
}

function stopRequest() {
  revision += 1;
  if (controller) controller.abort();
  controller = null;
  $("loading").hidden = true;
  $("result-stage").setAttribute("aria-busy", "false");
}

function clearAll() {
  stopRequest();
  clearResult();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  currentFile = null;
  input.value = "";
  $("original-image").removeAttribute("src");
  $("original-image").hidden = true;
  $("upload-placeholder").hidden = false;
  $("replace-hint").hidden = true;
  $("upload-status").textContent = "待上传";
  $("file-name").textContent = "尚未选择图片";
  $("file-size").textContent = "";
  $("clear-button").disabled = true;
  $("upload-button-text").textContent = "上传图片并识别";
  $("error-message").hidden = true;
}

function showError(message) {
  $("error-message").textContent = message;
  $("error-message").hidden = false;
  $("result-status").textContent = "识别失败";
  $("result-status").dataset.state = "error";
  $("result-summary").textContent = "请检查图片后重新上传";
}

function showModelError(message) {
  $("error-message").textContent = message;
  $("error-message").hidden = false;
}

function renderResult(data) {
  $("result-image").src = data.image;
  $("result-image").hidden = false;
  $("result-placeholder").hidden = true;
  $("result-status").textContent = "识别完成";
  $("result-status").dataset.state = "done";
  $("result-summary").textContent = `检测到 ${data.total} 个目标 · ${data.width} × ${data.height} px`;
  $("result-time").textContent = `推理耗时 ${data.inference_ms} ms`;
  for (const category of data.counts) $("count-" + category.id).textContent = category.count;
  const rows = document.createDocumentFragment();
  for (const item of data.detections) {
    const row = document.createElement("tr");
    const number = document.createElement("td");
    number.textContent = String(item.id).padStart(2, "0");
    const categoryCell = document.createElement("td");
    const category = document.createElement("span");
    category.className = "category-label";
    category.style.setProperty("--category-color", item.color);
    category.textContent = item.class_name;
    categoryCell.append(category);
    const score = document.createElement("td");
    score.textContent = (item.confidence * 100).toFixed(1) + "%";
    const track = document.createElement("span");
    track.className = "confidence-track";
    const bar = document.createElement("i");
    bar.style.width = item.confidence * 100 + "%";
    track.append(bar);
    score.append(track);
    const position = document.createElement("td");
    position.textContent = item.box.map(Math.round).join(", ");
    row.append(number, categoryCell, score, position);
    rows.append(row);
  }
  $("detection-rows").replaceChildren(rows);
  $("details-panel").hidden = false;
  $("no-detections").hidden = data.total !== 0;
  $("upload-status").textContent = "已识别";
}

async function detectCurrent() {
  if (!currentFile) return;
  stopRequest();
  clearResult();
  $("error-message").hidden = true;
  if (!modelReady()) {
    $("result-status").textContent = "等待加载模型";
    $("result-summary").textContent = "请先点击“加载模型”，加载完成后即可识别当前图片";
    $("upload-status").textContent = "等待加载模型";
    return;
  }

  const requestRevision = revision;
  const requestController = new AbortController();
  controller = requestController;
  $("loading").hidden = false;
  $("result-stage").setAttribute("aria-busy", "true");
  $("result-status").textContent = "识别中";
  $("result-status").dataset.state = "loading";
  $("result-summary").textContent = "正在调用本地模型";
  $("upload-status").textContent = "识别中";
  const form = new FormData();
  form.append("image", currentFile);
  form.append("confidence", Number($("confidence").value) / 100);
  try {
    let response = await fetch("/api/detect", {
      method: "POST", body: form, signal: requestController.signal, cache: "no-store",
    });
    while (response.status === 429 && requestRevision === revision) {
      await new Promise((resolve) => setTimeout(resolve, retryDelayMilliseconds(response)));
      if (requestRevision !== revision) return;
      response = await fetch("/api/detect", {
        method: "POST", body: form, signal: requestController.signal, cache: "no-store",
      });
    }
    const data = await response.json();
    if (requestRevision !== revision) return;
    if (!response.ok) throw new Error(data.error || "识别失败，请重新上传。");
    renderResult(data);
  } catch (error) {
    if (error.name === "AbortError" || requestRevision !== revision) return;
    showError(error instanceof TypeError ? "无法连接服务，请确认 Flask 正在运行。" : error.message);
    $("upload-status").textContent = "待重试";
  } finally {
    if (requestRevision === revision) {
      controller = null;
      $("loading").hidden = true;
      $("result-stage").setAttribute("aria-busy", "false");
    }
  }
}

async function loadModel() {
  if (modelBusy) return;
  stopRequest();
  clearResult();
  $("error-message").hidden = true;
  modelBusy = true;
  modelBusyMessage = "正在加载";
  updateModelStatus({
    loaded: modelLoaded,
    pending: modelPending,
    model_name: modelName,
    pending_name: pendingModelName,
  });
  let loaded = false;
  try {
    const response = await fetch("/api/model/load", { method: "POST", cache: "no-store" });
    const state = await response.json();
    updateModelStatus(state);
    if (!response.ok) throw new Error(state.error || "模型加载失败，请重试。");
    loaded = true;
  } catch (error) {
    showModelError(error instanceof TypeError ? "无法连接服务，请确认 Flask 正在运行。" : error.message);
  } finally {
    modelBusy = false;
    updateModelStatus({
      loaded: modelLoaded,
      pending: modelPending,
      model_name: modelName,
      pending_name: pendingModelName,
    });
  }
  if (loaded && modelReady() && currentFile) await detectCurrent();
}

async function uploadModel(file, fileInput) {
  if (!file || modelBusy) return;
  const expectedExtension = fileInput === $("pt-model-input") ? ".pt" : ".onnx";
  if (!file.name.toLowerCase().endsWith(expectedExtension)) {
    showModelError(`请选择 ${expectedExtension} 格式的模型文件。`);
    fileInput.value = "";
    return;
  }
  const maximumBytes = Number($("model-panel").dataset.maxModelBytes);
  if (file.size > maximumBytes) {
    showModelError(`模型文件超过 ${Math.round(maximumBytes / 1024 / 1024)} MB，请选择较小的文件。`);
    fileInput.value = "";
    return;
  }

  stopRequest();
  clearResult();
  $("error-message").hidden = true;
  modelBusy = true;
  modelBusyMessage = "正在上传";
  updateModelStatus({
    loaded: modelLoaded,
    pending: modelPending,
    model_name: modelName,
    pending_name: pendingModelName,
  });
  const form = new FormData();
  form.append("model", file);
  form.append("auto_load", $("auto-load-model").checked ? "1" : "0");
  let uploadedAndLoaded = false;
  try {
    const response = await fetch("/api/model/upload", {
      method: "POST", body: form, cache: "no-store",
    });
    const state = await response.json();
    updateModelStatus(state);
    if (!response.ok) throw new Error(state.error || "模型上传失败，请重试。");
    uploadedAndLoaded = modelReady();
  } catch (error) {
    showModelError(error instanceof TypeError ? "无法连接服务，请确认 Flask 正在运行。" : error.message);
  } finally {
    modelBusy = false;
    updateModelStatus({
      loaded: modelLoaded,
      pending: modelPending,
      model_name: modelName,
      pending_name: pendingModelName,
    });
    fileInput.value = "";
  }
  if (uploadedAndLoaded && currentFile) await detectCurrent();
}

function selectFile(file) {
  if (!file) return;
  clearAll();
  const supportedType = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/x-ms-bmp"].includes(file.type);
  if (!supportedType && !/\.(jpe?g|jfif|jpe|png|webp|bmp)$/i.test(file.name)) {
    showError("请选择 JPG、PNG、WEBP 或 BMP 格式的图片。");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError("图片超过 12 MB，请压缩后重新上传。");
    return;
  }
  currentFile = file;
  previewUrl = URL.createObjectURL(file);
  $("original-image").src = previewUrl;
  $("original-image").hidden = false;
  $("upload-placeholder").hidden = true;
  $("replace-hint").hidden = false;
  $("file-name").textContent = file.name;
  $("file-size").textContent = file.size >= 1024 * 1024
    ? (file.size / 1024 / 1024).toFixed(1) + " MB"
    : Math.max(1, Math.round(file.size / 1024)) + " KB";
  $("clear-button").disabled = false;
  $("upload-button-text").textContent = "更换图片并识别";
  detectCurrent();
}

async function refreshModelStatus() {
  try {
    const response = await fetch("/api/model", { cache: "no-store" });
    if (!response.ok) throw new Error("无法读取模型状态。");
    const state = await response.json();
    updateModelStatus(state);
    if (modelReady() && currentFile) await detectCurrent();
  } catch (error) {
    showModelError(error instanceof TypeError ? "无法连接服务，请确认 Flask 正在运行。" : error.message);
  }
}

input.addEventListener("change", () => selectFile(input.files[0]));
$("upload-button").addEventListener("click", () => input.click());
$("clear-button").addEventListener("click", clearAll);
$("model-pt-button").addEventListener("click", () => $("pt-model-input").click());
$("model-onnx-button").addEventListener("click", () => $("onnx-model-input").click());
$("pt-model-input").addEventListener("change", (event) => uploadModel(event.target.files[0], event.target));
$("onnx-model-input").addEventListener("change", (event) => uploadModel(event.target.files[0], event.target));
$("load-model-button").addEventListener("click", loadModel);
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
}
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
$("confidence").addEventListener("input", () => {
  $("confidence-value").value = $("confidence").value + "%";
});
$("confidence").addEventListener("change", detectCurrent);
window.addEventListener("pagehide", clearAll);
refreshModelStatus();
