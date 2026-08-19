"use strict";

const $ = (sel) => document.querySelector(sel);

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const info = await res.json();
    const device = info.cuda
      ? `CUDA · ${info.cuda_name || "GPU"}`
      : "CPU";
    const model = info.model_loaded ? "model loaded" : "loading model…";
    $("#health").textContent = `Running on ${device} — ${model}`;
  } catch {
    $("#health").textContent = "Could not reach backend.";
  }
}

function setupUpload() {
  const dropzone = $("#dropzone");
  const input = $("#file-input");
  const conf = $("#conf");
  const confVal = $("#conf-val");
  const result = $("#result");
  const resultImage = $("#result-image");
  const resultCount = $("#result-count");
  const resultList = $("#result-list");
  const status = $("#upload-status");

  conf.addEventListener("input", () => {
    confVal.textContent = Number(conf.value).toFixed(2);
  });

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) upload(input.files[0]);
  });

  async function upload(file) {
    const form = new FormData();
    form.append("file", file);
    const qs = `conf=${conf.value}`;
    status.className = "status";
    status.textContent = "Detecting…";
    try {
      const res = await fetch(`/api/detect/image?${qs}`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      resultImage.src = data.image;
      resultCount.textContent = data.count;
      resultList.innerHTML = "";
      for (const d of data.detections) {
        const li = document.createElement("li");
        const [x1, y1, x2, y2] = d.box;
        const label = document.createElement("span");
        label.textContent = `${d.label}  (${x1},${y1})–(${x2},${y2})`;
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = `${(d.score * 100).toFixed(1)}%`;
        li.appendChild(label);
        li.appendChild(score);
        resultList.appendChild(li);
      }
      result.classList.remove("hidden");
      status.className = "status ok";
      status.textContent = `Done — detected ${data.count} object(s).`;
    } catch (err) {
      status.className = "status err";
      status.textContent = `Error: ${err.message}`;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadHealth();
  setupUpload();
});
