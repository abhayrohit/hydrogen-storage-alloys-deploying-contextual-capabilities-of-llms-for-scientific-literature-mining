const API_BASE = (typeof window !== 'undefined' && window.API_BASE) ? window.API_BASE : "";
console.log("DEBUG: API_BASE=", API_BASE || "(same-origin)");

// Get all DOM elements
const form = document.getElementById("uploadForm");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("resultSection");
const resultTable = document.getElementById("resultTable");
const rawCsv = document.getElementById("rawCsv");
const downloadDiv = document.getElementById("download");
const submitBtn = document.getElementById("submitBtn");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const uploadArea = document.getElementById("uploadArea");
const progressContainer = document.getElementById("progressContainer");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const loadingSubText = document.getElementById("loadingSubText");

// Health check function
async function checkHealth() {
  try {
    console.log("Running health check...");
    const url = (API_BASE || '') + "/health";
    console.log("Health check URL:", url);
    const r = await fetch(url);
    console.log("Health check response status:", r.status);
    if (r.ok) {
      const j = await r.json();
      console.log("Health check data:", j);
      if (statusEl) {
        statusEl.innerHTML = `<span style="color: #48bb78;">✅ System Ready</span> • Model: <code>${j.model}</code>`;
      }
    } else {
      console.error("Health check non-OK response:", r.status, r.statusText);
      const text = await r.text();
      console.error("Response body:", text);
      if (statusEl) {
        statusEl.innerHTML = '<span style="color: #f56565;">❌ Backend returned error</span>';
      }
    }
  } catch (e) {
    console.error("Health check exception:", e);
    if (statusEl) {
      statusEl.innerHTML = '<span style="color: #f56565;">❌ Cannot reach backend</span>';
    }
  }
}

// File input change handler
fileInput.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    displayFileInfo(file);
    submitBtn.disabled = false;
  } else {
    hideFileInfo();
    submitBtn.disabled = true;
  }
});

// Drag and drop functionality
uploadArea.addEventListener("dragover", function (e) {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", function (e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", function (e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type === "application/pdf") {
    fileInput.files = files;
    displayFileInfo(files[0]);
    submitBtn.disabled = false;
  }
});

function displayFileInfo(file) {
  const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
  fileInfo.innerHTML = `
    <h4>📄 ${file.name}</h4>
    <p>Size: ${sizeInMB} MB • Type: PDF Document</p>
  `;
  fileInfo.classList.remove("hidden");
}

function hideFileInfo() {
  fileInfo.classList.add("hidden");
}

function showProgress(text = "Processing...") {
  progressText.textContent = text;
  progressContainer.classList.remove("hidden");
}

function updateProgress(percentage, text) {
  progressFill.style.width = percentage + "%";
  if (text) progressText.textContent = text;
}

function hideProgress() {
  progressContainer.classList.add("hidden");
  progressFill.style.width = "0%";
}

function showLoadingOverlay(text = "Uploading and analyzing your PDF…", sub = "This may take a minute for long papers.") {
  try {
    if (loadingText) loadingText.textContent = text;
    if (loadingSubText) loadingSubText.textContent = sub;
    loadingOverlay?.classList.remove("hidden");
    document.body.classList.add("no-scroll");
  } catch {}
}

function updateLoadingOverlay(text, sub) {
  try {
    if (text && loadingText) loadingText.textContent = text;
    if (sub && loadingSubText) loadingSubText.textContent = sub;
  } catch {}
}

function hideLoadingOverlay() {
  try {
    loadingOverlay?.classList.add("hidden");
    document.body.classList.remove("no-scroll");
  } catch {}
}

form.addEventListener("submit", async (e) => {
  // Prevent any default form navigation/refresh
  e.preventDefault();
  e.stopPropagation();
  try { if (document.activeElement) document.activeElement.blur(); } catch {}
  
  if (!fileInput.files.length) {
    return;
  }

  submitBtn.disabled = true;
  resultSection.classList.add("hidden");

  // Show progress
  showProgress("Uploading file...");
  updateProgress(10, "Uploading file...");

  statusEl.innerHTML =
    '<span style="color: #667eea;">📤 Uploading and processing your PDF...</span>';

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  // Use XHR to show real upload progress and a loading overlay during processing
  const data = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", (API_BASE || '') + "/extract");
    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable) {
        const pct = Math.min(100, Math.max(0, Math.floor((evt.loaded / evt.total) * 100)));
        const scaled = Math.round(10 + (pct * 0.5)); // 10..60
        updateProgress(scaled, pct < 100 ? `Uploading file... (${pct}%)` : "Upload complete");
        if (pct === 100) {
          showLoadingOverlay("Analyzing with AI…", "Extracting alloys and properties...");
          updateProgress(65, "Analyzing with AI model...");
        }
      }
    };
    xhr.onerror = () => {
      reject(new Error("Network error during upload"));
    };
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 2) {
        showLoadingOverlay("Analyzing with AI…", "This may take a minute for large documents.");
      }
    };
    xhr.onload = () => {
      updateProgress(95, "Finalizing results...");
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error("Failed to parse server response as JSON"));
        }
      } else {
        // Log full response for debugging
        console.error("Server error:", xhr.status, xhr.statusText, xhr.responseText);
        let errMsg = `Server error: ${xhr.status}`;
        try {
          const errJson = JSON.parse(xhr.responseText);
          if (errJson.detail) errMsg += ` - ${errJson.detail}`;
        } catch {}
        reject(new Error(errMsg));
      }
    };
    try {
      xhr.send(fd);
    } catch (e) {
      reject(e);
    }
  }).catch((err) => {
    statusEl.innerHTML = '<span style="color: #f56565;">❌ ' + (err.message || err) + '</span>';
    hideProgress();
    hideLoadingOverlay();
    submitBtn.disabled = false;
    return null;
  });
  
  if (!data) return;

  console.log("DEBUG: /extract data:", data);
  // Persist latest successful result so UI can restore after accidental refresh
  try { localStorage.setItem("lastResult", JSON.stringify(data)); } catch {}

  setTimeout(() => {
    updateProgress(100, "Complete!");
    statusEl.innerHTML =
      '<span style="color: #48bb78;">✅ Extraction completed successfully!</span>';

    renderTable(data.csv_text);
    rawCsv.textContent = data.csv_text;
    // Prefer absolute URL if provided; fallback to relative path for same-origin deployments
    const dlUrl = data.download_url || ((API_BASE || '').replace(/\/$/, '') + (data.download_path || ''));
    downloadDiv.innerHTML = `<a href="${dlUrl}" download class="dl">📥 Download CSV File</a>`;

    resultSection.classList.remove("hidden");
    hideProgress();
    hideLoadingOverlay();
    submitBtn.disabled = false;

    // Reset form
    setTimeout(() => {
      fileInput.value = "";
      hideFileInfo();
      submitBtn.disabled = true;
    }, 1000);
  }, 500);
});

// On load, attempt to restore last results in case the page refreshed
window.addEventListener("DOMContentLoaded", () => {
  try {
    const saved = localStorage.getItem("lastResult");
    if (saved) {
      const data = JSON.parse(saved);
      if (data && data.csv_text) {
        renderTable(data.csv_text);
        rawCsv.textContent = data.csv_text || "";
        const dlUrl = data.download_url || ((API_BASE || '').replace(/\/$/, '') + (data.download_path || ''));
        downloadDiv.innerHTML = `<a href="${dlUrl}" download class="dl">📥 Download CSV File</a>`;
        resultSection.classList.remove("hidden");
      }
    }
  } catch (e) {
    console.warn("DEBUG: failed restoring lastResult:", e);
  }
});

function renderTable(csv) {
  if (!csv) {
    resultTable.innerHTML = "";
    return;
  }
  const lines = csv.split(/\n/).filter((l) => l.trim());
  if (!lines.length) {
    resultTable.innerHTML = "";
    return;
  }
  const header = lines[0].split("|").map((h) => h.trim());
  let thead =
    "<thead><tr>" +
    header.map((h) => `<th>${escapeHtml(h)}</th>`).join("") +
    "</tr></thead>";
  let bodyRows = "";
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split("|");
    if (cols.length !== header.length) continue;
    bodyRows +=
      "<tr>" +
      cols.map((c) => `<td>${escapeHtml(c.trim())}</td>`).join("") +
      "</tr>";
  }
  resultTable.innerHTML = thead + "<tbody>" + bodyRows + "</tbody>";
}

function escapeHtml(str) {
  return str.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        c
      ] || c)
  );
}

// Run health check after a short delay to ensure DOM is ready
console.log("Script loaded, scheduling health check...");
setTimeout(() => {
  console.log("Executing health check...");
  checkHealth();
}, 200);
