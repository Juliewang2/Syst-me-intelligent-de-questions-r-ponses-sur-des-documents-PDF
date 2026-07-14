/**
 * upload.js
 * ----------
 * Handles the drag-and-drop / click-to-browse upload experience on
 * /upload, including per-file progress bars and status polling.
 */

(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const queue = document.getElementById("upload-queue");
  const browseBtn = document.getElementById("browse-btn");

  if (!dropzone || !fileInput || !queue) return;

  function humanFileSize(bytes) {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  }

  function createUploadItem(file) {
    const item = document.createElement("div");
    item.className = "upload-item";
    item.innerHTML = `
      <div class="file-icon">PDF</div>
      <div class="info">
        <div class="name">${escapeHtml(file.name)}</div>
        <div class="sub">
          <span class="size-label">${humanFileSize(file.size)}</span>
          &middot;
          <span class="status-label">Uploading&hellip;</span>
        </div>
        <div class="progress-bar"><div class="progress-bar-fill"></div></div>
      </div>
    `;
    queue.prepend(item);
    return item;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function setItemStatus(item, status, extra) {
    const statusLabel = item.querySelector(".status-label");
    const fill = item.querySelector(".progress-bar-fill");
    if (status === "ready") {
      statusLabel.innerHTML = `<span class="badge-pill ready">Ready</span> ${extra || ""}`;
      fill.style.width = "100%";
      fill.style.background = "var(--color-success)";
    } else if (status === "failed") {
      statusLabel.innerHTML = `<span class="badge-pill failed">Failed</span> ${extra || ""}`;
      fill.style.width = "100%";
      fill.style.background = "var(--color-danger)";
    } else if (status === "processing") {
      statusLabel.innerHTML = `<span class="badge-pill processing">Processing&hellip;</span>`;
      fill.style.width = "90%";
    }
  }

  function uploadFiles(files) {
    const validFiles = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (validFiles.length === 0) {
      window.PDFChatToast?.show("Please select PDF files only.", "error");
      return;
    }

    const formData = new FormData();
    const items = validFiles.map((file) => {
      formData.append("files", file);
      return { file, item: createUploadItem(file) };
    });

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.min(95, Math.round((e.loaded / e.total) * 100));
      items.forEach(({ item }) => {
        const fill = item.querySelector(".progress-bar-fill");
        fill.style.width = pct + "%";
      });
      if (pct >= 95) {
        items.forEach(({ item }) => setItemStatus(item, "processing"));
      }
    });

    xhr.onload = () => {
      try {
        const response = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          const docs = response.documents || [];
          items.forEach(({ item, file }, idx) => {
            const doc = docs.find((d) => d.filename === file.name) || docs[idx];
            if (doc && doc.status === "ready") {
              setItemStatus(item, "ready", `${doc.page_count} pages &middot; ${doc.chunk_count} chunks`);
            } else if (doc) {
              setItemStatus(item, "failed", doc.error_message || "Processing failed");
            } else {
              setItemStatus(item, "failed", "Unknown error");
            }
          });
          window.PDFChatToast?.show(`${docs.length} document(s) processed.`, "success");
          refreshDocumentSidebar();
        } else {
          items.forEach(({ item }) => setItemStatus(item, "failed", response.detail || "Upload failed"));
          window.PDFChatToast?.show(response.detail || "Upload failed.", "error");
        }
      } catch (err) {
        items.forEach(({ item }) => setItemStatus(item, "failed", "Unexpected server response"));
      }
    };

    xhr.onerror = () => {
      items.forEach(({ item }) => setItemStatus(item, "failed", "Network error"));
      window.PDFChatToast?.show("Network error during upload.", "error");
    };

    xhr.send(formData);
  }

  function refreshDocumentSidebar() {
    if (window.PDFChatSidebar && typeof window.PDFChatSidebar.reload === "function") {
      window.PDFChatSidebar.reload();
    }
  }

  // --- Drag & drop wiring ---
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length) uploadFiles(files);
  });

  dropzone.addEventListener("click", () => fileInput.click());
  browseBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length) {
      uploadFiles(fileInput.files);
      fileInput.value = "";
    }
  });
})();
