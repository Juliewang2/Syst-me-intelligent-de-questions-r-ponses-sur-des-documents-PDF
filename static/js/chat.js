/**
 * chat.js
 * --------
 * Drives the chat experience: document sidebar, conversation history,
 * sending messages, and rendering streamed Server-Sent Events
 * responses with Markdown formatting and source citations.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------
  // Toast notifications (shared utility, also used by upload.js)
  // -------------------------------------------------------------------
  const toastContainer = document.createElement("div");
  toastContainer.className = "toast-container";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(toastContainer));

  function showToast(message, type = "info") {
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = message;
    toastContainer.appendChild(el);
    setTimeout(() => el.remove(), 4500);
  }
  window.PDFChatToast = { show: showToast };

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  const state = {
    documents: [],
    conversations: [],
    currentConversationId: null,
    currentDocumentId: null, // null = search across all documents
    isStreaming: false,
  };

  // -------------------------------------------------------------------
  // DOM references (present on chat.html; guarded for other pages)
  // -------------------------------------------------------------------
  const chatScroll = document.getElementById("chat-scroll");
  const chatForm = document.getElementById("chat-form");
  const chatTextarea = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const docSelect = document.getElementById("scope-select");
  const documentListEl = document.getElementById("document-list");
  const conversationListEl = document.getElementById("conversation-list");
  const newChatBtn = document.getElementById("new-chat-btn");
  const emptyState = document.getElementById("empty-state");

  // -------------------------------------------------------------------
  // Markdown rendering (marked.js + DOMPurify, loaded via CDN in chat.html)
  // -------------------------------------------------------------------
  function renderMarkdown(text) {
    if (window.marked && window.DOMPurify) {
      const raw = window.marked.parse(text, { breaks: true });
      return window.DOMPurify.sanitize(raw);
    }
    // Fallback: escape HTML and preserve line breaks.
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, "<br>");
  }

  // -------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------

  // The session expired or was never there: go log in, then come back.
  function redirectToLogin() {
    window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
  }
  window.PDFChatAuth = { redirectToLogin };

  async function fetchJSON(url, options = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (res.status === 401) {
      redirectToLogin();
      throw new Error("Session expired. Please log in again.");
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  async function loadDocuments() {
    try {
      const data = await fetchJSON("/api/documents");
      state.documents = data.documents || [];
      renderDocumentSidebar();
      renderScopeSelect();
    } catch (err) {
      console.error(err);
    }
  }

  async function loadConversations() {
    if (!conversationListEl) return;
    try {
      const data = await fetchJSON("/api/conversations");
      state.conversations = data.conversations || [];
      renderConversationSidebar();
    } catch (err) {
      console.error(err);
    }
  }

  // -------------------------------------------------------------------
  // Sidebar rendering
  // -------------------------------------------------------------------
  function statusDot(status) {
    return `<span class="status-dot ${status}"></span>`;
  }

  function renderDocumentSidebar() {
    if (!documentListEl) return;
    if (state.documents.length === 0) {
      documentListEl.innerHTML = `<div style="padding: 10px; color: var(--color-text-muted); font-size: 13px;">
        No documents yet. <a href="/upload">Upload one</a> to get started.
      </div>`;
      return;
    }

    documentListEl.innerHTML = state.documents
      .map(
        (doc) => `
        <div class="doc-item" data-doc-id="${doc.id}">
          <div class="doc-icon">PDF</div>
          <div class="doc-meta">
            <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
            <div class="doc-sub">${statusDot(doc.status)} ${doc.page_count} pages &middot; ${doc.chunk_count} chunks</div>
          </div>
          <button class="doc-delete-btn" data-delete-id="${doc.id}" title="Delete document">&times;</button>
        </div>`
      )
      .join("");

    documentListEl.querySelectorAll(".doc-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".doc-delete-btn")) return;
        const docId = el.getAttribute("data-doc-id");
        state.currentDocumentId = state.currentDocumentId === docId ? null : docId;
        if (docSelect) docSelect.value = state.currentDocumentId || "";
        renderDocumentSidebar();
      });
      if (el.getAttribute("data-doc-id") === state.currentDocumentId) {
        el.classList.add("selected");
      }
    });

    documentListEl.querySelectorAll(".doc-delete-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const docId = btn.getAttribute("data-delete-id");
        if (!confirm("Delete this document? This cannot be undone.")) return;
        try {
          await fetchJSON(`/api/documents/${docId}`, { method: "DELETE" });
          showToast("Document deleted.", "success");
          if (state.currentDocumentId === docId) state.currentDocumentId = null;
          await loadDocuments();
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    });
  }

  function renderScopeSelect() {
    if (!docSelect) return;
    const options = [`<option value="">All documents</option>`].concat(
      state.documents
        .filter((d) => d.status === "ready")
        .map((d) => `<option value="${d.id}">${escapeHtml(d.filename)}</option>`)
    );
    docSelect.innerHTML = options.join("");
    docSelect.value = state.currentDocumentId || "";
  }

  function renderConversationSidebar() {
    if (!conversationListEl) return;
    if (state.conversations.length === 0) {
      conversationListEl.innerHTML = `<div style="padding: 10px; color: var(--color-text-muted); font-size: 13px;">
        No conversations yet.
      </div>`;
      return;
    }
    conversationListEl.innerHTML = state.conversations
      .map(
        (c) => `
        <div class="conv-item" data-conv-id="${c.id}">
          <div class="doc-meta">
            <div class="doc-name">${escapeHtml(c.title || "New Conversation")}</div>
            <div class="doc-sub">${c.message_count} messages</div>
          </div>
        </div>`
      )
      .join("");

    conversationListEl.querySelectorAll(".conv-item").forEach((el) => {
      el.addEventListener("click", () => loadConversation(el.getAttribute("data-conv-id")));
      if (el.getAttribute("data-conv-id") === state.currentConversationId) {
        el.classList.add("selected");
      }
    });
  }

  window.PDFChatSidebar = {
    reload: async () => {
      await loadDocuments();
      await loadConversations();
    },
  };

  // -------------------------------------------------------------------
  // Message rendering
  // -------------------------------------------------------------------
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function scrollToBottom() {
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  function hideEmptyState() {
    if (emptyState) emptyState.style.display = "none";
  }

  function appendMessage(role, content, sources) {
    if (!chatScroll) return null;
    hideEmptyState();

    const inner = chatScroll.querySelector(".chat-inner");
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role}`;
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubbleWrap = document.createElement("div");
    bubbleWrap.className = "bubble-wrap";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = role === "assistant" ? renderMarkdown(content) : escapeHtml(content);

    bubbleWrap.appendChild(bubble);

    if (sources && sources.length) {
      bubbleWrap.appendChild(buildSourcesPanel(sources));
    }

    row.appendChild(avatar);
    row.appendChild(bubbleWrap);
    inner.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  function buildSourcesPanel(sources) {
    const wrap = document.createElement("div");
    wrap.className = "sources-panel";

    const toggle = document.createElement("div");
    toggle.className = "sources-toggle";
    toggle.innerHTML = `&#128218; ${sources.length} source${sources.length > 1 ? "s" : ""}`;

    const chips = document.createElement("div");
    chips.style.display = "none";
    chips.style.flexDirection = "column";
    chips.style.gap = "6px";
    chips.innerHTML = sources
      .map(
        (s) => `
        <div class="source-chip">
          <div class="source-title"><span>${escapeHtml(s.document_name)}</span><span class="source-meta">${sourceMeta(s)}</span></div>
          <div class="source-snippet">${escapeHtml(s.text_snippet)}</div>
        </div>`
      )
      .join("");

    toggle.addEventListener("click", () => {
      chips.style.display = chips.style.display === "none" ? "flex" : "none";
    });

    wrap.appendChild(toggle);
    wrap.appendChild(chips);
    return wrap;
  }

  // "p.3" or "p.3–4", plus the reranker's relevance score when present.
  function sourceMeta(s) {
    const start = s.page ?? "?";
    const end = s.page_end ?? start;
    const pages = end !== start ? `p.${start}&ndash;${end}` : `p.${start}`;
    return s.rerank_score != null ? `${pages} &middot; relevance ${s.rerank_score}/10` : pages;
  }

  function appendTypingIndicator() {
    const bubble = appendMessage("assistant", "");
    if (bubble) {
      bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    }
    return bubble;
  }

  // -------------------------------------------------------------------
  // Conversation loading
  // -------------------------------------------------------------------
  async function loadConversation(conversationId) {
    state.currentConversationId = conversationId;
    const conv = state.conversations.find((c) => c.id === conversationId);
    if (conv) {
      state.currentDocumentId = conv.document_id || null;
      if (docSelect) docSelect.value = state.currentDocumentId || "";
    }

    const inner = chatScroll?.querySelector(".chat-inner");
    if (!inner) return;
    inner.innerHTML = "";

    try {
      const messages = await fetchJSON(`/api/conversations/${conversationId}/messages`);
      if (messages.length === 0) {
        if (emptyState) {
          inner.appendChild(emptyState);
          emptyState.style.display = "flex";
        }
      } else {
        hideEmptyState();
        messages.forEach((m) => appendMessage(m.role, m.content, m.sources));
      }
    } catch (err) {
      showToast(err.message, "error");
    }
    renderConversationSidebar();
    renderDocumentSidebar();
  }

  function startNewConversation() {
    state.currentConversationId = null;
    const inner = chatScroll?.querySelector(".chat-inner");
    if (inner && emptyState) {
      inner.innerHTML = "";
      inner.appendChild(emptyState);
      emptyState.style.display = "flex";
    }
    renderConversationSidebar();
  }

  // -------------------------------------------------------------------
  // Sending messages (Server-Sent Events streaming via fetch)
  // -------------------------------------------------------------------
  async function sendMessage(text) {
    if (!text.trim() || state.isStreaming) return;

    appendMessage("user", text);
    const assistantBubble = appendTypingIndicator();
    state.isStreaming = true;
    setSendingState(true);

    let accumulated = "";
    let sources = [];
    let firstToken = true;

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: state.currentConversationId,
          document_id: state.currentDocumentId,
          message: text,
        }),
      });

      if (response.status === 401) {
        redirectToLogin();
        throw new Error("Session expired. Please log in again.");
      }
      if (!response.ok || !response.body) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to reach the chat service.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop(); // keep any incomplete trailing chunk

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          let payload;
          try {
            payload = JSON.parse(jsonStr);
          } catch {
            continue;
          }

          if (payload.type === "start") {
            state.currentConversationId = payload.conversation_id;
          } else if (payload.type === "sources") {
            sources = payload.sources || [];
          } else if (payload.type === "token") {
            if (firstToken) {
              assistantBubble.innerHTML = "";
              firstToken = false;
            }
            accumulated += payload.content;
            assistantBubble.innerHTML = renderMarkdown(accumulated);
            scrollToBottom();
          } else if (payload.type === "error") {
            showToast(payload.message, "error");
          } else if (payload.type === "close") {
            // stream finished
          }
        }
      }

      if (!accumulated.trim()) {
        assistantBubble.innerHTML = renderMarkdown(
          "_I couldn't generate a response. Please try again._"
        );
      }

      if (sources.length) {
        const bubbleWrap = assistantBubble.closest(".bubble-wrap");
        bubbleWrap.appendChild(buildSourcesPanel(sources));
      }

      await loadConversations();
    } catch (err) {
      assistantBubble.innerHTML = renderMarkdown(`**Error:** ${err.message}`);
      showToast(err.message, "error");
    } finally {
      state.isStreaming = false;
      setSendingState(false);
      scrollToBottom();
    }
  }

  function setSendingState(isSending) {
    if (sendBtn) sendBtn.disabled = isSending;
    if (chatTextarea) chatTextarea.disabled = isSending;
  }

  // -------------------------------------------------------------------
  // Event wiring
  // -------------------------------------------------------------------
  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = chatTextarea.value;
      chatTextarea.value = "";
      autoResize();
      sendMessage(text);
    });
  }

  if (chatTextarea) {
    chatTextarea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm?.requestSubmit();
      }
    });
    chatTextarea.addEventListener("input", autoResize);
  }

  function autoResize() {
    if (!chatTextarea) return;
    chatTextarea.style.height = "auto";
    chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 160) + "px";
  }

  document.querySelectorAll(".suggested-chip").forEach((chip) => {
    chip.addEventListener("click", () => sendMessage(chip.textContent.trim()));
  });

  if (docSelect) {
    docSelect.addEventListener("change", () => {
      state.currentDocumentId = docSelect.value || null;
      renderDocumentSidebar();
    });
  }

  if (newChatBtn) {
    newChatBtn.addEventListener("click", startNewConversation);
  }

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
      window.location.href = "/login";
    });
  }

  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const sidebar = document.querySelector(".sidebar");
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", async () => {
    await loadDocuments();
    await loadConversations();
  });
})();
