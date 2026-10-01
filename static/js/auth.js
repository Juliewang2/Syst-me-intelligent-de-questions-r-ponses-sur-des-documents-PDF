/**
 * auth.js
 * --------
 * Login / registration form on /login. On success the server sets an
 * HttpOnly session cookie, so we only need to redirect.
 */

(function () {
  "use strict";

  const form = document.getElementById("auth-form");
  const submitBtn = document.getElementById("auth-submit");
  const errorEl = document.getElementById("auth-error");
  const tabs = document.querySelectorAll(".auth-tab");
  const passwordInput = form?.querySelector('input[name="password"]');
  let mode = "login";

  if (!form) return;

  // Only allow same-site relative redirects (e.g. "/chat"), never "//evil.com".
  function nextUrl() {
    const next = new URLSearchParams(window.location.search).get("next") || "/chat";
    return next.startsWith("/") && !next.startsWith("//") ? next : "/chat";
  }

  function setMode(newMode) {
    mode = newMode;
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
    form.querySelectorAll("[data-register-only]").forEach((el) => {
      el.hidden = mode !== "register";
    });
    submitBtn.textContent = mode === "login" ? "Log in" : "Create account";
    passwordInput.autocomplete = mode === "login" ? "current-password" : "new-password";
    passwordInput.minLength = mode === "login" ? 1 : 8;
    showError("");
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = !message;
  }

  function describeError(body, status) {
    // FastAPI validation errors come back as a list of {loc, msg}.
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join("; ");
    }
    return body.detail || `Request failed (${status})`;
  }

  tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!form.reportValidity()) return;

    const data = Object.fromEntries(new FormData(form).entries());
    const payload = { username: data.username.trim(), password: data.password };
    if (mode === "register" && data.invite_code !== undefined) payload.invite_code = data.invite_code;

    submitBtn.disabled = true;
    showError("");
    try {
      const res = await fetch(`/api/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(describeError(body, res.status));
      }
      window.location.href = nextUrl();
    } catch (err) {
      showError(err.message);
    } finally {
      submitBtn.disabled = false;
    }
  });

  setMode("login");
})();
