/**
 * theme.js
 * ---------
 * Handles dark/light theme toggling and persistence via localStorage.
 * Applied to `document.documentElement` as `data-theme="dark"|"light"`.
 */

(function () {
  const STORAGE_KEY = "pdf-chat-theme";

  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_KEY, theme);
    updateToggleButtons(theme);
  }

  function updateToggleButtons(theme) {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      const icon = btn.querySelector("[data-theme-icon]");
      const label = btn.querySelector("[data-theme-label]");
      if (icon) icon.textContent = theme === "dark" ? "\u2600\ufe0f" : "\ud83c\udf19";
      if (label) label.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    });
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(current === "dark" ? "light" : "dark");
  }

  // Apply theme immediately (before DOMContentLoaded) to avoid a flash
  // of incorrectly-themed content.
  applyTheme(getPreferredTheme());

  document.addEventListener("DOMContentLoaded", () => {
    updateToggleButtons(document.documentElement.getAttribute("data-theme") || "light");
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", toggleTheme);
    });
  });

  window.PDFChatTheme = { toggleTheme, applyTheme, getPreferredTheme };
})();
