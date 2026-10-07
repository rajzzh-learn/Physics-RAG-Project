// CS Tutor — bundled paste-image component
// Uses DOM `paste` event (Ctrl+V / Cmd+V) which works on HTTP localhost
// AND HTTPS without any browser clipboard-read permission prompt.
// Falls back to navigator.clipboard.read() silently on HTTPS if available.

function sendValue(dataUri) {
  Streamlit.setComponentValue(dataUri);
}

function readImageFromClipboardEvent(e) {
  const items = (e.clipboardData || {}).items || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      const reader = new FileReader();
      reader.onloadend = () => sendValue(reader.result);
      reader.readAsDataURL(blob);
      return true;
    }
  }
  return false;
}

function onRender(event) {
  if (window._rendered) return;
  window._rendered = true;

  const {
    label = "📋 Paste image",
    background_color = "#444654",
    hover_background_color = "#565869",
    text_color = "#ffffff",
  } = event.detail.args;

  const btn    = document.getElementById("paste-btn");
  const status = document.getElementById("status");

  // Apply theme colours
  btn.style.backgroundColor = background_color;
  btn.style.color            = text_color;
  if (event.detail.theme && event.detail.theme.font) {
    btn.style.fontFamily = event.detail.theme.font;
  }
  btn.textContent = label;
  const originalLabel = label;

  btn.addEventListener("mouseover", () => btn.style.backgroundColor = hover_background_color);
  btn.addEventListener("mouseout",  () => btn.style.backgroundColor = background_color);

  // ── Strategy 1: listen for a real paste event (Ctrl+V / Cmd+V) ────────
  // This is the most reliable approach — works on HTTP, HTTPS, all browsers,
  // no permission prompt. We register it once on button click.
  function waitForPaste() {
    btn.textContent = "Press Ctrl+V / Cmd+V …";
    status.textContent = "Waiting for paste…";

    function onPaste(e) {
      e.preventDefault();
      const found = readImageFromClipboardEvent(e);
      cleanup();
      if (!found) {
        status.textContent = "⚠️ No image found — copy a screenshot first.";
        btn.textContent = originalLabel;
        setTimeout(() => { status.textContent = ""; }, 3000);
      } else {
        status.textContent = "✅ Image captured!";
        btn.textContent = originalLabel;
        setTimeout(() => { status.textContent = ""; }, 2000);
      }
    }

    function onKeydown(e) {
      // If user presses Escape, cancel
      if (e.key === "Escape") {
        cleanup();
        btn.textContent = originalLabel;
        status.textContent = "";
      }
    }

    function cleanup() {
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("keydown", onKeydown);
    }

    window.addEventListener("paste", onPaste);
    window.addEventListener("keydown", onKeydown);
  }

  btn.addEventListener("click", async () => {
    // ── Strategy 2: try navigator.clipboard.read() first (HTTPS + permission) ──
    if (navigator.clipboard && navigator.clipboard.read) {
      try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
          const imageType = item.types.find(t => t.startsWith("image/"));
          if (imageType) {
            const blob   = await item.getType(imageType);
            const reader = new FileReader();
            reader.onloadend = () => {
              sendValue(reader.result);
              status.textContent = "✅ Image captured!";
              setTimeout(() => { status.textContent = ""; }, 2000);
            };
            reader.readAsDataURL(blob);
            return; // success — no need to wait for paste event
          }
        }
        // Clipboard API worked but had no image — fall through to paste event
        waitForPaste();
      } catch (_) {
        // Clipboard API blocked (HTTP localhost or permission denied) — use paste event
        waitForPaste();
      }
    } else {
      // Browser doesn't support clipboard.read() — use paste event
      waitForPaste();
    }
  });
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(56);
