"""
Bundled clipboard paste-image component for CS Tutor.

Works on HTTP localhost AND HTTPS Streamlit Cloud — no browser permission
prompt required. Uses the DOM `paste` event (Ctrl+V / Cmd+V) which every
browser allows without the Clipboard Permissions API.
"""
from pathlib import Path
import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).parent / "frontend"

_paste_image_button = components.declare_component(
    "paste_image_button",
    path=str(_FRONTEND_DIR),
)


class PasteResult:
    """Holds the result of a paste action."""
    def __init__(self, data_uri: str | None):
        self._data_uri = data_uri

    @property
    def image_data(self):
        """Return a PIL Image if an image was pasted, else None."""
        if not self._data_uri or self._data_uri.startswith("error"):
            return None
        try:
            import base64, io
            from PIL import Image
            header, b64 = self._data_uri.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64)))
        except Exception:
            return None


def paste_image_button(
    label: str = "📋 Paste image",
    background_color: str = "#444654",
    hover_background_color: str = "#565869",
    text_color: str = "#ffffff",
    key: str = "paste_button",
) -> PasteResult:
    """
    Render a paste-image button.

    The user clicks it, then presses Ctrl+V / Cmd+V to paste a screenshot.
    Returns a PasteResult whose .image_data is a PIL Image (or None).
    """
    raw = _paste_image_button(
        label=label,
        background_color=background_color,
        hover_background_color=hover_background_color,
        text_color=text_color,
        key=key,
        default=None,
    )
    return PasteResult(raw)
