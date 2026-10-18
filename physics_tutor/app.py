"""
Streamlit chat interface for the Physics Tutor RAG Agent.
Run with:  streamlit run physics_tutor/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path for Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from physics_tutor.config import get_config
from physics_tutor.ingest import build_vector_store
from physics_tutor.rag_chain import (
    build_rag_chain,
    convert_history,
    invoke_with_retry,
    is_rate_limit_error,
    normalize_math,
    sanitize_html,
    trim_text,
)
from physics_tutor.file_utils import (
    extract_text,
    image_to_base64_uri,
    is_image,
    SUPPORTED_EXTS,
)
from physics_tutor.components.paste_button import paste_image_button
from shared_tutor.validator import validate_answer

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Physics Tutor — Class 12 CBSE",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Class 12 Physics Tutor")
st.caption("Powered by NCERT textbooks & solutions, exemplar, notes, SSM test papers, and PYQs — 2027 Board Exam Edition")

# -- Sidebar ----------------------------------------------------------------
with st.sidebar:
    current_provider = get_config("LLM_PROVIDER", "openai").lower()
    st.caption(f"🤖 **Active Provider**: `{current_provider.upper()}`")
    st.header("🗂️ Chapters")
    st.markdown("""
**Part 1**

1. Electric Charges & Fields
2. Electrostatic Potential & Capacitance
3. Current Electricity
4. Moving Charges & Magnetism
5. Magnetism & Matter
6. Electromagnetic Induction
7. Alternating Current
8. Electromagnetic Waves

---

**Part 2**

1. Ray Optics & Optical Instruments
2. Wave Optics
3. Dual Nature of Radiation & Matter
4. Atoms
5. Nuclei
6. Semiconductor Electronics
""")
    st.divider()

    # Study Material Links
    REPO = "https://github.com/rajzzh-learn/Physics-RAG-Project/tree/main"
    st.header("📚 Study Material")
    st.markdown(f"""
| Folder | Link |
|--------|------|
| 📖 NCERT Book Part 1 | [leph1dd]({REPO}/book/leph1dd) |
| 📖 NCERT Book Part 2 | [leph2dd]({REPO}/book/leph2dd) |
| 🔬 Exemplar | [Exemplar]({REPO}/Exemplar) |
| 📝 Notes | [Notes]({REPO}/Notes) |
| ❓ Chapterwise Imp Questions | [Chapterwise imp questions]({REPO}/Chapterwise%20imp%20questions) |
| 📋 PYQ | [PYQ]({REPO}/PYQ) |
| 🎯 Competency Based Questions | [Competency based Questions]({REPO}/Competency%20based%20Questions) |
| 🔐 Secret Assignment | [Secret Assignment]({REPO}/Secret%20Assignment) |
| 💡 NCERT Solutions | [Ncert Solutions]({REPO}/Ncert%20Solutions) |
| 📄 SSM Question Papers | [SSM Question Paper so far]({REPO}/SSM%20Question%20Paper%20so%20far) |
""")
    st.divider()

    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Teach me Gauss's Law*
- *Give me 5 HOTS questions on Capacitance*
- *Explain NCERT exercise question on Electric Dipole*
- *Solve: A capacitor of 4μF …*
- *What questions appeared in the SSM test papers?*
- *What were the most repeated PYQ topics in 2024?*
""")
    st.divider()
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Re-ingesting all PDFs … (this takes a few minutes)"):
            st.session_state.pop("rag_chain", None)
            st.session_state.pop("vector_store", None)
            build_vector_store(force_rebuild=True)
        st.success("Vector store rebuilt!")

# -- Init vector store & chain ---------------------------------------------
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading your study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

target_provider = get_config("LLM_PROVIDER", "openai").lower()
if "rag_chain" not in st.session_state or st.session_state.get("_active_provider") != target_provider:
    try:
        st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
        st.session_state["_active_provider"] = target_provider
    except Exception as e:
        st.error(
            f"⚠️ **LLM Initialization Error**: {e}\n\n"
            "👉 If using Groq, ensure `LLM_PROVIDER = \"groq\"` and `GROQ_API_KEY = \"gsk_...\"` in Streamlit Secrets.\n"
            "👉 If using OpenAI, please ensure `OPENAI_API_KEY` is added to Streamlit Secrets."
        )
        st.stop()

# -- Chat history ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Physics teacher, here to help you "
                "ace your **2027 CBSE Board Exams**.\n\n"
                "Tell me which chapter or topic you want to study, "
                "or ask me to generate **HOTS questions** on any chapter! "
                "You can also **attach an image, PDF, or text file** using the 📎 button, "
                "or **paste a screenshot** with the 📋 button."
            ),
        }
    ]

def render_md(text: str) -> None:
    """Render Markdown with KaTeX math ($...$, $$...$$) and sanitized HTML."""
    st.markdown(sanitize_html(text), unsafe_allow_html=True)


def render_attachments(attachments: list[dict]) -> None:
    for att in attachments:
        st.caption(f"📎 **Attached:** `{att['name']}`")
        if att.get("preview") is not None:
            with st.expander(f"👁️ Preview — {att['name']}", expanded=False):
                if att.get("is_image"):
                    st.image(att["preview"], use_container_width=True)
                else:
                    st.text(str(att["preview"])[:2000])


# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        render_attachments(msg.get("attachments", []))
        render_md(msg["content"])


# ── Helper: call vision-capable LLM directly (image path) ──────────────
def _ask_vision_llm(
    question: str,
    image_data_uris: list[str],
    context_text: str,
    attached_text: str = "",
) -> str:
    """
    Send a multimodal (text + one or more images) message to Google Gemini Flash.
    Vision is always handled by Gemini regardless of LLM_PROVIDER,
    since Groq has no vision models and OpenAI requires paid credits.
    Falls back gracefully when GOOGLE_API_KEY is not set.
    """
    import base64, re as _re
    from google import genai
    from google.genai import types

    google_key = get_config("GOOGLE_API_KEY")
    if not google_key:
        return (
            "⚠️ **Image analysis requires a `GOOGLE_API_KEY`** (free).\n\n"
            "👉 Get one at https://aistudio.google.com/app/apikey — it's free, no billing needed.\n"
            "Then add `GOOGLE_API_KEY = \"AIza...\"` to your `.env` file or Streamlit Secrets and reload the app."
        )

    parts = []
    for uri in image_data_uris:
        # data:<mime>;base64,<data>
        match = _re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", uri)
        if not match:
            return "⚠️ Could not parse an attached image. Please try uploading it again."
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(match.group("data")),
                mime_type=match.group("mime"),
            )
        )

    prompt = (
        "You are an expert Class 12 CBSE Physics teacher.\n"
        "Write ALL math as LaTeX inside dollar signs: inline $...$ and display $$...$$ "
        "(never \\( \\) or \\[ \\]). Isotopes like $^{238}_{92}\\mathrm{U}$.\n\n"
        f"Context from the student's study materials:\n{context_text}\n\n"
        + (f"Attached documents:\n{attached_text}\n\n" if attached_text else "")
        + f"The student has attached {len(parts)} image(s) and asks:\n{question}"
    )

    client = genai.Client(api_key=google_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt, *parts],
    )
    return response.text


# -- Input area: file uploader + paste button + chat input -----------------
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_files = st.file_uploader(
        "📎 Attach files (multiple allowed)",
        type=sorted(SUPPORTED_EXTS),
        accept_multiple_files=True,
        label_visibility="visible",
        help=f"Images, PDF, Word, PowerPoint, Excel, CSV, HTML, Markdown, text. Supported: {ext_list}",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your Physics teacher …"):
    TOTAL_TEXT_BUDGET = 6000  # chars of attached-document text sent to the LLM
    attachments: list[dict] = []   # shown in the chat UI
    image_uris: list[str] = []     # sent to the vision model
    doc_texts: list[str] = []      # sent to the text model
    attach_errors: list[str] = []

    # Pasted clipboard image
    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        uri, _ = image_to_base64_uri(img_bytes, "pasted-image.png")
        image_uris.append(uri)
        attachments.append({"name": "pasted-image.png", "preview": img_bytes, "is_image": True})

    # Uploaded files (any mix of formats)
    docs_to_read = [f for f in (uploaded_files or []) if not is_image(f.name)]
    per_doc_budget = max(1500, TOTAL_TEXT_BUDGET // max(1, len(docs_to_read)))
    for f in uploaded_files or []:
        data = f.getvalue()
        if is_image(f.name):
            uri, _ = image_to_base64_uri(data, f.name)
            image_uris.append(uri)
            attachments.append({"name": f.name, "preview": data, "is_image": True})
            continue
        try:
            text = extract_text(data, f.name, max_chars=per_doc_budget)
        except Exception:
            attach_errors.append(f.name)
            continue
        if not text:
            attach_errors.append(f.name)
            continue
        doc_texts.append(f"--- Attached file: {f.name} ---\n{text}")
        attachments.append({"name": f.name, "preview": text, "is_image": False})

    attached_text = "\n\n".join(doc_texts)

    st.session_state["messages"].append(
        {"role": "user", "content": user_input, "attachments": attachments}
    )

    with st.chat_message("user"):
        render_attachments(attachments)
        for name in attach_errors:
            st.warning(f"⚠️ Could not read `{name}` (empty, corrupted or password-protected) — skipped.")
        render_md(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            # Keep the prompt small: Groq free tier allows ~8000 tokens/minute
            # (input + reserved output), so only the last 2 turns, truncated.
            # Pass full history — convert_history manages the token budget automatically
            recent_messages = st.session_state["messages"][:-1]
            clean_history = [
                {"role": m["role"], "content": m["content"]}
                for m in recent_messages
            ]
            chat_history = convert_history(clean_history)

            try:
                if image_uris:
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = normalize_math(
                        _ask_vision_llm(user_input, image_uris, context_text, attached_text)
                    )
                    sources = source_docs
                else:
                    question = user_input
                    if attached_text:
                        question = f"{user_input}\n\n{trim_text(attached_text, TOTAL_TEXT_BUDGET + 500)}"
                    result = invoke_with_retry(
                        lambda: st.session_state["rag_chain"].invoke(
                            {"question": question, "chat_history": chat_history}
                        )
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                # ── Layer 3: Gemini validation ──────────────────────
                context_for_validation = "\n\n".join(
                    d.page_content for d in sources[:4]
                ) if sources else ""
                validation = validate_answer(
                    subject="Physics",
                    question=user_input,
                    context=context_for_validation,
                    answer=answer,
                )
                answer = validation.answer
                if validation.status == "corrected":
                    st.warning("⚠️ **Answer corrected by Gemini validator**")
                    for corr in validation.corrections:
                        st.caption(f"• {corr}")
                elif validation.status == "verified":
                    st.success("✅ Answer verified")
                render_md(answer)

                if sources:
                    with st.expander("📎 Sources from your study material", expanded=False):
                        seen = set()
                        for doc in sources:
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            label = f"{src}  — page {page}"
                            if label not in seen:
                                st.markdown(f"- `{label}`")
                                seen.add(label)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = str(e)
                if "collection" in err_msg.lower() and "does not exist" in err_msg.lower():
                    st.session_state["vector_store"] = build_vector_store(force_rebuild=False)
                    st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
                    st.rerun()
                provider = get_config("LLM_PROVIDER", "openai").lower()
                if is_rate_limit_error(e):
                    if provider == "groq":
                        st.warning(
                            "⚠️ **Groq Rate Limit**: the request exceeded the free-tier "
                            "token budget (8000 tokens/min) even after automatic retries. "
                            "Wait about a minute and try again, or ask for one section "
                            "(e.g. Section A) at a time."
                        )
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Check billing on "
                            "[OpenAI Billing](https://platform.openai.com/account/billing/overview) "
                            "or switch to Groq in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
