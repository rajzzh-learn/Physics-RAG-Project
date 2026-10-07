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
from physics_tutor.rag_chain import build_rag_chain, convert_history
from physics_tutor.file_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    image_to_base64_uri,
    is_image,
    SUPPORTED_EXTS,
)
from physics_tutor.components.paste_button import paste_image_button

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Physics Tutor — Class 12 CBSE",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Class 12 Physics Tutor")
st.caption("Powered by your NCERT notes, exemplar, and PYQs — 2027 Board Exam Edition")

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
""")
    st.divider()

    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Teach me Gauss's Law*
- *Give me 5 HOTS questions on Capacitance*
- *What is the Bohr model?*
- *Solve: A capacitor of 4μF …*
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

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("attachment_name"):
            st.caption(f"📎 **Attached:** `{msg['attachment_name']}`")
            if msg.get("attachment_preview"):
                with st.expander("👁️ Attachment preview", expanded=False):
                    if msg.get("attachment_is_image"):
                        st.image(msg["attachment_preview"], use_container_width=True)
                    else:
                        st.text(msg["attachment_preview"][:2000])
        st.markdown(msg["content"])


# -- Helper: vision LLM call -----------------------------------------------
def _ask_vision_llm(question: str, image_data_uri: str, context_text: str) -> str:
    provider = get_config("LLM_PROVIDER", "openai").lower()
    vision_content = [
        {
            "type": "text",
            "text": (
                "You are an expert Class 12 CBSE Physics teacher.\n\n"
                f"Context from the student's study materials:\n{context_text}\n\n"
                f"The student has attached an image and asks:\n{question}"
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]
    if provider == "groq":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=get_config("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=get_config("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    else:
        return (
            "⚠️ **Image vision is not supported for the `watsonx` provider.** "
            "Please switch to `groq` or `openai`, or describe your question in text."
        )


# -- Input area: file uploader + paste button + chat input -----------------
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        f"📎 Attach a file — {ext_list}",
        type=list(SUPPORTED_EXTS),
        label_visibility="visible",
        help="Attach an image (diagram / question screenshot), PDF, or .txt file.",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your Physics teacher …"):
    attachment_name: str | None = None
    attachment_text: str | None = None
    attachment_image_uri: str | None = None
    attachment_preview = None
    attachment_is_image = False

    # Clipboard paste takes priority over file uploader
    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        attachment_name = "pasted-image.png"
        attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
        attachment_preview = file_bytes
        attachment_is_image = True
    elif uploaded_file is not None:
        attachment_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        if is_image(attachment_name):
            attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
            attachment_preview = file_bytes
            attachment_is_image = True
        elif attachment_name.lower().endswith(".pdf"):
            attachment_text = extract_text_from_pdf(file_bytes)
            attachment_preview = attachment_text
        else:
            attachment_text = extract_text_from_txt(file_bytes)
            attachment_preview = attachment_text

    user_msg: dict = {
        "role": "user",
        "content": user_input,
        "attachment_name": attachment_name,
        "attachment_preview": attachment_preview,
        "attachment_is_image": attachment_is_image,
    }
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        if attachment_name:
            st.caption(f"📎 **Attached:** `{attachment_name}`")
            if attachment_preview is not None:
                with st.expander("👁️ Attachment preview", expanded=False):
                    if attachment_is_image:
                        st.image(attachment_preview, use_container_width=True)
                    else:
                        st.text(str(attachment_preview)[:2000])
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]
            clean_history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
            chat_history = convert_history(clean_history)

            try:
                if attachment_image_uri is not None:
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = _ask_vision_llm(user_input, attachment_image_uri, context_text)
                    sources = source_docs
                elif attachment_text is not None:
                    augmented = f"{user_input}\n\n--- Attached file: {attachment_name} ---\n{attachment_text}"
                    result = st.session_state["rag_chain"].invoke(
                        {"question": augmented, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                else:
                    result = st.session_state["rag_chain"].invoke(
                        {"question": user_input, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                st.markdown(answer)

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
                provider = get_config("LLM_PROVIDER", "openai").lower()
                if "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg:
                    if provider == "groq":
                        st.warning("⚠️ **Groq Rate Limit**: Please wait ~10-15 seconds and try again.")
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Check billing on "
                            "[OpenAI Billing](https://platform.openai.com/account/billing/overview) "
                            "or switch to Groq in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
