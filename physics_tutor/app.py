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

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Physics Tutor — Class 12 CBSE",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Class 12 Physics Tutor")
st.caption("Powered by your NCERT notes, exemplar, and PYQs — 2027 Board Exam Edition")

# ── Sidebar ────────────────────────────────────────────────────────────────
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

# ── Init vector store & chain (cached in session state) ───────────────────
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading your study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

# Re-instantiate if provider changed or not initialized
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

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Physics teacher, here to help you "
                "ace your **2027 CBSE Board Exams**.\n\n"
                "Tell me which chapter or topic you want to study, "
                "or ask me to generate **HOTS questions** on any chapter!"
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your Physics teacher …"):
    # Show student message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            # Limit history to the last 4 exchanges (8 messages) to stay well within free token limits
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 8:
                recent_messages = recent_messages[-8:]
            chat_history = convert_history(recent_messages)

            try:
                result = st.session_state["rag_chain"].invoke({
                    "question": user_input,
                    "chat_history": chat_history,
                })
                answer = result["answer"]
                sources = result.get("source_documents", [])

                st.markdown(answer)

                # Show source references (collapsed)
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
                        st.warning(
                            "⚠️ **Groq Rate Limit (RPM/TPM)**: Groq's free tier has a per-minute token limit. Please wait ~10-15 seconds and try again."
                        )
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Your OpenAI account has run out of credits or reached its usage limit.\n\n"
                            "👉 Please check your billing on [OpenAI Billing](https://platform.openai.com/account/billing/overview) or switch to Groq (free) in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
