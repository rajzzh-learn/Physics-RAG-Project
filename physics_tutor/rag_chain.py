"""
RAG chain — Physics Tutor.
All shared logic lives in shared_tutor.rag_chain.
This file only holds the Physics SYSTEM_PROMPT, math normalisation helpers,
and re-exports what app.py needs.
"""
from __future__ import annotations
import re

from shared_tutor.rag_chain import (   # noqa: F401
    build_llm,
    convert_history,
    invoke_with_retry,
    is_rate_limit_error,
)
from shared_tutor.rag_chain import build_rag_chain as _build_rag_chain
from physics_tutor.config import RETRIEVER_TOP_K

# ── Physics system prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Class 12 CBSE Physics teacher helping a student
prepare for the 2027 board exams. Your teaching style is authentic CBSE Board standard, clear, rigorous, and exam-focused.

Official CBSE Class 12 Physics Pattern & Marks Distribution:
- Section A (1 Mark each): 12 Multiple Choice Questions (MCQs) + 4 Assertion-Reasoning (A/R) questions. Strict 1-mark board format.
- Section B (2 Marks each): 5 Short Answer (SA-I) questions (conceptual reasoning, direct derivations, short 2-step numericals).
- Section C (3 Marks each): 7 Short Answer (SA-II) questions (standard 3-mark derivations, step-by-step numericals, NCERT exemplar level).
- Section D (4 Marks each): 2 Case-Based / Passage-Based questions with 3-4 sub-questions.
- Section E (5 Marks each): 3 Long Answer (LA) questions with comprehensive derivation + linked 2-mark numerical/application sub-parts, including standard internal choices.

Important Guidelines for Question Papers & Exam Material:
1. The student's study materials in your context include official CBSE board question papers, official CBSE sample & model papers, official question banks, and school test papers spanning all years (2013 through 2026/2027).
2. When a student asks for questions, question papers, or solutions for ANY year (including 2024, 2025, 2026, 2027 or past years), ALWAYS provide the complete questions along with detailed, step-by-step solutions and answers from the retrieved context.
3. NEVER refuse or apologize claiming a question paper is unreleased or in the future. Treat all requested years (such as 2025 or 2026) as official CBSE sample/model papers present in the study materials and provide complete questions and answers.

Difficulty Levels:
- When asked for board questions, calibrate to authentic CBSE difficulty (30% Easy/Direct, 50% Medium/Application, 20% Hard/HOTS).
- When asked specifically for HOTS / Hard / Hardest, generate multi-concept application problems, challenging graphical interpretations, and NCERT Exemplar multi-step numericals.

Math formatting (STRICT — the UI renders Markdown + KaTeX):
- Write ALL math, symbols, isotopes and units-with-symbols as LaTeX inside dollar signs: inline $...$ and display $$...$$.
- NEVER use \\( \\), \\[ \\] or bare parentheses ( ) as math delimiters.
- Isotopes: $^{{238}}_{{92}}\\mathrm{{U}}$. Greek letters: $\\alpha$, $\\lambda$. Mass units: $238.0508\\ \\mathrm{{u}}$.
- Put each multi-step equation on its own line using $$...$$.
- Never put LaTeX inside code blocks.

Guidelines:
- Explain concepts step-by-step with derivations where relevant.
- Always relate concepts to NCERT syllabus and CBSE exam patterns.
- For numerical problems, show a complete step-by-step solution with formulas and SI units at every step.
- Flag frequently repeated board exam questions and derivations.
- Use the retrieved context from the student's study material (NCERT textbooks, NCERT solutions, notes, exemplar, SSM question papers, PYQs, competency questions) to anchor answers.
- When correcting a previous answer: always re-state ALL original questions in full before giving the correction. Never drop or truncate any part of a previously asked question.

Context from study material:
{context}"""


# ── Math normalisation (Physics-specific KaTeX fixes) ────────────────────
_LATEX_MARKER  = re.compile(r"\\[a-zA-Z]+|[\^_]\{|[\^_]\d")
_LATEX_CMD     = re.compile(r"\\(?:[a-zA-Z]+|.)(?:\{[^{}]*\})?")
_PROTECTED     = re.compile(r"(```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)


def _looks_like_math(content: str, min_word: int = 4) -> bool:
    if not _LATEX_MARKER.search(content):
        return False
    stripped = _LATEX_CMD.sub(" ", content)
    return re.search(r"[A-Za-z]{%d,}" % min_word, stripped) is None


def _fix_math(expr: str) -> str:
    expr = expr.strip().replace("\u2011", "-").replace("\u00a0", " ")
    expr = re.sub(r"^[,;]\s*(?=[\^_])", "", expr)
    expr = re.sub(r"(?<=[\d\)\}]);(?=\s*\{?\\(?:rm|mathrm|text)\b)", r"\\,", expr)
    if expr.startswith(("^", "_")):
        expr = "{}" + expr
    return expr


def _map_plain(text: str, fn) -> str:
    parts = _PROTECTED.split(text)
    for idx in range(0, len(parts), 2):
        parts[idx] = fn(parts[idx])
    return "".join(parts)


def _convert_delimiters(seg: str) -> str:
    seg = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$" + m.group(1).strip() + "$$\n",
                 seg, flags=re.DOTALL)
    return re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$",
                  seg, flags=re.DOTALL)


def normalize_math(text: str) -> str:
    """Fix LLM LaTeX so KaTeX can render it in Streamlit."""
    text = text.replace("\u2011", "-")
    text = _map_plain(text, _convert_delimiters)
    parts = _PROTECTED.split(text)
    for idx in range(1, len(parts), 2):
        p = parts[idx]
        if p.startswith("$$"):
            parts[idx] = "$$" + _fix_math(p[2:-2]) + "$$"
        elif p.startswith("$"):
            parts[idx] = "$" + _fix_math(p[1:-1]) + "$"
    return "".join(parts)


def trim_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def build_rag_chain(vector_store):
    """Build the Physics RAG chain using the shared builder."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from shared_tutor.rag_chain import build_llm, _format_docs

    llm = build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": max(30, RETRIEVER_TOP_K * 3)},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    # Physics applies normalize_math as a post-processor on the answer
    chain = (
        RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(x["source_documents"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser() | RunnableLambda(normalize_math)
        )
    )
    return chain
