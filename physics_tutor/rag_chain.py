"""
RAG chain — Physics Tutor persona.
Uses LangChain 1.x LCEL pipeline.
Supports OpenAI (gpt-4o) and IBM watsonx.ai (Granite) as LLM backends.
"""
import re
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from physics_tutor.config import (
    get_config,
    RETRIEVER_TOP_K,
)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Class 12 CBSE Physics teacher helping a student
prepare for the 2027 board exams. Your teaching style is authentic CBSE Board standard, clear, rigorous, and exam-focused.

Official CBSE Class 12 Physics Pattern & Marks Distribution:
- Section A (1 Mark each): 12 Multiple Choice Questions (MCQs) + 4 Assertion-Reasoning (A/R) questions. Strict 1-mark board format.
- Section B (2 Marks each): 5 Short Answer (SA-I) questions (conceptual reasoning, direct derivations, short 2-step numericals).
- Section C (3 Marks each): 7 Short Answer (SA-II) questions (standard 3-mark derivations, step-by-step numericals, NCERT exemplar level).
- Section D (4 Marks each): 2 Case-Based / Passage-Based questions with 3-4 sub-questions.
- Section E (5 Marks each): 3 Long Answer (LA) questions with comprehensive derivation + linked 2-mark numerical/application sub-parts, including standard internal choices.

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

Context from study material:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ── Math normalisation ─────────────────────────────────────────────────────
_LATEX_MARKER = re.compile(r"\\[a-zA-Z]+|[\^_]\{|[\^_]\d")
_LATEX_CMD = re.compile(r"\\(?:[a-zA-Z]+|.)(?:\{[^{}]*\})?")
_PROTECTED = re.compile(r"(```.*?```|`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)


def _looks_like_math(content: str, min_word: int = 4) -> bool:
    """True if a chunk contains LaTeX and no prose words (>= min_word letters)."""
    if not _LATEX_MARKER.search(content):
        return False
    stripped = _LATEX_CMD.sub(" ", content)
    return re.search(r"[A-Za-z]{%d,}" % min_word, stripped) is None


def _wrap_paren_math(text: str) -> str:
    """Convert '(\\alpha)' / '(m(^{238}{\\rm U}) = 1)' style groups to $...$."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] != "(":
            out.append(text[i])
            i += 1
            continue
        depth, j = 0, i
        while j < n:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:  # unbalanced — emit as is
            out.append(text[i])
            i += 1
            continue
        inner = text[i + 1:j]
        if _looks_like_math(inner):
            out.append("$" + inner.strip() + "$")
            i = j + 1
        else:
            out.append("(" + _wrap_paren_math(inner) + ")")
            i = j + 1
    return "".join(out)


def _fix_math(expr: str) -> str:
    """Repair common LLM LaTeX slips so KaTeX can render the expression."""
    expr = expr.strip().replace("\u2011", "-").replace("\u00a0", " ")
    expr = re.sub(r"^[,;]\s*(?=[\^_])", "", expr)            # '(,^{238}U)' -> '^{238}U'
    expr = re.sub(r"(?<=[\d\)\}]);(?=\s*\{?\\(?:rm|mathrm|text)\b)", r"\\,", expr)  # '238.0;{\rm u}'
    if expr.startswith(("^", "_")):                           # bare isotope prefix
        expr = "{}" + expr
    return expr


_BARE_LATEX_LINE = re.compile(r"^(\s*)([^\n$]*\\[a-zA-Z]+[^\n$]*)$", re.MULTILINE)


def _wrap_bare_latex_lines(seg: str) -> str:
    """Wrap whole lines that are LaTeX with no prose (e.g. 'Q = \\Delta m c^2')."""
    def repl(m):
        body = m.group(2).strip()
        if re.match(r"([-*+>#]|\d+[.)])\s", body) or not _looks_like_math(body, min_word=3):
            return m.group(0)
        return f"{m.group(1)}$${_fix_math(body)}$$"
    return _BARE_LATEX_LINE.sub(repl, seg)


def _map_plain(text: str, fn) -> str:
    """Apply fn only to text outside code spans and existing math spans."""
    parts = _PROTECTED.split(text)
    for idx in range(0, len(parts), 2):
        parts[idx] = fn(parts[idx])
    return "".join(parts)


def _convert_delimiters(seg: str) -> str:
    seg = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$" + m.group(1).strip() + "$$\n", seg, flags=re.DOTALL)
    return re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", seg, flags=re.DOTALL)


def normalize_math(text: str) -> str:
    """Make LLM output render in Streamlit (needs $...$ / $$...$$ delimiters)."""
    text = text.replace("\u2011", "-")
    # 1. \( \) and \[ \]  ->  $ $ and $$ $$   (outside code / existing math)
    text = _map_plain(text, _convert_delimiters)
    # 2. LaTeX written without delimiters (bare lines, parenthesised groups)
    text = _map_plain(text, _wrap_bare_latex_lines)
    text = _map_plain(text, _wrap_paren_math)
    # 3. Repair the content of every math span
    parts = _PROTECTED.split(text)
    for idx in range(1, len(parts), 2):
        p = parts[idx]
        if p.startswith("$$"):
            parts[idx] = "$$" + _fix_math(p[2:-2]) + "$$"
        elif p.startswith("$"):
            parts[idx] = "$" + _fix_math(p[1:-1]) + "$"
    return "".join(parts)


# ── Safe HTML rendering ────────────────────────────────────────────────────
_ALLOWED_TAGS = {
    "a", "b", "i", "u", "em", "strong", "sub", "sup", "br", "hr", "p", "div", "span",
    "mark", "small", "code", "pre", "blockquote", "ul", "ol", "li", "details", "summary",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td",
}
_ALLOWED_ATTRS = {"a": {"href", "title"}, "td": {"colspan", "rowspan"}, "th": {"colspan", "rowspan"}}


def sanitize_html(text: str) -> str:
    """Allow a safe subset of HTML in Markdown text (math and code untouched).

    Scripts, styles, event handlers, iframes, forms and javascript: URLs are removed.
    """
    import nh3
    parts = _PROTECTED.split(text)
    for idx in range(0, len(parts), 2):
        parts[idx] = nh3.clean(
            parts[idx],
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRS,
            url_schemes={"http", "https", "mailto"},
            link_rel="noopener noreferrer",
        )
    return "".join(parts)


# ── Rate-limit handling ────────────────────────────────────────────────────
def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate_limit" in msg or "rate limit" in msg or "429" in msg or "quota" in msg


def _retry_delay(exc: Exception, default: float) -> float:
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", str(exc), re.IGNORECASE)
    if m:
        return min(float(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0, 30.0)
    return default


def invoke_with_retry(fn, attempts: int = 4):
    """Call fn(); on rate-limit errors wait for the advertised delay and retry."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1 or not is_rate_limit_error(exc):
                raise
            time.sleep(_retry_delay(exc, default=8.0 * (attempt + 1)))


def trim_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _build_llm():
    """Instantiate the LLM based on dynamic configuration."""
    provider = get_config("LLM_PROVIDER", "openai").lower()

    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM
        apikey = get_config("WATSONX_APIKEY") or get_config("WATSONX_API_KEY")
        return WatsonxLLM(
            model_id=get_config("WATSONX_MODEL_ID") or get_config("WATSONX_MODEL", "ibm/granite-3-8b-instruct"),
            url=get_config("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            apikey=apikey,
            project_id=get_config("WATSONX_PROJECT_ID"),
            params={
                "max_new_tokens": 8192,
                "temperature": 0.3,
                "repetition_penalty": 1.1,
            },
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        api_key = get_config("GROQ_API_KEY")
        raw_model = get_config("GROQ_MODEL", "openai/gpt-oss-120b")

        # Map deprecated or alias names to active Groq models
        model_aliases = {
            "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
            "llama-3.1-8b-instant": "openai/gpt-oss-20b",
            "llama3-70b-8192": "openai/gpt-oss-120b",
            "llama3-8b-8192": "openai/gpt-oss-20b",
            "mixtral-8x7b-32768": "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b": "openai/gpt-oss-120b",
        }
        model = model_aliases.get(raw_model, raw_model)

        if not api_key:
            raise ValueError(
                "Missing `GROQ_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or in your `.env` file."
            )
        primary_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=int(get_config("GROQ_MAX_TOKENS", "4096")),
            max_retries=3,
            **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
        )
        fallback_llm = ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=int(get_config("GROQ_MAX_TOKENS", "4096")),
            max_retries=3,
        )
        return primary_llm.with_fallbacks([fallback_llm])
    else:
        from langchain_openai import ChatOpenAI
        api_key = get_config("OPENAI_API_KEY")
        model = get_config("OPENAI_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError(
                "Missing `OPENAI_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or provide it via sidebar."
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.3,
            max_tokens=8192,
        )


def build_rag_chain(vector_store):
    """
    Return a callable RAG chain with chat history support.
    Interface: chain.invoke({"question": str, "chat_history": list[dict]})
    Returns: {"answer": str, "source_documents": list}
    """
    llm = _build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Build LCEL chain (retrieve once, reuse for context + sources)
    chain = (
        RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"][:2000]))
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(x["source_documents"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser() | RunnableLambda(normalize_math)
        )
    )
    return chain


def convert_history(raw_history: list[dict], max_assistant_chars: int = 400) -> list:
    """Convert Streamlit message dicts to LangChain message objects with token-safe compaction."""
    messages = []
    for msg in raw_history:
        content = msg.get("content", "")
        if msg["role"] == "user":
            messages.append(HumanMessage(content=content[:1000]))
        elif msg["role"] == "assistant":
            if len(content) > max_assistant_chars:
                truncated = content[:max_assistant_chars].strip() + " ... [previous response summarized]"
            else:
                truncated = content
            messages.append(AIMessage(content=truncated))
    return messages
