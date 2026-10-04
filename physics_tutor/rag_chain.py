"""
RAG chain — Physics Tutor persona.
Uses LangChain 1.x LCEL pipeline.
Supports OpenAI (gpt-4o) and IBM watsonx.ai (Granite) as LLM backends.
"""
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
prepare for the 2027 board exams. Your teaching style is clear, rigorous, and
exam-focused.

Guidelines:
- Explain concepts step-by-step with derivations where relevant.
- Always relate concepts to NCERT syllabus and CBSE exam patterns.
- When asked for questions, generate them at Medium-to-Highest (HOTS) difficulty.
- For numerical problems, show a complete solution with units at every step.
- Flag important formulas and statements that are frequently asked in boards.
- Use the retrieved context from the student's own study material (notes,
  exemplar, PYQ) to anchor your answers.

Context from study material:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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
                "max_new_tokens": 4096,
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
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=4096,
            max_retries=3,
        )
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
            max_tokens=4096,
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

    # Build LCEL chain
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(
                retriever.invoke(x["question"])
            )),
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


def convert_history(raw_history: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    messages = []
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
