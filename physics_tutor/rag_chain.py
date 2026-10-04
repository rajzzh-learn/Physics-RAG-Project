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
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
    WATSONX_MODEL,
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
    """Instantiate the LLM based on LLM_PROVIDER env variable."""
    if LLM_PROVIDER == "watsonx":
        from langchain_ibm import WatsonxLLM
        return WatsonxLLM(
            model_id=WATSONX_MODEL,
            url=WATSONX_URL,
            apikey=WATSONX_API_KEY,
            project_id=WATSONX_PROJECT_ID,
            params={
                "max_new_tokens": 1024,
                "temperature": 0.3,
                "repetition_penalty": 1.1,
            },
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1024,
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
