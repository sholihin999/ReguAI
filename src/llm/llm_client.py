from langchain_groq import ChatGroq

from src.utils.config import MODEL_NAME, LLM_TEMPERATURE


def get_llm():
    """
    Membuat client LLM untuk ReguAI.

    Model:
    - Groq
    - llama-3.1-8b-instant

    Temperature dibuat 0 agar jawaban lebih konsisten,
    karena ReguAI menjawab berdasarkan dokumen hukum.
    """

    return ChatGroq(
        model=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )
