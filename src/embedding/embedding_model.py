from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Membuat model embedding yang digunakan oleh ReguAI.

    Model:
    - intfloat/multilingual-e5-base

    normalize_embeddings=True digunakan agar similarity search FAISS
    lebih stabil untuk pencarian semantik.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )