from pathlib import Path


# Root project ReguAI
BASE_DIR = Path(__file__).resolve().parents[2]


# Path utama
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db" / "reguai_faiss"
VECTOR_DB_PATH = str(VECTOR_DB_DIR)


# Model
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0


# Retrieval
TOP_K_DEFINITION = 100