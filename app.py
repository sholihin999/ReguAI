import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from src.embedding.embedding_model import get_embedding_model
from src.llm.llm_client import get_llm
from src.prompts.prompt_templates import build_legal_prompt, build_not_found_answer
from src.retrieval.retrieval_utils import (
    TOP_K_INITIAL,
    MAX_SCORE_THRESHOLD,
    build_search_query,
    is_in_scope,
    rerank_results,
)
from src.utils.config import VECTOR_DB_PATH

load_dotenv()

st.set_page_config(
    page_title="ReguAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Custom CSS
# =========================

# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #f7f8ff;
        --bg-main-2: #eef4ff;
        --bg-sidebar: #f1f5fb;

        --primary: #3157d5;
        --primary-dark: #223b8f;
        --primary-soft: #e9eeff;

        --accent: #7c3aed;
        --accent-soft: #f1eaff;

        --gold: #f6b73c;
        --gold-soft: #fff5df;

        --success: #10b981;
        --danger: #ef4444;

        --text-main: #172033;
        --text-soft: #475569;
        --text-muted: #718096;

        --card: #ffffff;
        --card-soft: #f9fbff;
        --border: #dce4f2;

        --shadow: rgba(41, 55, 120, 0.10);
        --shadow-soft: rgba(41, 55, 120, 0.06);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(124, 58, 237, 0.10), transparent 28%),
            radial-gradient(circle at 85% 5%, rgba(49, 87, 213, 0.13), transparent 30%),
            radial-gradient(circle at 50% 100%, rgba(246, 183, 60, 0.08), transparent 26%),
            linear-gradient(180deg, #fbfcff 0%, var(--bg-main) 45%, #ffffff 100%);
        color: var(--text-main);
    }

    header[data-testid="stHeader"] {
        background: rgba(247, 248, 255, 0.82);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(220, 228, 242, 0.75);
    }

    /* ================= Sidebar ================= */

    div[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #eef4ff 0%, #f7f9fd 55%, #f8fafc 100%);
        border-right: 1px solid var(--border);
    }

    div[data-testid="stSidebar"] > div {
        padding: 2rem 1.25rem 1.5rem 1.25rem;
    }

    .sidebar-brand {
        font-size: 30px;
        font-weight: 800;
        color: var(--primary-dark);
        margin-bottom: 2rem;
        letter-spacing: -0.04em;
    }

    .sidebar-card {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,251,255,0.96) 100%);
        border: 1px solid #d9e4f7;
        border-radius: 20px;
        padding: 20px 20px;
        margin-bottom: 24px;
        box-shadow: 0 14px 34px var(--shadow-soft);
    }

    .sidebar-card-title {
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: var(--primary-dark);
        margin-bottom: 16px;
    }

    .sidebar-card p {
        font-size: 14px;
        line-height: 1.6;
        color: var(--text-soft);
        margin-bottom: 14px;
    }

    .dataset-title {
        font-size: 11px;
        font-weight: 800;
        color: var(--primary);
        text-transform: uppercase;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .sidebar-card ul {
        padding-left: 20px;
        margin-top: 0;
        margin-bottom: 14px;
    }

    .sidebar-card li {
        color: var(--text-soft);
        font-size: 14px;
        margin-bottom: 7px;
        line-height: 1.45;
    }

    .sidebar-card li::marker {
        color: var(--primary);
    }

    .sidebar-note {
        font-size: 12px !important;
        color: var(--text-muted) !important;
        font-style: italic;
        margin-top: 12px !important;
        margin-bottom: 0 !important;
    }

    .example-title {
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: var(--primary-dark);
        margin-bottom: 14px;
        text-transform: uppercase;
    }

    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 54px;
        background: rgba(255, 255, 255, 0.95);
        color: var(--text-main);
        border: 1px solid #d5deee;
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 9px;
        font-size: 14px;
        font-weight: 600;
        line-height: 1.35;
        text-align: center;
        justify-content: center;
        white-space: normal;
        box-shadow: 0 6px 16px rgba(41, 55, 120, 0.055);
        transition: all 0.22s ease;
    }

    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(49, 87, 213, 0.23);
    }

    .reset-area {
        margin-top: 24px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
    }

    /* ================= Main Layout ================= */

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .main-container {
        max-width: 980px;
        margin: 0 auto;
        padding-top: 8px;
    }

    .hero {
        text-align: center;
        padding-top: 10px;
        padding-bottom: 34px;
    }

    .hero-mini {
        display: inline-flex;
        align-items: center;
        gap: 14px;
        justify-content: center;
        margin-bottom: 14px;
    }

    .hero-icon {
        font-size: 34px;
        background:
            linear-gradient(135deg, var(--gold-soft) 0%, #fffaf0 100%);
        border: 1px solid #f8dfaa;
        border-radius: 18px;
        padding: 10px 14px;
        box-shadow: 0 10px 24px rgba(246, 183, 60, 0.18);
    }

    .hero-title {
        font-size: 38px;
        line-height: 1.1;
        font-weight: 800;
        color: var(--primary-dark);
        letter-spacing: -0.05em;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #334155;
        font-weight: 500;
        margin-top: 4px;
    }

    .empty-space {
        height: 42vh;
    }

    /* ================= Chat Bubble ================= */

    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.55rem 0;
    }

    div[data-testid="stChatMessageContent"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dde6f5;
        border-radius: 20px;
        padding: 18px 22px;
        box-shadow: 0 14px 34px rgba(41, 55, 120, 0.07);
    }

    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
        font-size: 15.5px;
        line-height: 1.7;
        color: var(--text-main);
    }

    div[data-testid="stChatMessage"] p {
        margin-bottom: 0.75rem;
    }

    /* Avatar Chat */
    div[data-testid="stChatMessage"] img {
        border-radius: 12px;
    }

    /* ================= Chat Input ================= */

    div[data-testid="stChatInput"] {
        max-width: 760px;
        margin: 0 auto;
        background: transparent !important;
    }

    div[data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid #d8e2f0 !important;
        border-radius: 22px !important;
        box-shadow: 0 14px 34px rgba(41, 55, 120, 0.10) !important;
        padding: 8px 10px !important;
    }

    div[data-testid="stChatInput"] textarea {
        min-height: 54px !important;
        max-height: 160px !important;
        font-size: 15.5px !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--text-main) !important;
        padding: 14px 12px !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stChatInput"] button {
        width: 42px !important;
        height: 42px !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(49, 87, 213, 0.24) !important;
        margin-right: 4px !important;
    }

    section[data-testid="stBottom"] {
        background: transparent !important;
        border-top: none !important;
        padding-bottom: 20px !important;
    }
    /* ================= Sources ================= */

    .source-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 25px;
        font-weight: 800;
        color: var(--primary-dark);
        margin-top: 26px;
        margin-bottom: 18px;
    }

    .source-title span {
        font-size: 25px;
    }

    .source-empty {
        background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 100%);
        border: 1px solid #d4e2f6;
        color: var(--primary-dark);
        padding: 14px 16px;
        border-radius: 14px;
        font-size: 14px;
        font-weight: 500;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #dde6f5;
        border-radius: 15px;
        background: #ffffff;
        box-shadow: 0 7px 18px rgba(41, 55, 120, 0.05);
        margin-bottom: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        background: linear-gradient(90deg, #fbfdff 0%, #f5f8ff 100%);
        color: var(--text-main);
        font-weight: 600;
    }

    .source-meta {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
        border-radius: 12px;
        border: 1px solid #dde6f5;
        padding: 13px 15px;
        margin-bottom: 12px;
        color: #334155;
        line-height: 1.7;
    }

    .source-content {
        background: #ffffff;
        border-left: 4px solid var(--accent);
        padding: 12px 15px;
        border-radius: 10px;
        color: #334155;
        line-height: 1.75;
    }

    hr {
        border-color: var(--border);
        margin-top: 34px;
        margin-bottom: 28px;
    }

    @media screen and (max-width: 768px) {
        .block-container {
            padding-top: 2rem;
        }

        .hero-title {
            font-size: 28px;
        }

        .hero-subtitle {
            font-size: 14px;
        }

        .hero-icon {
            font-size: 28px;
            padding: 8px 10px;
        }

        .empty-space {
            height: 30vh;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load Resources
# =========================


@st.cache_resource
def load_resources():
    """
    Memuat resource utama aplikasi:
    - embedding model
    - FAISS vector database
    - LLM Groq
    """

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    llm = get_llm()

    return vectorstore, llm


vectorstore, llm = load_resources()


# =========================
# Helper Functions
# =========================


def is_definition_question(query: str) -> bool:
    """
    Mengecek apakah pertanyaan termasuk pertanyaan definisi.
    """

    q = query.lower().strip()

    definition_patterns = [
        "apa itu",
        "apa definisi",
        "definisi",
        "pengertian",
        "apa yang dimaksud",
        "yang dimaksud dengan",
    ]

    return any(pattern in q for pattern in definition_patterns)


def get_explicit_definition_status(query: str) -> bool:
    """
    Mengatur definisi mana yang boleh dijawab.

    Tujuannya:
    - Mencegah ReguAI membuat definisi umum di luar dokumen.
    - Hanya menjawab definisi yang memang tersedia di basis data.
    """

    q = query.lower().strip()

    blocked_exact = [
        "apa itu hukum",
        "apa itu hukum?",
        "apa definisi hukum",
        "apa definisi hukum?",
        "definisi hukum",
        "pengertian hukum",
        "apa itu pidana",
        "apa itu pidana?",
        "apa definisi pidana",
        "apa definisi pidana?",
        "definisi pidana",
        "pengertian pidana",
    ]

    if q in blocked_exact:
        return False

    allowed_terms = [
        "data pribadi",
        "subjek data pribadi",
        "pengendali data pribadi",
        "transaksi elektronik",
        "informasi elektronik",
        "dokumen elektronik",
        "sistem elektronik",
        "tindak pidana",
        "pencurian",
        "penipuan",
    ]

    return any(term in q for term in allowed_terms)


def is_too_general_query(query: str) -> bool:
    """
    Mengecek apakah pertanyaan terlalu umum sehingga tidak layak langsung
    dikirim ke LLM.

    Pertanyaan definisi yang tersedia di dokumen seperti:
    - Apa itu data pribadi?
    - Apa yang dimaksud dengan data pribadi?
    - Apa itu transaksi elektronik?
    - Apa yang dimaksud dengan tindak pidana?
    tidak dianggap terlalu umum.
    """

    q_original = query.lower().strip().replace("?", "")
    q = q_original

    allowed_definition_queries = [
        "apa itu data pribadi",
        "apa definisi data pribadi",
        "definisi data pribadi",
        "pengertian data pribadi",
        "apa yang dimaksud data pribadi",
        "apa yang dimaksud dengan data pribadi",
        "apa yang dimaksud dengan data pribadi menurut uu pdp",
        "apa yang dimaksud data pribadi menurut uu pdp",
        "apa itu subjek data pribadi",
        "apa definisi subjek data pribadi",
        "definisi subjek data pribadi",
        "pengertian subjek data pribadi",
        "apa yang dimaksud subjek data pribadi",
        "apa yang dimaksud dengan subjek data pribadi",
        "apa itu pengendali data pribadi",
        "apa definisi pengendali data pribadi",
        "definisi pengendali data pribadi",
        "pengertian pengendali data pribadi",
        "apa pengertian pengendali data pribadi",
        "apa yang dimaksud pengendali data pribadi",
        "apa yang dimaksud dengan pengendali data pribadi",
        "apa itu pemrosesan data pribadi",
        "apa definisi pemrosesan data pribadi",
        "definisi pemrosesan data pribadi",
        "pengertian pemrosesan data pribadi",
        "apa yang dimaksud pemrosesan data pribadi",
        "apa yang dimaksud dengan pemrosesan data pribadi",
        "apa itu transaksi elektronik",
        "apa definisi transaksi elektronik",
        "definisi transaksi elektronik",
        "pengertian transaksi elektronik",
        "apa yang dimaksud transaksi elektronik",
        "apa yang dimaksud dengan transaksi elektronik",
        "apa itu informasi elektronik",
        "apa definisi informasi elektronik",
        "apa yang dimaksud dengan informasi elektronik",
        "apa itu dokumen elektronik",
        "apa definisi dokumen elektronik",
        "apa yang dimaksud dengan dokumen elektronik",
        "apa itu sistem elektronik",
        "apa definisi sistem elektronik",
        "apa yang dimaksud dengan sistem elektronik",
        "apa yang dimaksud dengan tindak pidana",
        "apa yang dimaksud tindak pidana",
        "apa itu tindak pidana",
        "apa definisi tindak pidana",
        "pengertian tindak pidana",
    ]

    if q_original in allowed_definition_queries:
        return False

    definition_prefixes = [
        "apa itu ",
        "apa definisi ",
        "definisi ",
        "pengertian ",
        "apa yang dimaksud dengan ",
        "apa yang dimaksud ",
    ]

    for prefix in definition_prefixes:
        if q.startswith(prefix):
            q = q.replace(prefix, "", 1).strip()

    general_queries = [
        "uu ite",
        "ite",
        "undang undang ite",
        "undang-undang ite",
        "uu informasi dan transaksi elektronik",
        "informasi dan transaksi elektronik",
        "uu pdp",
        "pdp",
        "undang undang pdp",
        "undang-undang pdp",
        "pelindungan data pribadi",
        "perlindungan data pribadi",
        "kuhp",
        "uu kuhp",
        "kitab undang undang hukum pidana",
        "kitab undang-undang hukum pidana",
        "hukum digital",
        "hukum pidana",
        "pidana",
        "pasal",
        "uu",
    ]

    return q in general_queries


def build_too_general_answer(query: str = ""):
    """
    Jawaban standar ketika pertanyaan pengguna terlalu umum.
    """

    q = query.lower().strip()

    if "ite" in q or "transaksi elektronik" in q:
        return """
Pertanyaan masih terlalu umum.

Dalam basis data ReguAI tersedia dokumen UU ITE, tetapi sistem tidak membuat definisi umum sendiri jika tidak ada definisi eksplisit dalam pasal.

Silakan ajukan pertanyaan yang lebih spesifik, misalnya:
- Apa itu transaksi elektronik?
- Apa itu informasi elektronik?
- Apa itu dokumen elektronik?
- Apa itu sistem elektronik?
- Apa sanksi pencemaran nama baik di media sosial?
- Apa hukuman bagi seseorang yang menyalahgunakan sosial media untuk menyebarkan kebohongan?

Catatan:
ReguAI menjawab berdasarkan pasal dan dokumen hukum yang tersedia dalam basis data.
"""

    if (
        "pdp" in q
        or "data pribadi" in q
        or "pelindungan data" in q
        or "perlindungan data" in q
    ):
        return """
Pertanyaan masih terlalu umum.

Dalam basis data ReguAI tersedia dokumen UU Pelindungan Data Pribadi, tetapi sistem hanya menjawab berdasarkan ketentuan yang relevan di dalam pasal.

Silakan ajukan pertanyaan yang lebih spesifik, misalnya:
- Apa itu data pribadi?
- Apa itu subjek data pribadi?
- Apa itu pengendali data pribadi?
- Apa hak subjek data pribadi?
- Apa sanksi penyalahgunaan data pribadi?
- Apa hukuman untuk orang yang menyalahgunakan data orang lain?

Catatan:
ReguAI menjawab berdasarkan pasal dan dokumen hukum yang tersedia dalam basis data.
"""

    if "kuhp" in q or "pidana" in q:
        return """
Pertanyaan masih terlalu umum.

Dalam basis data ReguAI tersedia dokumen KUHP 2023 dan UU Penyesuaian Pidana 2026. Namun, pertanyaan perlu dibuat lebih spesifik agar sistem dapat mencari pasal yang relevan.

Silakan ajukan pertanyaan yang lebih spesifik, misalnya:
- Apa yang dimaksud dengan tindak pidana?
- Apa sanksi pencurian?
- Apa sanksi penipuan?
- Kategori II itu dendanya berapa?

Catatan:
ReguAI menjawab berdasarkan pasal dan dokumen hukum yang tersedia dalam basis data.
"""

    return """
Pertanyaan masih terlalu umum.

Silakan ajukan pertanyaan yang lebih spesifik, misalnya:
- Apa itu transaksi elektronik?
- Apa sanksi pencemaran nama baik di media sosial?
- Apa sanksi penyalahgunaan data pribadi?
- Apa hukuman untuk orang yang menyalahgunakan data orang lain?
- Apa yang dimaksud dengan tindak pidana?
- Kategori II itu dendanya berapa?

Catatan:
ReguAI menjawab berdasarkan pasal dan dokumen hukum yang tersedia dalam basis data.
"""


def is_unsafe_legal_strategy_query(query: str) -> bool:
    """
    Mengecek pertanyaan yang meminta strategi untuk menghindari hukuman,
    lolos dari tanggung jawab hukum, atau menghindari proses hukum.
    """

    q = query.lower().strip()

    unsafe_patterns = [
        "cara menghindari hukuman",
        "bagaimana cara menghindari hukuman",
        "menghindari hukuman",
        "agar tidak dihukum",
        "supaya tidak dihukum",
        "biar tidak dihukum",
        "biar gak dihukum",
        "biar tidak kena hukuman",
        "biar gak kena hukuman",
        "biar lolos",
        "agar lolos",
        "supaya lolos",
        "cara lolos dari hukuman",
        "cara lolos dari jerat hukum",
        "cara menghindari jerat hukum",
        "cara menghindari polisi",
        "cara menghindari pidana",
        "cara bebas dari hukuman",
        "tips menghindari hukuman",
        "trik menghindari hukuman",
        "bagaimana supaya tidak dipidana",
        "bagaimana agar tidak dipidana",
        "supaya tidak dipidana",
        "agar tidak dipidana",
    ]

    return any(pattern in q for pattern in unsafe_patterns)


def build_unsafe_legal_strategy_answer():
    """
    Jawaban standar untuk pertanyaan yang meminta strategi menghindari hukuman.
    """

    return """
ReguAI tidak memberikan panduan untuk menghindari hukuman, menghindari proses hukum, atau lolos dari tanggung jawab hukum.

Namun, ReguAI dapat membantu menjelaskan ketentuan hukum yang tersedia dalam basis data secara informatif. Silakan ajukan pertanyaan seperti:
- Apa sanksi pencemaran nama baik di media sosial?
- Apa hukuman jika menyebarkan kebohongan di media sosial?
- Apa sanksi penyalahgunaan data pribadi?
- Apa sanksi pencurian?
- Apa sanksi penipuan?

Catatan:
Jawaban ReguAI bersifat informatif, berdasarkan dokumen hukum yang tersedia, dan bukan merupakan nasihat hukum resmi.
"""


def build_llm_error_answer():
    """
    Jawaban standar ketika retrieval berhasil tetapi pemanggilan LLM gagal.
    """

    return """
Maaf, terjadi kendala saat menghubungi layanan LLM.

Silakan coba beberapa saat lagi atau gunakan pertanyaan yang lebih spesifik.

Catatan:
Hasil retrieval sumber hukum berhasil dilakukan, tetapi proses penyusunan jawaban oleh model bahasa gagal.
"""


def retrieve_documents(query: str):
    """
    Melakukan retrieval dokumen dari FAISS.
    """

    if not is_in_scope(query):
        return []

    wide_retrieval_terms = [
        "hukuman",
        "menyalahgunakan",
        "penyalahgunaan",
        "data orang lain",
        "data pribadi orang lain",
        "sosial media",
        "media sosial",
        "kebohongan",
        "berita bohong",
        "hoaks",
        "hoax",
        "informasi bohong",
        "informasi palsu",
        "informasi menyesatkan",
        "menyesatkan",
        "mencuri",
        "nyuri",
        "curi",
        "mengambil barang",
        "ngambil barang",
        "barang orang lain",
    ]

    q = query.lower().strip()

    if is_definition_question(query) or any(term in q for term in wide_retrieval_terms):
        retrieval_k = 100
    else:
        retrieval_k = TOP_K_INITIAL

    initial_results = vectorstore.similarity_search_with_score(
        build_search_query(query),
        k=retrieval_k,
    )

    if not initial_results:
        return []

    reranked_results = rerank_results(query, initial_results)

    if reranked_results:
        best_keyword_score = float(reranked_results[0][2])

        if best_keyword_score > 0:
            return reranked_results

    best_score = float(initial_results[0][1])

    if best_score > MAX_SCORE_THRESHOLD:
        return []

    return reranked_results


def save_assistant_message(answer: str):
    """
    Menyimpan jawaban assistant ke session state.
    """

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


def generate_answer(user_query: str, retrieved):
    """
    Membuat jawaban berdasarkan hasil retrieval.
    """

    if not retrieved:
        answer = build_not_found_answer()
        return answer, []

    docs = [item[0] for item in retrieved]

    if is_definition_question(user_query):
        allowed_definition = get_explicit_definition_status(user_query)

        if not allowed_definition:
            answer = build_not_found_answer()
            return answer, []

        docs_for_answer = docs[:1]
        prompt = build_legal_prompt(user_query, docs_for_answer)

        try:
            response = llm.invoke(prompt)
            return response.content, retrieved[:1]
        except Exception:
            return build_llm_error_answer(), retrieved[:1]

    prompt = build_legal_prompt(user_query, docs)

    try:
        response = llm.invoke(prompt)
        return response.content, retrieved
    except Exception:
        return build_llm_error_answer(), retrieved


# =========================
# Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# =========================
# Sidebar
# =========================

st.sidebar.markdown('<div class="sidebar-brand">ReguAI</div>', unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <div class="sidebar-card-title">TENTANG REGUAI</div>
        <p>ReguAI adalah chatbot informasi hukum berbasis RAG.</p>
        <div class="dataset-title">DATASET:</div>
        <ul>
            <li>UU PDP 2022</li>
            <li>UU ITE 2008</li>
            <li>UU ITE 2016</li>
            <li>UU ITE 2024</li>
            <li>KUHP 2023</li>
            <li>UU Penyesuaian Pidana 2026</li>
        </ul>
        <p class="sidebar-note">Jawaban bersifat informatif dan bukan nasihat hukum resmi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="example-title">CONTOH PERTANYAAN</div>', unsafe_allow_html=True
)

examples = [
    "Apa sanksi penyalahgunaan data pribadi?",
    "Apa hukuman untuk pencemaran nama baik?",
    "Kategori II itu dendanya berapa?",
    "Apa hak subjek data pribadi?",
    "Apa sanksi doxing?",
]

for idx, example in enumerate(examples):
    if st.sidebar.button(example, key=f"example_{idx}"):
        st.session_state.pending_query = example
        st.rerun()

st.sidebar.markdown('<div class="reset-area"></div>', unsafe_allow_html=True)

if st.sidebar.button("⟳  Reset Chat", key="reset_chat"):
    st.session_state.messages = []
    st.session_state.sources = []
    st.session_state.pending_query = None
    st.rerun()


# =========================
# Main UI
# =========================

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-mini">
            <span class="hero-icon">⚖️</span>
            <span class="hero-title">Selamat Datang di ReguAI</span>
        </div>
        <p class="hero-subtitle">
            Chatbot Hukum Digital dan Hukum Pidana Berbasis Retrieval-Augmented Generation (RAG)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def process_query(user_query: str):
    """
    Memproses pertanyaan pengguna dan menampilkan jawaban.
    """

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if is_too_general_query(user_query):
            answer = build_too_general_answer(user_query)
            st.markdown(answer)
            save_assistant_message(answer)
            st.session_state.sources = []

        elif is_unsafe_legal_strategy_query(user_query):
            answer = build_unsafe_legal_strategy_answer()
            st.markdown(answer)
            save_assistant_message(answer)
            st.session_state.sources = []

        else:
            with st.spinner("Mencari dasar hukum dan menyusun jawaban..."):
                retrieved = retrieve_documents(user_query)
                answer, sources = generate_answer(user_query, retrieved)

                st.markdown(answer)
                save_assistant_message(answer)
                st.session_state.sources = sources


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.markdown('<div class="empty-space"></div>', unsafe_allow_html=True)

if st.session_state.pending_query:
    query_from_button = st.session_state.pending_query
    st.session_state.pending_query = None
    process_query(query_from_button)
    st.rerun()

user_query = st.chat_input("Tanyakan persoalan hukum digital atau pidana...")

if user_query:
    process_query(user_query)
    st.rerun()


# =========================
# Sources
# =========================

if st.session_state.messages:
    st.divider()

    st.markdown(
        '<div class="source-title"><span>📚</span> Sumber yang Digunakan</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.sources:
        for i, (doc, distance_score, k_score, combined_score) in enumerate(
            st.session_state.sources,
            1,
        ):
            nama_dokumen = doc.metadata.get("nama_dokumen", "-")
            pasal = doc.metadata.get("pasal", "-")
            domain = doc.metadata.get("domain", "-")
            tahun = doc.metadata.get("tahun", "-")
            clean_content = doc.page_content.replace("passage: ", "")

            with st.expander(f"Sumber {i}: Pasal {pasal} - {nama_dokumen}"):
                st.markdown(
                    f"""
                    <div class="source-meta">
                        <b>Domain:</b> {domain}<br>
                        <b>Tahun:</b> {tahun}<br>
                        <b>Dokumen:</b> {nama_dokumen}<br>
                        <b>Pasal:</b> {pasal}<br>
                        <b>Distance Score:</b> {distance_score:.4f}<br>
                        <b>Keyword Score:</b> {k_score:.2f}<br>
                        <b>Final Score:</b> {combined_score:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Isi potongan dokumen:**")
                st.markdown(
                    f"""
                    <div class="source-content">
                        {clean_content[:2500]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="source-empty">
                Sumber akan muncul setelah sistem menemukan dokumen hukum yang relevan.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)
