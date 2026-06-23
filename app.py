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

st.set_page_config(page_title="ReguAI", page_icon="⚖️", layout="wide")


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

    Contoh:
    - "uu ite"
    - "apa itu uu ite"
    - "kuhp"
    - "apa itu uu pdp"
    """

    q = query.lower().strip().replace("?", "")

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
        "data pribadi",
        "tindak pidana",
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

    Catatan:
    Reranking dilakukan sebelum threshold final agar query natural seperti
    "hukuman menyalahgunakan sosial media untuk menyebarkan kebohongan"
    tidak langsung ditolak hanya karena distance semantic awal cukup besar.
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

    Untuk pertanyaan definisi:
    - Sistem hanya menggunakan dokumen ranking pertama.
    - Sumber yang ditampilkan juga hanya ranking pertama agar konsisten.

    Untuk pertanyaan non-definisi:
    - Sistem menggunakan seluruh hasil reranking.

    Fungsi ini juga diberi try-except agar aplikasi tidak crash jika Groq error.
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
# UI Streamlit
# =========================

st.title("⚖️ ReguAI")
st.caption(
    "Chatbot Hukum Digital dan Hukum Pidana Indonesia berbasis Retrieval-Augmented Generation"
)

st.sidebar.title("Tentang ReguAI")
st.sidebar.info("""
ReguAI adalah chatbot informasi hukum berbasis RAG.

Dataset:
- UU PDP 2022
- UU ITE 2008
- UU ITE 2016
- UU ITE 2024
- KUHP 2023
- UU Penyesuaian Pidana 2026

Jawaban bersifat informatif dan bukan nasihat hukum resmi.
""")

st.sidebar.markdown("### Contoh Pertanyaan")

examples = [
    "Apa sanksi penyalahgunaan data pribadi?",
    "Apa hukuman untuk orang yang menyalahgunakan data orang lain?",
    "Apa sanksi pencemaran nama baik di media sosial?",
    "Apa hukuman bagi seseorang yang menyalahgunakan sosial media untuk menyebarkan kebohongan?",
    "Kategori II itu dendanya berapa?",
    "Apa yang dimaksud dengan tindak pidana?",
    "Apa hak subjek data pribadi?",
    "Apa sanksi doxing?",
]

for example in examples:
    st.sidebar.code(example)

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.sources = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Tanyakan persoalan hukum digital atau pidana...")

if user_query:
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

        else:
            with st.spinner("Mencari dasar hukum dan menyusun jawaban..."):
                retrieved = retrieve_documents(user_query)
                answer, sources = generate_answer(user_query, retrieved)

                st.markdown(answer)
                save_assistant_message(answer)
                st.session_state.sources = sources


st.divider()

st.subheader("📚 Sumber yang Digunakan")

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
            st.write(f"**Domain:** {domain}")
            st.write(f"**Tahun:** {tahun}")
            st.write(f"**Dokumen:** {nama_dokumen}")
            st.write(f"**Pasal:** {pasal}")
            st.write(f"**Distance Score:** {distance_score:.4f}")
            st.write(f"**Keyword Score:** {k_score:.2f}")
            st.write(f"**Final Score:** {combined_score:.4f}")
            st.markdown("**Isi potongan dokumen:**")
            st.write(clean_content[:2500])
else:
    st.info("Sumber akan muncul setelah kamu mengajukan pertanyaan.")
