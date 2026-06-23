import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="ReguAI", page_icon="⚖️", layout="wide")

st.title("⚖️ ReguAI")
st.caption("Chatbot Hukum Digital dan Hukum Pidana Indonesia berbasis RAG")

st.sidebar.title("Tentang ReguAI")
st.sidebar.info("""
    ReguAI adalah chatbot informasi hukum berbasis
    Retrieval-Augmented Generation (RAG).

    Dataset saat ini:
    - UU PDP 2022
    - UU ITE 2024
    - KUHP 2023
    - UU Penyesuaian Pidana 2026

    Jawaban bersifat informatif dan bukan nasihat hukum resmi.
    """)

st.sidebar.markdown("### Contoh Pertanyaan")
example_questions = [
    "Apa sanksi penyalahgunaan data pribadi?",
    "Apa yang dimaksud dengan tindak pidana?",
    "Apa sanksi pencemaran nama baik di media sosial?",
    "Apa perbedaan hukum tertulis dan hukum tidak tertulis?",
]

for q in example_questions:
    st.sidebar.code(q)


@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        "vector_db/reguai_faiss", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    return vectorstore, llm


vectorstore, llm = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []


def build_prompt(query, docs):
    context = "\n\n".join([f"""
Sumber:
Dokumen: {doc.metadata.get("nama_dokumen")}
Pasal: {doc.metadata.get("pasal")}
Isi:
{doc.page_content}
""" for doc in docs])

    prompt = f"""
Kamu adalah ReguAI, asisten informasi hukum Indonesia.

Tugasmu:
1. Jawab pertanyaan pengguna hanya berdasarkan konteks hukum yang diberikan.
2. Jangan mengarang pasal, sanksi, tahun, atau nama undang-undang.
3. Jika konteks tidak cukup, katakan bahwa informasi tidak ditemukan dalam dokumen yang tersedia.
4. Gunakan bahasa Indonesia yang jelas dan mudah dipahami.
5. Jangan menyebut sumber yang tidak ada di konteks.
6. Jangan memberikan nasihat hukum resmi.

Konteks hukum:
{context}

Pertanyaan pengguna:
{query}

Format jawaban wajib:

Jawaban singkat:
...

Dasar hukum:
...

Pasal terkait:
...

Catatan:
Jawaban ini bersifat informatif dan bukan nasihat hukum resmi.
"""
    return prompt


def retrieve_documents(query, k=5):
    search_query = "query: " + query
    results = vectorstore.similarity_search_with_score(search_query, k=k)
    return results


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Tanyakan persoalan hukum digital atau pidana...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Mencari pasal relevan dan menyusun jawaban..."):
            retrieved = retrieve_documents(user_query, k=5)
            docs = [item[0] for item in retrieved]

            prompt = build_prompt(user_query, docs)
            response = llm.invoke(prompt)
            answer = response.content

            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.session_state.sources = retrieved

st.divider()

st.subheader("📚 Sumber yang Digunakan")

if st.session_state.sources:
    for i, (doc, score) in enumerate(st.session_state.sources, 1):
        nama_dokumen = doc.metadata.get("nama_dokumen", "-")
        pasal = doc.metadata.get("pasal", "-")
        domain = doc.metadata.get("domain", "-")
        tahun = doc.metadata.get("tahun", "-")

        with st.expander(
            f"Sumber {i}: Pasal {pasal} - {nama_dokumen} | Score: {score:.4f}"
        ):
            st.write(f"**Domain:** {domain}")
            st.write(f"**Tahun:** {tahun}")
            st.write(f"**Dokumen:** {nama_dokumen}")
            st.write(f"**Pasal:** {pasal}")
            st.markdown("**Isi potongan dokumen:**")
            st.write(doc.page_content.replace("passage: ", "")[:2000])
else:
    st.info("Sumber akan muncul setelah kamu mengajukan pertanyaan.")

if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.sources = []
    st.rerun()
