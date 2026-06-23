from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local(
    "vector_db/reguai_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

query = "Apa sanksi penyalahgunaan data pribadi?"

docs = vectorstore.similarity_search(
    "query: " + query,
    k=5
)

context = "\n\n".join([
    f"Sumber: {doc.metadata['nama_dokumen']} Pasal {doc.metadata['pasal']}\n{doc.page_content}"
    for doc in docs
])

prompt = f"""
Kamu adalah asisten informasi hukum Indonesia.

Jawab pertanyaan pengguna hanya berdasarkan konteks hukum berikut.
Jangan mengarang.
Jika informasi tidak ada di konteks, katakan bahwa informasi tidak ditemukan dalam dokumen.

Konteks:
{context}

Pertanyaan:
{query}

Format jawaban:
Jawaban singkat:
Dasar hukum:
Pasal terkait:
Catatan:
Jawaban ini bersifat informatif dan bukan nasihat hukum resmi.
"""

response = llm.invoke(prompt)

print(response.content)