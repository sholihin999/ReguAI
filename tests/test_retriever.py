from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading FAISS...")

vectorstore = FAISS.load_local(
    "vector_db/reguai_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "query: Apa saja hak subjek data pribadi menurut UU PDP?"

results = vectorstore.similarity_search(
    query,
    k=5
)

print("\nHASIL RETRIEVAL\n")

for i, doc in enumerate(results, 1):
    print("=" * 80)
    print(f"Ranking {i}")
    print("Pasal:", doc.metadata["pasal"])
    print("Dokumen:", doc.metadata["nama_dokumen"])
    print()
    print(doc.page_content[:1000])