# ReguAI

ReguAI adalah chatbot informasi hukum Indonesia berbasis **Retrieval-Augmented Generation (RAG)**. Aplikasi ini membantu pengguna mencari informasi seputar **Hukum Digital** dan **Hukum Pidana** berdasarkan dokumen hukum resmi.

## Demo
https://reguaiuasnlp.streamlit.app/

## Dataset

Dataset yang digunakan berupa dokumen hukum resmi dalam format PDF, yaitu:

1. UU Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi
2. UU Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik
3. UU Nomor 19 Tahun 2016 tentang Perubahan atas UU ITE
4. UU Nomor 1 Tahun 2024 tentang Perubahan Kedua atas UU ITE
5. UU Nomor 1 Tahun 2023 tentang Kitab Undang-Undang Hukum Pidana
6. UU Nomor 1 Tahun 2026 tentang Penyesuaian Pidana

## Teknologi

- Python
- Streamlit
- LangChain
- FAISS
- PyMuPDF
- Hugging Face Embedding: `intfloat/multilingual-e5-base`
- Groq LLM: `llama-3.1-8b-instant`

## Alur Sistem

```text
PDF Dokumen Hukum
→ Ekstraksi Teks
→ Cleaning
→ Chunking Berbasis Pasal
→ Embedding
→ FAISS Vector Database
→ Retrieval
→ Reranking
→ LLM
→ Jawaban + Sumber Hukum
