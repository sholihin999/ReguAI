# ReguAI

ReguAI adalah chatbot informasi hukum Indonesia berbasis Retrieval-Augmented Generation (RAG).

## Dataset

- UU Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi
- UU Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik
- UU Nomor 19 Tahun 2016 tentang Perubahan atas UU ITE
- UU Nomor 1 Tahun 2024 tentang Perubahan Kedua atas UU ITE
- UU Nomor 1 Tahun 2023 tentang KUHP
- UU Nomor 1 Tahun 2026 tentang Penyesuaian Pidana

## Arsitektur

PDF → Text Extraction → Chunking per Pasal → Embedding → FAISS Vector Database → Retrieval → Reranking → LLM Groq → Streamlit UI

## Cara Menjalankan

Install dependency:

```bash
pip install -r requirements.txt