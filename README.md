# ReguAI ⚖️

ReguAI adalah chatbot informasi hukum Indonesia berbasis **Retrieval-Augmented Generation (RAG)**. Aplikasi ini membantu pengguna mencari informasi seputar **Hukum Digital** dan **Hukum Pidana** berdasarkan dokumen hukum resmi.

> **Disclaimer:** Jawaban ReguAI bersifat informatif dan bukan merupakan nasihat hukum resmi.

## Demo

🔗 https://reguaiuasnlp.streamlit.app/

## Fitur

- Tanya jawab seputar hukum digital dan hukum pidana dalam bahasa Indonesia.
- Jawaban disusun hanya dari pasal yang ditemukan di basis data dokumen hukum.
- Menampilkan sumber yang digunakan (dokumen, pasal, tahun, dan skor relevansi).
- Penanganan pertanyaan di luar cakupan, terlalu umum, atau meminta strategi menghindari hukuman.

## Dataset

Dataset berupa dokumen hukum resmi dalam format PDF:

| No | Dokumen |
|----|---------|
| 1 | UU Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi |
| 2 | UU Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik |
| 3 | UU Nomor 19 Tahun 2016 tentang Perubahan atas UU ITE |
| 4 | UU Nomor 1 Tahun 2024 tentang Perubahan Kedua atas UU ITE |
| 5 | UU Nomor 1 Tahun 2023 tentang Kitab Undang-Undang Hukum Pidana |
| 6 | UU Nomor 1 Tahun 2026 tentang Penyesuaian Pidana |

## Teknologi

| Komponen | Teknologi |
|----------|-----------|
| Bahasa | Python |
| Antarmuka | Streamlit |
| Framework RAG | LangChain |
| Vector database | FAISS |
| Ekstraksi PDF | PyMuPDF |
| Embedding | Hugging Face `intfloat/multilingual-e5-base` |
| LLM | Groq `openai/gpt-oss-120b` |

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
```

## Struktur Proyek

```text
ReguAI/
├── app.py            # Aplikasi Streamlit
├── data/             # Dokumen hukum
├── vector_db/        # Index FAISS (reguai_faiss)
├── src/
│   ├── api/
│   ├── chunking/     # Chunking berbasis pasal
│   ├── embedding/    # Model embedding
│   ├── evaluation/   # Evaluasi sistem
│   ├── ingestion/    # Pemrosesan dokumen
│   ├── llm/          # Client LLM (Groq)
│   ├── prompts/      # Template prompt
│   ├── retrieval/    # Retrieval dan reranking
│   └── utils/        # Konfigurasi
└── requirements.txt
```

## Menjalankan Secara Lokal

1. Clone repositori dan masuk ke foldernya:

   ```bash
   git clone https://github.com/sholihin999/ReguAI.git
   cd ReguAI
   ```

2. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Buat file `.env` di root proyek dan isi API key Groq:

   ```env
   GROQ_API_KEY=isi_api_key_kamu
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

## Konfigurasi

Pengaturan model ada di `src/utils/config.py`:

```python
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0
```

Saat deploy di Streamlit Cloud, isi `GROQ_API_KEY` melalui menu **Settings → Secrets**, karena file `.env` tidak ikut ter-deploy.

> **Catatan:** Groq secara berkala menghentikan model lama. Jika muncul error `model_not_found`, periksa daftar model aktif di [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations) dan perbarui `MODEL_NAME`.

## Batasan

- Cakupan terbatas pada enam dokumen hukum di atas.
- Sistem tidak membuat definisi umum di luar dokumen.
- Jawaban tidak menggantikan konsultasi dengan ahli hukum.
