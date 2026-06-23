import fitz
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.chunking.pasal_chunker import split_by_pasal
from src.embedding.embedding_model import get_embedding_model
from src.utils.config import DATA_DIR, VECTOR_DB_PATH


PDF_FILES = [
    {
        "path": DATA_DIR / "UU_PDP_2022.pdf",
        "nama_dokumen": "UU Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi",
        "domain": "Hukum Digital",
        "tahun": "2022",
    },
    {
        "path": DATA_DIR / "UU_ITE_2008.pdf",
        "nama_dokumen": "UU Nomor 11 Tahun 2008 tentang Informasi dan Transaksi Elektronik",
        "domain": "Hukum Digital",
        "tahun": "2008",
    },
    {
        "path": DATA_DIR / "UU_ITE_2016.pdf",
        "nama_dokumen": "UU Nomor 19 Tahun 2016 tentang Perubahan atas UU ITE",
        "domain": "Hukum Digital",
        "tahun": "2016",
    },
    {
        "path": DATA_DIR / "UU_ITE_2024.pdf",
        "nama_dokumen": "UU Nomor 1 Tahun 2024 tentang Perubahan Kedua atas UU ITE",
        "domain": "Hukum Digital",
        "tahun": "2024",
    },
    {
        "path": DATA_DIR / "KUHP_2023.pdf",
        "nama_dokumen": "UU Nomor 1 Tahun 2023 tentang Kitab Undang-Undang Hukum Pidana",
        "domain": "Hukum Pidana",
        "tahun": "2023",
    },
    {
        "path": DATA_DIR / "UU_PENYESUAIAN_PIDANA_2026.pdf",
        "nama_dokumen": "UU Nomor 1 Tahun 2026 tentang Penyesuaian Pidana",
        "domain": "Hukum Pidana",
        "tahun": "2026",
    },
]


def extract_text(pdf_path: Path) -> str:
    """
    Mengekstrak teks dari PDF menggunakan PyMuPDF.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text() + "\n"

    doc.close()
    return text


def clean_text(text: str) -> str:
    """
    Membersihkan teks hasil ekstraksi PDF dari noise umum:
    - kesalahan OCR ringan,
    - header/footer,
    - nomor halaman,
    - bagian penjelasan/lampiran yang tidak dipakai sebagai sumber utama.
    """

    replacements = {
        "Pasa1": "Pasal",
        "PasaI": "Pasal",
        "PasaJ": "Pasal",
        "Pasa722": "Pasal 22",
        "Pasal2T": "Pasal 27",
        "Pasal 7O": "Pasal 70",
        "Pasal 6O5": "Pasal 605",
        "Pasal 6O9": "Pasal 609",
        "Pasal 7L": "Pasal 71",
        "Pdsal": "Pasal",
        "Pasal T2": "Pasal 72",
        "Pasal I": "Pasal 1",
        "Pasal 1X": "Pasal IX",
        "Pasal 1II": "Pasal III",
        "PRESIOEN": "PRESIDEN",
        "PRESlDEN": "PRESIDEN",
        "FRESIDEN": "PRESIDEN",
        "REPUELIK": "REPUBLIK",
        "REPUBUK": "REPUBLIK",
        "REPI.JBLIK": "REPUBLIK",
        "REPIIBLIK": "REPUBLIK",
        "REPUEUK": "REPUBLIK",
        "INOONESIA": "INDONESIA",
        "INDONESTA": "INDONESIA",
        "INDONES]A": "INDONESIA",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    split_markers = [
        "PENJELASAN",
        "PENJEI,ASAN",
        "TAMBAHAN LEMBARAN NEGARA",
        "LAMPIRAN I",
        "DAFTAR PERUBAHAN KETENTUAN PIDANA",
        "Agar setiap orang mengetahuinya",
    ]

    for marker in split_markers:
        if marker in text:
            text = text.split(marker)[0]

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("SK No"):
            continue

        if line in [
            "PRESIDEN",
            "REPUBLIK INDONESIA",
            "PRESIDEN REPUBLIK INDONESIA",
        ]:
            continue

        if re.match(r"^-?\s*\d+\s*-?$", line):
            continue

        if line.lower().startswith("salinan"):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text


def build_all_chunks():
    """
    Pipeline:
    PDF -> extract_text -> clean_text -> split_by_pasal -> list chunk dengan metadata.
    """

    all_chunks = []

    for pdf in PDF_FILES:
        print(f"\nMemproses: {pdf['path']}")

        raw_text = extract_text(pdf["path"])
        cleaned_text = clean_text(raw_text)
        pasal_chunks = split_by_pasal(cleaned_text)

        print(f"Jumlah chunk dari dokumen ini: {len(pasal_chunks)}")

        for chunk in pasal_chunks:
            all_chunks.append(
                {
                    "nama_dokumen": pdf["nama_dokumen"],
                    "domain": pdf["domain"],
                    "tahun": pdf["tahun"],
                    "pasal": chunk["pasal"],
                    "text": chunk["text"],
                }
            )

    return all_chunks


def build_documents(chunks):
    """
    Mengubah chunk pasal menjadi Document LangChain dengan metadata.
    """

    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content="passage: " + chunk["text"],
                metadata={
                    "nama_dokumen": chunk["nama_dokumen"],
                    "domain": chunk["domain"],
                    "tahun": chunk["tahun"],
                    "pasal": chunk["pasal"],
                },
            )
        )

    return documents


def preview_chunks(chunks, preview_count: int = 5):
    """
    Menampilkan beberapa chunk pertama dan terakhir untuk pengecekan manual.
    """

    print(f"\nTotal chunk: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError(
            "Tidak ada chunk yang berhasil dibuat. Cek nama file PDF atau pola Pasal."
        )

    print(f"\n{preview_count} chunk pertama:")
    for chunk in chunks[:preview_count]:
        print("=" * 80)
        print("Dokumen:", chunk["nama_dokumen"])
        print("Pasal:", chunk["pasal"])
        print(chunk["text"][:500])

    print(f"\n{preview_count} chunk terakhir:")
    for chunk in chunks[-preview_count:]:
        print("=" * 80)
        print("Dokumen:", chunk["nama_dokumen"])
        print("Pasal:", chunk["pasal"])
        print(chunk["text"][:500])


def create_vector_db(documents):
    """
    Membuat embedding dan menyimpan FAISS vector database.
    """

    print("\nMembuat embedding...")

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    print(f"\nVector DB berhasil disimpan di {VECTOR_DB_PATH}")


def main():
    chunks = build_all_chunks()
    preview_chunks(chunks)

    documents = build_documents(chunks)
    create_vector_db(documents)


if __name__ == "__main__":
    main()