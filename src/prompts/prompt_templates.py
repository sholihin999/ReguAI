def format_legal_context(docs):
    """
    Mengubah hasil retrieval menjadi konteks hukum yang siap dikirim ke LLM.

    Setiap dokumen berisi:
    - nama dokumen
    - domain
    - tahun
    - pasal
    - isi potongan dokumen
    """

    context_parts = []

    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"""
[Sumber {i}]
Dokumen: {doc.metadata.get("nama_dokumen")}
Domain: {doc.metadata.get("domain")}
Tahun: {doc.metadata.get("tahun")}
Pasal: {doc.metadata.get("pasal")}
Isi:
{doc.page_content.replace("passage: ", "")}
"""
        )

    return "\n\n".join(context_parts)


def build_legal_prompt(query, docs):
    """
    Prompt utama ReguAI untuk menjawab pertanyaan hukum berbasis RAG.

    Prompt ini membatasi LLM agar:
    - hanya menjawab berdasarkan konteks hasil retrieval,
    - tidak membuat definisi sendiri,
    - tidak mengarang pasal atau angka sanksi,
    - selalu menampilkan dasar hukum dan pasal terkait.
    """

    context = format_legal_context(docs)

    return f"""
Kamu adalah ReguAI, asisten informasi hukum Indonesia.

Aturan wajib:
1. Jawab HANYA berdasarkan konteks hukum yang diberikan.
2. Jangan menggunakan pengetahuan umum di luar konteks.
3. Jangan membuat definisi sendiri.
4. Jangan mengarang pasal, sanksi, tahun, nama undang-undang, atau angka denda.
5. Jangan menyebut pasal yang tidak ada dalam konteks.
6. Jika konteks tidak cukup untuk menjawab, tulis persis:
   "Informasi tidak ditemukan dalam dokumen yang tersedia."
7. Gunakan bahasa Indonesia yang jelas dan mudah dipahami.
8. Jangan memberi nasihat hukum resmi.
9. Untuk pertanyaan definisi atau "apa itu", jawab hanya jika definisi tersebut tersedia dalam konteks.
10. Jika pertanyaan meminta definisi tetapi konteks hanya berisi pasal sanksi atau pasal umum, jangan menyimpulkan definisi sendiri.
11. Untuk pertanyaan tentang sanksi, prioritaskan pasal yang memuat kata "dipidana", "pidana penjara", "pidana denda", atau "dilarang".
12. Untuk pertanyaan tentang penyalahgunaan data pribadi, prioritaskan UU PDP Pasal 65 sampai Pasal 69 jika ada dalam konteks.
13. Untuk pertanyaan tentang pencemaran nama baik di media sosial, prioritaskan UU ITE jika ada dalam konteks.
14. Untuk pertanyaan tentang hak subjek data pribadi, prioritaskan UU PDP Pasal 5 sampai Pasal 13 jika ada dalam konteks.
15. Untuk pertanyaan tentang definisi tindak pidana, prioritaskan KUHP Pasal 12 jika ada dalam konteks.
16. Untuk rumusan delik seperti pencurian atau penipuan, boleh menjelaskan unsur perbuatannya hanya dari pasal yang tersedia.
17. Jika isi konteks memuat daftar bernomor seperti "1.", "2.", atau "3.", jangan menganggap nomor tersebut sebagai nomor pasal. Nomor pasal resmi hanya boleh diambil dari metadata "Pasal:" pada setiap sumber.
18. Jika definisi berada dalam daftar bernomor di dalam suatu pasal, tulis sebagai "Pasal X angka Y", bukan "Pasal Y".

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
Jawaban ini disusun berdasarkan dokumen hukum yang tersedia dalam basis data ReguAI dan bersifat informatif. Jawaban bukan merupakan nasihat hukum resmi maupun pendapat hukum yang mengikat.
"""


def build_not_found_answer():
    """
    Jawaban standar ketika pertanyaan berada di luar cakupan
    atau dokumen tidak cukup untuk menjawab.
    """

    return """
Informasi tidak ditemukan dalam dokumen yang tersedia.

Catatan:
ReguAI hanya menjawab berdasarkan dokumen hukum digital dan hukum pidana yang tersedia dalam basis data.
"""