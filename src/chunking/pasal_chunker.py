import re


ROMAN_ARTICLE_MARKERS = {
    "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"
}


def split_by_pasal(text: str):
    """
    Memecah teks dokumen hukum menjadi chunk berbasis Pasal.

    Alasan:
    - Dokumen hukum memiliki struktur formal berbasis pasal.
    - Satu pasal dijadikan satu chunk agar konteks hukum tetap utuh.
    - Metadata pasal nantinya dipakai untuk sumber jawaban dan evaluasi retrieval.

    Return:
    [
        {
            "pasal": "67",
            "text": "Pasal 67 ..."
        },
        ...
    ]
    """

    pattern = r"(?m)^Pasal\s+([0-9]+[A-Z]?|[IVX]+)\s*$"
    matches = list(re.finditer(pattern, text))

    chunks = []

    for i, match in enumerate(matches):
        pasal_raw = match.group(1).strip()

        # Pada beberapa UU perubahan, Pasal II, III, IV, dst. sering merupakan
        # pasal transisi/penutup. Untuk kasus ReguAI, fokus utama adalah pasal
        # substansi bernomor agar sumber jawaban lebih presisi.
        if pasal_raw in ROMAN_ARTICLE_MARKERS:
            continue

        # Beberapa PDF membaca "Pasal I" sebagai angka romawi,
        # padahal konteksnya adalah Pasal 1.
        pasal = "1" if pasal_raw == "I" else pasal_raw

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()

        if len(chunk_text) > 50:
            chunks.append(
                {
                    "pasal": pasal,
                    "text": chunk_text,
                }
            )

    return chunks