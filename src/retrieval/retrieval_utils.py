import re


TOP_K_INITIAL = 30
TOP_K_FINAL = 5
MAX_SCORE_THRESHOLD = 0.90


PDP_MISUSE_PATTERNS = [
    "penyalahgunaan data",
    "menyalahgunakan data",
    "menyalahgunakan data pribadi",
    "penyalahgunaan data pribadi",
    "data orang lain",
    "data pribadi orang lain",
    "menggunakan data orang lain",
    "menggunakan data pribadi orang lain",
    "menyebarkan data orang lain",
    "menyebarkan data pribadi",
    "menyebarkan data pribadi orang lain",
    "membocorkan data pribadi",
    "membocorkan data orang lain",
    "memakai data orang lain",
    "memakai data pribadi orang lain",
    "hukuman untuk orang yang menyalahgunakan data",
    "hukuman menyalahgunakan data",
]

ITE_HOAX_PATTERNS = [
    "berita bohong",
    "kebohongan",
    "menyebarkan kebohongan",
    "penyebaran kebohongan",
    "hoaks",
    "hoax",
    "informasi bohong",
    "informasi palsu",
    "informasi menyesatkan",
    "menyesatkan",
    "menyebarkan hoaks",
    "menyebarkan hoax",
    "menyebarkan berita bohong",
    "sosial media",
    "media sosial",
    "menyalahgunakan sosial media",
    "menyalahgunakan media sosial",
    "penyalahgunaan sosial media",
    "penyalahgunaan media sosial",
]

THEFT_PATTERNS = [
    "pencurian",
    "mencuri",
    "nyuri",
    "curi",
    "ngambil barang",
    "mengambil barang",
    "mengambil barang orang lain",
    "mengambil milik orang lain",
    "barang orang lain",
    "mencuri barang",
    "mencuri sedikit",
    "ambil barang orang",
    "mengambil barang milik orang lain",
]


def normalize(text: str) -> str:
    return text.lower().strip()


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def is_definition_query(query: str) -> bool:
    q = normalize(query)
    return any(
        x in q
        for x in [
            "apa itu",
            "apa yang dimaksud",
            "apa yang dimaksud dengan",
            "pengertian",
            "definisi",
        ]
    )


def get_definition_target(query: str):
    q = normalize(query)

    if is_definition_query(q):
        if "pengendali data pribadi" in q:
            return "pdp_pasal_1"

        if "subjek data pribadi" in q:
            return "pdp_pasal_1"

        if "data pribadi" in q:
            return "pdp_pasal_1"

        if "informasi elektronik" in q:
            return "ite2008_pasal_1"

        if "dokumen elektronik" in q:
            return "ite2008_pasal_1"

        if "sistem elektronik" in q:
            return "ite2008_pasal_1"

        if "transaksi elektronik" in q:
            return "ite2008_pasal_1"

        if "tindak pidana" in q:
            return "kuhp_pasal_12"

    return None


def is_uu_pdp(doc_name: str) -> bool:
    d = normalize(doc_name)
    return "pelindungan data pribadi" in d


def is_uu_ite_2008(doc_name: str) -> bool:
    d = normalize(doc_name)
    return "nomor 11 tahun 2008" in d and "informasi dan transaksi elektronik" in d


def is_uu_ite_2016(doc_name: str) -> bool:
    d = normalize(doc_name)
    return "nomor 19 tahun 2016" in d


def is_uu_ite_2024(doc_name: str) -> bool:
    d = normalize(doc_name)
    return "nomor 1 tahun 2024" in d


def is_kuhp_2023(doc_name: str) -> bool:
    d = normalize(doc_name)
    return "nomor 1 tahun 2023" in d and "kitab undang-undang hukum pidana" in d


def expand_query(query: str) -> str:
    q = normalize(query)
    expansions = []

    synonyms = {
        "hak subjek data pribadi": (
            "hak subjek data pribadi berhak mendapatkan informasi berhak melengkapi data "
            "berhak memperbarui data berhak memperbaiki data berhak mengakses data "
            "berhak menghapus data menarik persetujuan berhak menggugat ganti rugi "
            "UU PDP Pasal 5 Pasal 6 Pasal 7 Pasal 8 Pasal 9 Pasal 10 Pasal 11 Pasal 12 Pasal 13"
        ),
        "hak pemilik data pribadi": (
            "hak subjek data pribadi berhak mendapatkan informasi berhak melengkapi data "
            "berhak mengakses data berhak menghapus data menarik persetujuan "
            "berhak menggugat ganti rugi UU PDP Pasal 5 Pasal 6 Pasal 7 Pasal 8 "
            "Pasal 9 Pasal 10 Pasal 11 Pasal 12 Pasal 13"
        ),
        "hak data pribadi": (
            "hak subjek data pribadi berhak mendapatkan informasi berhak mengakses data "
            "berhak menghapus data menarik persetujuan berhak menggugat ganti rugi "
            "UU PDP Pasal 5 Pasal 6 Pasal 7 Pasal 8 Pasal 9 Pasal 10 Pasal 11 Pasal 12 Pasal 13"
        ),
        "doxing": (
            "penyebaran data pribadi tanpa izin memperoleh mengumpulkan mengungkapkan "
            "menggunakan data pribadi bukan miliknya melawan hukum dipidana pidana penjara "
            "pidana denda UU PDP Pasal 65 Pasal 66 Pasal 67 Pasal 68 Pasal 69"
        ),
        "doksing": (
            "penyebaran data pribadi tanpa izin memperoleh mengumpulkan mengungkapkan "
            "menggunakan data pribadi bukan miliknya melawan hukum dipidana pidana penjara "
            "pidana denda UU PDP Pasal 65 Pasal 66 Pasal 67 Pasal 68 Pasal 69"
        ),
        "penyalahgunaan data pribadi": (
            "memperoleh mengumpulkan mengungkapkan menggunakan data pribadi bukan miliknya "
            "melawan hukum menguntungkan diri sendiri merugikan subjek data pribadi "
            "dipidana pidana penjara pidana denda UU PDP Pasal 65 Pasal 66 Pasal 67 Pasal 68 Pasal 69"
        ),
        "kebocoran data": (
            "pelanggaran pemrosesan data pribadi pengungkapan data pribadi tanpa izin "
            "melawan hukum UU PDP Pasal 65 Pasal 66 Pasal 67 Pasal 68 Pasal 69"
        ),
        "pencemaran nama baik": (
            "menyerang kehormatan nama baik menuduhkan hal penghinaan informasi elektronik "
            "dokumen elektronik sistem elektronik media sosial UU ITE Pasal 27A Pasal 45"
        ),
        "fitnah": "pencemaran nama baik penghinaan kehormatan menuduhkan hal",
        "media sosial": "informasi elektronik dokumen elektronik sistem elektronik UU ITE",
        "sosial media": "informasi elektronik dokumen elektronik sistem elektronik UU ITE media sosial",
        "hoaks": "berita bohong informasi palsu informasi menyesatkan informasi elektronik UU ITE Pasal 28 Pasal 45A",
        "hoax": "berita bohong informasi palsu informasi menyesatkan informasi elektronik UU ITE Pasal 28 Pasal 45A",
        "berita bohong": "hoaks hoax informasi palsu informasi menyesatkan informasi elektronik UU ITE Pasal 28 Pasal 45A",
        "informasi menyesatkan": "berita bohong hoaks hoax informasi palsu informasi elektronik UU ITE Pasal 28 Pasal 45A",
        "kebohongan": "berita bohong hoaks hoax informasi palsu informasi menyesatkan informasi elektronik UU ITE Pasal 28 Pasal 45A",
        "tindak pidana": "perbuatan pidana ancaman pidana sanksi pidana KUHP",
        "penipuan": "tipu muslihat rangkaian kebohongan menguntungkan diri sendiri KUHP",
        "pencurian": "mengambil barang milik orang lain melawan hukum KUHP Pasal 476 Pasal 477 Pasal 478",
        "mencuri": "pencurian mengambil barang milik orang lain melawan hukum KUHP Pasal 476 Pasal 477 Pasal 478",
        "nyuri": "pencurian mengambil barang milik orang lain melawan hukum KUHP Pasal 476 Pasal 477 Pasal 478",
        "kategori ii": "kategori II pidana denda paling banyak sepuluh juta rupiah KUHP Pasal 79",
        "kategori 2": "kategori II pidana denda paling banyak sepuluh juta rupiah KUHP Pasal 79",
        "denda kategori": "pidana denda kategori besaran denda KUHP Pasal 79",
    }

    for key, value in synonyms.items():
        if key in q:
            expansions.append(value)

    if contains_any(q, PDP_MISUSE_PATTERNS):
        expansions.append(
            "penyalahgunaan data pribadi data pribadi orang lain memperoleh mengumpulkan "
            "mengungkapkan menggunakan data pribadi bukan miliknya secara melawan hukum "
            "menguntungkan diri sendiri merugikan subjek data pribadi dilarang dipidana "
            "pidana penjara pidana denda hukuman sanksi UU PDP Pasal 65 Pasal 66 "
            "Pasal 67 Pasal 68 Pasal 69"
        )

    if contains_any(q, ITE_HOAX_PATTERNS):
        expansions.append(
            "berita bohong hoaks hoax informasi palsu informasi bohong informasi menyesatkan "
            "media sosial sosial media informasi elektronik dokumen elektronik sistem elektronik "
            "mendistribusikan mentransmisikan membuat dapat diakses sanksi pidana hukuman "
            "pidana penjara pidana denda UU ITE Pasal 28 Pasal 45A"
        )

    if contains_any(q, THEFT_PATTERNS):
        expansions.append(
            "pencurian mencuri mengambil barang sesuatu yang seluruhnya atau sebagian "
            "milik orang lain dengan maksud untuk dimiliki secara melawan hukum "
            "dipidana pidana penjara pidana denda KUHP Pasal 476 Pasal 477 Pasal 478"
        )

    if "hukuman" in q:
        expansions.append("sanksi pidana dipidana pidana penjara pidana denda")

    return query + (" " + " ".join(expansions) if expansions else "")


def detect_domain_hint(query: str) -> str:
    q = normalize(query)

    if any(
        x in q
        for x in [
            "data pribadi",
            "data orang lain",
            "data pribadi orang lain",
            "doxing",
            "doksing",
            "kebocoran data",
            "privasi",
            "penyalahgunaan data",
            "menyalahgunakan data",
            "hak subjek data pribadi",
            "hak pemilik data pribadi",
            "subjek data",
            "pengendali data",
        ]
    ):
        return "pdp"

    if any(
        x in q
        for x in [
            "media sosial",
            "sosial media",
            "ite",
            "elektronik",
            "informasi elektronik",
            "dokumen elektronik",
            "sistem elektronik",
            "transaksi elektronik",
            "hoaks",
            "hoax",
            "berita bohong",
            "kebohongan",
            "informasi bohong",
            "informasi palsu",
            "informasi menyesatkan",
            "menyebarkan kebohongan",
            "menyalahgunakan sosial media",
            "pencemaran nama baik",
            "penghinaan online",
        ]
    ):
        return "ite"

    if any(
        x in q
        for x in [
            "tindak pidana",
            "pidana",
            "hukuman",
            "denda",
            "kategori",
            "pencurian",
            "mencuri",
            "nyuri",
            "curi",
            "mengambil barang",
            "ngambil barang",
            "barang orang lain",
            "penipuan",
            "pembunuhan",
            "narkotika",
            "kuhp",
        ]
    ):
        return "pidana"

    return "umum"


def is_in_scope(query: str) -> bool:
    q = normalize(query)

    allowed_terms = [
        "data pribadi",
        "data orang lain",
        "data pribadi orang lain",
        "menyalahgunakan data",
        "penyalahgunaan data",
        "menggunakan data orang lain",
        "membocorkan data",
        "pdp",
        "privasi",
        "doxing",
        "doksing",
        "kebocoran data",
        "subjek data",
        "pengendali data",
        "ite",
        "elektronik",
        "informasi elektronik",
        "dokumen elektronik",
        "sistem elektronik",
        "transaksi elektronik",
        "media sosial",
        "sosial media",
        "penyalahgunaan media sosial",
        "penyalahgunaan sosial media",
        "menyalahgunakan media sosial",
        "menyalahgunakan sosial media",
        "hoaks",
        "hoax",
        "berita bohong",
        "kebohongan",
        "menyebarkan kebohongan",
        "penyebaran kebohongan",
        "informasi bohong",
        "informasi palsu",
        "informasi menyesatkan",
        "menyesatkan",
        "pencemaran nama baik",
        "penghinaan",
        "pidana",
        "hukuman",
        "tindak pidana",
        "kuhp",
        "denda",
        "kategori",
        "pencurian",
        "mencuri",
        "nyuri",
        "curi",
        "mengambil barang",
        "ngambil barang",
        "mengambil barang orang lain",
        "barang orang lain",
        "mencuri barang",
        "penipuan",
        "narkotika",
        "sanksi",
        "pasal",
        "uu",
    ]

    return any(term in q for term in allowed_terms)


def build_search_query(query: str) -> str:
    q = normalize(query)
    target = get_definition_target(q)

    if target == "pdp_pasal_1":
        if "pengendali data pribadi" in q:
            return (
                "query: UU Nomor 27 Tahun 2022 Pasal 1 Pengendali Data Pribadi adalah "
                "setiap orang badan publik dan organisasi internasional yang bertindak "
                "sendiri-sendiri atau bersama-sama dalam menentukan tujuan dan melakukan "
                "kendali pemrosesan Data Pribadi"
            )

        if "subjek data pribadi" in q:
            return (
                "query: UU Nomor 27 Tahun 2022 Pasal 1 Subjek Data Pribadi adalah "
                "orang perseorangan yang pada dirinya melekat Data Pribadi"
            )

        return (
            "query: UU Nomor 27 Tahun 2022 Pasal 1 Data Pribadi adalah data tentang "
            "orang perseorangan yang teridentifikasi atau dapat diidentifikasi"
        )

    if target == "ite2008_pasal_1":
        if "transaksi elektronik" in q:
            return (
                "query: UU Nomor 11 Tahun 2008 Pasal 1 Transaksi Elektronik adalah "
                "perbuatan hukum yang dilakukan dengan menggunakan Komputer jaringan "
                "Komputer dan atau media elektronik lainnya"
            )

        if "sistem elektronik" in q:
            return (
                "query: UU Nomor 11 Tahun 2008 Pasal 1 Sistem Elektronik adalah "
                "serangkaian perangkat dan prosedur elektronik yang berfungsi "
                "mempersiapkan mengumpulkan mengolah menganalisis menyimpan "
                "menampilkan mengumumkan mengirimkan dan menyebarkan Informasi Elektronik"
            )

        if "dokumen elektronik" in q:
            return (
                "query: UU Nomor 11 Tahun 2008 Pasal 1 Dokumen Elektronik adalah "
                "setiap Informasi Elektronik yang dibuat diteruskan dikirimkan diterima "
                "atau disimpan dalam bentuk analog digital elektromagnetik optikal atau sejenisnya"
            )

        return (
            "query: UU Nomor 11 Tahun 2008 Pasal 1 Informasi Elektronik adalah "
            "satu atau sekumpulan data elektronik termasuk tulisan suara gambar peta "
            "rancangan foto electronic data interchange surat elektronik telegram teleks "
            "telecopy huruf tanda angka kode akses simbol atau perforasi"
        )

    if target == "kuhp_pasal_12":
        return (
            "query: UU Nomor 1 Tahun 2023 Pasal 12 Tindak Pidana merupakan perbuatan "
            "yang oleh peraturan perundang-undangan diancam dengan sanksi pidana "
            "dan atau tindakan"
        )

    expanded = expand_query(query)
    domain_hint = detect_domain_hint(query)

    if domain_hint == "pdp":
        expanded += " UU PDP Pelindungan Data Pribadi"
    elif domain_hint == "ite":
        expanded += " UU ITE Informasi dan Transaksi Elektronik"
    elif domain_hint == "pidana":
        expanded += " KUHP Kitab Undang-Undang Hukum Pidana"

    return "query: " + expanded


def keyword_score(query: str, text: str, metadata: dict) -> float:
    q = normalize(query)
    t = normalize(text)
    doc_name = normalize(metadata.get("nama_dokumen", ""))
    pasal = str(metadata.get("pasal", "")).upper()

    score = 0.0
    target = get_definition_target(q)

    if target == "pdp_pasal_1":
        if is_uu_pdp(doc_name):
            score += 100
        if pasal == "1":
            score += 100
        if "adalah" in t:
            score += 15
        if "pengendali data pribadi" in q and "pengendali data pribadi" in t:
            score += 50
        if "subjek data pribadi" in q and "subjek data pribadi" in t:
            score += 50
        if "data pribadi" in q and "data pribadi adalah" in t:
            score += 50
        return score

    if target == "ite2008_pasal_1":
        if is_uu_ite_2008(doc_name):
            score += 120
        if pasal == "1":
            score += 120
        if is_uu_ite_2016(doc_name) or is_uu_ite_2024(doc_name):
            score -= 100
        if "adalah" in t:
            score += 15
        if "informasi elektronik" in q and "informasi elektronik" in t:
            score += 50
        if "dokumen elektronik" in q and "dokumen elektronik" in t:
            score += 50
        if "sistem elektronik" in q and "sistem elektronik" in t:
            score += 50
        if "transaksi elektronik" in q and "transaksi elektronik" in t:
            score += 50
        return score

    if target == "kuhp_pasal_12":
        if is_kuhp_2023(doc_name):
            score += 120
        if pasal == "12":
            score += 120
        if "tindak pidana merupakan" in t or "tindak pidana adalah" in t:
            score += 60
        return score

    important_terms = [
        "data pribadi",
        "data orang lain",
        "subjek data",
        "pengendali data",
        "informasi elektronik",
        "dokumen elektronik",
        "sistem elektronik",
        "transaksi elektronik",
        "media sosial",
        "sosial media",
        "berita bohong",
        "informasi bohong",
        "informasi palsu",
        "informasi menyesatkan",
        "hoaks",
        "hoax",
        "penghinaan",
        "pencemaran",
        "nama baik",
        "kehormatan",
        "pidana",
        "hukuman",
        "denda",
        "kategori",
        "tindak pidana",
        "penipuan",
        "pencurian",
        "mencuri",
        "nyuri",
        "curi",
        "mengambil barang",
        "barang orang lain",
        "milik orang lain",
        "narkotika",
        "penyalahgunaan",
        "melawan hukum",
        "dilarang",
        "dipidana",
        "pidana penjara",
        "pidana denda",
        "mengungkapkan",
        "menggunakan",
        "memperoleh",
        "mengumpulkan",
        "berhak",
        "mengakses",
        "menghapus",
        "menarik persetujuan",
        "menggugat",
    ]

    for term in important_terms:
        if term in q and term in t:
            score += 2.0

    if any(
        x in q
        for x in [
            "hak subjek data pribadi",
            "hak pemilik data pribadi",
            "hak data pribadi",
            "hak subjek data",
        ]
    ):
        if is_uu_pdp(doc_name):
            score += 10
        if pasal in ["5", "6", "7", "8", "9", "10", "11", "12", "13"]:
            score += 20
        if pasal in ["65", "66", "67", "68", "69"]:
            score -= 15
        if any(
            term in t
            for term in [
                "berhak",
                "subjek data pribadi berhak",
                "mengakses",
                "menghapus",
                "menarik kembali persetujuan",
                "menggugat",
                "ganti rugi",
            ]
        ):
            score += 8

    if any(
        x in q
        for x in [
            "data pribadi",
            "data orang lain",
            "data pribadi orang lain",
            "doxing",
            "doksing",
            "kebocoran data",
            "penyalahgunaan data",
            "menyalahgunakan data",
            "membocorkan data",
            "menggunakan data orang lain",
        ]
    ):
        if is_uu_pdp(doc_name):
            score += 12

        if any(
            x in q
            for x in [
                "sanksi",
                "pidana",
                "hukuman",
                "penyalahgunaan",
                "menyalahgunakan",
                "melawan hukum",
                "denda",
                "dilarang",
                "doxing",
                "doksing",
            ]
        ):
            if any(
                term in t
                for term in [
                    "dilarang",
                    "dipidana",
                    "pidana penjara",
                    "pidana denda",
                    "melawan hukum",
                    "data pribadi yang bukan miliknya",
                    "menggunakan data pribadi",
                    "mengungkapkan data pribadi",
                    "memperoleh data pribadi",
                    "mengumpulkan data pribadi",
                ]
            ):
                score += 14
            if pasal in ["65", "66", "67", "68", "69"]:
                score += 24
            if pasal in ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]:
                score -= 5
            if pasal in ["56", "57", "58", "59", "60"]:
                score -= 6

    if any(
        x in q
        for x in [
            "media sosial",
            "sosial media",
            "ite",
            "hoaks",
            "hoax",
            "berita bohong",
            "kebohongan",
            "menyebarkan kebohongan",
            "informasi bohong",
            "informasi palsu",
            "informasi menyesatkan",
            "menyesatkan",
            "menyalahgunakan sosial media",
            "menyalahgunakan media sosial",
            "pencemaran nama baik",
            "penghinaan",
        ]
    ):
        if "ite" in doc_name or "informasi dan transaksi elektronik" in doc_name:
            score += 14

        if any(
            term in t
            for term in [
                "berita bohong",
                "menyesatkan",
                "informasi elektronik",
                "dokumen elektronik",
                "sistem elektronik",
                "media sosial",
                "menyerang kehormatan",
                "nama baik",
                "mendistribusikan",
                "mentransmisikan",
                "membuat dapat diakses",
                "dipidana",
                "pidana penjara",
                "pidana denda",
            ]
        ):
            score += 12

        if pasal in ["28", "45A"]:
            score += 28

        if pasal in ["27A", "45", "45B"]:
            score += 10

        if is_kuhp_2023(doc_name) and pasal in ["433", "434", "435", "436"]:
            score -= 8

    if contains_any(q, THEFT_PATTERNS):
        if is_kuhp_2023(doc_name):
            score += 10

        if any(
            term in t
            for term in [
                "mengambil",
                "barang",
                "milik orang lain",
                "melawan hukum",
                "dipidana",
                "pidana penjara",
                "pidana denda",
            ]
        ):
            score += 12

        if pasal in ["476", "477", "478"]:
            score += 30

        if pasal in ["479", "480", "481"]:
            score += 8

    if any(
        x in q
        for x in ["kategori ii", "kategori 2", "denda kategori", "dendanya berapa"]
    ):
        if is_kuhp_2023(doc_name):
            score += 5
        if pasal == "79":
            score += 30

    if any(x in q for x in ["penipuan"]):
        if is_kuhp_2023(doc_name):
            score += 5
        if pasal == "492":
            score += 25

    pasal_match = re.search(r"pasal\s+([0-9]+[a-z]?)", q)
    if pasal_match:
        target_pasal = pasal_match.group(1).upper()
        if pasal == target_pasal:
            score += 20

    return score


def rerank_results(question, results, top_k=TOP_K_FINAL):
    reranked = []
    q = normalize(question)
    target = get_definition_target(q)

    for doc, distance_score in results:
        doc_name = normalize(doc.metadata.get("nama_dokumen", ""))
        pasal = str(doc.metadata.get("pasal", "")).upper()

        if target == "pdp_pasal_1":
            if not is_uu_pdp(doc_name):
                continue
            if pasal != "1":
                continue

        elif target == "ite2008_pasal_1":
            if not is_uu_ite_2008(doc_name):
                continue
            if pasal != "1":
                continue

        elif target == "kuhp_pasal_12":
            if not is_kuhp_2023(doc_name):
                continue
            if pasal != "12":
                continue

        text = doc.page_content.replace("passage: ", "")
        k_score = keyword_score(question, text, doc.metadata)
        combined_score = k_score - float(distance_score)
        reranked.append((doc, distance_score, k_score, combined_score))

    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked[:top_k]