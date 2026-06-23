"""
Modul API ReguAI.

Catatan:
Saat ini ReguAI menggunakan Streamlit sebagai antarmuka utama,
sehingga endpoint API belum diaktifkan.

Folder ini disiapkan jika ke depannya ReguAI dikembangkan menjadi
backend API menggunakan FastAPI atau Flask.
"""


def api_status():
    """
    Fungsi sederhana untuk menunjukkan bahwa modul API tersedia.
    """

    return {
        "status": "ready",
        "message": "API module ReguAI tersedia, tetapi belum digunakan dalam versi Streamlit.",
    }
