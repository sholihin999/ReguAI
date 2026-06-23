import fitz

pdf_path = "data/UU_PDP_2022.pdf"

doc = fitz.open(pdf_path)
print("Jumlah halaman:", len(doc))
print(doc[0].get_text()[:2000])