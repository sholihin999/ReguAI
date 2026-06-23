import fitz
import re

doc = fitz.open("data/UU_PPP_2011.pdf")

text = ""

for page in doc:
    text += page.get_text() + "\n"

# Ambil isi utama sebelum bagian penjelasan
split_marker = "Agar setiap orang mengetahuinya"

if split_marker in text:
    text = text.split(split_marker)[0]

pattern = r'(?m)^Pasal\s+([0-9]+)\s*$'
matches = re.findall(pattern, text)

print("Jumlah heading Pasal ditemukan:", len(matches))

print("\n50 Pasal pertama:")

for p in matches[:50]:
    print("Pasal", p)