import pickle
import spacy

# Inisialisasi model bahasa Spacy
nlp = spacy.load('xx_ent_wiki_sm')

# Fungsi untuk mengidentifikasi aspek dari ulasan
def identifikasi_aspek(ulasan):
    daftar_aspek = {
        'HARGA': ['harga', 'mahal', 'murah'],
        'TEMPAT': ['tempat', 'lokasi', 'area'],
        'KEBERSIHAN': ['kotor', 'bersih'],
        'PELAYANAN': ['pelayanan', 'pelayan', 'layanan', 'waiter', 'waitress'],
    }

    doc = nlp(ulasan.lower())
    aspek_teridentifikasi = []

    for token in doc:
        for aspek, kata_kunci in daftar_aspek.items():
            if token.text in kata_kunci:
                aspek_teridentifikasi.append(aspek)

    aspek_teridentifikasi = list(set(aspek_teridentifikasi))  # Menghilangkan duplikat aspek
    return aspek_teridentifikasi

# Fungsi untuk melakukan analisis sentimen dan identifikasi aspek
def aspek(ulasan):
    aspek = identifikasi_aspek(ulasan)

    return aspek