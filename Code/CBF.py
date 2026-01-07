# CBF.py
from typing import Tuple
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Processing import preprocess_text

# -------------------------
# GLOBAL VARIABLES / MODULE LEVEL
# -------------------------
df_metadata = pd.read_csv("data/metadata_cleaned.csv")  # ganti sesuai path
# Bisa pilih 'content' atau 'keywords' untuk vektorisasi
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_metadata['content'].astype(str))

tfidf_kw = TfidfVectorizer()
tfidf_kw_matrix = tfidf_kw.fit_transform(df_metadata['keywords'].astype(str))


# -------------------------
# CBF: REKOMENDASI DOSEN
# -------------------------
def recommend_dosen_from_input(judul: str, abstrak: str, top_n_docs: int = 10, top_n_dosen: int = 5) -> pd.DataFrame:
    """
    Memberikan rekomendasi dosen berdasarkan kemiripan konten 
    antara input judul + abstrak dengan data penelitian dosen.
    """
    # Preprocess input teks
    input_text = preprocess_text(f"{judul} {abstrak}")

    # Hitung similarity dengan semua dokumen dosen
    input_vec = tfidf.transform([input_text])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # Ambil top-N dokumen paling mirip
    top_doc_indices = sim_scores.argsort()[-top_n_docs:][::-1]
    top_docs = df_metadata.iloc[top_doc_indices].copy()
    top_docs['Similarity Score'] = sim_scores[top_doc_indices]

    # Kelompokkan berdasarkan dosen dan hitung skor rata-rata
    dosen_group = top_docs.groupby('Dosen').agg({'Similarity Score': ['count', 'mean']}).reset_index()
    dosen_group.columns = ['Dosen', 'Jumlah_Kemunculan', 'Rata2_Kemiripan']
    dosen_group['Rata2_Kemiripan'] = dosen_group['Rata2_Kemiripan'].round(4)

    # Urutkan dari yang paling relevan
    dosen_group = dosen_group.sort_values(by=['Jumlah_Kemunculan', 'Rata2_Kemiripan'], ascending=False).reset_index(drop=True)
    return dosen_group.head(top_n_dosen)


# -------------------------
# KOMBINASI: REKOMENDASI SAJA
# -------------------------
def recommend_with_topic(
    judul: str,
    abstrak: str,
    top_n_docs_for_cbf: int = 10,
    top_n_dosen: int = 5
) -> pd.DataFrame:
    """
    Mengembalikan daftar dosen yang direkomendasikan berdasarkan input judul + abstrak.
    (Topik utama tidak diekstrak lagi.)
    """
    dosen_df = recommend_dosen_from_input(
        judul, 
        abstrak, 
        top_n_docs=top_n_docs_for_cbf, 
        top_n_dosen=top_n_dosen
    )
    return dosen_df
