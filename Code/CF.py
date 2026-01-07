import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from collections import Counter
from Processing import remove_punctuation, remove_stopwords, remove_numbers
from CBF import tfidf, tfidf_matrix, df_metadata  # pastikan tfidf & df_metadata ada di cbf.py

# === 1️⃣ Buat user-item matrix ===
user_item_matrix = df_metadata.pivot_table(
    index='user_id',
    columns='Dosen',
    aggfunc='size',
    fill_value=0
)

user_similarity = cosine_similarity(user_item_matrix)

# ===== ✅ Fungsi Prediksi Dosen TANPA Topik =====
def predict_cf_dosen(judul, abstrak, top_k=10, top_dosen_n=5, debug=True, return_df=False):
    """
    Prediksi dosen menggunakan Collaborative Filtering.
    (Tidak lagi menghitung atau menampilkan topik utama.)
    """
    # --- Preprocessing input ---
    input_text = f"{judul.lower()} {abstrak.lower()}"
    input_text = remove_punctuation(input_text)
    input_text = remove_stopwords(input_text)
    input_text = remove_numbers(input_text)

    # --- TF-IDF vektorisasi ---
    input_vec = tfidf.transform([input_text])
    similarity_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # --- Cari mahasiswa paling mirip ---
    top_k_indices = similarity_scores.argsort()[-top_k:][::-1]
    top_k_user_ids = df_metadata.iloc[top_k_indices]['user_id'].values
    top_k_scores = similarity_scores[top_k_indices]

    if debug:
        print("\n📌 Top-k mahasiswa mirip berdasarkan TF-IDF:\n")
        debug_df = df_metadata.iloc[top_k_indices][['user_id', 'title', 'Dosen', 'keywords']].copy()
        debug_df['similarity'] = top_k_scores
        print(tabulate(debug_df.values.tolist(), headers=list(debug_df.columns), tablefmt='grid', showindex=False))

    # --- Hitung skor prediksi dosen ---
    cf_matrix_top_k = user_item_matrix.loc[list(top_k_user_ids)]
    weighted_scores = cf_matrix_top_k.mul(top_k_scores[:, np.newaxis], axis=0)
    predicted_scores = weighted_scores.sum(axis=0) / (top_k_scores.sum() + 1e-8)
    predicted_scores = predicted_scores.sort_values(ascending=False)

    # --- Buat tabel hasil ---
    hasil = []
    for dosen, score in predicted_scores.head(top_dosen_n).items():
        mask = cf_matrix_top_k[dosen] > 0
        mhs_mirip = mask.sum()
        rata2_kemiripan = (top_k_scores[mask.values].mean() if mhs_mirip > 0 else 0)

        hasil.append({
            'Dosen': dosen,
            'CF Score': round(score, 4),
            'Jumlah Mahasiswa Mirip': int(mhs_mirip),
            'Rata-rata Kemiripan': round(rata2_kemiripan, 4)
        })

    hasil_df = pd.DataFrame(hasil)

    if return_df:
        return hasil_df
    else:
        print("\n📊 Rekomendasi Dosen Berdasarkan Collaborative Filtering:\n")
        print(tabulate(hasil_df.values.tolist(), headers=list(hasil_df.columns), tablefmt='grid', showindex=False))
