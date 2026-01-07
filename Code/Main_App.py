import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Hybrid Recommender", layout="wide")

# -------------------------
# Fungsi preprocess sederhana
# -------------------------
def preprocess_text(text):
    return str(text).lower().strip()

# -------------------------
# Load dataset & matrices
# -------------------------
@st.cache_data
def load_data():
    df_metadata = pd.read_csv("data/metadata_cleaned.csv")
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_metadata['content'].astype(str))
    
    user_item_matrix = pd.read_csv("data/user_item_matrix.csv", index_col=0)
    
    return df_metadata, tfidf, tfidf_matrix, user_item_matrix

df_metadata, tfidf, tfidf_matrix, user_item_matrix = load_data()

# -------------------------
# Hybrid Recommendation
# -------------------------
def hybrid_recommendation(judul, abstrak, w_cbf=0.2, w_cf=0.8, top_k=10, top_n=5):
    input_text = preprocess_text(f"{judul} {abstrak}")
    
    input_vec = tfidf.transform([input_text])
    sim_docs = cosine_similarity(input_vec, tfidf_matrix).flatten()

    if sim_docs.size == 0:
        return pd.DataFrame(columns=['Dosen', 'Hybrid_Score'])

    min_similarity_threshold = 0.15
    max_sim = float(sim_docs.max())
    topk = min(5, sim_docs.size)
    topk_mean = float(np.sort(sim_docs)[-topk:].mean())

    if (max_sim < min_similarity_threshold) or (topk_mean < 0.08):
        return pd.DataFrame(columns=['Dosen', 'Hybrid_Score'])

    df_temp = df_metadata.copy().reset_index(drop=True)
    df_temp['__sim'] = sim_docs
    cbf_series = df_temp.groupby('Dosen')['__sim'].mean()

    if 'user_id' not in df_temp.columns:
        df_temp['user_id'] = ['mhs_' + str(i) for i in range(len(df_temp))]

    top_k_indices = np.argsort(sim_docs)[-min(top_k, len(sim_docs)):][::-1]
    top_k_scores = sim_docs[top_k_indices]
    top_k_user_ids = df_temp.iloc[top_k_indices]['user_id'].values.astype(str)

    existing_users = user_item_matrix.index.intersection(list(top_k_user_ids))
    if existing_users.empty:
        cf_series = pd.Series(0.0, index=user_item_matrix.columns)
    else:
        cf_matrix_top_k = user_item_matrix.loc[existing_users].fillna(0)
        user_score_map = dict(zip(top_k_user_ids, top_k_scores))
        scores_for_existing_users = np.array([user_score_map[user] for user in existing_users])

        sum_scores = scores_for_existing_users.sum()
        if sum_scores == 0:
            cf_series = pd.Series(0.0, index=user_item_matrix.columns)
        else:
            weighted = cf_matrix_top_k.mul(scores_for_existing_users[:, np.newaxis], axis=0)
            cf_series = weighted.sum(axis=0) / (sum_scores + 1e-8)

    cbf_series_aligned, cf_series_aligned = cbf_series.align(cf_series, join='outer', fill_value=0)
    merged = pd.concat(
        [
            cbf_series_aligned.rename('CBF_Score'),
            cf_series_aligned.rename('CF_Score')
        ],
        axis=1
    )

    merged['Hybrid_Score'] = w_cbf * merged['CBF_Score'] + w_cf * merged['CF_Score']
    merged = merged.reset_index().rename(columns={'index': 'Dosen'})
    merged = merged.sort_values('Hybrid_Score', ascending=False)

    return merged[['Dosen', 'Hybrid_Score']].head(top_n).copy()

# -------------------------
# Streamlit UI
# -------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🎓 Sistem Rekomendasi Dosen Pembimbing Skripsi</h1>
        <p>
            Masukkan <b>judul</b> dan <b>abstrak</b> skripsi Anda untuk mendapatkan
            <b>5 rekomendasi dosen terbaik</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

judul_input = st.text_input("📘 Judul Skripsi")
abstrak_input = st.text_area("📄 Abstrak Skripsi")

if st.button("🔎 Proses Rekomendasi"):
    if not judul_input or not abstrak_input:
        st.warning("⚠️ Harap isi judul dan abstrak terlebih dahulu.")
    else:
        with st.spinner("🔍 Sedang menganalisis rekomendasi..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            df_hybrid = hybrid_recommendation(
                judul=judul_input,
                abstrak=abstrak_input,
                w_cbf=0.2,
                w_cf=0.8,
                top_k=10,
                top_n=5
            )

        st.markdown("---")

        if df_hybrid.empty:
            st.error("❌ Tidak ada hasil rekomendasi yang ditemukan.")
        else:
            # ===== INI SATU-SATUNYA PENYESUAIAN =====
            df_show = df_hybrid[['Dosen', 'Hybrid_Score']].copy()
            df_show['Hybrid_Score'] = df_show['Hybrid_Score'].map('{:.4f}'.format)

            st.dataframe(df_show, hide_index=True)

            csv = df_show.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Rekomendasi (CSV)",
                data=csv,
                file_name="hasil_rekomendasi.csv",
                mime="text/csv"
            )