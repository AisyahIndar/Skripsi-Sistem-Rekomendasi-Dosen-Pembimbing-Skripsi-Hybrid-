import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from CBF import recommend_dosen_from_input
from Processing import preprocess_dataframe

# === 🧠 Load dan Preprocess Dataset ===
st.set_page_config(page_title="Sistem Rekomendasi Dosen CBF", layout="wide")

st.title("🎓 Sistem Rekomendasi Dosen Pembimbing (CBF)")
st.markdown("Masukkan judul dan abstrak skripsi kamu untuk mendapatkan rekomendasi dosen pembimbing yang relevan.")

# === 📂 Load dataset ===
df_metadata = pd.read_csv("data/metadata_cleaned.csv")
df_metadata, _ = preprocess_dataframe(df_metadata)

# === 🔤 Input dari user ===
judul_mahasiswa = st.text_input("📘 Judul Skripsi", placeholder="Masukkan judul skripsi...")
abstrak_mahasiswa = st.text_area("📄 Abstrak Skripsi", placeholder="Masukkan abstrak skripsi...")

# === 🚀 Jalankan rekomendasi saat tombol diklik ===
if st.button("🔍 Cari Rekomendasi Dosen"):
    if judul_mahasiswa.strip() == "" or abstrak_mahasiswa.strip() == "":
        st.warning("⚠️ Harap masukkan judul dan abstrak terlebih dahulu.")
    else:
        with st.spinner("🔎 Memproses dan mencari rekomendasi..."):

            # === 🤖 Rekomendasi dosen ===
            rekomendasi_dosen = recommend_dosen_from_input(
                judul=judul_mahasiswa,
                abstrak=abstrak_mahasiswa,
                top_n_docs=10,
                top_n_dosen=5
            )

        # === 📊 Tampilkan hasil ===
        st.success("✅ Rekomendasi berhasil ditemukan!")
        
        st.subheader("👨‍🏫 Rekomendasi Dosen:")
        st.dataframe(rekomendasi_dosen)

        # 📤 Tombol download hasil rekomendasi
        csv = rekomendasi_dosen.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Rekomendasi (CSV)", csv, "rekomendasi_dosen.csv", "text/csv")
