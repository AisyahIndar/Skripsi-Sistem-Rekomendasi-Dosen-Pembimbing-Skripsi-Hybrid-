import streamlit as st
from CF import predict_cf_dosen

st.set_page_config(page_title="Rekomendasi Dosen (CF)", layout="centered")
st.title("🎓 Sistem Rekomendasi Dosen - Collaborative Filtering")
st.markdown("Masukkan judul dan abstrak skripsi untuk mendapatkan rekomendasi dosen.")

# === 📥 Input user ===
judul_input = st.text_input("📘 Judul Skripsi")
abstrak_input = st.text_area("📄 Abstrak Skripsi")

# ✅ Tetapkan nilai tetap (tidak bisa diubah user)
TOP_K = 10           # jumlah mahasiswa mirip
TOP_DOSEN_N = 5      # jumlah dosen direkomendasikan

if st.button("🔎 Cari Rekomendasi Dosen"):
    if not judul_input or not abstrak_input:
        st.warning("⚠️ Harap isi judul dan abstrak terlebih dahulu.")
    else:
        # --- Prediksi CF ---
        hasil_df = predict_cf_dosen(
            judul=judul_input,
            abstrak=abstrak_input,
            top_k=TOP_K,
            top_dosen_n=TOP_DOSEN_N,
            debug=False,
            return_df=True
        )

        # --- Tampilkan hasil ---
        if hasil_df is not None and not hasil_df.empty:
            st.success("✅ Rekomendasi berhasil ditemukan!")

            st.subheader("👨‍🏫 Rekomendasi Dosen:")
            st.dataframe(hasil_df)

            # Tombol download CSV
            csv = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil Rekomendasi (CSV)", csv, "rekomendasi_dosen_cf.csv", "text/csv")
        else:
            st.error("❌ Tidak ada hasil rekomendasi yang ditemukan. Silakan cek input atau hubungi admin.")
