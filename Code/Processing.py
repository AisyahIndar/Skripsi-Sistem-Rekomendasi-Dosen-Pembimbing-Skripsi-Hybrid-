import pandas as pd
import ast
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# ===============================
# 1️⃣ LOAD DATA
# ===============================
def load_data(metadata_path: str, data_uji_path: str, dosen_list_path: str):
    # Load CSV
    df_metadata = pd.read_csv(metadata_path, encoding='latin-1')
    df_data_uji = pd.read_csv(data_uji_path, encoding='latin-1', keep_default_na=True, na_values=[''])

    # Drop rows tanpa keywords
    df_data_uji.dropna(subset=['keywords'], inplace=True)

    # Hapus kolom tidak dipakai
    for df in [df_metadata, df_data_uji]:
        df.drop(columns=['date', 'url'], inplace=True, errors='ignore')

    # Parse kolom 'creators' → 'Dosen'
    df_metadata['author_list'] = df_metadata['creators'].apply(parse_creators)
    df_metadata = df_metadata.explode('author_list').rename(columns={'author_list': 'Dosen'})

    df_data_uji['author_list'] = df_data_uji['creators'].apply(lambda x: parse_creators(x, is_data_uji=True))
    df_data_uji = df_data_uji.rename(columns={'author_list': 'Dosen'})

    # Hapus kolom creators
    df_metadata.drop(columns=['creators'], inplace=True, errors='ignore')
    df_data_uji.drop(columns=['creators'], inplace=True, errors='ignore')

    # read_csv untuk baca data
    df2 = pd.read_csv(dosen_list_path, encoding='latin-1')
    df2['clean_nama_dosen'] = df2['Nama Dosen'].apply(remove_titles)

    df_metadata = df_metadata[df_metadata['Dosen'].isin(df2['clean_nama_dosen'])]
    df_data_uji = df_data_uji[df_data_uji['Dosen'].apply(
        lambda lst: any(d in df2['clean_nama_dosen'].values for d in lst) if isinstance(lst, list) else False
    )]

    return df_metadata, df_data_uji


# ===============================
# 2️⃣ UTILITAS PEMROSESAN
# ===============================
def parse_creators(raw_string, is_data_uji=False):
    if not isinstance(raw_string, str):
        return []
    formatted_names, name_list_raw = [], []

    if raw_string.strip().startswith('[') and raw_string.strip().endswith(']'):
        try:
            name_list_raw = ast.literal_eval(raw_string)
        except (SyntaxError, ValueError):
            pass
    elif ';' in raw_string:
        name_list_raw = [name.strip() for name in raw_string.split(';')]
        if is_data_uji and len(name_list_raw) > 1:
            name_list_raw = name_list_raw[1:]
    else:
        name_list_raw = [raw_string.strip()]

    for name in name_list_raw:
        parts = name.split(',')
        formatted = f"{parts[1].strip()} {parts[0].strip()}" if len(parts) == 2 else name.strip()
        formatted_names.append(formatted.lower().title())

    return [n for n in formatted_names if n]


def remove_titles(name):
    if isinstance(name, str):
        nama_bersih = name.split(',')[0]
        nama_bersih = re.sub(r'[.]', '', nama_bersih)
        nama_bersih = re.sub(r'\s+', ' ', nama_bersih).strip()
        return nama_bersih.title()
    return name


def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    return text


indonesian_stopwords = stopwords.words('indonesian')
english_stopwords = stopwords.words('english')
stop_words = set(indonesian_stopwords + english_stopwords)

custom_academic_stopwords = [
    # --- Grup Proses Penelitian ---
    'penelitian', 'research', 'studi', 'study', 'analisis', 'analysis', 
    'metode', 'method', 'methods', 'hasil', 'result', 'results', 'tujuan', 
    'bertujuan', 'aims', 'masalah', 'problem', 'pembahasan', 'kesimpulan', 
    'implementasi', 'pengembangan', 'perancangan', 'pengujian', 'testing', 
    'training', 'penulis', 'abstrak', 'skripsi', 'tugas', 'akhir', 'jurnal',

    # --- Grup Kata Benda Generik ---
    'data', 'dataset', 'sistem', 'system', 'model', 'algoritma', 'algorithm', 
    'algorithms', 'informasi', 'information', 'aplikasi', 'application', 
    'feature', 'selection', 'proses', 'process', 'pakar', 'expert',
    'pengguna', 'user',

    # --- Grup Evaluasi & Verba Umum ---
    'menggunakan', 'using', 'used', 'use', 'uses', 'berbasis', 'based', 
    'berdasarkan', 'akurasi', 'accuracy', 'nilai', 'value', 'values', 'level',
    'performansi', 'performance', 'klasifikasi', 'classification', 'clustering',
    'prediksi', 'prediction', 'diperoleh', 'obtained', 'didapat', 'menunjukkan', 
    'dilakukan', 'carried', 'terhadap', 'pengaruh', 'dapat', 'fuzzy', 'naive',
    'bayes', 'svm', 'decision', 'tree', 'machine', 'learning', 'sentimen', 
    'sentiment', 'text'
]

stop_words.update(custom_academic_stopwords)

def remove_stopwords(text):
    if isinstance(text, str):
        return ' '.join([word for word in text.split() if word not in stop_words])
    return text


# ===============================
# 3️⃣ PREPROCESS TEKS
# ===============================
# 🧹 1️⃣ Untuk DataFrame
def preprocess_dataframe(df_metadata: pd.DataFrame, df_data_uji: pd.DataFrame = None): # type: ignore
    """
    Membersihkan dan melakukan preprocessing pada dataframe metadata 
    dan optional dataframe data uji.
    """
    # Simpan semua dataframe valid dalam list
    dataframes = [df_metadata]
    if isinstance(df_data_uji, pd.DataFrame):  # ✅ hanya tambahkan jika benar-benar DataFrame
        dataframes.append(df_data_uji)

    # 🔄 Lakukan preprocessing untuk setiap dataframe
    for df in dataframes:
        for col in ['title', 'keywords', 'abstract', 'Dosen']:
            if col in df.columns:  # ✅ cek kolom ada agar tidak error
                df[col] = df[col].astype(str).str.lower().apply(remove_punctuation)
        for col in ['title', 'keywords', 'abstract']:
            if col in df.columns:
                df[col] = df[col].apply(remove_stopwords)

    # 🧠 Buat kolom 'content' khusus df_metadata
    if all(c in df_metadata.columns for c in ['title', 'keywords', 'abstract']):
        df_metadata['content'] = (
            df_metadata['title'] + ' ' +
            df_metadata['keywords'] + ' ' +
            df_metadata['abstract']
        )

    # 🔑 Tambahkan kolom user_id jika belum ada
    if 'user_id' not in df_metadata.columns:
        df_metadata['user_id'] = ['mhs_' + str(i) for i in range(len(df_metadata))]

    return df_metadata, df_data_uji



# ✂️ 2️⃣ Untuk teks biasa (judul + abstrak)
def preprocess_text(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = remove_punctuation(text)
        text = remove_stopwords(text)
        return text
    return ""


# ===============================
# 4️⃣ SIMPAN HASIL PREPROCESS
# ===============================
def save_cleaned(df_metadata, df_data_uji, save_metadata_path, save_data_uji_path):
    df_metadata.to_csv(save_metadata_path, index=False)
    df_data_uji.to_csv(save_data_uji_path, index=False)


# ===============================
# 🚀 EKSEKUSI PROGRAM
# ===============================
metadata_path = "data/metadata5.csv"
data_uji_path = "data/data_uji1.csv"
dosen_list_path = "data/List_Nama_Dosen.csv"  

# 1️⃣ Load data
df_metadata, df_data_uji = load_data(metadata_path, data_uji_path, dosen_list_path)

# 2️⃣ Preprocess teks
df_metadata, df_data_uji = preprocess_dataframe(df_metadata, df_data_uji)

# 3️⃣ Simpan hasil
save_cleaned(df_metadata, df_data_uji, "data/metadata_cleaned.csv", "data/data_uji_cleaned.csv")

print("✅ Preprocessing selesai dan hasil sudah disimpan!")
