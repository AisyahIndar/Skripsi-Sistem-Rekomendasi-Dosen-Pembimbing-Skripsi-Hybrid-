# hybrid.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from CF import df_metadata, tfidf, tfidf_matrix, user_item_matrix
from CBF import preprocess_text

def hybrid_recommendation(
    judul,
    abstrak,
    w_cbf=0.8,
    w_cf=0.2,
    top_k=10,
    top_n=5,
    debug=False,
    return_df=False
):
    """
    Hybrid recommender: gabungan CBF + CF
    """

    # --- Validasi bobot ---
    if not np.isclose(w_cbf + w_cf, 1.0):
        total_weight = w_cbf + w_cf
        if total_weight == 0:
            w_cbf, w_cf = 0.8, 0.2
        else:
            w_cbf = w_cbf / total_weight
            w_cf = w_cf / total_weight

    # --- Preprocess input ---
    input_text = preprocess_text(f"{judul} {abstrak}")

    # ===== CBF part =====
    input_vec = tfidf.transform([input_text])
    sim_with_docs = cosine_similarity(input_vec, tfidf_matrix).flatten()
    if sim_with_docs.size == 0 or len(sim_with_docs) != len(df_metadata):
        empty_df = pd.DataFrame(columns=['Dosen', 'CBF_Score', 'CF_Score', 'Hybrid_Score'])
        return empty_df

    df_temp = df_metadata.copy().reset_index(drop=True)
    df_temp['__sim'] = sim_with_docs
    cbf_series = df_temp.groupby('Dosen')['__sim'].mean()
    cbf_series.name = 'CBF_Score'

    # ===== CF part =====
    actual_top_k = min(top_k, len(sim_with_docs))
    top_k_indices = np.argsort(sim_with_docs)[-actual_top_k:][::-1]
    top_k_scores = sim_with_docs[top_k_indices]

    if 'user_id' not in df_temp.columns:
        df_temp['user_id'] = ['mhs_' + str(i) for i in range(len(df_temp))]

    top_k_user_ids = df_temp.iloc[top_k_indices]['user_id'].values.astype(str)
    cf_matrix_top_k = user_item_matrix.loc[top_k_user_ids].fillna(0)

    top_k_scores_np = np.array(top_k_scores)
    sum_top_k_scores = top_k_scores_np.sum()
    if sum_top_k_scores == 0:
        predicted_scores = pd.Series(0.0, index=user_item_matrix.columns)
    else:
        weighted = cf_matrix_top_k.mul(top_k_scores_np[:, np.newaxis], axis=0)
        predicted_scores = weighted.sum(axis=0) / (sum_top_k_scores + 1e-8)

    cf_series = predicted_scores
    cf_series.name = 'CF_Score'

    # ===== Gabungkan CBF & CF =====
    cbf_series_aligned, cf_series_aligned = cbf_series.align(cf_series, join='outer', fill_value=0)
    merged = pd.concat(
        [cbf_series_aligned.rename('CBF_Score'), cf_series_aligned.rename('CF_Score')],
        axis=1
    )
    merged['Hybrid_Score'] = w_cbf * merged['CBF_Score'] + w_cf * merged['CF_Score']
    merged = merged.reset_index().rename(columns={'index': 'Dosen'}).sort_values('Hybrid_Score', ascending=False)
    result_df = merged.head(top_n).copy()

    # ===== Tambahkan sample Title/Keywords =====
    titles, keywords = [], []
    if all(col in df_metadata.columns for col in ['title', 'keywords', 'Dosen']):
        dosen_info_map = df_metadata.groupby('Dosen').agg({'title': 'first', 'keywords': 'first'}).to_dict(orient='index')
        for dos in result_df['Dosen']:
            info = dosen_info_map.get(str(dos), {'title': '', 'keywords': ''})
            titles.append(info['title'])
            keywords.append(info['keywords'])
    else:
        titles = [''] * len(result_df)
        keywords = [''] * len(result_df)

    result_df['Title'] = titles
    result_df['Keywords'] = keywords

    if debug:
        print(result_df.head())

    return result_df