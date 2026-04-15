import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier

# 【ステップ1】Google Driveのマウントを削除し、ローカル環境で動くように変更

CATEGORIES = ['順当', '準順当', '穴', '大穴']

def load_and_preprocess_boatracer():
    # パスをローカル（同じフォルダ）に変更
    boatracer_df = pd.read_csv("./boatracer.data.csv", header=1)

    def clean_pct(val):
        if pd.isna(val): return np.nan
        val_str = str(val).replace('%', '').strip()
        if val_str in ['- -', '-', '']: return np.nan
        try: return float(val_str)
        except ValueError: return np.nan

    pct_cols = ['3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)']
    for col in pct_cols:
        boatracer_df[col] = boatracer_df[col].apply(clean_pct)

    boatracer_df['平均ST'] = pd.to_numeric(boatracer_df['平均ST'], errors='coerce').fillna(0.17)
    boatracer_df['コース'] = pd.to_numeric(boatracer_df['コース'], errors='coerce')

    for col in pct_cols:
        valid_df = boatracer_df[boatracer_df[col].notna()]
        if not valid_df.empty:
            idx_max_course = valid_df.groupby('登録番号')['コース'].idxmax()
            ref_data = boatracer_df.loc[idx_max_course, ['登録番号', 'コース', col]]
            ref_data = ref_data.rename(columns={'コース': 'ref_course', col: 'ref_val'})
            temp_df = boatracer_df.merge(ref_data, on='登録番号', how='left')
            missing_mask = boatracer_df[col].isna() & temp_df['ref_val'].notna()
            imputed_vals = temp_df.loc[missing_mask, 'ref_val'] * temp_df.loc[missing_mask, 'ref_course'] / boatracer_df.loc[missing_mask, 'コース']
            boatracer_df.loc[missing_mask, col] = imputed_vals.clip(upper=100.0)

    boatracer_df['3連対率(%)'] = boatracer_df['3連対率(%)'].fillna(20.0)
    boatracer_df['1着率(%)'] = boatracer_df['1着率(%)'].fillna(5.0)
    boatracer_df['2着率(%)'] = boatracer_df['2着率(%)'].fillna(5.0)
    boatracer_df['3着率(%)'] = boatracer_df['3着率(%)'].fillna(10.0)

    return boatracer_df, pct_cols

def assign_dividend_category(val):
    if pd.isna(val): return np.nan
    if val <= 2000: return '順当'
    elif val <= 7000: return '準順当'
    elif val <= 13000: return '穴'
    else: return '大穴'

def train_models():
    # 【ステップ2】print()をStreamlit用の出力st.write()等に変更
    st.write("データの読み込みと前処理を開始します...")
    
    # パスをローカル（同じフォルダ）に変更
    race_df = pd.read_csv("./race.data.csv", header=1)
    boatracer_df, pct_cols = load_and_preprocess_boatracer()

    valid_racers = set(boatracer_df['登録番号'].unique())
    race_df['is_valid_racer'] = race_df['登録番号'].isin(valid_racers)

    valid_race_mask = race_df.groupby(['日付', 'レース'])['is_valid_racer'].transform('all')
    waku_count = race_df.groupby(['日付', 'レース'])['枠番'].transform('count')

    initial_race_count = race_df[['日付', 'レース']].drop_duplicates().shape[0]
    race_df = race_df[valid_race_mask & (waku_count == 6)].drop(columns=['is_valid_racer']).copy()
    final_race_count = race_df[['日付', 'レース']].drop_duplicates().shape[0]
    
    st.info(f"不完全なレースを除外しました（{initial_race_count}レース -> {final_race_count}レースに厳選）")

    def parse_currency(val):
        if pd.isna(val) or val == 'NaN': return np.nan
        try: return float(str(val).replace('¥', '').replace(',', '').strip())
        except: return np.nan

    race_df['3連単払戻金'] = race_df['3連単払戻金'].apply(parse_currency)
    race_df['着順'] = pd.to_numeric(race_df['着順'], errors='coerce')
    race_df = race_df.dropna(subset=['3連単払戻金', '着順']).copy()
    race_df['払戻金カテゴリ'] = race_df['3連単払戻金'].apply(assign_dividend_category)

    num_cols = ['全国勝率', '全国3連', 'モーター3連', '展示タイム', 'F数']
    for col in num_cols:
        race_df[col] = pd.to_numeric(race_df[col], errors='coerce')

    race_df['F数'] = race_df['F数'].fillna(0)

    def get_all_course_mean_rank(row):
        cols = ['コース1_平均着順', 'コース2_平均着順', 'コース3_平均着順', 'コース4_平均着順', 'コース5_平均着順', 'コース6_平均着順']
        vals = pd.to_numeric(row[cols], errors='coerce').values
        valid_vals = [v for v in vals if not np.isnan(v)]
        return np.mean(valid_vals) if len(valid_vals) > 0 else 5.0

    def get_smoothed_course_rank(row):
        try:
            course = int(row['枠番'])
            cols = ['コース1_平均着順', 'コース2_平均着順', 'コース3_平均着順', 'コース4_平均着順', 'コース5_平均着順', 'コース6_平均着順']
            vals = pd.to_numeric(row[cols], errors='coerce').values

            if course == 1:
                val_c1 = vals[0]
                if not np.isnan(val_c1):
                    return val_c1
                valid_vals = [v for v in vals[0:2] if not np.isnan(v)]
            elif course == 6:
                valid_vals = [v for v in vals[4:6] if not np.isnan(v)]
            elif 1 < course < 6:
                valid_vals = [v for v in vals[course-2:course+1] if not np.isnan(v)]
            else:
                return 5.0

            return np.mean(valid_vals) if len(valid_vals) > 0 else 5.0
        except: return 5.0

    race_df['全コース平均着順'] = race_df.apply(get_all_course_mean_rank, axis=1)
    race_df['smoothed_course_rank'] = race_df.apply(get_smoothed_course_rank, axis=1)

    race_df['モーター_mean'] = race_df.groupby(['日付', 'レース'])['モーター3連'].transform('mean')
    race_df['モーター_num'] = race_df['モーター3連'] - race_df['モーター_mean']
    race_df['展示タイム_mean'] = race_df.groupby(['日付', 'レース'])['展示タイム'].transform('mean')
    race_df['展示タイム_diff'] = race_df['展示タイム'] - race_df['展示タイム_mean']

    train_df = pd.merge(race_df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')

    train_df['平均ST_num'] = train_df['平均ST'] - train_df.groupby(['日付', 'レース'])['平均ST'].transform('mean')

    train_df['is_1st'] = (train_df['着順'] == 1).astype(int)
    train_df['is_2nd'] = (train_df['着順'] == 2).astype(int)
    train_df['is_top3'] = (train_df['着順'] <= 3).astype(int)

    features = [
        '枠番', '全国勝率', '全国3連', 'モーター_num',
        '展示タイム_diff', '平均ST_num',
        '3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)',
        'smoothed_course_rank', '全コース平均着順', 'F数'
    ]

    for col in features:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    st.write("---")
    st.write("🤖 **[Gatekeeper] 学習中...**")
    gate_base_cols = ['全国勝率', '3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)', 'F数', 'smoothed_course_rank']

    gate_wide = train_df.pivot_table(
        index=['日付', 'レース'],
        columns='枠番',
        values=gate_base_cols,
        aggfunc='first'
    )

    gate_wide.columns = [f"{int(waku)}号艇_{col}" for col, waku in gate_wide.columns]
    gate_wide = gate_wide.reset_index()

    race_target = train_df.groupby(['日付', 'レース'])['払戻金カテゴリ'].first().reset_index()
    gate_train_df = pd.merge(gate_wide, race_target, on=['日付', 'レース'])

    gate_features = [col for col in gate_train_df.columns if col not in ['日付', 'レース', '払戻金カテゴリ']]

    X_gate = gate_train_df[gate_features]
    y_gate = gate_train_df['払戻金カテゴリ']

    gate_model = HistGradientBoostingClassifier(random_state=42).fit(X_gate, y_gate)
    pickle.dump(gate_model, open("./model_gatekeeper.pkl", "wb"))

    for cat in CATEGORIES:
        cat_df = train_df[train_df['払戻金カテゴリ'] == cat]
        X = cat_df[features]
        st.write(f"- [{cat}] カテゴリ学習中... (データ数: {len(cat_df)})")
        pickle.dump(HistGradientBoostingClassifier(random_state=42).fit(X, cat_df['is_1st']), open(f"./model_1st_{cat}.pkl", "wb"))
        pickle.dump(HistGradientBoostingClassifier(random_state=42).fit(X, cat_df['is_2nd']), open(f"./model_2nd_{cat}.pkl", "wb"))
        pickle.dump(HistGradientBoostingClassifier(random_state=42).fit(X, cat_df['is_top3']), open(f"./model_top3_{cat}.pkl", "wb"))

    pickle.dump(features, open("./model_features.pkl", "wb"))
    pickle.dump(gate_features, open("./model_gate_features.pkl", "wb"))
    st.success("全ての学習プロセスが完了し、pklファイルが作成されました！")


# --- Streamlit UI部分 ---
if __name__ == "__main__":
    st.title("競艇AI 学習モデル構築システム")
    st.markdown("過去データからAIの学習を実行し、予測に必要なモデル（pklファイル）を生成します。")

    st.warning("実行前に `boatracer.data.csv` と `race.data.csv` がこのアプリと同じフォルダに配置されているか確認してください。")

    # ボタンが押されたら学習開始
    if st.button("学習を開始する"):
        # ファイルの存在チェックを事前に行う
        if not os.path.exists("./boatracer.data.csv") or not os.path.exists("./race.data.csv"):
            st.error("エラー: 必要なCSVデータが見つかりません。ファイルをアップロードしてから再度お試しください。")
        else:
            with st.spinner("AIが学習を行っています。完了までしばらくお待ちください..."):
                train_models()