import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import unicodedata
import pickle
import warnings
import itertools
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

# ==========================================
# 設定・定数
# ==========================================
BEST_W1 = 1
BEST_W2 = 1
BEST_W3 = 1
MODEL_DIR = "."

# ==========================================
# 関数定義
# ==========================================
def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").strip()

def create_robust_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    return session

def parse_values(td_element):
    if not td_element: return [np.nan, np.nan, np.nan]
    raw_texts = td_element.get_text(separator="|").split("|")
    vals = []
    for t in raw_texts:
        clean_t = t.strip().replace('%', '')
        if re.match(r'^[0-9.]+$', clean_t): vals.append(float(clean_t))
    while len(vals) < 3: vals.append(np.nan)
    return vals[:3]

@st.cache_resource(show_spinner=False)
def load_and_preprocess_boatracer():
    try:
        csv_path = os.path.join(MODEL_DIR, "boatracer.data.csv")
        df_csv = pd.read_csv(csv_path, header=1)
        df_csv = df_csv[['登録番号', 'コース', '3連対率(%)', '1着率(%)']].copy()
        df_csv['登録番号'] = pd.to_numeric(df_csv['登録番号'], errors='coerce').fillna(-1).astype(int)
        df_csv['コース'] = pd.to_numeric(df_csv['コース'], errors='coerce').fillna(-1).astype(int)
        df_csv = df_csv[df_csv['登録番号'] != -1]

        def clean_pct(val):
            if pd.isna(val): return np.nan
            val_str = str(val).replace('%', '').strip()
            if val_str in ['- -', '-', '']: return np.nan
            try: return float(val_str)
            except ValueError: return np.nan

        for col in ['3連対率(%)', '1着率(%)']:
            df_csv[col] = df_csv[col].apply(clean_pct)
        df_csv = df_csv.drop_duplicates(subset=['登録番号', 'コース'])
        return df_csv
    except Exception as e:
        st.error(f"⚠️ 選手データの読み込みエラー: {e}")
        return None

@st.cache_resource(show_spinner="AIモデルを読み込み中...")
def load_models():
    try:
        features = pickle.load(open(os.path.join(MODEL_DIR, "model_features.pkl"), "rb"))
        boat1_features = pickle.load(open(os.path.join(MODEL_DIR, "model_boat1_features.pkl"), "rb"))
        expert_models = {
            '1st': pickle.load(open(os.path.join(MODEL_DIR, "model_1st.pkl"), "rb")),
            '2nd': pickle.load(open(os.path.join(MODEL_DIR, "model_2nd.pkl"), "rb")),
            'top3': pickle.load(open(os.path.join(MODEL_DIR, "model_top3.pkl"), "rb"))
        }
        model_1st_boat = pickle.load(open(os.path.join(MODEL_DIR, "model_1st_boat_win.pkl"), "rb"))
        boatracer_df = load_and_preprocess_boatracer()
        return features, boat1_features, expert_models, model_1st_boat, boatracer_df
    except Exception as e:
        st.error(f"❌ モデルのロードエラー: {e}")
        return None

def scrape_target_race_basic(hd, rno, jcd):
    session = create_robust_session()
    params = {"rno": str(rno), "jcd": str(jcd), "hd": hd}
    try:
        res = session.get("https://www.boatrace.jp/owpc/pc/race/racelist", params=params, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")
        racers_info = {}
        tbodies = soup.select("tbody.is-fs12")
        if len(tbodies) < 6: return None
        for tbody in tbodies[:6]:
            rows = tbody.find_all("tr")
            if len(rows) < 4: continue
            waku_td = tbody.select_one("td[class*='is-boatColor']")
            if not waku_td: continue
            waku = int(normalize_text(waku_td.get_text(strip=True)))
            reg_info = tbody.select_one(".is-fs11").get_text(strip=True)
            reg_match = re.search(r'\d{4}', reg_info)
            if not reg_match: continue
            reg_no = int(reg_match.group())
            stats_tds = tbody.select("td.is-lineH2")
            nat = parse_values(stats_tds[1]) if len(stats_tds) > 1 else [np.nan]*3
            course_row = rows[1].find_all("td")
            st_row = rows[2].find_all("td")
            rank_row = rows[3].find_all("td")
            series_data = []
            min_len = min(len(course_row), len(st_row), len(rank_row))
            for i in range(min_len):
                c_txt = normalize_text(course_row[i].get_text(strip=True))
                s_txt = normalize_text(st_row[i].get_text(strip=True))
                r_txt = normalize_text(rank_row[i].get_text(strip=True))
                if c_txt.isdigit() and c_txt in "123456":
                    course = int(c_txt)
                    st_val = np.nan
                    st_match = re.match(r'^([FL]?)\.(\d{2})$', s_txt)
                    if st_match: st_val = float("0." + st_match.group(2))
                    rank_val = np.nan
                    if r_txt.isdigit() and r_txt in "123456": rank_val = int(r_txt)
                    if not np.isnan(rank_val) or not np.isnan(st_val):
                        series_data.append({'course': course, 'rank': rank_val, 'st': st_val})
            c1_ranks = [d['rank'] for d in series_data if d['course'] == 1 and not np.isnan(d['rank'])]
            c123_ranks = [d['rank'] for d in series_data if d['course'] in [1, 2, 3] and not np.isnan(d['rank'])]
            c456_ranks = [d['rank'] for d in series_data if d['course'] in [4, 5, 6] and not np.isnan(d['rank'])]
            c_all_ranks = [d['rank'] for d in series_data if not np.isnan(d['rank'])]
            st_values = [d['st'] for d in series_data if not np.isnan(d['st'])]
            racers_info[waku] = {
                "登録番号": reg_no, "枠番": waku, "全国勝率": nat[0],
                "節間_コース1_平均着順": np.mean(c1_ranks) if c1_ranks else np.nan,
                "節間_スロー成績": np.mean(c123_ranks) if c123_ranks else np.nan,
                "節間_ダッシュ成績": np.mean(c456_ranks) if c456_ranks else np.nan,
                "節間_平均着順": np.mean(c_all_ranks) if c_all_ranks else np.nan,
                "節間平均ST": np.mean(st_values) if st_values else np.nan, "展示タイム": np.nan
            }
        try:
            res_before = session.get("https://www.boatrace.jp/owpc/pc/race/beforeinfo", params=params, timeout=15)
            soup_before = BeautifulSoup(res_before.content, "html.parser")
            for bt_tbody in soup_before.select("tbody.is-fs12"):
                tds = bt_tbody.find("tr").find_all("td")
                if len(tds) >= 5:
                    b_waku = normalize_text(tds[0].get_text(strip=True))
                    if b_waku.isdigit() and int(b_waku) in racers_info:
                        try: racers_info[int(b_waku)]["展示タイム"] = float(tds[4].get_text(strip=True))
                        except: pass
        except: pass
        return racers_info
    except: return None

def get_custom_series_rank(row):
    try:
        course = int(row['枠番'])
        dash_val = row['節間_ダッシュ成績']
        if course == 1:
            val = row['節間_コース1_平均着順']
            return val if pd.notna(val) else (dash_val if pd.notna(dash_val) else np.nan)
        elif course in [2, 3]:
            val = row['節間_スロー成績']
            return val if pd.notna(val) else (dash_val if pd.notna(dash_val) else np.nan)
        else:
            return dash_val if pd.notna(dash_val) else np.nan
    except: return np.nan

def evaluate_single_race(hd_input, rno, jcd, jcd_name, loaded_data):
    features, boat1_features, expert_models, model_1st_boat, boatracer_df = loaded_data
    st.markdown(f"### ▼▼ {hd_input} 第{rno}R ({jcd_name}) AI予測 ▼▼")

    with st.spinner("出走表データを取得中..."):
        r_info = scrape_target_race_basic(hd_input, rno, jcd)

    if not r_info:
        st.error(f"❌ 出走表データが取得できませんでした。")
        return

    try:
        df = pd.DataFrame.from_dict(r_info, orient='index').reset_index(drop=True)
        if boatracer_df is not None:
            df['登録番号'] = pd.to_numeric(df['登録番号'], errors='coerce').fillna(-1).astype(int)
            df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(-1).astype(int)
            df = pd.merge(df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')

        df = df.sort_values('枠番').reset_index(drop=True)
        df['適性_節間成績'] = df.apply(get_custom_series_rank, axis=1)

        df['適性_節間成績'] = df.apply(get_custom_series_rank, axis=1)

        # -------------------------------------------------------------
        # 【追加】展示タイム、3連対率(%)、1着率(%)、全国勝率の欠損チェック
        # ※データフレームに列が存在しないエラーを防ぐため、リストで指定して一括判定
        required_cols = ['適性_節間成績', '節間平均ST', '展示タイム', '3連対率(%)', '1着率(%)', '全国勝率']
        
        # 必要な列がそもそも取得できていないか、いずれかの列に欠損(NaN)が含まれているかチェック
        if any(col not in df.columns for col in required_cols) or df[required_cols].isna().any().any():
            st.warning("⚠️ 必要なデータ（節間成績、ST、展示タイム、勝率など）が不足しているため、予測を中止します。")
            return
        # -------------------------------------------------------------

        # ※欠損値で弾くようになったため、展示タイムのfillna(平均値補完)は不要になりますが、
        # 念のため残すか、削除しても問題ありません。
        df['展示タイム'] = df['展示タイム'].fillna(df['展示タイム'].mean() if not df['展示タイム'].isna().all() else 6.80)
        
        df['展示タイム_mean'] = df['展示タイム'].mean()
        df['展示タイム_diff'] = df['展示タイム'] - df['展示タイム_mean']
        df['節間平均ST_mean'] = df['節間平均ST'].mean()
        df['節間平均ST_num'] = df['節間平均ST'] - df['節間平均ST_mean']
        df['全国勝率_mean'] = df['全国勝率'].mean()
        df['全国勝率_num'] = df['全国勝率'] - df['全国勝率_mean']
        if '節間_平均着順' not in df.columns: df['節間_平均着順'] = np.nan

        X_boat1 = df[df['枠番'] == 1][boat1_features]
        boat1_win_prob = model_1st_boat.predict_proba(X_boat1)[0][1]

        # 1号艇1着確率の閾値設定
        THRESHOLD = 0.66
        for col in features:
            if col not in df.columns: df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X_pred = df[features]
        p1 = expert_models['1st'].predict_proba(X_pred)[:, 1]
        p2 = expert_models['2nd'].predict_proba(X_pred)[:, 1]
        p_top3 = expert_models['top3'].predict_proba(X_pred)[:, 1]
        p_top2 = np.clip(p1 + p2, 0.0, 1.0)

        sanrentan_results = []
        st.info(f"🚤 **1号艇1着確率**: {boat1_win_prob*100:.1f}%")
        
        if boat1_win_prob >= THRESHOLD:
            st.success("確率が高い為、【**1号艇1着固定 (1-X-X)**】 で予想を展開します。")
            b1 = 1
            for perm in itertools.permutations(range(2, 7), 2):
                b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))
        else:
            st.warning("確率が基準未満の為、【**全6艇 (X-Y-Z)**】 から広く予想を展開します。")
            for perm in itertools.permutations(range(1, 7), 3):
                b1, b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))

        sanrentan_results.sort(key=lambda x: x[3], reverse=True)

        # スコア125以上の買い目があるかチェック (スコア表示は res[3]*1000)
        bet_targets = [res for res in sanrentan_results[:5] if boat1_win_prob >= 0.66 and (res[3]*1000) >= 240]
        
        if bet_targets:
            st.markdown("### 🔔 【推奨】条件達成！")
            st.success(f"**1号艇1着確率が66%以上、かつスコアが240以上の買い目があります。予想1〜２位について積極的に購入を検討してください。**")

        st.markdown("#### 🎯 AI予測 3連単 上位5通り")
        result_df = pd.DataFrame([
            {"順位": f"{i+1}位", "買い目": f"{res[0]} - {res[1]} - {res[2]}", "スコア": f"{res[3]*1000:.3f}"}
            for i, res in enumerate(sanrentan_results[:5])
        ])
        st.table(result_df)

    except Exception as e:
        st.error(f"❌ 処理エラー: {e}")

# ==========================================
# UI表示 (Streamlit メイン)
# ==========================================
st.set_page_config(page_title="競艇AI予測", page_icon="🚤")
st.title("🚤 競艇AI予測アプリケーション")

loaded_data = load_models()

if loaded_data is not None:
    st.sidebar.header("🎯 予測設定")
    jcd_dict = {"20": "若松", "13": "尼崎"}
    jcd_name = st.sidebar.selectbox("競艇場を選択", list(jcd_dict.values()))
    jcd_code = [k for k, v in jcd_dict.items() if v == jcd_name][0]
    selected_date = st.sidebar.date_input("対象日付")
    hd_input = selected_date.strftime("%Y%m%d")
    rno = st.sidebar.slider("レース番号", min_value=1, max_value=12, value=1)
    
    if st.sidebar.button("予測を開始する", type="primary"):
        evaluate_single_race(hd_input, rno, jcd_code, jcd_name, loaded_data)
else:
    st.warning("システムを再起動するか、ファイルを確認してください。")
