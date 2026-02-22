import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import unicodedata
import pickle
import warnings
warnings.filterwarnings('ignore')

# 基本設定（尼崎固定）
JCD = "13"

# --- ページ設定 ---
st.set_page_config(page_title="競艇AI予想システム（尼崎）", page_icon="🚤", layout="centered")
st.title("競艇AI予想システム（尼崎専用）")
st.markdown("最新のデータとAI（HistGradientBoosting）を用いて、3連単の予想トップ30以内を算出します。")

# --- キャッシュ機能 ---
# モデルやマスターデータは重いため、一度だけ読み込んでキャッシュ（保存）します
@st.cache_resource
def load_models():
    clf_top3 = pickle.load(open("model_top3.pkl", "rb"))
    clf_1st = pickle.load(open("model_1st.pkl", "rb"))
    clf_2nd = pickle.load(open("model_2nd.pkl", "rb"))
    features = pickle.load(open("model_features.pkl", "rb"))
    return clf_top3, clf_1st, clf_2nd, features

@st.cache_data
def load_and_preprocess_boatracer():
    boatracer_df = pd.read_csv("boatracer.data.csv", header=1)
    
    def clean_pct(val):
        if pd.isna(val): return np.nan
        val_str = str(val).replace('%', '').strip()
        if val_str in ['- -', '-', '']: return np.nan
        try: return float(val_str)
        except ValueError: return np.nan
        
    def clean_st(val):
        if pd.isna(val): return np.nan
        val_str = str(val).strip()
        if val_str in ['- -', '-', '']: return np.nan
        try: return float(val_str)
        except ValueError: return np.nan

    pct_cols = ['3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)']
    st_cols = ['平均ST', '平均スタート順']
    
    for col in pct_cols: boatracer_df[col] = boatracer_df[col].apply(clean_pct)
    for col in st_cols: boatracer_df[col] = boatracer_df[col].apply(clean_st)
    return boatracer_df

@st.cache_data
def load_race_data():
    return pd.read_csv("race.data.csv", header=1)


# --- スクレイピング関数 ---
def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").strip()

def scrape_target_race_basic(hd, rno):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    params = {"rno": str(rno), "jcd": JCD, "hd": hd}

    url_list = "https://www.boatrace.jp/owpc/pc/race/racelist"
    res_list = session.get(url_list, params=params, timeout=15)
    res_list.encoding = res_list.apparent_encoding
    soup_list = BeautifulSoup(res_list.text, "html.parser")
    
    if not soup_list or not soup_list.select("tbody.is-fs12"):
        raise ValueError("出走表のデータが取得できません。日程とレース番号を確認してください。")

    def parse_values(td_element):
        if not td_element: return [np.nan, np.nan, np.nan]
        raw_texts = td_element.get_text(separator="|").split("|")
        vals = []
        for t in raw_texts:
            clean_t = t.strip().replace('%', '')
            if re.match(r'^[0-9.]+$', clean_t): vals.append(float(clean_t))
        while len(vals) < 3: vals.append(np.nan)
        return vals[:3]

    racers_info = {}
    for tbody in soup_list.select("tbody.is-fs12"):
        waku_td = tbody.select_one("td[class*='is-boatColor']")
        if not waku_td: continue
        waku = normalize_text(waku_td.get_text(strip=True))
        
        reg_info = tbody.select_one(".is-fs11").get_text(strip=True)
        reg_match = re.search(r'\d{4}', reg_info)
        
        if waku.isdigit() and reg_match:
            stats_tds = tbody.select("td.is-lineH2")
            nat = parse_values(stats_tds[1]) if len(stats_tds) > 1 else [np.nan]*3
            mot = parse_values(stats_tds[3]) if len(stats_tds) > 3 else [np.nan]*3
            
            racers_info[int(waku)] = {
                "登録番号": int(reg_match.group()),
                "全国勝率": nat[0],
                "全国3連": nat[2],
                "モーター3連": mot[2],
                "展示タイム": np.nan
            }

    url_before = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    res_before = session.get(url_before, params=params, timeout=15)
    if res_before.status_code == 200:
        res_before.encoding = res_before.apparent_encoding
        soup_before = BeautifulSoup(res_before.text, "html.parser")
        for bt_tbody in soup_before.select("tbody.is-fs12"):
            first_tr = bt_tbody.find("tr")
            if first_tr:
                tds = first_tr.find_all("td")
                if len(tds) >= 5:
                    b_waku = normalize_text(tds[0].get_text(strip=True))
                    if b_waku.isdigit() and int(b_waku) in racers_info:
                        t_val = tds[4].get_text(strip=True)
                        try: racers_info[int(b_waku)]["展示タイム"] = float(t_val)
                        except: pass

    if len(racers_info) < 6:
        raise ValueError("6艇分のデータが揃っていません。")
    return racers_info


# --- 予測関数 ---
def generate_predictions(hd_input, rno_input):
    racers_info = scrape_target_race_basic(hd_input, rno_input)
    clf_top3, clf_1st, clf_2nd, features = load_models()
    race_df = load_race_data()
    boatracer_df = load_and_preprocess_boatracer()

    mock_features = []
    exhibit_times = [v["展示タイム"] for v in racers_info.values() if not pd.isna(v["展示タイム"])]
    avg_exhibit = np.mean(exhibit_times) if exhibit_times else 6.80

    for c in range(1, 7):
        r_num = racers_info[c]["登録番号"]
        e_time = racers_info[c]["展示タイム"]
        if pd.isna(e_time): e_time = avg_exhibit
        
        hist = race_df[race_df['登録番号'] == r_num]
        
        row_dict = {
            '枠番': c, '登録番号': r_num,
            '全国勝率': racers_info[c]["全国勝率"],
            '全国3連': racers_info[c]["全国3連"],
            'モーター3連': racers_info[c]["モーター3連"],
            '展示タイム_diff': e_time - avg_exhibit
        }

        if not hist.empty:
            rec = hist.iloc[-1]
            cols = ['コース1_平均着順', 'コース2_平均着順', 'コース3_平均着順', 'コース4_平均着順', 'コース5_平均着順', 'コース6_平均着順']
            vals = pd.to_numeric(rec[cols], errors='coerce').values
            
            valid_all_vals = [v for v in vals if not np.isnan(v)]
            row_dict['全コース平均着順'] = np.mean(valid_all_vals) if len(valid_all_vals) > 0 else 5.0

            if c == 1: valid_vals = [v for v in vals[0:2] if not np.isnan(v)]
            elif c == 6: valid_vals = [v for v in vals[4:6] if not np.isnan(v)]
            elif 1 < c < 6: valid_vals = [v for v in vals[c-2:c+1] if not np.isnan(v)]
            row_dict['smoothed_course_rank'] = np.mean(valid_vals) if len(valid_vals) > 0 else 5.0
        else:
            row_dict['全コース平均着順'] = 5.0
            row_dict['smoothed_course_rank'] = 5.0
            
        mock_features.append(row_dict)

    target_df = pd.DataFrame(mock_features)
    target_df['枠番'] = pd.to_numeric(target_df['枠番'], errors='coerce')
    target_df = pd.merge(target_df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')

    target_df['3連対率(%)'] = target_df['3連対率(%)'].fillna(10.0)
    target_df['1着率(%)'] = target_df['1着率(%)'].fillna(0.0)
    target_df['2着率(%)'] = target_df['2着率(%)'].fillna(5.0)
    target_df['3着率(%)'] = target_df['3着率(%)'].fillna(5.0)

    for col in features:
        target_df[col] = pd.to_numeric(target_df[col], errors='coerce')

    X_pred = target_df[features]
    
    target_df['prob_top3'] = clf_top3.predict_proba(X_pred)[:, 1]
    target_df['prob_1st'] = clf_1st.predict_proba(X_pred)[:, 1]
    target_df['prob_2nd'] = clf_2nd.predict_proba(X_pred)[:, 1]

    excluded_boats = target_df[target_df['prob_top3'] <= 0.025]['枠番'].tolist()
    valid_df = target_df[~target_df['枠番'].isin(excluded_boats)]
    
    top_1st = valid_df.nlargest(2, 'prob_1st')['枠番'].tolist()
    top_2nd = valid_df.nlargest(3, 'prob_2nd')['枠番'].tolist()

    combinations = []
    for c1 in top_1st:
        for c2 in top_2nd:
            if c1 == c2: continue
            for c3 in valid_df['枠番'].tolist():
                if c3 == c1 or c3 == c2: continue
                p1 = target_df[target_df['枠番'] == c1]['prob_1st'].values[0]
                p2 = target_df[target_df['枠番'] == c2]['prob_2nd'].values[0]
                p3 = target_df[target_df['枠番'] == c3]['prob_top3'].values[0]
                score = p1 * p2 * p3
                combinations.append({"買い目": f"{int(c1)}-{int(c2)}-{int(c3)}", "AIスコア": round(score, 4)})

    combinations.sort(key=lambda x: x["AIスコア"], reverse=True)
    return combinations[:30], excluded_boats


# --- UI部分 ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        hd_input = st.text_input("📅 レース日程 (例: 20260221)", value="20260221")
    with col2:
        rno_input = st.number_input("🏁 レース番号 (1〜12)", min_value=1, max_value=12, value=1)
    
    submitted = st.form_submit_button("AI予想を実行する")

if submitted:
    with st.spinner("出走表のスクレイピングとAI予想を実行中..."):
        try:
            results, excluded = generate_predictions(hd_input, rno_input)
            
            st.success("予想が完了しました！")
            
            if excluded:
                st.warning(f"⚠️ 除外艇（3着以内の確率2.5%以下）: {', '.join([str(int(x)) for x in excluded])}号艇")
            else:
                st.info("ℹ️ 今回のレースで除外された艇はありません。")

            st.subheader("🏆 3連単 予想")
            # 結果を綺麗なテーブルで表示
            result_df = pd.DataFrame(results)
            result_df.index = np.arange(1, len(result_df) + 1) # インデックスを1からにする
            st.dataframe(result_df, use_container_width=True)

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")