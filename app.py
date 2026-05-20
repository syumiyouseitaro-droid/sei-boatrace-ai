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
BEST_W1 = 1.0
BEST_W2 = 1.0
BEST_W3 = 1.0
MODEL_DIR = "."

# ==========================================
# ページ設定 (最初に記述する必要があります)
# ==========================================
st.set_page_config(page_title="競艇AI予測アプリケーション", page_icon="🚤", layout="wide")

# ==========================================
# 関数定義
# ==========================================
def normalize_text(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").strip()

def create_robust_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    return session

def parse_values(td_element) -> list:
    if not td_element: return [np.nan, np.nan, np.nan]
    raw_texts = td_element.get_text(separator="|").split("|")
    vals = []
    for t in raw_texts:
        clean_t = t.strip().replace('%', '')
        if re.match(r'^[0-9.]+$', clean_t): 
            vals.append(float(clean_t))
    while len(vals) < 3: 
        vals.append(np.nan)
    return vals[:3]

@st.cache_resource(show_spinner=False)
def load_and_preprocess_boatracer() -> pd.DataFrame:
    try:
        csv_path = os.path.join(MODEL_DIR, "boatracer.data.csv")
        if not os.path.exists(csv_path):
            return None
            
        df_csv = pd.read_csv(csv_path, header=1)
        df_csv = df_csv[['登録番号', 'コース', '3連対率(%)', '2連対率(%)', '1着率(%)']].copy()
        
        df_csv['登録番号'] = pd.to_numeric(df_csv['登録番号'], errors='coerce').fillna(-1).astype(int)
        df_csv['コース'] = pd.to_numeric(df_csv['コース'], errors='coerce').fillna(-1).astype(int)
        df_csv = df_csv[df_csv['登録番号'] != -1]

        def clean_pct(val):
            if pd.isna(val): return np.nan
            val_str = str(val).replace('%', '').strip()
            if val_str in ['- -', '-', '']: return np.nan
            try: return float(val_str)
            except ValueError: return np.nan

        for col in ['3連対率(%)', '2連対率(%)', '1着率(%)']:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(clean_pct)
        df_csv = df_csv.drop_duplicates(subset=['登録番号', 'コース'])
        return df_csv
    except Exception as e:
        st.error(f"⚠️ 選手データの読み込みエラー: {e}")
        return None

# 【変更】競艇場コード(JCD)を受け取り、その場専用のモデルをロードするように分離
@st.cache_resource(show_spinner="AIモデルを読み込み中...")
def load_venue_models(jcd_code: str) -> tuple:
    n = str(int(jcd_code)) # 例: "01" -> "1", "20" -> "20" に変換（プレフィックス用）
    try:
        expert_models = {
            '1st': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_1st.pkl"), "rb")),
            '2nd': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_2nd.pkl"), "rb")),
            'top3': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_top3.pkl"), "rb"))
        }
        model_1st_boat = pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_1st_boat_win.pkl"), "rb"))
        return expert_models, model_1st_boat
    except Exception as e:
        st.error(f"❌ モデルのロードエラー: 指定のディレクトリ({MODEL_DIR})に『 {n}_model_*.pkl 』が存在するか確認してください。詳細: {e}")
        return None, None

def scrape_target_race_basic(hd: str, rno: int, jcd: str) -> dict:
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
                "コース1_平均着順": np.mean(c1_ranks) if c1_ranks else np.nan,
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
    except Exception:
        return None

def get_custom_series_rank(row) -> float:
    try:
        course = int(row['枠番'])
        course1_val = row.get('コース1_平均着順', np.nan)
        slow_val = row.get('節間_スロー成績', np.nan)
        dash_val = row.get('節間_ダッシュ成績', np.nan)
        mean_val = row.get('節間_平均着順', np.nan)

        if course == 1:
            if pd.notna(course1_val): return course1_val
            if pd.notna(slow_val): return slow_val
            return mean_val
        elif course in [2, 3]:
            if pd.notna(slow_val): return slow_val
            if pd.notna(dash_val): return dash_val
            return mean_val
        elif course in [4, 5, 6]:
            if pd.notna(dash_val): return dash_val
            return 5.0
        return np.nan
    except:
        return np.nan

def evaluate_single_race(hd_input: str, rno: int, jcd: str, jcd_name: str, loaded_data: tuple):
    expert_models, model_1st_boat, boatracer_df = loaded_data
    
    boat1_features = ['1着率(%)', '全国勝率_num', '適性_節間成績', '展示タイム_diff']
    features = [
        '節間平均ST_num', '適性_節間成績', '展示タイム_diff',
        '全国勝率_num', '2連対率(%)', '3連対率(%)', '1着率(%)'
    ]

    st.markdown(f"## 🏁 {hd_input} 第{rno}R ({jcd_name}) AI予測結果")
    st.divider()

    with st.spinner("WEBから最新の出走表・展示データを取得中..."):
        r_info = scrape_target_race_basic(hd_input, rno, jcd)

    if not r_info:
        st.error("❌ 出走表データが取得できませんでした。レース番号や日付を確認してください。")
        return

    try:
        df = pd.DataFrame.from_dict(r_info, orient='index').reset_index(drop=True)
        if boatracer_df is not None:
            df['登録番号'] = pd.to_numeric(df['登録番号'], errors='coerce').fillna(-1).astype(int)
            df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(-1).astype(int)
            df = pd.merge(df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')

        df = df.sort_values('枠番').reset_index(drop=True)
        df['適性_節間成績'] = df.apply(get_custom_series_rank, axis=1)

        imputed_messages = []
        idx_6 = df[df['枠番'] == 6].index
        if not idx_6.empty:
            i = idx_6[0]
            if '3連対率(%)' in df.columns and pd.isna(df.at[i, '3連対率(%)']):
                df.at[i, '3連対率(%)'] = 10.0
                imputed_messages.append("6号艇の『3連対率(%)』が欠損していたため、10.0% で補完しました。")
            if '2連対率(%)' in df.columns and pd.isna(df.at[i, '2連対率(%)']):
                df.at[i, '2連対率(%)'] = 5.0
                imputed_messages.append("6号艇の『2連対率(%)』が欠損していたため、5.0% で補完しました。")
            if '1着率(%)' in df.columns and pd.isna(df.at[i, '1着率(%)']):
                df.at[i, '1着率(%)'] = 0.0
                imputed_messages.append("6号艇の『1着率(%)』が欠損していたため、0.0% で補完しました。")
        
        if imputed_messages:
            for msg in imputed_messages:
                st.info(f"ℹ️ {msg}")

        df['展示タイム_mean'] = df['展示タイム'].mean()
        df['展示タイム_diff'] = df['展示タイム'] - df['展示タイム_mean']
        
        df['節間平均ST_mean'] = df['節間平均ST'].mean()
        df['節間平均ST_num'] = df['節間平均ST'] - df['節間平均ST_mean']
        
        df['全国勝率_mean'] = df['全国勝率'].mean()
        df['全国勝率_num'] = df['全国勝率'] - df['全国勝率_mean']

        missing_details = []
        col_unique = list(set(features + boat1_features))
        for col in col_unique:
            if col not in df.columns:
                df[col] = np.nan

        missing_cols = [col for col in col_unique if df[col].isna().any()]
        if missing_cols:
            for col in missing_cols:
                missing_wakubans = df[df[col].isna()]['枠番'].tolist()
                wakuban_str = ', '.join(f"{w}号艇" for w in missing_wakubans)
                missing_details.append(f"【{col}】 (該当: {wakuban_str})")

        if missing_details:
            st.warning("⚠️ 展示タイムなど、予測に必要なデータが未発表または不足しているため予測を中止します。")
            for detail in missing_details:
                st.error(f"・不足データ: {detail}")
            return

        for col in set(features + boat1_features):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X_boat1 = df[df['枠番'] == 1][boat1_features]
        boat1_win_prob = model_1st_boat.predict_proba(X_boat1)[0][1]

        X_pred = df[features]
        p1 = expert_models['1st'].predict_proba(X_pred)[:, 1]
        p2 = expert_models['2nd'].predict_proba(X_pred)[:, 1]
        p_top3 = expert_models['top3'].predict_proba(X_pred)[:, 1]
        p_top2 = np.clip(p1 + p2, 0.0, 1.0)

        with st.expander("内部データとAI評価の可視化 (詳細を確認する)", expanded=False):
            st.markdown("モデルがどのような確率を算出したか、どの特徴量を重視したかを確認できます。")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**各号艇のAI評価確率**")
                prob_df = pd.DataFrame({
                    "枠番": df['枠番'].astype(str) + "号艇",
                    "1着確率": [f"{val * 100:.1f}%" for val in p1],
                    "2着確率": [f"{val * 100:.1f}%" for val in p2],
                    "3着内確率": [f"{val * 100:.1f}%" for val in p_top3] 
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

            with col_b:
                st.markdown("**🔢 モデルに入力された相対特徴量**")
                feature_display_df = df[['枠番'] + features].copy()
                feature_display_df['枠番'] = feature_display_df['枠番'].astype(str) + "号艇"
                for col in features:
                    feature_display_df[col] = feature_display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
                st.dataframe(feature_display_df, hide_index=True, use_container_width=True)

        st.metric(label="イン逃げ期待度 (1号艇1着確率)", value=f"{boat1_win_prob*100:.1f}%")

        THRESHOLD = 0.99
        sanrentan_results = []
        
        if boat1_win_prob >= THRESHOLD:
            st.success(f"イン逃げ確率が基準({THRESHOLD*100:.0f}%)を満たしている為、【**1号艇1着固定 (1-X-X)**】 で予想を展開します。")
            b1 = 1
            for perm in itertools.permutations(range(2, 7), 2):
                b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))
        else:
            st.warning(f"イン逃げ確率が基準未満の為、波乱を含めて【**全6艇 (X-Y-Z)**】 から広く予想を展開します。")
            for perm in itertools.permutations(range(1, 7), 3):
                b1, b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))

        sanrentan_results.sort(key=lambda x: x[3], reverse=True)
        
        bet_targets = [res for res in sanrentan_results[:5] if boat1_win_prob >= 0.79 and (res[3]*1000) >= 240]
        if bet_targets:
            st.markdown("###【鉄板推奨】条件達成")
            st.success("**1号艇1着確率が81%以上、かつスコアが222以上の買い目があります。予想上位3位以内について購入してください。**")

        st.markdown("AI予測 3連単 上位5通り")
        result_df = pd.DataFrame([
            {"順位": f"{i+1}位", "買い目": f"{res[0]} - {res[1]} - {res[2]}", "AIスコア": f"{res[3]*1000:.3f}"}
            for i, res in enumerate(sanrentan_results[:5])
        ])
        st.dataframe(result_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 分析処理中にエラーが発生しました: {e}")

# ==========================================
# UI表示 (Streamlit メイン)
# ==========================================
st.title("競艇AI予測モデル")

st.sidebar.header("予測設定")
# 【変更】競艇場の辞書に桐生(01)を追加（APIの仕様上01などの2桁フォーマットにしておきます）
jcd_dict = {"01": "桐生", "13": "尼崎", "20": "若松"} 
jcd_name = st.sidebar.selectbox("競艇場を選択", list(jcd_dict.values()))
jcd_code = [k for k, v in jcd_dict.items() if v == jcd_name][0]

# 【変更】UIで選択された競艇場に応じたモデルをロードする
expert_models, model_1st_boat = load_venue_models(jcd_code)
boatracer_df = load_and_preprocess_boatracer()

if expert_models is not None and boatracer_df is not None:
    selected_date = st.sidebar.date_input("対象日付")
    hd_input = selected_date.strftime("%Y%m%d")
    rno = st.sidebar.slider("レース番号", min_value=1, max_value=12, value=1)
    
    if st.sidebar.button("予測を開始する", type="primary", use_container_width=True):
        evaluate_single_race(hd_input, rno, jcd_code, jcd_name, (expert_models, model_1st_boat, boatracer_df))
else:
    n = str(int(jcd_code))
    st.warning(f"選択した競艇場（{jcd_name}）のモデルファイル（ {n}_model_*.pkl ）または boatracer.data.csv が見つかりません。ディレクトリ({MODEL_DIR})を確認してください。")
