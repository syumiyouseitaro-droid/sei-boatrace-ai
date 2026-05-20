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
st.set_page_config(page_title="競艇AI予測モデル", layout="wide")

# ==========================================
# カスタムCSS (スマホ最適化・絵文字なし・モダンデザイン)
# ==========================================
st.markdown("""
<style>
    /* 全体のフォント設定 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif;
    }
    
    /* Metric(指標)カードのデザイン */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    div[data-testid="metric-container"] > label {
        font-weight: bold;
        color: #333;
        font-size: 1.1rem;
    }
    div[data-testid="metric-container"] > div {
        color: #0066cc;
        font-size: 2rem !important;
        font-weight: 800;
    }
    
    /* 買い目ランキングのカード風デザイン */
    .ranking-card {
        background: linear-gradient(135deg, #ffffff, #f0f4f8);
        border-left: 5px solid #0056b3;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 1px 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .ranking-rank {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        background-color: #0056b3;
        width: 30px;
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 50%;
    }
    .ranking-bet {
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 2px;
        color: #2c3e50;
        flex-grow: 1;
        text-align: center;
    }
    .ranking-score {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: bold;
    }

    /* スマホ向け（画面幅が768px以下の場合）の設定 */
    @media (max-width: 768px) {
        .ranking-bet { font-size: 1.3rem; }
        .ranking-rank { width: 25px; height: 25px; font-size: 1rem; }
        .ranking-score { font-size: 0.8rem; }
        /* 指標カードの余白を減らして画面を広く使う */
        div[data-testid="metric-container"] { padding: 10px; }
        /* Streamlit全体の左右の余白を削る */
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)


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
        if not os.path.exists(csv_path): return None
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
        st.error(f"選手データの読み込みエラー: {e}")
        return None

@st.cache_resource(show_spinner="AIモデルを読み込み中...")
def load_venue_models(jcd_code: str) -> tuple:
    n = jcd_code 
    try:
        expert_models = {
            '1st': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_1st.pkl"), "rb")),
            '2nd': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_2nd.pkl"), "rb")),
            'top3': pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_top3.pkl"), "rb"))
        }
        model_1st_boat = pickle.load(open(os.path.join(MODEL_DIR, f"{n}_model_1st_boat_win.pkl"), "rb"))
        return expert_models, model_1st_boat
    except Exception as e:
        st.error(f"モデルのロードエラー: 指定のディレクトリ({MODEL_DIR})に『 {n}_model_*.pkl 』が存在するか確認してください。")
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

# ==========================================
# 評価処理 (デザイン改修版)
# ==========================================
def evaluate_single_race(hd_input: str, rno: int, jcd: str, jcd_name: str, loaded_data: tuple):
    expert_models, model_1st_boat, boatracer_df = loaded_data
    
    boat1_features = ['1着率(%)', '全国勝率_num', '適性_節間成績', '展示タイム_diff']
    features = [
        '節間平均ST_num', '適性_節間成績', '展示タイム_diff',
        '全国勝率_num', '2連対率(%)', '3連対率(%)', '1着率(%)'
    ]

    st.markdown(f"## {jcd_name} 第{rno}R ({hd_input[:4]}/{hd_input[4:6]}/{hd_input[6:]}) AI予測結果")
    
    with st.spinner("WEBから最新の出走表・展示データを取得・解析中..."):
        r_info = scrape_target_race_basic(hd_input, rno, jcd)

    if not r_info:
        st.error("出走表データが取得できませんでした。レース番号や日付を確認してください。")
        return

    try:
        df = pd.DataFrame.from_dict(r_info, orient='index').reset_index(drop=True)
        if boatracer_df is not None:
            df['登録番号'] = pd.to_numeric(df['登録番号'], errors='coerce').fillna(-1).astype(int)
            df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(-1).astype(int)
            df = pd.merge(df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')

        df = df.sort_values('枠番').reset_index(drop=True)
        df['適性_節間成績'] = df.apply(get_custom_series_rank, axis=1)

        # 欠損補完処理
        idx_6 = df[df['枠番'] == 6].index
        if not idx_6.empty:
            i = idx_6[0]
            if '3連対率(%)' in df.columns and pd.isna(df.at[i, '3連対率(%)']): df.at[i, '3連対率(%)'] = 10.0
            if '2連対率(%)' in df.columns and pd.isna(df.at[i, '2連対率(%)']): df.at[i, '2連対率(%)'] = 5.0
            if '1着率(%)' in df.columns and pd.isna(df.at[i, '1着率(%)']): df.at[i, '1着率(%)'] = 0.0

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
            st.warning("展示タイムなど、予測に必要なデータが未発表または不足しているため予測を中止します。")
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
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(label="イン逃げ期待度 (1号艇1着確率)", value=f"{boat1_win_prob*100:.1f}%")

        st.divider()

        THRESHOLD = 0.99
        sanrentan_results = []
        
        if boat1_win_prob >= THRESHOLD:
            st.info(f"[AI判断] イン逃げ確率が非常に高いため、【1号艇1着固定 (1-X-X)】 で予想を展開します。")
            b1 = 1
            for perm in itertools.permutations(range(2, 7), 2):
                b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))
        else:
            st.info(f"[AI判断] 波乱の可能性を含め、【全6艇 (X-Y-Z)】 から広く予想を展開します。")
            for perm in itertools.permutations(range(1, 7), 3):
                b1, b2, b3 = perm
                score = (p1[b1-1] ** BEST_W1) * (p_top2[b2-1] ** BEST_W2) * (p_top3[b3-1] ** BEST_W3)
                sanrentan_results.append((b1, b2, b3, score))

        sanrentan_results.sort(key=lambda x: x[3], reverse=True)
        
        bet_targets = [res for res in sanrentan_results[:5] if boat1_win_prob >= 0.90 and (res[3]*1000) >= 240]
        if bet_targets:
            st.success("[回収率プラス条件達成] 1号艇1着確率が90%以上、かつスコアが240以上の強力な買い目があります。")

        st.markdown("### AI予測 3連単 上位5通り")
        ranks = ["1", "2", "3", "4", "5"]
        for i, res in enumerate(sanrentan_results[:5]):
            html_card = f"""
            <div class="ranking-card">
                <div class="ranking-rank">{ranks[i]}</div>
                <div class="ranking-bet">{res[0]} - {res[1]} - {res[2]}</div>
                <div class="ranking-score">Score: {res[3]*1000:.1f}</div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)
            
        st.divider()

        with st.expander("内部データとAI評価の可視化 (詳細)", expanded=False):
            st.markdown("モデルが算出した各号艇の確率と、入力された特徴量を確認できます。")
            
            # スマホ最適化: 横に長い表をタブで分割
            tab1, tab2 = st.tabs(["各号艇のAI評価", "入力された特徴量データ"])
            
            with tab1:
                prob_df = pd.DataFrame({
                    "枠番": [f"{i}号艇" for i in range(1, 7)],
                    "1着確率(%)": p1 * 100,
                    "2着確率(%)": p2 * 100,
                    "3着内確率(%)": p_top3 * 100
                })
                st.dataframe(
                    prob_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "1着確率(%)": st.column_config.ProgressColumn("1着確率", format="%.1f%%", min_value=0, max_value=100),
                        "2着確率(%)": st.column_config.ProgressColumn("2着確率", format="%.1f%%", min_value=0, max_value=100),
                        "3着内確率(%)": st.column_config.ProgressColumn("3着内確率", format="%.1f%%", min_value=0, max_value=100),
                    }
                )

            with tab2:
                feature_display_df = df[['枠番'] + features].copy()
                feature_display_df['枠番'] = feature_display_df['枠番'].astype(str) + "号艇"
                # エラー回避のため、少数点第3位までの文字列に変換
                for col in features:
                    feature_display_df[col] = feature_display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
                st.dataframe(feature_display_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"分析処理中にエラーが発生しました: {e}")

# ==========================================
# メイン画面 (スマホ最適化: サイドバー廃止)
# ==========================================
st.title("競艇AI予測モデル")
st.markdown("---")

st.markdown("### 予測設定")

# 入力項目をメイン画面上部に配置
col1, col2 = st.columns(2)
with col1:
    jcd_dict = {"01": "桐生", "13": "尼崎", "20": "若松", "24": "大村"} 
    jcd_name = st.selectbox("競艇場を選択", list(jcd_dict.values()))
    jcd_code = [k for k, v in jcd_dict.items() if v == jcd_name][0]
    
with col2:
    selected_date = st.date_input("対象日付")
    hd_input = selected_date.strftime("%Y%m%d")

rno = st.number_input("レース番号", min_value=1, max_value=12, value=1)

start_button = st.button("予測を開始する", type="primary", use_container_width=True)
st.markdown("---")

# 実行
if start_button:
    expert_models, model_1st_boat = load_venue_models(jcd_code)
    boatracer_df = load_and_preprocess_boatracer()
    
    if expert_models is not None and boatracer_df is not None:
        evaluate_single_race(hd_input, rno, jcd_code, jcd_name, (expert_models, model_1st_boat, boatracer_df))
    else:
        st.error("モデルまたはデータが不足しています。")
