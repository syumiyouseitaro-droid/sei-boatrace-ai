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
# ページ設定
# ==========================================
st.set_page_config(page_title="競艇AI予測モデル", layout="wide")

# ==========================================
# カスタムCSS (スマホ最適化・モダンデザイン)
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
        text-align: left;
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
    .ranking-stats {
        text-align: right;
        font-size: 0.9rem;
        color: #555;
    }
    .ranking-stats span {
        display: block;
        margin-bottom: 2px;
    }
    .ev-text {
        color: #e74c3c;
        font-weight: bold;
    }

    /* スマホ向けの設定 */
    @media (max-width: 768px) {
        .ranking-bet { font-size: 1.3rem; }
        .ranking-rank { width: 25px; height: 25px; font-size: 1rem; }
        .ranking-stats { font-size: 0.8rem; }
        div[data-testid="metric-container"] { padding: 10px; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 関数定義
# ==========================================
def normalize_text(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace(" ", "").strip()

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
    except Exception:
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

def scrape_odds_3t(hd: str, rno: int, jcd: str) -> dict:
    session = create_robust_session()
    params = {"rno": str(rno), "jcd": str(jcd), "hd": hd}
    odds_dict = {}
    try:
        res = session.get("https://www.boatrace.jp/owpc/pc/race/odds3t", params=params, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")
        
        odds_elements = soup.select("td.oddsPoint")
        if len(odds_elements) == 120:
            columns = {}
            for b1 in range(1, 7):
                others = [x for x in
