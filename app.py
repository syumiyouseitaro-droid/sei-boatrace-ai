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
                others = [x for x in range(1, 7) if x != b1]
                columns[b1] = [(b1, b2, b3) for b2, b3 in itertools.permutations(others, 2)]
            
            correct_combos = []
            for row in range(20):
                for b1 in range(1, 7):
                    correct_combos.append(columns[b1][row])
            
            for combo, el in zip(correct_combos, odds_elements):
                text = el.get_text(strip=True)
                try:
                    odds_dict[combo] = float(text)
                except ValueError:
                    odds_dict[combo] = 0.0
    except Exception:
        pass
    return odds_dict

def get_target_race_result(hd: str, rno: int, jcd: str):
    session = create_robust_session()
    url = f"https://www.boatrace.jp/owpc/pc/race/resultlist?jcd={jcd}&hd={hd}"
    try:
        res = session.get(url, timeout=15)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')

        rows = soup.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            row_data = [col.get_text(strip=True) for col in cols if col.get_text(strip=True)]
            if len(row_data) >= 3 and re.match(r'^([1-9]|1[0-2])R$', row_data[0]):
                current_rno = int(row_data[0].replace('R', ''))
                if current_rno == rno:
                    nums = row_data[1].split('-')
                    payout_str = row_data[2].replace('¥', '').replace(',', '').strip()
                    payout_val = int(payout_str) if payout_str.isdigit() else row_data[2]
                    if len(nums) == 3 and all(n.isdigit() for n in nums):
                        return {"result": (int(nums[0]), int(nums[1]), int(nums[2])), "payout": payout_val}
    except Exception:
        pass
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
# 評価処理 (ダイレクト評価版・オッズ反映・フォーメーション分析)
# ==========================================
def evaluate_single_race(hd_input: str, rno: int, jcd: str, jcd_name: str, loaded_data: tuple):
    expert_models, model_1st_boat, boatracer_df = loaded_data
    
    boat1_features = ['1着率(%)', '全国勝率_num', '適性_節間成績', '展示タイム_diff']
    features = [
        '節間平均ST_num', '適性_節間成績', '展示タイム_diff',
        '全国勝率_num', '2連対率(%)', '3連対率(%)', '1着率(%)'
    ]

    st.markdown(f"## {jcd_name} 第{rno}R ({hd_input[:4]}/{hd_input[4:6]}/{hd_input[6:]}) AI予測結果")
    
    with st.spinner("WEBから最新の出走表・オッズデータを取得・解析中..."):
        r_info = scrape_target_race_basic(hd_input, rno, jcd)
        odds_dict = scrape_odds_3t(hd_input, rno, jcd)
        actual_result = get_target_race_result(hd_input, rno, jcd)

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

        # モデルが記憶している正しい順番を取得する関数
        def get_expected_cols(model):
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_) 
            else:
                return boat1_features if model == model_1st_boat else features

        # 1号艇1着確率の算出
        expected_boat1_cols = get_expected_cols(model_1st_boat)
        X_boat1 = df[df['枠番'] == 1].reindex(columns=expected_boat1_cols, fill_value=0)
        boat1_win_prob = model_1st_boat.predict_proba(X_boat1)[0][1]

        # エキスパート予測用データ準備（ダイレクト評価）
        expected_pred_cols = get_expected_cols(expert_models['1st'])
        X_pred = df.reindex(columns=expected_pred_cols, fill_value=0)
        
        p1 = expert_models['1st'].predict_proba(X_pred)[:, 1]
        p_2nd = expert_models['2nd'].predict_proba(X_pred)[:, 1] 
        p_top3 = expert_models['top3'].predict_proba(X_pred)[:, 1]
        
        # 降順ソートで艇番抽出
        rank_1st = np.argsort(p1)[::-1] + 1
        rank_2nd = np.argsort(p_2nd)[::-1] + 1
        rank_top3 = np.argsort(p_top3)[::-1] + 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="イン逃げ期待度 (1号艇1着確率)", value=f"{boat1_win_prob*100:.1f}%")

        st.divider()

        THRESHOLD = 0.95
        if boat1_win_prob >= THRESHOLD:
            st.info(f"🚀 **[AI判断] 1号艇1着確率 {boat1_win_prob*100:.1f}%** → 【1号艇1着固定フォーメーション (1-4-4)】")
            candidates_1st = [1]
            candidates_2nd = rank_2nd[:4].tolist()
            candidates_3rd = rank_top3[:5].tolist()
        else:
            st.info(f"🌊 **[AI判断] 1号艇1着確率 {boat1_win_prob*100:.1f}%** → 【通常フォーメーション (2-4-5)】")
            candidates_1st = rank_1st[:2].tolist()
            candidates_2nd = rank_2nd[:4].tolist()
            candidates_3rd = rank_top3[:5].tolist()

        form_str_1st = "".join(map(str, sorted(candidates_1st)))
        form_str_2nd = "".join(map(str, sorted(candidates_2nd)))
        form_str_3rd = "".join(map(str, sorted(candidates_3rd)))
        formation_display = f"{form_str_1st}-{form_str_2nd}-{form_str_3rd}"
        st.write(f"**展開フォーメーション**: {formation_display}")

        # 各買い目の計算（期待値とスコア）
        formation_bets = []
        for b1 in candidates_1st:
            for b2 in candidates_2nd:
                if b1 == b2: continue
                for b3 in candidates_3rd:
                    if b1 == b3 or b2 == b3: continue
                    
                    prob_1st = p1[b1-1]
                    prob_2nd = p_2nd[b2-1]
                    prob_top3 = p_top3[b3-1]
                    
                    prob = prob_1st * prob_2nd * prob_top3
                    odds = odds_dict.get((b1, b2, b3), 0.0)
                    ev = prob * odds
                    score = (prob_1st ** BEST_W1) * (prob_2nd ** BEST_W2) * (prob_top3 ** BEST_W3)
                    
                    formation_bets.append({
                        'combo': (b1, b2, b3),
                        'score': score,
                        'prob': prob,
                        'odds': odds,
                        'ev': ev
                    })

        formation_bets.sort(key=lambda x: x['score'], reverse=True)
        bet_count = len(formation_bets)

        st.markdown(f"### AI予測 3連単 上位5通り (実質買い目: 計{bet_count}点)")
        ranks = ["1", "2", "3", "4", "5"]
        for i, res in enumerate(formation_bets[:5]):
            ev_str = f"{res['ev']:.2f}" if res['ev'] > 0 else "算出不可"
            odds_str = f"{res['odds']:.1f}倍" if res['odds'] > 0 else "未発表"
            html_card = f"""
            <div class="ranking-card">
                <div class="ranking-rank">{ranks[i]}</div>
                <div class="ranking-bet">{res['combo'][0]} - {res['combo'][1]} - {res['combo'][2]}</div>
                <div class="ranking-stats">
                    <span>オッズ: {odds_str}</span>
                    <span class="ev-text">期待値(EV): {ev_str}</span>
                    <span>Score: {res['score']*1000:.1f}</span>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

        st.divider()

        # 実結果の表示処理（レースが終了している場合）
        if actual_result:
            actual_tuple = actual_result["result"]
            actual_payout = actual_result["payout"]
            is_hit = any(b['combo'] == actual_tuple for b in formation_bets)
            profit = actual_payout - (bet_count * 100) if is_hit else -(bet_count * 100)
            
            st.markdown("### 🏁 レース結果")
            if is_hit:
                st.success(f"**結果:** 🎯 的中 【{actual_tuple[0]}-{actual_tuple[1]}-{actual_tuple[2]}】\n\n**払戻:** {actual_payout}円 | **投資:** {bet_count * 100}円 | **収支:** {profit}円")
            else:
                st.error(f"**結果:** ❌ ハズレ 【{actual_tuple[0]}-{actual_tuple[1]}-{actual_tuple[2]}】\n\n**払戻:** 0円 | **投資:** {bet_count * 100}円 | **収支:** {profit}円")
            st.divider()

        with st.expander("内部データとAI評価の可視化 (詳細)", expanded=False):
            st.markdown("モデルが算出した各号艇の確率と、入力された特徴量を確認できます。")
            
            tab1, tab2 = st.tabs(["各号艇のAI評価", "入力された特徴量データ"])
            
            with tab1:
                prob_df = pd.DataFrame({
                    "枠番": [f"{i}号艇" for i in range(1, 7)],
                    "1着確率(%)": p1 * 100,
                    "2着確率(%)": p_2nd * 100,
                    "3着内確率(%)": p_top3 * 100
                })
                st.dataframe(
                    prob_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "1着確率(%)": st.column_config.ProgressColumn("1着確率", format="%.1f%%", min_value=0, max_value=100),
                        "2着確率(%)": st.column_config.ProgressColumn("2着ダイレクト評価", format="%.1f%%", min_value=0, max_value=100),
                        "3着内確率(%)": st.column_config.ProgressColumn("3着内確率", format="%.1f%%", min_value=0, max_value=100),
                    }
                )

            with tab2:
                feature_display_df = df[['枠番'] + features].copy()
                feature_display_df['枠番'] = feature_display_df['枠番'].astype(str) + "号艇"
                for col in features:
                    feature_display_df[col] = feature_display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
                st.dataframe(feature_display_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"分析処理中にエラーが発生しました: {e}")

# ==========================================
# メイン画面
# ==========================================
st.title("競艇AI予測モデル")
st.markdown("---")

st.markdown("### 予測設定")

col1, col2 = st.columns(2)
with col1:
    jcd_dict = {"01": "桐生", "13": "尼崎", "15": "丸亀", "20": "若松", "24": "大村"} 
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
