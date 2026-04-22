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
import datetime
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# 全国のボートレース場リスト
RACE_COURSES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
    "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津（三重）", "10": "三国（福井）",
    "11": "びわこ（滋賀）", "12": "住之江（大阪）", "13": "尼崎（兵庫）", "14": "鳴門（徳島）",
    "15": "丸亀（香川）", "16": "児島（岡山）", "17": "宮島（広島）", "18": "徳山（山口）",
    "19": "下関（山口）", "20": "若松（福岡）", "21": "芦屋（福岡）", "22": "福岡（福岡）",
    "23": "唐津（佐賀）", "24": "大村（長崎）"
}

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").strip()

def load_and_preprocess_boatracer():
    boatracer_df = pd.read_csv("./boatracer.data.csv", header=1)
    def clean_pct(val):
        if pd.isna(val): return np.nan
        val_str = str(val).replace('%', '').strip()
        if val_str in ['- -', '-', '']: return np.nan
        try: return float(val_str)
        except ValueError: return np.nan

    pct_cols = ['2連対率(%)', '3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)']
    for col in pct_cols:
        if col in boatracer_df.columns:
            boatracer_df[col] = boatracer_df[col].apply(clean_pct)

    boatracer_df['コース'] = pd.to_numeric(boatracer_df['コース'], errors='coerce')

    # コース実績の欠損値を補完
    for col in pct_cols:
        if col not in boatracer_df.columns:
            continue
        valid_df = boatracer_df[boatracer_df[col].notna()]
        if not valid_df.empty:
            idx_max_course = valid_df.groupby('登録番号')['コース'].idxmax()
            ref_data = boatracer_df.loc[idx_max_course, ['登録番号', 'コース', col]]
            ref_data = ref_data.rename(columns={'コース': 'ref_course', col: 'ref_val'})
            temp_df = boatracer_df.merge(ref_data, on='登録番号', how='left')
            missing_mask = boatracer_df[col].isna() & temp_df['ref_val'].notna()
            imputed_vals = temp_df.loc[missing_mask, 'ref_val'] * temp_df.loc[missing_mask, 'ref_course'] / boatracer_df.loc[missing_mask, 'コース']
            boatracer_df.loc[missing_mask, col] = imputed_vals.clip(upper=100.0)

    if '2連対率(%)' in boatracer_df.columns: boatracer_df['2連対率(%)'] = boatracer_df['2連対率(%)'].fillna(10.0)
    if '3連対率(%)' in boatracer_df.columns: boatracer_df['3連対率(%)'] = boatracer_df['3連対率(%)'].fillna(20.0)
    if '1着率(%)' in boatracer_df.columns: boatracer_df['1着率(%)'] = boatracer_df['1着率(%)'].fillna(5.0)

    return boatracer_df

def scrape_target_race_basic(hd, rno, jcd):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    params = {"rno": str(rno), "jcd": jcd, "hd": hd}

    try:
        # 1. 出走表（racelist）から基礎データと節間成績を取得
        res_list = session.get("https://www.boatrace.jp/owpc/pc/race/racelist", params=params, timeout=15)
        res_list.raise_for_status()
        soup_list = BeautifulSoup(res_list.content, "html.parser")

        racers_info = {}
        tbodys = soup_list.select("tbody.is-fs12")

        # 6艇分のデータがない場合は失敗とみなす
        if len(tbodys) < 6:
            return None

        for tbody in tbodys[:6]:
            waku_td = tbody.select_one("td[class*='is-boatColor']")
            if not waku_td: continue
            waku = int(normalize_text(waku_td.get_text(strip=True)))

            reg_info = tbody.select_one(".is-fs11").get_text(strip=True)
            reg_match = re.search(r'\d{4}', reg_info)
            if not reg_match: continue
            reg_no = int(reg_match.group())

            # 全国2連・3連の取得
            stats_tds = tbody.select("td.is-lineH2")
            nat = [np.nan, np.nan, np.nan]
            if len(stats_tds) > 1:
                raw_texts = stats_tds[1].get_text(separator="|").split("|")
                vals = [float(t.strip().replace('%', '')) for t in raw_texts if re.match(r'^[0-9.]+$', t.strip().replace('%', ''))]
                if len(vals) >= 3: nat = vals[:3]

            tbody_text = tbody.get_text(separator=" ", strip=True)
            f_match = re.search(r'F\s*([0-9])', tbody_text)
            f_count = int(f_match.group(1)) if f_match else 0

            # --- 節間成績の解析 ---
            series_data = []
            for boat_span in tbody.select('span[class^="is-boatColor"]'):
                cls_name = [c for c in boat_span.get('class', []) if 'is-boatColor' in c]
                if not cls_name: continue
                course_str = cls_name[0].replace('is-boatColor', '')
                if not course_str.isdigit(): continue
                course = int(course_str)

                parent = boat_span.parent
                rank_val = np.nan
                # 着順の取得
                rank_a = parent.select_one('a')
                if rank_a:
                    t = rank_a.get_text(strip=True)
                    if t in ['1','2','3','4','5','6']: rank_val = int(t)
                else:
                    t = parent.get_text(strip=True).replace(boat_span.get_text(strip=True), '')
                    m = re.search(r'[1-6]', t)
                    if m: rank_val = int(m.group(0))

                # STの取得
                st_val = np.nan
                st_span = parent.select_one('span.is-fs10')
                if st_span:
                    try: st_val = float(st_span.get_text(strip=True))
                    except: pass

                if not np.isnan(rank_val) or not np.isnan(st_val):
                    series_data.append({'course': course, 'rank': rank_val, 'st': st_val})

            # コース別の平均着順を計算
            c1_ranks = [d['rank'] for d in series_data if d['course'] == 1 and not np.isnan(d['rank'])]
            c123_ranks = [d['rank'] for d in series_data if d['course'] in [1, 2, 3] and not np.isnan(d['rank'])]
            c456_ranks = [d['rank'] for d in series_data if d['course'] in [4, 5, 6] and not np.isnan(d['rank'])]
            sts = [d['st'] for d in series_data if not np.isnan(d['st'])]

            racers_info[waku] = {
                "登録番号": reg_no,
                "枠番": waku,
                "全国2連": nat[1],
                "全国3連": nat[2],
                "F数": f_count,
                "節間_コース1_平均着順": np.mean(c1_ranks) if c1_ranks else np.nan,
                "節間_コース1,2,3_平均着順": np.mean(c123_ranks) if c123_ranks else np.nan,
                "節間_コース4,5,6_平均着順": np.mean(c456_ranks) if c456_ranks else np.nan,
                "節間平均ST": np.mean(sts) if sts else np.nan,
                "展示タイム": np.nan
            }

        # 2. 直前情報（beforeinfo）から展示タイムを取得
        res_before = session.get("https://www.boatrace.jp/owpc/pc/race/beforeinfo", params=params, timeout=15)
        res_before.raise_for_status()
        soup_before = BeautifulSoup(res_before.content, "html.parser")
        for bt_tbody in soup_before.select("tbody.is-fs12"):
            tds = bt_tbody.find("tr").find_all("td")
            if len(tds) >= 5:
                b_waku = int(normalize_text(tds[0].get_text(strip=True)))
                if b_waku in racers_info:
                    try: racers_info[b_waku]["展示タイム"] = float(tds[4].get_text(strip=True))
                    except: pass

        if len(racers_info) != 6:
            return None

        return racers_info
    except Exception as e:
        return None

def plot_probability_chart(p1, p_top2, p_top3, rno):
    labels = [f'{i}号艇' for i in range(1, 7)]
    
    # 棒グラフ用に値を分割 (積み上げで元の確率になるよう調整)
    p1_only = p1
    p2_only = np.clip(p_top2 - p1, 0.0, 1.0)
    p3_only = np.clip(p_top3 - p_top2, 0.0, 1.0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels, x=p1_only, name='1着率', orientation='h',
        marker=dict(color='#FFD700')
    ))
    fig.add_trace(go.Bar(
        y=labels, x=p2_only, name='2着以内率', orientation='h',
        marker=dict(color='#C0C0C0')
    ))
    fig.add_trace(go.Bar(
        y=labels, x=p3_only, name='3着以内率', orientation='h',
        marker=dict(color='#CD7F32')
    ))

    annotations = []
    for i in range(len(labels)):
        total_val = p_top3[i]
        if total_val > 0:
            annotations.append(dict(
                x=total_val + 0.01, 
                y=labels[i],
                text=f'{total_val:.2f}',
                font=dict(size=14, color="black"),
                showarrow=False,
                xanchor='left',
                yanchor='middle'
            ))

    fig.update_layout(
        barmode='stack',
        title=f'第{rno}レース：各艇の着順確率予測',
        xaxis_title='確率',
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 1.1]),
        annotations=annotations,
        margin=dict(l=50, r=50, t=50, b=50),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

def get_custom_series_rank(row):
    try:
        course = int(row['枠番'])
        dash_val = row['節間_コース4,5,6_平均着順']

        if course == 1:
            val = row['節間_コース1_平均着順']
            if pd.notna(val): return val
            return dash_val if pd.notna(dash_val) else 5.0
        elif course in [2, 3]:
            val = row['節間_コース1,2,3_平均着順']
            if pd.notna(val): return val
            return dash_val if pd.notna(dash_val) else 5.0
        elif course in [4, 5, 6]:
            return dash_val if pd.notna(dash_val) else 5.0
        else:
            return 5.0
    except:
        return 5.0

def predict_single_race(hd_input, rno, jcd):
    try:
        features = pickle.load(open("./model_features.pkl", "rb"))
        gate_model = pickle.load(open("./model_gatekeeper.pkl", "rb"))
        gate_features = pickle.load(open("./model_gate_features.pkl", "rb"))

        expert_models = {}
        for cat in ['順当', '準順当']:
            expert_models[cat] = {
                '1st': pickle.load(open(f"./model_1st_{cat}.pkl", "rb")),
                '2nd': pickle.load(open(f"./model_2nd_{cat}.pkl", "rb")),
                'top3': pickle.load(open(f"./model_top3_{cat}.pkl", "rb"))
            }

        boatracer_df = load_and_preprocess_boatracer()

    except Exception as e:
        st.error(f"モデルのロードエラーが発生しました: {e}\n\nCSVファイルとPKLファイルが全て同じフォルダに配置されているか確認してください。")
        return

    st.subheader(f"▼▼ {hd_input} {RACE_COURSES[jcd]} 第{rno}レース 予測結果 ▼▼")

    r_info = scrape_target_race_basic(hd_input, rno, jcd)
    if not r_info:
        st.error("データが取得できませんでした。日程やレース番号が間違っているか、展示タイムがまだ公開されていない可能性があります。")
        return

    try:
        df = pd.DataFrame.from_dict(r_info, orient='index').reset_index(drop=True)
        df['展示タイム'] = df['展示タイム'].fillna(df['展示タイム'].mean() if not df['展示タイム'].isna().all() else 6.80)

        df = pd.merge(df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')
        df = df.sort_values('枠番').reset_index(drop=True)

        df['適性_節間成績'] = df.apply(get_custom_series_rank, axis=1)

        df['展示タイム_mean'] = df['展示タイム'].mean()
        df['展示タイム_diff'] = df['展示タイム'] - df['展示タイム_mean']

        df['節間平均ST'] = df['節間平均ST'].fillna(0.17)
        df['節間平均ST_mean'] = df['節間平均ST'].mean()
        df['節間平均ST_num'] = df['節間平均ST'] - df['節間平均ST_mean']

        gate_base_cols = ['全国3連', '全国2連', 'F数', '適性_節間成績', '節間平均ST', '2連対率(%)', '3連対率(%)', '1着率(%)']
        gate_row = {}
        for idx, row in df.iterrows():
            w = int(row['枠番'])
            for c in gate_base_cols:
                gate_row[f"{w}号艇_{c}"] = row.get(c, np.nan)

        X_gate_df = pd.DataFrame([gate_row])
        for col in gate_features:
            if col not in X_gate_df.columns:
                X_gate_df[col] = np.nan
        X_gate = X_gate_df[gate_features]

        predicted_cat = gate_model.predict(X_gate)[0]

        st.markdown("### 【レース荒れ具合】")
        if predicted_cat in ['穴', '大穴']:
            st.warning(f"荒れる可能性が高い（AI予測カテゴリ: **{predicted_cat}**）ため、予想対象外とします。")
            return
        else:
            st.text(f"順当/準順当/荒れる可能性が高い（AI予測カテゴリ: {predicted_cat}）")

        for col in features:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X_pred = df[features]

        p1_junto = expert_models['順当']['1st'].predict_proba(X_pred)[:, 1]
        p2_junto = expert_models['順当']['2nd'].predict_proba(X_pred)[:, 1]
        p_top3_junto = expert_models['順当']['top3'].predict_proba(X_pred)[:, 1]

        p1_semi = expert_models['準順当']['1st'].predict_proba(X_pred)[:, 1]
        p2_semi = expert_models['準順当']['2nd'].predict_proba(X_pred)[:, 1]
        p_top3_semi = expert_models['準順当']['top3'].predict_proba(X_pred)[:, 1]

        p1_mean = np.clip((p1_junto + p1_semi) / 2, 0.0, 1.0)
        p2_mean = np.clip((p2_junto + p2_semi) / 2, 0.0, 1.0)
        p_top3_mean = np.clip((p_top3_junto + p_top3_semi) / 2, 0.0, 1.0)
        
        prob_1st = p1_mean
        prob_top2 = np.clip(p1_mean + p2_mean, 0.0, 1.0)
        prob_top3 = p_top3_mean

        # --- 【修正】総合レーティング（期待値）の計算 ---
        p_top2_junto = np.clip(p1_junto + p2_junto, 0.0, 1.0)
        p_top2_semi = np.clip(p1_semi + p2_semi, 0.0, 1.0)

        # 新しい重み付け：(1着確率 × 4) + (2着以内確率 × 2) + (3着以内確率 × 1)
        rating_junto = (p1_junto * 4) + (p_top2_junto * 2) + (p_top3_junto * 1)
        rating_semi = (p1_semi * 4) + (p_top2_semi * 2) + (p_top3_semi * 1)
        total_rating = rating_junto + rating_semi

        st.markdown("### 【各艇AI総合レーティング】")
        for w in range(len(total_rating)):
            waku = int(df.iloc[w]['枠番'])
            reg_num = int(df.iloc[w]['登録番号'])
            st.text(f"  {waku}号艇 (登録番号: {reg_num}) : {total_rating[w]:.2f} pt")

        st.markdown("### 【各艇の着順確率予想】")
        plot_probability_chart(prob_1st, prob_top2, prob_top3, rno)

        sanrentan_results = []
        for perm in itertools.permutations(range(1, 7), 3):
            b1, b2, b3 = perm
            score_junto = p1_junto[b1-1] * p2_junto[b2-1] * rating_junto[b3-1]
            score_semi = p1_semi[b1-1] * p2_semi[b2-1] * rating_semi[b3-1]
            total_score = score_junto + score_semi
            sanrentan_results.append((b1, b2, b3, total_score))

        sanrenpuku_scores = {}
        for combo in itertools.combinations(range(1, 7), 3):
            combo_tuple = tuple(sorted(combo))
            sanrenpuku_scores[combo_tuple] = 0.0

        for r in sanrentan_results:
            b1, b2, b3, score = r
            combo_tuple = tuple(sorted((b1, b2, b3)))
            sanrenpuku_scores[combo_tuple] += score

        sanrenpuku_results = sorted(sanrenpuku_scores.items(), key=lambda x: x[1], reverse=True)
        sanrentan_results.sort(key=lambda x: x[3], reverse=True)

        st.markdown("### 【3連単予想上位5点】")
        for i in range(5):
            r = sanrentan_results[i]
            st.text(f"  {i+1}位: {r[0]}-{r[1]}-{r[2]} (Score: {r[3]*1000:.3f})")

        st.markdown("### 【厳選3連複予想】")
        for i in range(2):
            combo, score = sanrenpuku_results[i]
            st.text(f"  ★ {combo[0]} = {combo[1]} = {combo[2]} (Score: {score*1000:.3f})")

    except Exception as e:
        st.error(f"データ処理中にエラーが発生しました: {e}")

# --- アプリのメイン画面 ---
if __name__ == "__main__":
    st.title("ボートレースAI予測システム")
    st.markdown("---")
    
    # ユーザー入力エリアを3列に分割して配置
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 日程選択UI (カレンダー形式)
        today = datetime.date.today()
        selected_date = st.date_input("📅 対象の日程", today)
        input_hd = selected_date.strftime("%Y%m%d")
        
    with col2:
        # ボートレース場選択UI (プルダウン)
        course_options = [f"{code} {name}" for code, name in RACE_COURSES.items()]
        # デフォルトを尼崎(13)にするためにindexを指定
        selected_course = st.selectbox("📍 ボートレース場", course_options, index=12)
        # 選択された文字列の先頭2文字(コード)を抽出
        input_jcd = selected_course[:2]
        
    with col3:
        # レース番号選択UI (プルダウン)
        input_rno = st.selectbox("🚤 レース番号", list(range(1, 13)), index=0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("予測を開始する", type="primary", use_container_width=True):
        with st.spinner("データ取得およびAIによる予測を実行中..."):
            predict_single_race(input_hd, int(input_rno), input_jcd)
