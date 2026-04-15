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
import time
import datetime
import plotly.graph_objects as go

# インストールした日本語フォントを使うよう指定
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

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

    pct_cols = ['3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)']
    for col in pct_cols:
        boatracer_df[col] = boatracer_df[col].apply(clean_pct)

    boatracer_df['平均ST'] = pd.to_numeric(boatracer_df['平均ST'], errors='coerce').fillna(0.17)
    boatracer_df['コース'] = pd.to_numeric(boatracer_df['コース'], errors='coerce')

    return boatracer_df

def scrape_target_race_basic(hd, rno, jcd):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    params = {"rno": str(rno), "jcd": jcd, "hd": hd}

    res_list = session.get("https://www.boatrace.jp/owpc/pc/race/racelist", params=params, timeout=15)
    soup_list = BeautifulSoup(res_list.content, "html.parser")

    def parse_values(td_element):
        if not td_element: return [np.nan, np.nan, np.nan]
        raw_texts = td_element.get_text(separator="|").split("|")
        vals = [float(t.strip().replace('%', '')) for t in raw_texts if re.match(r'^[0-9.]+$', t.strip().replace('%', ''))]
        while len(vals) < 3: vals.append(np.nan)
        return vals[:3]

    racers_info = {}
    for tbody in soup_list.select("tbody.is-fs12"):
        waku_td = tbody.select_one("td[class*='is-boatColor']")
        if not waku_td: continue
        waku = int(normalize_text(waku_td.get_text(strip=True)))
        reg_info = tbody.select_one(".is-fs11").get_text(strip=True)
        reg_match = re.search(r'\d{4}', reg_info)

        if reg_match:
            stats_tds = tbody.select("td.is-lineH2")
            nat = parse_values(stats_tds[1]) if len(stats_tds) > 1 else [np.nan]*3
            mot = parse_values(stats_tds[3]) if len(stats_tds) > 3 else [np.nan]*3

            tbody_text = tbody.get_text(separator=" ", strip=True)
            f_match = re.search(r'F\s*([0-9])', tbody_text)
            f_count = int(f_match.group(1)) if f_match else 0

            # 選手名のスクレイピングを追加
            name_node = tbody.select_one("a[href*='/owpc/pc/data/racer']")
            if name_node:
                racer_name = normalize_text(name_node.get_text())
            else:
                name_node_fallback = tbody.select_one(".is-fs18")
                racer_name = normalize_text(name_node_fallback.get_text()) if name_node_fallback else "不明"

            racers_info[waku] = {
                "登録番号": int(reg_match.group()),
                "選手名": racer_name,  # 選手名を追加
                "全国勝率": nat[0],
                "全国3連": nat[2],
                "モーター3連": mot[2],
                "展示タイム": np.nan,
                "F数": f_count
            }

    res_before = session.get("https://www.boatrace.jp/owpc/pc/race/beforeinfo", params=params, timeout=15)
    soup_before = BeautifulSoup(res_before.content, "html.parser")
    for bt_tbody in soup_before.select("tbody.is-fs12"):
        tds = bt_tbody.find("tr").find_all("td")
        if len(tds) >= 5:
            b_waku = int(normalize_text(tds[0].get_text(strip=True)))
            if b_waku in racers_info:
                try: racers_info[b_waku]["展示タイム"] = float(tds[4].get_text(strip=True))
                except: pass

    return racers_info

def plot_probability_chart(p1, p_top2, p_top3, rno):
    labels = [f'{i}号艇' for i in range(1, 7)]
    
    p1_only = p1
    p2_only = np.clip(p_top2 - p1, 0.0, 1.0)
    p3_only = np.clip(p_top3 - p_top2, 0.0, 1.0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels, x=p1_only, name='1着率', orientation='h',
        marker=dict(color='#FFD700')
    ))
    fig.add_trace(go.Bar(
        y=labels, x=p2_only, name='2着率', orientation='h',
        marker=dict(color='#C0C0C0')
    ))
    fig.add_trace(go.Bar(
        y=labels, x=p3_only, name='3着率', orientation='h',
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
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

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

        race_df_hist = pd.read_csv("./race.data.csv", header=1)
        boatracer_df = load_and_preprocess_boatracer()

    except Exception as e:
        st.error(f"モデルのロードエラーが発生しました: {e}\n\nCSVファイルとPKLファイルが全て同じフォルダにアップロードされているか確認してください。")
        return

    st.subheader(f"▼▼ {hd_input} {RACE_COURSES[jcd]} 第{rno}レース 予測結果 ▼▼")

    try:
        r_info = scrape_target_race_basic(hd_input, rno, jcd)
        if not r_info:
            st.error("データが取得できませんでした。日程やレース番号を確認してください。")
            return

        df = pd.DataFrame.from_dict(r_info, orient='index').reset_index().rename(columns={'index': '枠番'})
        df['展示タイム'] = df['展示タイム'].fillna(df['展示タイム'].mean() if not df['展示タイム'].isna().all() else 6.80)

        df = pd.merge(df, boatracer_df, left_on=['登録番号', '枠番'], right_on=['登録番号', 'コース'], how='left')
        df = df.sort_values('枠番').reset_index(drop=True)

        mock_course_stats = []
        for idx, row in df.iterrows():
            reg_num = row['登録番号']
            waku = int(row['枠番'])
            hist = race_df_hist[race_df_hist['登録番号'] == reg_num]
            cols = ['コース1_平均着順', 'コース2_平均着順', 'コース3_平均着順', 'コース4_平均着順', 'コース5_平均着順', 'コース6_平均着順']

            s_rank = 5.0
            a_rank = 5.0

            if not hist.empty:
                last_rec = hist.iloc[-1]
                vals = pd.to_numeric(last_rec[cols], errors='coerce').values

                if waku == 1:
                    val_c1 = vals[0]
                    if not np.isnan(val_c1):
                        s_rank = float(val_c1)
                    else:
                        v_vals = [v for v in vals[0:2] if not np.isnan(v)]
                        if v_vals: s_rank = np.mean(v_vals)
                elif waku == 6:
                    v_vals = [v for v in vals[4:6] if not np.isnan(v)]
                    if v_vals: s_rank = np.mean(v_vals)
                elif 1 < waku < 6:
                    v_vals = [v for v in vals[waku-2:waku+1] if not np.isnan(v)]
                    if v_vals: s_rank = np.mean(v_vals)

                if np.any(~np.isnan(vals)):
                    a_rank = np.mean([v for v in vals if not np.isnan(v)])

            mock_course_stats.append({'smoothed_course_rank': s_rank, '全コース平均着順': a_rank})

        stats_df = pd.DataFrame(mock_course_stats)
        df['smoothed_course_rank'] = stats_df['smoothed_course_rank']
        df['全コース平均着順'] = stats_df['全コース平均着順']

        df['モーター_mean'] = df['モーター3連'].mean()
        df['モーター_num'] = df['モーター3連'] - df['モーター_mean']
        df['展示タイム_mean'] = df['展示タイム'].mean()
        df['展示タイム_diff'] = df['展示タイム'] - df['展示タイム_mean']
        df['平均ST_num'] = df['平均ST'] - df['平均ST'].mean()

        gate_base_cols = ['全国勝率', '3連対率(%)', '1着率(%)', '2着率(%)', '3着率(%)', 'F数', 'smoothed_course_rank']
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

        p3_junto = np.clip(p_top3_junto - (p1_junto + p2_junto), 0.0, 1.0)
        p3_semi = np.clip(p_top3_semi - (p1_semi + p2_semi), 0.0, 1.0)

        rating_junto = (p1_junto * 10) + (p2_junto * 7) + (p3_junto * 4)
        rating_semi = (p1_semi * 10) + (p2_semi * 7) + (p3_semi * 4)
        total_rating = rating_junto + rating_semi

        st.markdown("### 【各艇AI総合レーティング】")
        for w in range(len(total_rating)):
            waku = int(df.iloc[w]['枠番'])
            # dfに保存された選手名を取得
            racer_name = df.iloc[w]['選手名'] if '選手名' in df.columns else ""
            st.text(f"  {waku}号艇 {racer_name} : {total_rating[w]:.2f} pt")

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

        st.markdown("### 【3連複予想】")
        for i in range(2):
            combo, score = sanrenpuku_results[i]
            st.text(f"  ★ {combo[0]} = {combo[1]} = {combo[2]} (Score: {score*1000:.3f})")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# --- アプリのメイン画面 ---
if __name__ == "__main__":
    st.title("ボートレース予測システム")
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

    st.markdown("<br>", unsafe_allow_html=True) # 少し余白を空ける

    if st.button("予測を開始する", type="primary", use_container_width=True):
        with st.spinner("データ取得およびAIによる予測を実行中..."):
            predict_single_race(input_hd, int(input_rno), input_jcd)
