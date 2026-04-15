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
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# import japanize_matplotlib  <-- この行を削除します

# 代わりに以下の1行を追加して、インストールしたフォントを使うよう指定します
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

warnings.filterwarnings('ignore')

# 尼崎競艇場の場コード
JCD = "13"

def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFKC', text).replace(" ", "").replace("　", "").strip()

def load_and_preprocess_boatracer():
    # パスをローカルに変更
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

def scrape_target_race_basic(hd, rno):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    params = {"rno": str(rno), "jcd": JCD, "hd": hd}

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

            racers_info[waku] = {
                "登録番号": int(reg_match.group()),
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
    x = np.arange(len(labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, p1, width, label='1着率', color='gold')
    rects2 = ax.bar(x, p_top2, width, label='2着以内率', color='silver')
    rects3 = ax.bar(x + width, p_top3, width, label='3着以内率', color='peru')

    ax.set_ylabel('確率')
    ax.set_title(f'第{rno}レース：各艇の着順確率予測')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    # Streamlit用にグラフを出力
    st.pyplot(fig)

def predict_single_race(hd_input, rno):
    try:
        # パスをローカルに変更
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

    st.subheader(f"▼▼ {hd_input} 第{rno}レース 予測結果 ▼▼")

    try:
        r_info = scrape_target_race_basic(hd_input, rno)
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

        if predicted_cat in ['穴', '大穴']:
            st.warning(f"荒れる可能性が高い（AI予測カテゴリ: **{predicted_cat}**）ため、予想対象外とします。")
            return

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

        plot_probability_chart(prob_1st, prob_top2, prob_top3, rno)

        p3_junto = np.clip(p_top3_junto - (p1_junto + p2_junto), 0.0, 1.0)
        p3_semi = np.clip(p_top3_semi - (p1_semi + p2_semi), 0.0, 1.0)

        rating_junto = (p1_junto * 10) + (p2_junto * 7) + (p3_junto * 4)
        rating_semi = (p1_semi * 10) + (p2_semi * 7) + (p3_semi * 4)
        total_rating = rating_junto + rating_semi

        st.markdown(f"### AI総合レーティング (予測カテゴリ: {predicted_cat})")
        for w in range(len(total_rating)):
            st.text(f"  {w+1}号艇: {total_rating[w]:.2f} pt")

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

        st.markdown("### 3連単 総合予想スコア上位5点")
        for i in range(5):
            r = sanrentan_results[i]
            st.text(f"  {i+1}位: {r[0]}-{r[1]}-{r[2]} (Score: {r[3]*1000:.3f})")

        st.markdown("### 厳選3連複予想")
        for i in range(2):
            combo, score = sanrenpuku_results[i]
            st.success(f"  ★ {combo[0]} = {combo[1]} = {combo[2]} (3連複Score: {score*1000:.3f})")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# --- アプリのメイン画面 ---
if __name__ == "__main__":
    st.title("競艇 AI予測システム")
    st.markdown("---")
    
    # ユーザー入力エリア
    col1, col2 = st.columns(2)
    with col1:
        input_hd = st.text_input("対象の日程 (例: 20260415)", value="20260415")
    with col2:
        input_rno = st.number_input("対象のレース番号", min_value=1, max_value=12, value=1)

    if st.button("予測を開始する", type="primary"):
        with st.spinner("データ取得およびAIによる予測を実行中..."):
            predict_single_race(input_hd, int(input_rno))
