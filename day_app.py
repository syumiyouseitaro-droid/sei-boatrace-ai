import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import re
import unicodedata
import time
from datetime import datetime
import pytz

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="厳選！勝負レース自動抽出", page_icon="🚤", layout="centered")

# ==========================================
# スクレイピング関数
# ==========================================
def scrape_todays_target_races(target_date):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    jcd_dict = {
        f"{i:02d}": name for i, name in enumerate([
            "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
            "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
            "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
            "下関", "若松", "芦屋", "福岡", "唐津", "大村"
        ], 1)
    }

    results = []
    
    # UI用のプログレスバーとステータス表示
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (jcd, venue_name) in enumerate(jcd_dict.items()):
        # 進捗の更新
        progress = (idx + 1) / len(jcd_dict)
        progress_bar.progress(progress)
        status_text.text(f"🔍 検索中... 【{venue_name}】 を確認しています ({idx+1}/24)")

        url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={jcd}&hd={target_date}"
        
        try:
            res = session.get(url, timeout=30)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, "html.parser")
            
            text_content = soup.get_text()
            if "該当のデータがありません" in text_content or "中止" in text_content:
                continue
                
            # ① 一般戦かどうかの判定
            is_ippan = True
            for img in soup.find_all('img'):
                src = img.get('src', '').lower()
                if any(g in src for g in ['_sg.', '_g1.', '_g2.', '_g3.']) or \
                   any(f'icon_{g.lower()}' in src for g in ['sg', 'g1', 'g2', 'g3']) or \
                   (img.get('class') and any('g1' in c or 'g2' in c or 'g3' in c or 'sg' in c for c in img.get('class'))):
                    if 'ippan' not in src:
                        is_ippan = False
                        break

            heading_title = soup.find(class_=re.compile(r'heading2_title'))
            if heading_title and heading_title.get('class'):
                title_classes = " ".join(heading_title.get('class')).lower()
                if re.search(r'title(sg|g1|g2|g3)$', title_classes):
                    is_ippan = False

            if not is_ippan:
                continue

            # ② 現在何日目かを抽出する
            current_day = ""
            day_pattern = re.compile(r'(初日|[1-7]日目|最終日)')
            
            detail_div = soup.find(class_=re.compile(r'heading2_titleDetail'))
            if detail_div:
                norm_text = unicodedata.normalize('NFKC', detail_div.get_text(strip=True))
                match = day_pattern.search(norm_text)
                if match:
                    current_day = match.group(1)
            
            if not current_day:
                active_elements = soup.find_all(class_=re.compile(r'is-active'))
                for el in active_elements:
                    norm_text = unicodedata.normalize('NFKC', el.get_text(strip=True))
                    match = day_pattern.search(norm_text)
                    if match:
                        current_day = match.group(1)
                        break

            if not current_day:
                for el in soup.find_all('li'):
                    if 'active' in " ".join(el.get('class', [])).lower():
                        norm_text = unicodedata.normalize('NFKC', el.get_text(strip=True))
                        match = day_pattern.search(norm_text)
                        if match:
                            current_day = match.group(1)
                            break

            valid_days = ["4日目", "5日目", "6日目", "7日目", "最終日"]
            if current_day not in valid_days:
                continue

            # ③ 1Rの締切時間を抽出する
            first_race_time = None
            all_times = []
            
            for td in soup.find_all("td"):
                txt = td.get_text(strip=True)
                if re.match(r'^\d{1,2}:\d{2}$', txt):
                    all_times.append(txt)
            
            if all_times:
                def time_to_minutes(t_str):
                    h, m = map(int, t_str.split(':'))
                    return h * 60 + m
                
                all_times_unique = sorted(list(set(all_times)), key=time_to_minutes)
                first_race_time = all_times_unique[0]

            if first_race_time:
                results.append({
                    "venue": venue_name,
                    "day": current_day,
                    "1r_time": first_race_time,
                    "url": url
                })
            
            time.sleep(0.5) # Streamlit上での待ち時間を少し短縮

        except Exception:
            pass # エラー時はスキップして次へ

    # 検索完了後のクリーンアップ
    status_text.empty()
    progress_bar.empty()
    
    # 1Rの時間が早い順（時間割順）にソートして返す
    return sorted(results, key=lambda x: x['1r_time'])

# ==========================================
# UI表示部分 (Streamlit メイン)
# ==========================================
st.title("🚤 本日の勝負レース抽出ツール")
st.markdown("全国24箇所のボートレース場から、**「一般戦」**かつ**「4日目以降（最終日含む）」**の開催場を自動で探し出します。")

# 日本時間の「現在」を取得（サーバーの場所に依存しないようにする）
jst = pytz.timezone('Asia/Tokyo')
now_jst = datetime.now(jst)
today_str = now_jst.strftime("%Y%m%d")
display_date = now_jst.strftime("%Y年%m月%d日")

st.info(f"📅 取得対象日: **{display_date}** (自動取得)")

if st.button("🚀 検索を開始する", type="primary", use_container_width=True):
    with st.spinner("公式サイトからデータを取得しています...（約10〜15秒かかります）"):
        extracted_races = scrape_todays_target_races(today_str)
        
    st.divider()
    
    if not extracted_races:
        st.warning("😭 本日、条件（一般戦 ＆ 4日目以降）に一致する開催レースはありません。")
    else:
        st.success(f"🎯 条件に一致する開催が **{len(extracted_races)}場** 見つかりました！")
        
        # 結果を綺麗なカード形式で表示
        for race in extracted_races:
            html_card = f"""
            <div style="
                background: linear-gradient(135deg, #ffffff, #f0f4f8);
                border-left: 5px solid #0056b3;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                box-shadow: 1px 2px 5px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">🌊 {race['venue']}</h3>
                    <span style="font-size: 0.9rem; color: #7f8c8d; font-weight: bold;">{race['day']}</span>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">1R締切</p>
                    <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: #e74c3c;">🕒 {race['1r_time']}</p>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)
