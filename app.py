import os
import re
import csv
import math
import numpy as np
import pandas as pd
import streamlit as st
import folium

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_folium import st_folium

# =====================================================
# SAYFA AYARLARI
# =====================================================
st.set_page_config(
    page_title="Akıllı Kuaför Tavsiye Sistemi",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #0f172a;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    border: 1px solid #e5e7eb;
    text-align: center;
}
.big-reco {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    padding: 28px;
    border-radius: 22px;
    box-shadow: 0 14px 34px rgba(37, 99, 235, 0.24);
    margin-top: 10px;
    margin-bottom: 20px;
}
.small-note {
    color: #475569;
    font-size: 14px;
}
.score-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.18);
    margin-right: 8px;
    margin-bottom: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 💇 Akıllı Kuaför / Berber / Güzellik Merkezi Öneri Sistemi")
st.markdown(
    "<div class='small-note'>SBERT + Cosine Similarity + Yıldız + Harita + Gerçek Mesafe ile öneri</div>",
    unsafe_allow_html=True
)

# =====================================================
# MODEL VE GEOCODER
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def get_geocoder():
    geolocator = Nominatim(user_agent="kuafor_tavsiye_sistemi_v1")
    geocode_func = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=1,
        swallow_exceptions=True
    )
    return geocode_func

model = load_model()
geocode = get_geocoder()

@st.cache_data(show_spinner=False)
def embed_texts(texts):
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

# =====================================================
# YARDIMCI FONKSİYONLAR
# =====================================================
def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("ı", "i").replace("İ", "i").replace("ş", "s").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ç", "c")
    s = re.sub(r"\s+", " ", s)
    return s

def find_col(df, candidates):
    cols = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        candn = norm_col(cand)
        if candn in cols:
            return cols[candn]
    for cand in candidates:
        candn = norm_col(cand)
        for cn, orig in cols.items():
            if candn in cn:
                return orig
    return None

def clean_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_float(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", ".")
    x = re.sub(r"[^0-9.]+", "", x)
    try:
        return float(x)
    except:
        return np.nan

def minmax(series):
    arr = np.asarray(series, dtype=float)
    if len(arr) == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0)
    mn, mx = np.min(arr), np.max(arr)
    if abs(mx - mn) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def load_csv_robust(path_or_upload, default_sep=None):
    if path_or_upload is None:
        return None

    if isinstance(path_or_upload, str):
        if not os.path.exists(path_or_upload):
            return None
        source = path_or_upload
    else:
        source = path_or_upload

    if default_sep:
        try:
            return pd.read_csv(source, sep=default_sep, encoding="utf-8-sig")
        except:
            pass

    try:
        return pd.read_csv(source, encoding="utf-8-sig")
    except:
        pass

    try:
        return pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    except:
        pass

    try:
        return pd.read_csv(
            source,
            sep=None,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
            encoding="utf-8-sig"
        )
    except Exception as e:
        st.error(f"CSV okunamadı: {e}")
        return None

def build_business_docs(df, col_biz, col_text, col_cat=None, col_rating=None, col_addr=None):
    d = df.copy()
    d[col_text] = d[col_text].apply(clean_text)

    if col_rating and col_rating in d.columns:
        d[col_rating] = d[col_rating].apply(to_float)

    agg_dict = {
        col_text: lambda x: " ".join([t for t in x.tolist() if t])
    }

    if col_cat and col_cat in d.columns:
        agg_dict[col_cat] = "first"
    if col_rating and col_rating in d.columns:
        agg_dict[col_rating] = "mean"
    if col_addr and col_addr in d.columns:
        agg_dict[col_addr] = "first"

    agg = d.groupby(col_biz).agg(agg_dict).reset_index()

    if col_cat and col_cat not in agg.columns:
        agg[col_cat] = ""
    if col_rating and col_rating not in agg.columns:
        agg[col_rating] = np.nan
    if col_addr and col_addr not in agg.columns:
        agg[col_addr] = ""

    return agg

def geocode_single_address(address):
    address = clean_text(address)
    if not address:
        return np.nan, np.nan
    try:
        location = geocode(address)
        if location is None:
            return np.nan, np.nan
        return float(location.latitude), float(location.longitude)
    except:
        return np.nan, np.nan

def enrich_business_coordinates(df, address_col, cache_file="isletme_koordinat_cache.csv"):
    df = df.copy()

    if address_col not in df.columns:
        df["Latitude"] = np.nan
        df["Longitude"] = np.nan
        return df

    df[address_col] = df[address_col].fillna("").astype(str)

    cache_cols = ["AdresKey", "Latitude", "Longitude"]
    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file, encoding="utf-8-sig")
            if not set(cache_cols).issubset(cache_df.columns):
                cache_df = pd.DataFrame(columns=cache_cols)
        except:
            cache_df = pd.DataFrame(columns=cache_cols)
    else:
        cache_df = pd.DataFrame(columns=cache_cols)

    cache_map = dict(zip(cache_df["AdresKey"].astype(str), zip(cache_df["Latitude"], cache_df["Longitude"])))

    lat_list = []
    lon_list = []
    new_rows = []

    progress = st.progress(0, text="Adresler koordinata çevriliyor...")
    total = max(len(df), 1)

    for i, addr in enumerate(df[address_col].tolist()):
        addr_key = clean_text(addr)

        if addr_key in cache_map:
            lat, lon = cache_map[addr_key]
        else:
            lat, lon = geocode_single_address(addr_key)
            new_rows.append({"AdresKey": addr_key, "Latitude": lat, "Longitude": lon})

        lat_list.append(lat)
        lon_list.append(lon)
        progress.progress(min((i + 1) / total, 1.0), text=f"Adresler koordinata çevriliyor... {i+1}/{total}")

    progress.empty()

    df["Latitude"] = lat_list
    df["Longitude"] = lon_list

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache_df = pd.concat([cache_df, new_df], ignore_index=True).drop_duplicates(subset=["AdresKey"], keep="last")
        cache_df.to_csv(cache_file, index=False, encoding="utf-8-sig")

    return df

def haversine_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

def compute_distance_scores(df, user_lat, user_lon):
    dists = []
    for _, row in df.iterrows():
        d = haversine_km(user_lat, user_lon, row["Latitude"], row["Longitude"])
        dists.append(d)

    dists = np.array(dists, dtype=float)
    valid = ~np.isnan(dists)

    if valid.sum() == 0:
        return dists, np.zeros(len(dists), dtype=float)

    max_dist = np.nanmax(dists[valid])
    if max_dist < 1e-12:
        score = np.ones(len(dists), dtype=float)
    else:
        score = np.where(valid, 1 - (dists / max_dist), 0.0)
        score = np.clip(score, 0, 1)

    return dists, score

def recommend_one_business(
    agg,
    emb_biz,
    query,
    user_lat,
    user_lon,
    col_biz,
    col_addr=None,
    col_cat=None,
    col_rating=None,
    top_n=10,
    min_rating=0.0,
    max_distance_km=20.0,
    sel_cats=None,
    w_text=0.50,
    w_rating=0.30,
    w_distance=0.20
):
    q = clean_text(query)
    if not q:
        return pd.DataFrame(), None, agg.copy()

    q_vec = model.encode([q], show_progress_bar=False, normalize_embeddings=True)
    text_scores = cosine_similarity(q_vec, emb_biz).flatten()

    out = agg.copy()
    out["Text_Score"] = text_scores

    if sel_cats and col_cat and col_cat in out.columns:
        out = out[out[col_cat].astype(str).isin(sel_cats)]

    if col_rating and col_rating in out.columns:
        out["Rating_Num"] = out[col_rating].apply(to_float).fillna(0)
    else:
        out["Rating_Num"] = 0.0

    if min_rating > 0:
        out = out[out["Rating_Num"] >= float(min_rating)]

    if "Latitude" not in out.columns or "Longitude" not in out.columns:
        out["Latitude"] = np.nan
        out["Longitude"] = np.nan

    out["Distance_km"], out["Distance_Score"] = compute_distance_scores(out, user_lat, user_lon)

    if max_distance_km > 0:
        out = out[(out["Distance_km"].isna()) | (out["Distance_km"] <= max_distance_km)]

    out["Text_N"] = minmax(out["Text_Score"])
    out["Rating_N"] = (out["Rating_Num"] / 5.0).clip(0, 1)
    out["Distance_N"] = out["Distance_Score"].clip(0, 1)

    out["Final_Score"] = (
        w_text * out["Text_N"] +
        w_rating * out["Rating_N"] +
        w_distance * out["Distance_N"]
    )

    out = out.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    result = pd.DataFrame()
    result["İşletme"] = out[col_biz].astype(str)
    result["Kategori"] = out[col_cat].astype(str) if col_cat and col_cat in out.columns else ""
    result["Adres"] = out[col_addr].astype(str) if col_addr and col_addr in out.columns else ""
    result["Yıldız"] = out["Rating_Num"].round(2)
    result["Mesafe (km)"] = out["Distance_km"].round(2)
    result["Benzerlik Skoru"] = out["Text_N"].round(4)
    result["Yıldız Skoru"] = out["Rating_N"].round(4)
    result["Konum Skoru"] = out["Distance_N"].round(4)
    result["Final Skor"] = out["Final_Score"].round(4)
    result["Latitude"] = out["Latitude"]
    result["Longitude"] = out["Longitude"]

    best_one = result.iloc[0].to_dict() if len(result) > 0 else None
    top_list = result.head(top_n).copy()

    return top_list, best_one, out

def build_google_links(user_lat, user_lon, dest_lat, dest_lon):
    konum_link = f"https://www.google.com/maps/search/?api=1&query={dest_lat},{dest_lon}"

    if not pd.isna(user_lat) and not pd.isna(user_lon):
        yol_tarifi = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={user_lat},{user_lon}"
            f"&destination={dest_lat},{dest_lon}"
            f"&travelmode=driving"
        )
    else:
        yol_tarifi = f"https://www.google.com/maps/dir/?api=1&destination={dest_lat},{dest_lon}"

    return konum_link, yol_tarifi

def create_map(user_lat, user_lon, selected_df):
    center_lat = user_lat if not pd.isna(user_lat) else 37.5858
    center_lon = user_lon if not pd.isna(user_lon) else 36.9371
    zoom_level = 13

    if len(selected_df) > 0:
        first_row = selected_df.iloc[0]
        if not pd.isna(first_row["Latitude"]) and not pd.isna(first_row["Longitude"]):
            center_lat = float(first_row["Latitude"])
            center_lon = float(first_row["Longitude"])
            zoom_level = 14

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, control_scale=True)

    if not pd.isna(user_lat) and not pd.isna(user_lon):
        folium.Marker(
            [user_lat, user_lon],
            tooltip="Senin konumun",
            popup="Başlangıç konumu",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(fmap)

    for _, row in selected_df.iterrows():
        lat = row.get("Latitude", np.nan)
        lon = row.get("Longitude", np.nan)

        if pd.isna(lat) or pd.isna(lon):
            continue

        google_maps_link, directions_link = build_google_links(user_lat, user_lon, lat, lon)

        popup_html = f"""
        <b>{row['İşletme']}</b><br>
        Kategori: {row['Kategori']}<br>
        Yıldız: {row['Yıldız']}<br>
        Mesafe: {row['Mesafe (km)']} km<br>
        Final Skor: {row['Final Skor']}<br>
        Adres: {row['Adres']}<br><br>
        <a href="{google_maps_link}" target="_blank">📍 Konumu Aç</a><br>
        <a href="{directions_link}" target="_blank">🚗 Yol Tarifi Al</a>
        """

        folium.Marker(
            [lat, lon],
            tooltip=row["İşletme"],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(fmap)

        if not pd.isna(user_lat) and not pd.isna(user_lon):
            folium.PolyLine(
                locations=[[user_lat, user_lon], [float(lat), float(lon)]],
                weight=4,
                opacity=0.8
            ).add_to(fmap)

    return fmap

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("⚙️ Ayarlar")

    dataset_choice = st.radio(
        "Veri seti seç",
        ["Google Maps Veri Seti", "Hazır (Yelp/Kaggle) Veri Seti"],
        index=0
    )

    st.divider()

    maps_upload = st.file_uploader("Google Maps CSV yükle", type=["csv"])
    yelp_upload = st.file_uploader("Yelp/Kaggle CSV yükle", type=["csv"])

    st.divider()
    st.subheader("💬 Hizmet Tercihi")

    default_query = "hijyenik, ilgili personel, uygun fiyat, kaliteli saç kesimi"
    query = st.text_area("Nasıl bir yer arıyorsun?", value=default_query, height=120)

    st.subheader("📍 Konum Bilgileri")
    city = st.text_input("Şehir", value="Kahramanmaraş")
    district = st.text_input("İlçe", value="Onikişubat")
    user_address = st.text_input("Açık adres / mahalle / cadde", value="")

    st.divider()
    st.subheader("⭐ Filtreler")
    min_rating = st.slider("Minimum yıldız", 0.0, 5.0, 3.5, 0.1)
    max_distance_km = st.slider("Maksimum mesafe (km)", 1, 50, 15)
    top_n = st.slider("Alternatif öneri sayısı", 3, 15, 5)

    st.divider()
    st.subheader("🧠 Kullanılan Ağırlıklar")

    w_text = 0.50
    w_rating = 0.30
    w_distance = 0.20

    st.info(
        f"Benzerlik: {w_text:.2f}\n\nYıldız: {w_rating:.2f}\n\nKonum: {w_distance:.2f}"
    )

    run_btn = st.button("🚀 En Uygun İşletmeyi Bul", use_container_width=True)

# =====================================================
# VERİ YOLU
# =====================================================
DEFAULT_MAPS = r"D:\kuafor_ui\kahramanmaras_yorumlar.csv"
DEFAULT_YELP = r"D:\kuafor_ui\yelp_beauty_reviews.csv"

df_maps = load_csv_robust(maps_upload if maps_upload else DEFAULT_MAPS, default_sep=";")
df_yelp = load_csv_robust(yelp_upload if yelp_upload else DEFAULT_YELP)

df_active = df_maps if dataset_choice == "Google Maps Veri Seti" else df_yelp

if "first_run" not in st.session_state:
    st.session_state.first_run = True
    run_btn = True

if df_active is None:
    st.error("Veri seti okunamadı. Dosya yolunu veya CSV yüklemeyi kontrol et.")
    st.stop()

# =====================================================
# KOLON TESPİTİ
# =====================================================
col_biz = find_col(df_active, ["Isletme Adi", "İsletme Adi", "Isletme", "name", "business"])
col_text = find_col(df_active, ["Yorum", "text", "review", "comment"])
col_cat = find_col(df_active, ["Kategori", "categories", "category"])
col_rating = find_col(df_active, ["Puan", "stars", "rating", "Isletme Ort Puan"])
col_addr = find_col(df_active, ["Adres", "address", "formatted_address"])
col_user = find_col(df_active, ["Yorum Sahibi", "user", "user_id", "author"])

if col_biz is None or col_text is None:
    st.error("Veri setinde işletme adı veya yorum kolonu bulunamadı.")
    st.write("Bulunan kolonlar:", df_active.columns.tolist())
    st.stop()

if col_addr is None:
    st.error("Harita ve mesafe hesabı için adres kolonu gerekli.")
    st.write("Bulunan kolonlar:", df_active.columns.tolist())
    st.stop()

# =====================================================
# KULLANICI KONUMU
# =====================================================
full_user_address = ", ".join([x for x in [user_address, district, city, "Türkiye"] if clean_text(x)])
user_lat, user_lon = geocode_single_address(full_user_address)

if pd.isna(user_lat) or pd.isna(user_lon):
    st.warning("Kullanıcı adresi koordinata çevrilemedi. Daha açık bir adres, mahalle veya cadde bilgisi gir.")
    st.stop()

# =====================================================
# ÜST METRİKLER
# =====================================================
total_reviews = len(df_active)
total_business = df_active[col_biz].nunique()
total_users = df_active[col_user].nunique() if col_user else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><h3>{total_business}</h3><div>İşletme Sayısı</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><h3>{total_reviews}</h3><div>Yorum Sayısı</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><h3>{total_users}</h3><div>Kullanıcı Sayısı</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# VERİ HAZIRLAMA
# =====================================================
agg = build_business_docs(
    df_active,
    col_biz=col_biz,
    col_text=col_text,
    col_cat=col_cat,
    col_rating=col_rating,
    col_addr=col_addr
)

st.info("İlk çalıştırmada işletme adresleri koordinata çevrileceği için biraz bekleyebilirsin. Sonraki açılışlarda cache kullanılacak.")

agg = enrich_business_coordinates(
    agg,
    address_col=col_addr,
    cache_file="isletme_koordinat_cache.csv"
)

sel_cats = []
if col_cat and col_cat in agg.columns:
    cat_list = sorted([x for x in agg[col_cat].dropna().astype(str).unique().tolist() if x.strip()])
    sel_cats = st.multiselect("Kategori filtresi", cat_list)

with st.spinner("İşletmeler analiz ediliyor..."):
    emb_biz = embed_texts(agg[col_text].tolist())

    top_list, best_one, result_full = recommend_one_business(
        agg=agg,
        emb_biz=emb_biz,
        query=query,
        user_lat=user_lat,
        user_lon=user_lon,
        col_biz=col_biz,
        col_addr=col_addr,
        col_cat=col_cat,
        col_rating=col_rating,
        top_n=top_n,
        min_rating=min_rating,
        max_distance_km=max_distance_km,
        sel_cats=sel_cats,
        w_text=w_text,
        w_rating=w_rating,
        w_distance=w_distance
    )

# =====================================================
# SONUÇ
# =====================================================
if best_one is None:
    st.warning("Uygun işletme bulunamadı. Mesafe filtresini veya minimum yıldızı düşürmeyi dene.")
    st.stop()

st.markdown("## 🌟 Sana En Uygun Tek İşletme")
st.markdown(f"""
<div class="big-reco">
    <h2 style="margin-top:0;">{best_one['İşletme']}</h2>
    <p><b>Kategori:</b> {best_one['Kategori']}</p>
    <p><b>Adres:</b> {best_one['Adres']}</p>
    <p>
        <span class="score-badge">⭐ Yıldız: {best_one['Yıldız']}</span>
        <span class="score-badge">📏 Mesafe: {best_one['Mesafe (km)']} km</span>
        <span class="score-badge">🧠 Benzerlik Skoru: {best_one['Benzerlik Skoru']}</span>
        <span class="score-badge">⭐ Yıldız Skoru: {best_one['Yıldız Skoru']}</span>
        <span class="score-badge">📍 Konum Skoru: {best_one['Konum Skoru']}</span>
        <span class="score-badge">🏆 Final Skor: {best_one['Final Skor']}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# HARİTA - ANA ÖNERİ
# =====================================================
st.markdown("## 🗺️ Önerilen İşletmenin Haritadaki Konumu")

best_map_df = top_list[top_list["İşletme"].astype(str) == str(best_one["İşletme"])].copy()

fmap_best = create_map(user_lat, user_lon, best_map_df)
st_folium(fmap_best, width=None, height=500, key="map_best")

best_row = best_map_df.iloc[0]
best_lat = best_row["Latitude"]
best_lon = best_row["Longitude"]

if not pd.isna(best_lat) and not pd.isna(best_lon):
    best_google_maps_link, best_directions_link = build_google_links(user_lat, user_lon, best_lat, best_lon)

    c1, c2 = st.columns(2)
    with c1:
        st.link_button("📍 Önerilen işletmenin konumunu aç", best_google_maps_link, use_container_width=True)
    with c2:
        st.link_button("🚗 Önerilen işletme için yol tarifi al", best_directions_link, use_container_width=True)

# =====================================================
# ALTERNATİFLER
# =====================================================
st.markdown("## 📋 Alternatif İşletmeler")

show_cols = [
    "İşletme",
    "Kategori",
    "Adres",
    "Yıldız",
    "Mesafe (km)",
    "Benzerlik Skoru",
    "Yıldız Skoru",
    "Konum Skoru",
    "Final Skor"
]

st.dataframe(top_list[show_cols], use_container_width=True)

alternative_df = top_list[top_list["İşletme"].astype(str) != str(best_one["İşletme"])].copy()

if len(alternative_df) > 0:
    st.markdown("## 📍 Alternatif İşletme Seç")

    selected_alt = st.selectbox(
        "Haritada görmek ve yol tarifi almak istediğin alternatif işletmeyi seç",
        options=alternative_df["İşletme"].astype(str).tolist()
    )

    selected_alt_df = alternative_df[alternative_df["İşletme"].astype(str) == str(selected_alt)].copy()

    st.markdown("## 🗺️ Seçilen Alternatif İşletmenin Konumu")
    fmap_alt = create_map(user_lat, user_lon, selected_alt_df)
    st_folium(
        fmap_alt,
        width=None,
        height=450,
        key=f"map_alt_{selected_alt}"
    )

    alt_row = selected_alt_df.iloc[0]
    alt_lat = alt_row["Latitude"]
    alt_lon = alt_row["Longitude"]

    if not pd.isna(alt_lat) and not pd.isna(alt_lon):
        alt_google_maps_link, alt_directions_link = build_google_links(user_lat, user_lon, alt_lat, alt_lon)

        c1, c2 = st.columns(2)
        with c1:
            st.link_button(
                f"📍 {selected_alt} konumunu aç",
                alt_google_maps_link,
                use_container_width=True
            )
        with c2:
            st.link_button(
                f"🚗 {selected_alt} için yol tarifi al",
                alt_directions_link,
                use_container_width=True
            )
else:
    st.info("Gösterilecek alternatif işletme bulunamadı.")

# =====================================================
# FORMÜL AÇIKLAMASI
# =====================================================
st.caption("""
Final skor şu formülle hesaplanır:

Final Skor = 0.50 × Benzerlik Skoru + 0.30 × Yıldız Skoru + 0.20 × Konum Skoru

Tüm skorlar 0–1 aralığına normalize edilmiştir.
""")