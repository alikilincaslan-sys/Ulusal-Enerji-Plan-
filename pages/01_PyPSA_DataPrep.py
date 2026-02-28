# pages/01_PyPSA_DataPrep.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PyPSA DataPrep", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("PyPSA DataPrep (Türkiye Tek-Node)")

st.markdown(
"""
Bu sayfa ham Excel'leri okur, **29 Şubat'ı düşürür**, lisanslı+lisanssız üretimleri birleştirir,
kaynakları gruplar ve `data/processed` altına parquet üretir.

✅ Hidro split:
- **Barajlı → hydro_res_shape**
- **Akarsu → hydro_ror_shape**
"""
)

# -----------------------------
# Helpers
# -----------------------------
def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Tarih" not in df.columns:
        raise ValueError("Kolon bulunamadı: 'Tarih'")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    dt = pd.to_datetime(df["Tarih"], errors="coerce")

    # Eğer Tarih parse edilemiyorsa ve Saat varsa birleştir
    if dt.isna().mean() > 0.2 and "Saat" in df.columns:
        date_part = pd.to_datetime(df["Tarih"], errors="coerce").dt.strftime("%Y-%m-%d")
        s = df["Saat"].astype(str).str.strip()
        s2 = s.where(s.str.contains(":"), s.str.zfill(2) + ":00")
        dt = pd.to_datetime(date_part + " " + s2, errors="coerce")

    df["timestamp"] = dt
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

def drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    return df.loc[~((idx.month == 2) & (idx.day == 29))]

def ensure_hourly_continuous(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.index.duplicated().any():
        st.warning(f"{name}: duplicate timestamp var. İlkini tutuyorum.")
        df = df[~df.index.duplicated(keep="first")]
    return df

def sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols_exist = [c for c in cols if c in df.columns]
    if not cols_exist:
        return pd.Series(0.0, index=df.index)
    return df[cols_exist].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

def normalize_shape_max(s: pd.Series) -> pd.Series:
    # p_max_pu gibi: max'a böl -> 0..1
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    mx = float(s.max()) if len(s) else 0.0
    if mx <= 0:
        return s * 0.0
    return (s / mx).clip(0, 1)

def pick_unlic(unlic: pd.DataFrame, contains: str):
    hits = [c for c in unlic.columns if contains.lower() in str(c).lower()]
    return hits[0] if hits else None

@st.cache_data(show_spinner=False)
def read_excel_cached(path: str) -> pd.DataFrame:
    # path string olmalı ki cache stabil olsun
    return pd.read_excel(path)

def read_excel_any(obj):
    """
    obj:
      - streamlit UploadedFile
      - pathlib.Path
      - str path
    """
    if obj is None:
        raise ValueError("Dosya seçilmedi.")
    if hasattr(obj, "read"):  # UploadedFile gibi
        return pd.read_excel(obj)
    p = Path(obj)
    return read_excel_cached(str(p))

def list_raw_excels() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    files = sorted(list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls")))
    return files

# -----------------------------
# File selection
# -----------------------------
st.subheader("1) Ham dosyalar")

use_upload = st.toggle("Excel yükleyerek çalıştır (Streamlit Cloud için en kolayı)", value=True)

raw_files = list_raw_excels()
if not use_upload:
    if not raw_files:
        st.error(
            f"`{RAW_DIR}` içinde Excel bulunamadı.\n\n"
            "Upload kapalı modda çalıştırmak için Excel dosyalarını repo içine koymalısın:\n"
            "- data/raw/ dosyasına .xlsx kopyala\n"
            "- GitHub’a commit et (veya Streamlit Cloud dosya yapısına ekle)\n\n"
            "Sonra bu sayfayı yenileyip dropdown’dan seçebilirsin."
        )
        st.stop()

if use_upload:
    cons_file = st.file_uploader("Tüketim Excel'i (2021–2025)", type=["xlsx", "xls"])
    lic_gen_file = st.file_uploader("Lisanslı Üretim Excel'i (2021–2025)", type=["xlsx", "xls"])
    unlic_gen_file = st.file_uploader("Lisanssız Üretim Excel'i (2021–2025)", type=["xlsx", "xls"])
else:
    st.caption(f"Repo içinden seçiliyor: {RAW_DIR}")
    cons_file = st.selectbox("Tüketim dosyası", options=raw_files, format_func=lambda p: p.name)
    lic_gen_file = st.selectbox("Lisanslı üretim dosyası", options=raw_files, format_func=lambda p: p.name)
    unlic_gen_file = st.selectbox("Lisanssız üretim dosyası", options=raw_files, format_func=lambda p: p.name)

base_year = st.selectbox("Baz yıl (8760 profil üretimi için)", [2021, 2022, 2023, 2024, 2025], index=3)
run_btn = st.button("Hazırla (Parquet üret)", type="primary")

# -----------------------------
# Main
# -----------------------------
if run_btn:
    if use_upload and (cons_file is None or lic_gen_file is None or unlic_gen_file is None):
        st.error("Lütfen 3 dosyayı da yükle.")
        st.stop()

    # If not upload, validate paths exist
    if not use_upload:
        for p in [cons_file, lic_gen_file, unlic_gen_file]:
            p = Path(p)
            if not p.exists():
                st.error(f"Dosya bulunamadı: {p}")
                st.stop()

    # Read
    with st.spinner("Excel'ler okunuyor..."):
        df_cons = read_excel_any(cons_file)
        df_lic = read_excel_any(lic_gen_file)
        df_unlic = read_excel_any(unlic_gen_file)

    # Parse datetime
    with st.spinner("Timestamp hazırlanıyor (Tarih+Saat) ve 29 Şubat düşürülüyor..."):
        cons = _to_datetime(df_cons)
        lic = _to_datetime(df_lic)
        unlic = _to_datetime(df_unlic)

        cons = drop_feb29(cons)
        lic = drop_feb29(lic)
        unlic = drop_feb29(unlic)

        cons = ensure_hourly_continuous(cons, "Tüketim")
        lic = ensure_hourly_continuous(lic, "Lisanslı Üretim")
        unlic = ensure_hourly_continuous(unlic, "Lisanssız Üretim")

    # Normalize columns
    lic.columns = [str(c).strip() for c in lic.columns]
    unlic.columns = [str(c).strip() for c in unlic.columns]
    cons.columns = [str(c).strip() for c in cons.columns]

    # Find consumption column
    cons_col_candidates = [c for c in cons.columns if "tüketim" in c.lower() and "mwh" in c.lower()]
    if not cons_col_candidates:
        st.error("Tüketim kolonunu bulamadım. Örn: 'Tüketim Miktarı(MWh)' gibi bir kolon olmalı.")
        st.write("Mevcut kolonlar:", list(cons.columns))
        st.stop()
    cons_col = cons_col_candidates[0]

    # Align indexes
    with st.spinner("Seriler hizalanıyor..."):
        idx = cons.index.intersection(lic.index).intersection(unlic.index)
        if len(idx) == 0:
            st.error("Dosyalar arasında ortak timestamp bulunamadı. Tarih aralıklarını ve formatı kontrol et.")
            st.stop()
        cons = cons.loc[idx]
        lic = lic.loc[idx]
        unlic = unlic.loc[idx]

    # Build hourly history (MWh per hour)
    with st.spinner("Gruplama yapılıyor..."):
        out = pd.DataFrame(index=idx)
        out["consumption_mwh"] = pd.to_numeric(cons[cons_col], errors="coerce").fillna(0)

        # Licensed VRE
        out["solar_lic_mwh"] = pd.to_numeric(lic.get("Güneş", 0), errors="coerce").fillna(0)
        out["wind_lic_mwh"] = pd.to_numeric(lic.get("Rüzgar", 0), errors="coerce").fillna(0)

        # ✅ Hydro split: Reservoir=Barajlı, RoR=Akarsu
        if "Barajlı" not in lic.columns:
            st.warning("Lisanslı üretimde 'Barajlı' kolonu bulunamadı. 'Hidro' varsa ona düşeceğim.")
        out["hydro_res_mwh"] = pd.to_numeric(lic.get("Barajlı", lic.get("Hidro", 0)), errors="coerce").fillna(0)
        out["hydro_ror_mwh"] = pd.to_numeric(lic.get("Akarsu", 0), errors="coerce").fillna(0)
        out["hydro_mwh"] = out["hydro_res_mwh"] + out["hydro_ror_mwh"]  # compat

        # Coal groups
        out["coal_imported_mwh"] = pd.to_numeric(lic.get("İthal Kömür", 0), errors="coerce").fillna(0)
        out["coal_hard_mwh"] = pd.to_numeric(lic.get("Taş Kömür", 0), errors="coerce").fillna(0)
        out["coal_asphaltite_mwh"] = pd.to_numeric(lic.get("Asfaltit Kömür", 0), errors="coerce").fillna(0)
        out["lignite_mwh"] = pd.to_numeric(lic.get("Linyit", 0), errors="coerce").fillna(0)

        # Gas split
        out["gas_natural_mwh"] = pd.to_numeric(lic.get("Doğal Gaz", 0), errors="coerce").fillna(0)
        out["gas_other_mwh"] = sum_cols(lic, ["LNG", "Nafta", "Fuel Oil", "Atık Isı"])

        # Others
        out["geothermal_mwh"] = pd.to_numeric(lic.get("Jeotermal", 0), errors="coerce").fillna(0)
        out["biomass_lic_mwh"] = pd.to_numeric(lic.get("Biyokütle", 0), errors="coerce").fillna(0)

        # Unlicensed VRE
        col_solar_u = pick_unlic(unlic, "Güneş")
        col_wind_u  = pick_unlic(unlic, "Rüzgar")

        out["solar_unlic_mwh"] = pd.to_numeric(unlic.get(col_solar_u, 0), errors="coerce").fillna(0)
        out["wind_unlic_mwh"]  = pd.to_numeric(unlic.get(col_wind_u, 0), errors="coerce").fillna(0)

        out["solar_total_mwh"] = out["solar_lic_mwh"] + out["solar_unlic_mwh"]
        out["wind_total_mwh"]  = out["wind_lic_mwh"] + out["wind_unlic_mwh"]

        # Net load (optional)
        out["net_load_mwh"] = (out["consumption_mwh"] - out["solar_unlic_mwh"]).clip(lower=0)

    # Create base year profiles
    with st.spinner(f"{base_year} için 8760 profilleri üretiliyor..."):
        base = out[out.index.year == base_year].copy()
        if len(base) != 8760:
            st.warning(f"Baz yıl {base_year} için satır sayısı 8760 değil: {len(base)} (eksik saat olabilir).")

        prof_out = pd.DataFrame(index=base.index)
        prof_out["load_base"] = base["consumption_mwh"]
        prof_out["net_load_base"] = base["net_load_mwh"]

        prof_out["solar_shape"] = normalize_shape_max(base["solar_total_mwh"])
        prof_out["wind_shape"]  = normalize_shape_max(base["wind_total_mwh"])

        prof_out["hydro_res_shape"] = normalize_shape_max(base["hydro_res_mwh"])  # Barajlı
        prof_out["hydro_ror_shape"] = normalize_shape_max(base["hydro_ror_mwh"])  # Akarsu
        prof_out["hydro_shape"] = normalize_shape_max(base["hydro_mwh"])          # compat

    # Save parquet
    with st.spinner("Parquet yazılıyor..."):
        history_path  = OUT_DIR / "history_hourly.parquet"
        profiles_path = OUT_DIR / f"profiles_{base_year}.parquet"

        out.reset_index().rename(columns={"index": "timestamp"}).to_parquet(history_path, index=False)
        prof_out.reset_index().rename(columns={"index": "timestamp"}).to_parquet(profiles_path, index=False)

    st.success("Hazır! Parquet üretildi.")
    st.write("✅", str(history_path))
    st.write("✅", str(profiles_path))

    # Quick checks
    st.subheader("Kontroller (baz yıl)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("profiles saat", f"{len(prof_out):,}")
    k2.metric("solar_shape max", f"{prof_out['solar_shape'].max():.3f}")
    k3.metric("hydro_res_shape max", f"{prof_out['hydro_res_shape'].max():.3f}")
    k4.metric("hydro_ror_shape max", f"{prof_out['hydro_ror_shape'].max():.3f}")

    st.dataframe(prof_out.head(48))

    # Download buttons (Cloud için pratik)
    st.download_button("history_hourly.parquet indir", data=history_path.read_bytes(), file_name=history_path.name)
    st.download_button(f"profiles_{base_year}.parquet indir", data=profiles_path.read_bytes(), file_name=profiles_path.name)
