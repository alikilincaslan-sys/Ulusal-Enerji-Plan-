import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PyPSA DataPrep", layout="wide")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root (app.py'nin olduğu klasör)
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("PyPSA DataPrep (Türkiye Tek-Node)")

st.markdown(
"""
Bu sayfa ham Excel'leri okur, **29 Şubat'ı düşürür**, lisanslı+lisanssız üretimleri birleştirir,
kaynakları senin istediğin şekilde **gruplar**, sonra **parquet** üretir.
"""
)

# -----------------------------
# Helpers
# -----------------------------
def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns: 'Tarih' and optionally 'Saat'.
    - If 'Tarih' already includes time, we just parse it.
    - If 'Saat' exists and 'Tarih' date-only, we combine.
    """
    if "Tarih" not in df.columns:
        raise ValueError("Kolon bulunamadı: 'Tarih'")

    # normalize column names (strip)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Try parse Tarih directly (may already include hour)
    dt = pd.to_datetime(df["Tarih"], errors="coerce")

    # If many NaT, combine Tarih + Saat
    if dt.isna().mean() > 0.2 and "Saat" in df.columns:
        # 'Saat' sometimes like 0-23 or '00:00' etc.
        # create string "YYYY-MM-DD HH:MM"
        date_part = pd.to_datetime(df["Tarih"], errors="coerce").dt.strftime("%Y-%m-%d")
        # saat parse
        s = df["Saat"].astype(str).str.strip()
        # if '0'..'23' -> format HH:00
        s2 = s.where(s.str.contains(":"), s.str.zfill(2) + ":00")
        dt = pd.to_datetime(date_part + " " + s2, errors="coerce")

    df["timestamp"] = dt
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Feb 29 to make leap years 8760."""
    idx = df.index
    mask = ~((idx.month == 2) & (idx.day == 29))
    return df.loc[mask]


def ensure_hourly_continuous(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Basic checks: duplicates, frequency gaps. We won't force-fill big gaps here,
    just warn.
    """
    if df.index.duplicated().any():
        st.warning(f"{name}: duplicate timestamp var. İlkini tutuyorum.")
        df = df[~df.index.duplicated(keep="first")]

    # Try infer frequency
    inf = pd.infer_freq(df.index[:2000]) if len(df) >= 10 else None
    if inf not in ("H", "h"):
        st.info(f"{name}: saatlik frekans otomatik algılanamadı (infer_freq={inf}). Bu normal olabilir; yine de kontrol et.")
    return df


def read_excel_any(path: Path) -> pd.DataFrame:
    """Read first sheet by default."""
    return pd.read_excel(path)


def sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols_exist = [c for c in cols if c in df.columns]
    if not cols_exist:
        return pd.Series(0.0, index=df.index)
    out = df[cols_exist].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return out


def normalize_shape(s: pd.Series) -> pd.Series:
    s = s.clip(lower=0)
    mx = float(s.max()) if len(s) else 0.0
    if mx <= 0:
        return s * 0.0
    return (s / mx).clip(0, 1)


# -----------------------------
# File selection
# -----------------------------
st.subheader("1) Ham dosyalar")

use_upload = st.toggle("Dosyaları arayüzden yükle (önerilir)", value=True)

if use_upload:
    cons_file = st.file_uploader("Tüketim Excel'i (2021–2025)", type=["xlsx"])
    lic_gen_file = st.file_uploader("Lisanslı Üretim Excel'i (2021–2025)", type=["xlsx"])
    unlic_gen_file = st.file_uploader("Lisanssız Üretim Excel'i (2021–2025)", type=["xlsx"])
else:
    st.caption("Dosyaları data/raw altına koyup aşağıya isimlerini yaz.")
    cons_name = st.text_input("Tüketim dosya adı", "Gercek_Zamanli_Tuketim-01012021-31122025.xlsx")
    lic_name = st.text_input("Lisanslı üretim dosya adı", "Lisanslı Gercek_Zamanli_Uretim-01012021-31012025.xlsx")
    unlic_name = st.text_input("Lisanssız üretim dosya adı", "Lisanssiz_Uretim_Miktari-01012021-31122025 (1).xlsx")
    cons_file = RAW_DIR / cons_name
    lic_gen_file = RAW_DIR / lic_name
    unlic_gen_file = RAW_DIR / unlic_name


base_year = st.selectbox("Baz yıl (8760 profil üretimi için)", [2021, 2022, 2023, 2024, 2025], index=2)
run_btn = st.button("Hazırla (Parquet üret)", type="primary")

# -----------------------------
# Main
# -----------------------------
if run_btn:
    if use_upload and (cons_file is None or lic_gen_file is None or unlic_gen_file is None):
        st.error("Lütfen 3 dosyayı da yükle.")
        st.stop()

    # Read Excel
    with st.spinner("Excel'ler okunuyor..."):
        if use_upload:
            df_cons = pd.read_excel(cons_file)
            df_lic = pd.read_excel(lic_gen_file)
            df_unlic = pd.read_excel(unlic_gen_file)
        else:
            df_cons = read_excel_any(Path(cons_file))
            df_lic = read_excel_any(Path(lic_gen_file))
            df_unlic = read_excel_any(Path(unlic_gen_file))

    # Parse datetime index
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

    # Normalize column names
    lic.columns = [str(c).strip() for c in lic.columns]
    unlic.columns = [str(c).strip() for c in unlic.columns]
    cons.columns = [str(c).strip() for c in cons.columns]

    # Identify consumption column
    # expected: "Tüketim Miktarı(MWh)" from your file
    cons_col_candidates = [c for c in cons.columns if "tüketim" in c.lower() and "mwh" in c.lower()]
    if not cons_col_candidates:
        st.error("Tüketim kolonunu bulamadım. Beklenen: 'Tüketim Miktarı(MWh)' benzeri.")
        st.stop()
    cons_col = cons_col_candidates[0]

    # Align indexes (inner join)
    with st.spinner("Seriler hizalanıyor..."):
        idx = cons.index.intersection(lic.index).intersection(unlic.index)
        cons = cons.loc[idx]
        lic = lic.loc[idx]
        unlic = unlic.loc[idx]

    # Build grouped hourly series (MWh per hour; treat as average MW for PyPSA p_set)
    with st.spinner("Gruplama yapılıyor (kömür ayrımı, gas ayrımı, RES/GES toplamı)..."):
        out = pd.DataFrame(index=idx)
        out["consumption_mwh"] = pd.to_numeric(cons[cons_col], errors="coerce").fillna(0)

        # Licensed columns
        # Renewable totals
        out["solar_lic_mwh"] = pd.to_numeric(lic.get("Güneş", 0), errors="coerce").fillna(0)
        out["wind_lic_mwh"] = pd.to_numeric(lic.get("Rüzgar", 0), errors="coerce").fillna(0)

        out["hydro_mwh"] = sum_cols(lic, ["Barajlı", "Akarsu"])

        # Coal groups (separate)
        out["coal_imported_mwh"] = pd.to_numeric(lic.get("İthal Kömür", 0), errors="coerce").fillna(0)
        out["coal_hard_mwh"] = pd.to_numeric(lic.get("Taş Kömür", 0), errors="coerce").fillna(0)
        out["coal_asphaltite_mwh"] = pd.to_numeric(lic.get("Asfaltit Kömür", 0), errors="coerce").fillna(0)
        out["lignite_mwh"] = pd.to_numeric(lic.get("Linyit", 0), errors="coerce").fillna(0)

        # Gas split
        out["gas_natural_mwh"] = pd.to_numeric(lic.get("Doğal Gaz", 0), errors="coerce").fillna(0)
        out["gas_other_mwh"] = sum_cols(lic, ["LNG", "Nafta", "Fuel Oil", "Atık Isı"])

        # Optional others (keep for later)
        out["geothermal_mwh"] = pd.to_numeric(lic.get("Jeotermal", 0), errors="coerce").fillna(0)
        out["biomass_lic_mwh"] = pd.to_numeric(lic.get("Biyokütle", 0), errors="coerce").fillna(0)

        # Unlicensed columns (with MWh labels)
        # (names like "Güneş (MWh)")
        def pick_unlic(colname_contains: str):
            hits = [c for c in unlic.columns if colname_contains.lower() in c.lower()]
            return hits[0] if hits else None

        col_solar_u = pick_unlic("Güneş")
        col_wind_u = pick_unlic("Rüzgar")
        out["solar_unlic_mwh"] = pd.to_numeric(unlic.get(col_solar_u, 0), errors="coerce").fillna(0)
        out["wind_unlic_mwh"] = pd.to_numeric(unlic.get(col_wind_u, 0), errors="coerce").fillna(0)

        out["solar_total_mwh"] = out["solar_lic_mwh"] + out["solar_unlic_mwh"]
        out["wind_total_mwh"] = out["wind_lic_mwh"] + out["wind_unlic_mwh"]

        # Net load (optional: subtract unlicensed solar; you can later choose gross vs net)
        out["net_load_mwh"] = (out["consumption_mwh"] - out["solar_unlic_mwh"]).clip(lower=0)

    # Create base year 8760 profiles (shape)
    with st.spinner(f"{base_year} için 8760 shape profilleri üretiliyor..."):
        base = out[out.index.year == base_year].copy()

        if base_year == 2024:
            # After dropping Feb29, it should be 8760
            pass

        # sanity: expect 8760
        if len(base) != 8760:
            st.warning(f"Baz yıl {base_year} için satır sayısı 8760 değil: {len(base)}. (Eksik saat veya veri boşluğu olabilir.)")

        prof = pd.DataFrame(index=base.index)
        prof["load_base"] = base["consumption_mwh"]
        prof["net_load_base"] = base["net_load_mwh"]

        prof["solar_shape"] = normalize_shape(base["solar_total_mwh"])
        prof["wind_shape"] = normalize_shape(base["wind_total_mwh"])
        # hydro: istersen shape üret (şimdilik üretelim)
        prof["hydro_shape"] = normalize_shape(base["hydro_mwh"])

    # Save parquet
    with st.spinner("Parquet dosyaları yazılıyor..."):
        history_path = OUT_DIR / "history_hourly.parquet"
        profiles_path = OUT_DIR / f"profiles_{base_year}.parquet"
        out.reset_index().rename(columns={"index": "timestamp"}).to_parquet(history_path, index=False)
        prof.reset_index().rename(columns={"index": "timestamp"}).to_parquet(profiles_path, index=False)

    st.success("Hazır! Parquet üretildi.")
    st.write("✅", history_path)
    st.write("✅", profiles_path)

    # Quick checks
    st.subheader("Kontroller")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Toplam saat (history)", f"{len(out):,}")
        st.metric(f"{base_year} saat (profiles)", f"{len(prof):,}")
    with c2:
        st.metric("Solar shape max", f"{prof['solar_shape'].max():.3f}")
        st.metric("Wind shape max", f"{prof['wind_shape'].max():.3f}")
    with c3:
        st.metric("Tüketim max (baz yıl)", f"{prof['load_base'].max():,.0f} MWh")
        st.metric("Net yük min (baz yıl)", f"{prof['net_load_base'].min():,.0f} MWh")

    st.subheader("Örnek tablo (baz yıl)")
    st.dataframe(prof.head(24))

    # Download buttons (optional)
    st.download_button("history_hourly.parquet indir", data=history_path.read_bytes(), file_name=history_path.name)
    st.download_button(f"profiles_{base_year}.parquet indir", data=profiles_path.read_bytes(), file_name=profiles_path.name)
