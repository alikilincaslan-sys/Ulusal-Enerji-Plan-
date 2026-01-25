# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Power Generation Dashboard", layout="wide")

# -----------------------------
# Config
# -----------------------------
# DEFAULT max year (will be overwritten by sidebar year range)
MAX_YEAR = 2050

# Electricity supply (GWh) – reference
TOTAL_SUPPLY_LABEL = "Total Domestic Power Generation - Gross"

# Installed capacity total row (GW) - label kept for reference, but KG total uses row formula below
TOTAL_CAPACITY_LABEL = "Gross Installed Capacity (in GWe)"

# Storage & PTX total rows (GW / GWh depending on block)
TOTAL_STORAGE_LABEL = "Total Storage"
TOTAL_PTX_LABEL = "Total Power to X"

# --- KG TOTAL RULE (Excel 1-indexed rows in Power_Generation sheet) ---
KG_BASE_ROW_1IDX = 79
KG_SUB_ROW_1IDX_1 = 102
KG_SUB_ROW_1IDX_2 = 106

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_cols(col):
    if col is None:
        return ""
    s = str(col).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _maybe_numeric_year(x):
    try:
        return int(float(x))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_sheet_raw(xlsx_file, sheet_name):
    return pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)


def _extract_years_row(raw, years_row_1idx=3, start_col_idx=2):
    r0 = years_row_1idx - 1
    years = []
    cols = []
    for c in range(start_col_idx, raw.shape[1]):
        y = _maybe_numeric_year(raw.iloc[r0, c])
        if y is None:
            continue
        years.append(y)
        cols.append(c)
    return years, cols


def _row_series(raw, row_1idx, year_cols_idx, years):
    r0 = row_1idx - 1
    vals = []
    for c in year_cols_idx:
        vals.append(_safe_float(raw.iloc[r0, c]))
    return pd.DataFrame({"year": years, "value": vals})


def _long_from_rows(raw, years_row_1idx, start_col_idx, label_col_idx, start_row_1idx, end_row_1idx):
    years, cols = _extract_years_row(raw, years_row_1idx=years_row_1idx, start_col_idx=start_col_idx)

    rows = []
    for r in range(start_row_1idx - 1, end_row_1idx):
        label = raw.iloc[r, label_col_idx] if label_col_idx < raw.shape[1] else None
        label = _normalize_cols(label)
        if not label:
            continue

        vals = []
        for c in cols:
            vals.append(_safe_float(raw.iloc[r, c]))

        tmp = pd.DataFrame({"year": years, "value": vals})
        tmp["label"] = label
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["year", "value", "label"])
    return pd.concat(rows, ignore_index=True)


def _filter_years(df, start_year, end_year):
    if df is None or df.empty:
        return df
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()


def _energy_unit_label():
    return st.session_state.get("energy_unit_label", "GWh")


def _energy_value_format():
    return st.session_state.get("energy_value_format", ",.0f")


def _convert_energy_df(df):
    # Placeholder: keep as-is (your existing conversion logic remains)
    return df


def _line_chart(df, title, y_title, value_format=",.2f"):
    if df is None or df.empty:
        st.info(f"{title}: veri yok")
        return

    base = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("year:O", title="Yıl"),
        y=alt.Y("value:Q", title=y_title),
        tooltip=[
            alt.Tooltip("year:O", title="Yıl"),
            alt.Tooltip("value:Q", title=y_title, format=value_format),
        ],
    )
    st.markdown(f"**{title}**")
    st.altair_chart(base.properties(height=280), use_container_width=True)


def _render_stacked(df, title, x_field, stack_field, y_title, category_title, value_format=",.0f", order=None):
    if df is None or df.empty:
        st.info(f"{title}: veri yok")
        return

    year_vals = sorted(df[x_field].unique().tolist())

    base = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            f"{x_field}:O",
            title="Yıl",
            sort=year_vals,
            axis=alt.Axis(values=year_vals, labelAngle=0, labelPadding=14),
        ),
        y=alt.Y("value:Q", title=y_title, stack="zero"),
        color=alt.Color(f"{stack_field}:N", title=category_title, sort=order),
        tooltip=[
            alt.Tooltip(f"{x_field}:O", title="Yıl"),
            alt.Tooltip(f"{stack_field}:N", title=category_title),
            alt.Tooltip("value:Q", title=y_title, format=value_format),
        ],
    )

    st.markdown(f"**{title}**")
    st.altair_chart(base.properties(height=360, padding={"bottom": 28}), use_container_width=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Dosyalar")
    uploaded_files = st.file_uploader("Excel dosyalarını seçin", type=["xlsx"], accept_multiple_files=True)

    st.divider()
    st.header("Ayarlar")

    # Energy unit selection placeholder (keep your original)
    # Example:
    energy_unit = st.selectbox("Enerji birimi", ["GWh", "TWh", "ktoe"], index=0)
    if energy_unit == "GWh":
        st.session_state["energy_unit_label"] = "GWh"
        st.session_state["energy_value_format"] = ",.0f"
    elif energy_unit == "TWh":
        st.session_state["energy_unit_label"] = "TWh"
        st.session_state["energy_value_format"] = ",.2f"
    else:
        st.session_state["energy_unit_label"] = "ktoe"
        st.session_state["energy_value_format"] = ",.0f"

    # Year range + preset buttons (your existing logic remains)
    YEAR_OPTIONS = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (2025, 2050)

    year_range = st.select_slider(
        "Senaryo yıl aralığı",
        options=YEAR_OPTIONS,
        value=st.session_state["year_range"],
        key="year_range",
    )
    start_year, MAX_YEAR = year_range

    st.caption(
        "Metodolojik not: Model çıktıları 5 yıllık zaman adımlarında üretilmiştir. "
        "Görsel süreklilik ve eğilimlerin okunabilirliği için ara yıllar lineer "
        "interpolasyon yöntemiyle tahmini olarak doldurulmuştur. "
        "Bu değerler doğrudan model çıktısı değildir."
    )

    st.caption("Hızlı dönem seçimi")
    if st.button("Net Zero (2025–2050)", use_container_width=True):
        st.session_state["year_range"] = (2025, 2050)
        st.rerun()
    if st.button("TUEP (2025–2035)", use_container_width=True):
        st.session_state["year_range"] = (2025, 2035)
        st.rerun()

    st.divider()
    st.header("Paneller")
    show_electricity = st.checkbox("Elektrik", value=True)
    show_energy = st.checkbox("Enerji", value=True)
    show_emissions = st.checkbox("Sera Gazı Emisyonları", value=True)

selected_panels = []
if show_electricity:
    selected_panels.append("Elektrik")
if show_energy:
    selected_panels.append("Enerji")
if show_emissions:
    selected_panels.append("Emisyon")

# -----------------------------
# Main
# -----------------------------
st.title("Türkiye Ulusal Enerji Planı Modeli Arayüzü")

# ===== İçindekiler (TOC) =====
st.markdown("""
<style>
a { text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.subheader("📌 İçindekiler")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**⚡ Elektrik**")
    st.markdown("- [Elektrik (Genel)](#elektrik)")
    st.markdown("- [Elektrik Üretimi](#elektrik-uretim)")
    st.markdown("- [Kurulu Güç](#kurulu-guc)")
    st.markdown("- [Elektrik Tüketimi](#elektrik-tuketim)")

with c2:
    st.markdown("**🔥 Enerji**")
    st.markdown("- [Enerji (Genel)](#enerji)")
    st.markdown("- [Birincil Enerji](#birincil-enerji)")
    st.markdown("- [Nihai Enerji](#nihai-enerji)")

with c3:
    st.markdown("**🌍 Emisyonlar**")
    st.markdown("- [Emisyonlar (Genel)](#emisyonlar)")
    st.markdown("- [Türkiye Seragazı Emisyonları](#tr-emisyon)")
    st.markdown("- [Net Zero Takibi](#net-zero)")

st.divider()

if not uploaded_files:
    st.info("Başlamak için sol menüden Excel dosyası yükle.")
    st.stop()

# NOTE: Aşağıdaki okuma/hesaplama blokları senin mevcut kodundaki gibi kalmalıdır.
# Burada örnek/placeholder veri hazırlıyorum.
# Senin dosyanda zaten df_pop, df_gdp, df_genmix, df_capmix, df_sector_el vb. üretiliyor.

# -----------------------------
# Placeholder DF'ler (SENİN KODUNDA ZATEN VAR)
# -----------------------------
years = [2025, 2030, 2035, 2040, 2045, 2050]
df_pop = pd.DataFrame({"year": years, "value": [88, 90, 92, 94, 96, 98]})
df_gdp = pd.DataFrame({"year": years, "value": [900, 1000, 1100, 1200, 1300, 1400]})
df_electrification = pd.DataFrame({"year": years, "value": [20, 23, 26, 30, 33, 36]})

df_genmix = pd.DataFrame({
    "year": years * 3,
    "value": [100, 110, 120, 130, 140, 150, 60, 62, 65, 66, 68, 70, 40, 45, 50, 55, 60, 65],
    "group": ["Hydro"] * 6 + ["Wind (RES)"] * 6 + ["Solar (GES)"] * 6
})

df_capmix = pd.DataFrame({
    "year": years * 2,
    "value": [30, 32, 34, 36, 38, 40, 20, 22, 24, 26, 28, 30],
    "group": ["Hydro"] * 6 + ["Wind (RES)"] * 6
})

df_sector_el = pd.DataFrame({
    "year": years * 2,
    "value": [50, 52, 55, 57, 60, 62, 40, 42, 44, 46, 48, 50],
    "sector": ["Sanayi"] * 6 + ["Binalar"] * 6
})

df_primary = pd.DataFrame({
    "year": years * 2,
    "value": [300, 320, 340, 360, 380, 400, 200, 210, 220, 230, 240, 250],
    "source": ["Fosil"] * 6 + ["Yenilenebilir"] * 6
})

df_final = pd.DataFrame({
    "year": years * 2,
    "value": [250, 260, 270, 285, 300, 315, 150, 155, 160, 165, 170, 175],
    "source": ["Fosil"] * 6 + ["Elektrik"] * 6
})

df_emissions = pd.DataFrame({
    "year": years * 3,
    "value": [450, 460, 470, 480, 490, 500, 120, 125, 130, 135, 140, 145, -60, -62, -65, -68, -70, -72],
    "label": ["Energy"] * 6 + ["Non-energy"] * 6 + ["LULUCF"] * 6
})

# -----------------------------
# PANELS
# -----------------------------
# ELECTRICITY PANEL
if "Elektrik" in selected_panels:
    st.markdown('<a id="elektrik"></a>', unsafe_allow_html=True)
    st.markdown("## Elektrik")

    _line_chart(df_pop, "Türkiye Nüfus Gelişimi", "Nüfus (milyon)", value_format=",.3f")
    _line_chart(df_gdp, "GSYH (Milyar ABD Doları, 2015 fiyatlarıyla)", "Milyar ABD Doları (2015)", value_format=",.2f")
    _line_chart(df_electrification, "Nihai Enerjide Elektrifikasyon Oranı (%)", "%", value_format=",.1f")

    st.divider()

    st.markdown('<a id="elektrik-uretim"></a>', unsafe_allow_html=True)
    order_gen = ["Hydro", "Wind (RES)", "Solar (GES)"]
    _render_stacked(
        _convert_energy_df(df_genmix).rename(columns={"group": "category"}),
        title=f"Kaynaklarına Göre Elektrik Üretimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Kaynak/Teknoloji",
        value_format=_energy_value_format(),
        order=order_gen,
    )

    st.divider()

    st.markdown('<a id="kurulu-guc"></a>', unsafe_allow_html=True)
    order_cap = ["Hydro", "Wind (RES)"]
    _render_stacked(
        _convert_energy_df(df_capmix).rename(columns={"group": "category"}),
        title="Elektrik Kurulu Gücü (GW) – Depolama & PTX Hariç",
        x_field="year",
        stack_field="category",
        y_title="GW",
        category_title="Teknoloji",
        value_format=",.2f",
        order=order_cap,
    )

    st.divider()

    st.markdown('<a id="elektrik-tuketim"></a>', unsafe_allow_html=True)
    _render_stacked(
        _convert_energy_df(df_sector_el).rename(columns={"sector": "category"}),
        title=f"Sektörlere Göre Elektrik Tüketimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Sektör",
        value_format=_energy_value_format(),
    )

# ENERGY PANEL
if "Enerji" in selected_panels:
    st.markdown('<a id="enerji"></a>', unsafe_allow_html=True)
    st.markdown("## Enerji")

    st.markdown('<a id="birincil-enerji"></a>', unsafe_allow_html=True)
    _render_stacked(
        _convert_energy_df(df_primary).rename(columns={"source": "category"}),
        title=f"Kaynaklarına Göre Birincil Enerji Tüketimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Kaynak",
        value_format=_energy_value_format(),
    )

    st.divider()

    st.markdown('<a id="nihai-enerji"></a>', unsafe_allow_html=True)
    _render_stacked(
        _convert_energy_df(df_final).rename(columns={"source": "category"}),
        title=f"Kaynaklarına Göre Nihai Enerji Tüketimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Kaynak",
        value_format=_energy_value_format(),
    )

# EMISSIONS PANEL
if "Emisyon" in selected_panels:
    st.markdown('<a id="emisyonlar"></a>', unsafe_allow_html=True)
    st.markdown("## Sera Gazı Emisyonları")

    st.markdown('<a id="tr-emisyon"></a>', unsafe_allow_html=True)
    st.markdown("### Türkiye Seragazı Emisyonları — Net Zero Hedefi Takibi (CO₂e)")

    st.markdown('<a id="net-zero"></a>', unsafe_allow_html=True)
    _render_stacked(
        df_emissions.rename(columns={"label": "category"}),
        title="Toplam Emisyon Bileşenleri (Energy / Non-energy / LULUCF)",
        x_field="year",
        stack_field="category",
        y_title="MtCO₂e",
        category_title="Bileşen",
        value_format=",.1f",
    )
