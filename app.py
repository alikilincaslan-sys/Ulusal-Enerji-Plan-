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

# --- NEW: Extra rows to add into "Elektrik Kurulu Gücü (GW) – Depolama & PTX Hariç" ---
# Power_Generation sheet (Excel 1-indexed)
CAPACITY_EXTRA_ROWS_1IDX = [98, 99, 100, 101]

# NEW: bu satırlar boş geliyorsa isimleri buradan ver
EXTRA_ROW_NAME_MAP = {
    98: "Biomass",
    99: "Biomass",
    100: "Geothermal",
    101: "Geothermal",
}

# --- GDP / POP / CARBON PRICE RULES (Scenario_Assumptions sheet) ---
GDP_ROW_1IDX = 6  # Scenario_Assumptions sekmesinde 6. satır (GSYH)
POP_ROW_1IDX = 5  # Scenario_Assumptions sekmesinde 5. satır (Nüfus)
SCENARIO_ASSUMP_YEARS_ROW_1IDX = 3  # Scenario_Assumptions sekmesinde 3. satır (Yıllar)
CARBON_PRICE_ROW_1IDX = 15  # Scenario_Assumptions sekmesinde 15. satır (Carbon price ETS sectors, US$ '15/tnCO2)

# Exclude headers/subtotals – installed capacity
# NOTE: we DO NOT exclude Total Storage / Total Power to X, because we use them as totals.
CAPACITY_EXCLUDE_EXACT = {
    "Renewables",
    "Combustion Plants",
}
CAPACITY_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)(?!Power to X\b)", flags=re.IGNORECASE)

# Exclude headers/subtotals – gross generation block
GEN_EXCLUDE_EXACT = {
    "Renewables",
    "Combustion Plants",
}
GEN_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)(?!Power to X\b)", flags=re.IGNORECASE)

# Components that should NOT be counted if Total Storage exists (avoid double count)
STORAGE_COMPONENT_REGEX = re.compile(
    r"(pumped\s+storage|\bbattery\b|demand\s+response)",
    flags=re.IGNORECASE,
)

# PTX components (if Total Power to X not present)
PTX_COMPONENT_REGEX = re.compile(
    r"^power\s+to\s+(hydrogen|gas|liquids)\b|power\s+to\s+x",
    flags=re.IGNORECASE,
)

# Natural gas should be ONLY the sum of these items (avoid over-counting)
NATURAL_GAS_ITEMS_EXACT = {
    "Industrial CHP Plant Solid fuels",
    "CCGT without CCS",
    "CCGT with CCS",
    "Open cycle, IC and GT",
    "Industrial CHP plants Oil/Gas",
}
NATURAL_GAS_REGEX = re.compile(
    r"^(industrial\s+chp\s+plant\s+solid\s+fuels|ccgt\s+without\s+ccs|ccgt\s+with\s+ccs|open\s+cycle,\s*ic\s+and\s+gt|industrial\s+chp\s+plants?\s+oil/gas)$",
    flags=re.IGNORECASE,
)

# Technology mapping
TECH_GROUPS = {
    "Hydro": [r"\bhydro\b"],
    "Wind (RES)": [r"\bwind\b", r"\bres\b"],
    "Solar (GES)": [r"\bsolar\b", r"\bges\b", r"\bpv\b"],
    "Coal": [r"\bcoal\b(?!.*lignite)"],
    "Lignite": [r"\blignite\b"],
    "Nuclear": [r"\bnuclear\b"],
    "Other Renewables": [
        r"\bgeothermal\b",
        r"\bbiomass\b",
        r"\bbiogas\b",
        r"\bwaste\b",
        r"\bwave\b",
        r"\btidal\b",
    ],
}

RENEWABLE_GROUPS = {"Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables"}
INTERMITTENT_RE_GROUPS = {"Wind (RES)", "Solar (GES)"}


# -----------------------------
# Helpers
# -----------------------------
def _as_int_year(x):
    """Yıl hücresini mümkün olduğunca sağlam parse et.
    - Sayı / float / numpy türleri
    - '2025E', 'Year 2030', '2035*' gibi stringlerde 4 haneli yılı yakala
    """
    try:
        v = int(float(x))
        if 1900 <= v <= 2100:
            return v
    except Exception:
        pass

    try:
        sx = str(x)
        m = re.search(r"(19\d{2}|20\d{2}|2100)", sx)
        if m:
            v = int(m.group(1))
            if 1900 <= v <= 2100:
                return v
    except Exception:
        pass
    return None


def _find_year_row(raw: pd.DataFrame, scan_rows=25):
    best_idx, best_score = None, -1
    for i in range(min(scan_rows, len(raw))):
        row = raw.iloc[i, :].tolist()
        years = [_as_int_year(v) for v in row]
        score = sum([1 for y in years if y is not None])
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _extract_years(raw: pd.DataFrame, year_row_idx: int):
    years, cols = [], []
    for j, v in enumerate(raw.iloc[year_row_idx, :].tolist()):
        y = _as_int_year(v)
        if y is not None:
            years.append(y)
            cols.append(j)
    return years, cols


def _extract_block(raw: pd.DataFrame, first_col_idx: int, year_cols_idx: list[int], years: list[int], block_title_regex: str):
    c0 = raw.iloc[:, first_col_idx].astype(str)
    mask = c0.str.contains(block_title_regex, case=False, na=False, regex=True)
    if not mask.any():
        return None

    title_row = mask[mask].index[0]
    start = title_row + 1

    end = start
    while end < len(raw):
        v = raw.iloc[end, first_col_idx]
        if pd.isna(v):
            break
        if isinstance(v, str) and re.search(r"\(in\s+gw", v.strip(), flags=re.IGNORECASE):
            break
        end += 1
    end = max(start, end - 1)

    use_cols = [first_col_idx] + year_cols_idx
    blk = raw.iloc[start : end + 1, use_cols].copy()
    blk.columns = ["item"] + years
    blk["item"] = blk["item"].astype(str).str.strip()
    blk = blk[(blk["item"] != "") & (blk["item"].str.lower() != "nan")]

    keep_years = [y for y in years if y <= MAX_YEAR]
    blk = blk[["item"] + keep_years]
    for y in keep_years:
        blk[y] = pd.to_numeric(blk[y], errors="coerce")
    return blk


def _to_long(df_wide: pd.DataFrame, value_name="value"):
    year_cols = [c for c in df_wide.columns if isinstance(c, int)]
    long = df_wide.melt(id_vars=["item"], value_vars=year_cols, var_name="year", value_name=value_name)
    return long.dropna(subset=[value_name])


def _filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()


def _cagr(start_value: float, end_value: float, n_years: int) -> float:
    # returns decimal (e.g., 0.035)
    if n_years <= 0:
        return np.nan
    if start_value is None or end_value is None:
        return np.nan
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        return np.nan
    if start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1.0 / n_years) - 1.0


# -----------------------------
# Reading: Power_Generation
# -----------------------------
@st.cache_data(show_spinner=False)
def read_power_generation(xlsx_file):
    raw = pd.read_excel(xlsx_file, sheet_name="Power_Generation", header=None)
    year_row = _find_year_row(raw)
    years, year_cols_idx = _extract_years(raw, year_row)
    first_col_idx = 0

    return {
        "electricity_balance": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Electricity\s+Balance"),
        "gross_generation": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Gross\s+Electricity\s+Generation\s+by\s+plant\s+type"),
        "net_generation": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Net\s+Electricity\s+Generation\s+by\s+plant\s+type"),
        "installed_capacity": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Gross\s+Installed\s+Capacity"),
        # meta for fixed-row reading
        "_raw": raw,
        "_years": years,
        "_year_cols_idx": year_cols_idx,
    }


def get_series_from_block(block_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if block_df is None or block_df.empty:
        return pd.DataFrame(columns=["year", "value"])

    row = block_df[block_df["item"].str.strip().eq(label)]
    if row.empty:
        row = block_df[block_df["item"].str.contains(re.escape(label), case=False, na=False)]
    if row.empty:
        return pd.DataFrame(columns=["year", "value"])

    year_cols = [c for c in row.columns if isinstance(c, int)]
    s = pd.to_numeric(row.iloc[0][year_cols], errors="coerce")
    out = pd.DataFrame({"year": year_cols, "value": s.values}).dropna()
    return out.sort_values("year")


def _strict_match_group(item: str) -> str:
    s = (item or "").lower().strip()
    for grp, pats in TECH_GROUPS.items():
        for p in pats:
            if re.search(p, s, flags=re.IGNORECASE):
                return grp
    return "Other"


def _is_natural_gas_item(item: str) -> bool:
    it = (item or "").strip()
    if it in NATURAL_GAS_ITEMS_EXACT:
        return True
    return bool(NATURAL_GAS_REGEX.search(it))


# -----------------------------
# NEW: Fixed-row reading for extra capacity rows (Power_Generation)
# -----------------------------
def _read_power_generation_fixed_rows_as_stack(xlsx_file, value_rows_1idx: list[int]) -> pd.DataFrame:
    """
    Power_Generation sekmesinden sabit satırları stacked veri olarak okur.
    - Yıl satırı: otomatik (en çok yıl bulunan satır)
    - Etiket: satırın 1. sütunu (A sütunu); boşsa 'Ek Satır {row}'
    Çıktı: year, group, value
    """
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Power_Generation", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "group", "value"])

    year_row = _find_year_row(raw)
    if year_row is None:
        return pd.DataFrame(columns=["year", "group", "value"])

    years, year_cols_idx = _extract_years(raw, year_row)
    if not years:
        return pd.DataFrame(columns=["year", "group", "value"])

    recs = []
    for r1 in value_rows_1idx:
        r0 = r1 - 1
        if r0 < 0 or r0 >= len(raw):
            continue

        label = raw.iloc[r0, 0]
        label = "" if pd.isna(label) else str(label).strip()

        if (not label) or (label.lower() == "nan"):
            # önce haritadan isim ata, yoksa fallback
            label = EXTRA_ROW_NAME_MAP.get(r1, f"Ek Satır {r1}")

        for y, c in zip(years, year_cols_idx):
            if int(y) > MAX_YEAR:
                continue
            v = pd.to_numeric(raw.iloc[r0, c], errors="coerce")
            if pd.isna(v):
                continue
            recs.append({"year": int(y), "group": label, "value": float(v)})

    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])
    return df.sort_values(["year", "group"])


# -----------------------------
# Reading: Scenario assumptions (Nüfus & GSYH & Karbon)
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_scenario_assumptions_row(xlsx_file, value_row_1idx: int, series_name: str) -> pd.DataFrame:
    candidates = [
        "Scenario_Assumptions",
        "Scenario_Assumption",
        "Scenario assumption",
        "Scenario_assumption",
        "Scenario Assumption",
        "Scenario_Assumption",
        "ScenarioAssumption",
    ]

    raw = None
    used_sheet = None
    for sh in candidates:
        try:
            raw = pd.read_excel(xlsx_file, sheet_name=sh, header=None)
            used_sheet = sh
            break
        except Exception:
            continue

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["year", "value", "series"])

    year_r0 = SCENARIO_ASSUMP_YEARS_ROW_1IDX - 1
    if year_r0 < 0 or year_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series"])

    years, year_cols_idx = _extract_years(raw, year_r0)
    if not years:
        return pd.DataFrame(columns=["year", "value", "series"])

    val_r0 = value_row_1idx - 1
    if val_r0 < 0 or val_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series"])

    out_years, out_vals = [], []
    for y, c in zip(years, year_cols_idx):
        if y <= MAX_YEAR:
            out_years.append(y)
            out_vals.append(pd.to_numeric(raw.iloc[val_r0, c], errors="coerce"))

    df = pd.DataFrame({"year": out_years, "value": out_vals})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("year")
    df["series"] = series_name
    df["sheet"] = used_sheet
    return df


@st.cache_data(show_spinner=False)
def _read_fixed_row_sheet(xlsx_file, sheet_name: str, years_row_1idx: int, value_row_1idx: int, series_name: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    yr_r0 = years_row_1idx - 1
    val_r0 = value_row_1idx - 1
    if yr_r0 < 0 or val_r0 < 0 or yr_r0 >= len(raw) or val_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    row_year = raw.iloc[yr_r0, :].tolist()
    years, year_cols_idx = [], []
    for j, v in enumerate(row_year):
        y = _as_int_year(v)
        if y is not None:
            years.append(int(y))
            year_cols_idx.append(j)
    if len(years) == 0:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    out_years, out_vals = [], []
    for y, c in zip(years, year_cols_idx):
        if y <= MAX_YEAR:
            out_years.append(int(y))
            out_vals.append(pd.to_numeric(raw.iloc[val_r0, c], errors="coerce"))

    df = pd.DataFrame({"year": out_years, "value": out_vals})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("year")
    df["series"] = series_name
    df["sheet"] = sheet_name
    return df


def read_co2_emissions_series(xlsx_file) -> pd.DataFrame:
    return _read_fixed_row_sheet(
        xlsx_file,
        sheet_name="PowerGeneration-Indicators",
        years_row_1idx=3,
        value_row_1idx=30,
        series_name="CO2 Emisyonları (ktn CO2)",
    )


# -----------------------------
# CO2 -> CO2e (CRF 2023 tabanli varsayim)
# Summary&Indicators: 72,73,74,75,77,78,79,80,81 satirlarinda sektor bazli
# enerji kaynakli CO2 (bin ton) yer aliyor. Bu degerleri CO2e'ye donusturmek
# icin (CO2 / CO2e) oranlari kullanilir.
#
# Varsayimlar (CO2/CO2e):
# - Elektrik (Power Generation): 0.99
# - Ulastirma (Transport): 0.94
# - Sanayi (Industry + Industrial Plants): 0.97
# - Tarim: 0.01  (bu dosyada satirlar icinde yok; ileride gerekirse)
# - Diger sektorler: 0.99
# -----------------------------

SECTOR_CO2_OVER_CO2E = {
    "power": 0.99,
    "transport": 0.94,
    "industry": 0.97,
    "agriculture": 0.01,
    "other": 0.99,
}


def _sector_ratio_from_label(label: str) -> float:
    s = (label or "").strip().lower()
    if "transport" in s:
        return SECTOR_CO2_OVER_CO2E["transport"]
    if "power generation" in s or "electric" in s:
        return SECTOR_CO2_OVER_CO2E["power"]
    if s == "industry" or "industrial plants" in s or "industry" in s:
        return SECTOR_CO2_OVER_CO2E["industry"]
    if "agric" in s:
        return SECTOR_CO2_OVER_CO2E["agriculture"]
    return SECTOR_CO2_OVER_CO2E["other"]


def read_energy_emissions_sectoral_co2e(xlsx_file) -> pd.DataFrame:
    """
    Summary&Indicators sekmesinden sektor bazli enerji kaynakli emisyonlari okur.
    Girdi birimi: bin ton CO2 (ktn CO2)
    Cikti birimi: bin ton CO2e (ktn CO2e)

    Satirlar (1-indexed): 72,73,74,75,77,78,79,80,81
    Yillar: 3. satir
    """
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet", "ratio_co2_over_co2e"])

    YEARS_ROW_1IDX = 3
    START_COL_IDX = 2  # C
    SECTOR_ROWS_1IDX = [72, 73, 74, 75, 77, 78, 79, 80, 81]

    yr_r0 = YEARS_ROW_1IDX - 1
    if yr_r0 < 0 or yr_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet", "ratio_co2_over_co2e"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()
    years = []
    for y_cell in years_row:
        y = _as_int_year(y_cell)
        if y is not None and int(y) <= MAX_YEAR:
            years.append(int(y))
        else:
            years.append(None)

    records = []
    for r1 in SECTOR_ROWS_1IDX:
        r0 = r1 - 1
        if r0 < 0 or r0 >= len(raw):
            continue

        label = raw.iloc[r0, 0]
        label = "" if pd.isna(label) else str(label).strip()
        if label == "" or label.lower() == "nan":
            continue

        ratio = float(_sector_ratio_from_label(label))
        vals_row = raw.iloc[r0, START_COL_IDX : START_COL_IDX + len(years)].tolist()
        for y, v_cell in zip(years, vals_row):
            if y is None:
                continue
            v_co2 = pd.to_numeric(v_cell, errors="coerce")
            if pd.isna(v_co2):
                continue
            v_co2 = float(v_co2)
            # CO2e = CO2 / (CO2/CO2e)
            v_co2e = v_co2 / ratio if ratio and np.isfinite(ratio) else np.nan
            if not np.isfinite(v_co2e):
                continue
            records.append(
                {"year": int(y), "sector": label, "value": float(v_co2e), "ratio_co2_over_co2e": ratio}
            )

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet", "ratio_co2_over_co2e"])

    df = df.sort_values(["year", "sector"])
    df["series"] = "Sektorel Enerji Emisyonlari (CO2e, ktn)"
    df["sheet"] = "Summary&Indicators"
    return df


def energy_share_assumption(year: int) -> float:
    """Enerji kaynakli emisyonlarin toplam SGE icindeki payi (varsayim).

    2025: %70
    2050: %75
    Arasi lineer.
    """
    if year <= 2025:
        return 0.70
    if year >= 2050:
        return 0.75
    return 0.70 + (year - 2025) * (0.75 - 0.70) / (2050 - 2025)


def lulucf_assumption_ktn(year: int) -> float:
    """LULUCF net yutak varsayimi (ktn CO2e).

    2025: -56 MtCO2e  -> -56,000 ktn
    2050: -100 MtCO2e -> -100,000 ktn
    Arasi lineer.
    """
    y0, v0 = 2025, -56_000.0
    y1, v1 = 2050, -100_000.0
    if year <= y0:
        return v0
    if year >= y1:
        return v1
    return v0 + (year - y0) * (v1 - v0) / (y1 - y0)


def read_primary_energy_consumption_by_source(xlsx_file) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3
    START_COL_IDX = 2  # column C
    ROW_START_1IDX = 33
    ROW_END_1IDX = 42

    yr_r0 = YEARS_ROW_1IDX - 1
    if yr_r0 < 0 or yr_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()

    years = []
    for y_cell in years_row:
        y = _as_int_year(y_cell)
        if y is not None and int(y) <= MAX_YEAR:
            years.append(int(y))
        else:
            years.append(None)

    records = []
    for r1 in range(ROW_START_1IDX, ROW_END_1IDX + 1):
        r0 = r1 - 1
        if r0 < 0 or r0 >= len(raw):
            continue

        label = raw.iloc[r0, 0]
        label = "" if pd.isna(label) else str(label).strip()
        if label == "" or label.lower() == "nan":
            continue

        vals_row = raw.iloc[r0, START_COL_IDX : START_COL_IDX + len(years)].tolist()
        for y, v_cell in zip(years, vals_row):
            if y is None:
                continue
            v = pd.to_numeric(v_cell, errors="coerce")
            if pd.isna(v):
                continue
            records.append({"year": int(y), "source": label, "value": float(v)})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    df = df.sort_values(["year", "source"])
    df["series"] = "Birincil Enerji Talebi"
    df["sheet"] = "Summary&Indicators"
    return df


def read_final_energy_consumption_by_source(xlsx_file) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3
    START_COL_IDX = 2  # column C
    ROW_START_1IDX = 47
    ROW_END_1IDX = 52

    yr_r0 = YEARS_ROW_1IDX - 1
    if yr_r0 < 0 or yr_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()

    years = []
    for y_cell in years_row:
        y = _as_int_year(y_cell)
        if y is not None and int(y) <= MAX_YEAR:
            years.append(int(y))
        else:
            years.append(None)

    records = []
    for r1 in range(ROW_START_1IDX, ROW_END_1IDX + 1):
        r0 = r1 - 1
        if r0 < 0 or r0 >= len(raw):
            continue

        label = raw.iloc[r0, 0]
        label = "" if pd.isna(label) else str(label).strip()
        if label == "" or label.lower() == "nan":
            continue

        vals_row = raw.iloc[r0, START_COL_IDX : START_COL_IDX + len(years)].tolist()
        for y, v_cell in zip(years, vals_row):
            if y is None:
                continue
            v = pd.to_numeric(v_cell, errors="coerce")
            if pd.isna(v):
                continue
            records.append({"year": int(y), "source": label, "value": float(v)})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["year", "source", "value", "series", "sheet"])

    df = df.sort_values(["year", "source"])
    df["series"] = "Nihai Enerji Tüketimi"
    df["sheet"] = "Summary&Indicators"
    return df


def read_final_energy_electrification_ratio(xlsx_file) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3
    START_COL_IDX = 2  # C sütunu
    TOTAL_FINAL_ROW_1IDX = 46
    ELECTRICITY_ROW_1IDX = 50

    yr_r0 = YEARS_ROW_1IDX - 1
    tot_r0 = TOTAL_FINAL_ROW_1IDX - 1
    elc_r0 = ELECTRICITY_ROW_1IDX - 1

    if any(r < 0 for r in [yr_r0, tot_r0, elc_r0]) or yr_r0 >= len(raw) or tot_r0 >= len(raw) or elc_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()
    tot_row = raw.iloc[tot_r0, START_COL_IDX:].tolist()
    elc_row = raw.iloc[elc_r0, START_COL_IDX:].tolist()

    recs = []
    for y_cell, t_cell, e_cell in zip(years_row, tot_row, elc_row):
        y = _as_int_year(y_cell)
        if y is None:
            continue
        y = int(y)
        if y > MAX_YEAR:
            continue

        t = pd.to_numeric(t_cell, errors="coerce")
        e = pd.to_numeric(e_cell, errors="coerce")
        if pd.isna(t) or pd.isna(e) or float(t) == 0.0:
            continue

        recs.append({"year": y, "value": float(e) / float(t) * 100.0})

    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    df = df.sort_values("year")
    df["series"] = "Elektrifikasyon Oranı (%)"
    df["sheet"] = "Summary&Indicators"
    return df


def read_energy_import_dependency_ratio(xlsx_file) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3
    START_COL_IDX = 2  # column C
    NUM_ROW_1IDX = 14  # numerator: row 14
    DEN_ROW_1IDX = 32  # denominator: row 32

    yr_r0 = YEARS_ROW_1IDX - 1
    num_r0 = NUM_ROW_1IDX - 1
    den_r0 = DEN_ROW_1IDX - 1

    if any(r < 0 for r in [yr_r0, num_r0, den_r0]) or yr_r0 >= len(raw) or num_r0 >= len(raw) or den_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()
    num_row = raw.iloc[num_r0, START_COL_IDX:].tolist()
    den_row = raw.iloc[den_r0, START_COL_IDX:].tolist()

    years = []
    for y_cell in years_row:
        y = _as_int_year(y_cell)
        years.append(int(y) if y is not None and int(y) <= MAX_YEAR else None)

    recs = []
    for y, n_cell, d_cell in zip(years, num_row, den_row):
        if y is None:
            continue
        n = pd.to_numeric(n_cell, errors="coerce")
        d = pd.to_numeric(d_cell, errors="coerce")
        if pd.isna(n) or pd.isna(d) or d == 0:
            continue
        val = (1.0 - (float(n) / float(d))) * 100.0
        recs.append({"year": int(y), "value": float(val)})

    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])
    df = df.sort_values("year")
    df["series"] = "Dışa Bağımlılık Oranı (%)"
    df["sheet"] = "Summary&Indicators"
    return df


def read_gdp_series(xlsx_file) -> pd.DataFrame:
    return _read_scenario_assumptions_row(xlsx_file, GDP_ROW_1IDX, "GSYH")


def read_carbon_price_series(xlsx_file) -> pd.DataFrame:
    return _read_scenario_assumptions_row(xlsx_file, CARBON_PRICE_ROW_1IDX, "Karbon Fiyatı (Varsayım)")


def read_population_series(xlsx_file) -> pd.DataFrame:
    return _read_scenario_assumptions_row(xlsx_file, POP_ROW_1IDX, "Nüfus")


def read_electricity_consumption_by_sector(xlsx_file) -> pd.DataFrame:
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Power_Generation", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3
    yr_r0 = YEARS_ROW_1IDX - 1
    if yr_r0 < 0 or yr_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet"])

    years, year_cols_idx = _extract_years(raw, yr_r0)
    if not years:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet"])

    ROW_START_1IDX = 6
    ROW_END_1IDX = 10

    records = []
    for r1 in range(ROW_START_1IDX, ROW_END_1IDX + 1):
        r0 = r1 - 1
        if r0 < 0 or r0 >= len(raw):
            continue

        label = raw.iloc[r0, 0]
        label = "" if pd.isna(label) else str(label).strip()
        if label == "" or label.lower() == "nan":
            continue

        for y, c in zip(years, year_cols_idx):
            if int(y) > MAX_YEAR:
                continue
            v = pd.to_numeric(raw.iloc[r0, c], errors="coerce")
            if pd.isna(v):
                continue
            records.append({"year": int(y), "sector": label, "value": float(v)})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet"])

    df = df.sort_values(["year", "sector"])
    df["series"] = "Sektörlere Göre Elektrik Tüketimi"
    df["sheet"] = "Power_Generation"
    return df


# -----------------------------
# Generation (GWh) – Mix
# -----------------------------
def _gen_is_excluded(item: str) -> bool:
    it = (item or "").strip()
    if it in GEN_EXCLUDE_EXACT:
        return True
    if GEN_EXCLUDE_REGEX.search(it):
        return True
    return False


def generation_mix_from_block(gross_gen_df: pd.DataFrame) -> pd.DataFrame:
    """Gross generation mix (GWh) – Total Storage HARİÇ."""
    if gross_gen_df is None or gross_gen_df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = gross_gen_df.copy()
    df = df[~df["item"].apply(_gen_is_excluded)]

    # --- Total Storage ve storage alt bileşenlerini tamamen çıkar ---
    df = df[~df["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))]
    df = df[df["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    # Natural gas: sadece belirlenen satırların toplamı
    natgas_rows = df[df["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"
    natgas_series = natgas_series[["year", "group", "value"]]

    # Kalanlar teknoloji gruplarına
    df_rest = df[~df["item"].apply(_is_natural_gas_item)].copy()
    long = _to_long(df_rest, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)
    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()

    mix = pd.concat([mix, natgas_series], ignore_index=True)
    return mix


def share_series_from_mix(mix_df: pd.DataFrame, total_series_df: pd.DataFrame, groups: set[str], name: str) -> pd.DataFrame:
    if mix_df.empty or total_series_df.empty:
        return pd.DataFrame(columns=["year", "series", "value"])

    num = mix_df[mix_df["group"].isin(groups)].groupby("year", as_index=False)["value"].sum()
    den = total_series_df.rename(columns={"value": "total"}).copy()
    out = num.merge(den, on="year", how="inner")
    out["value"] = (out["value"] / out["total"]) * 100.0
    out["series"] = name
    return out[["year", "series", "value"]]


# -----------------------------
# Installed Capacity (GW)
# -----------------------------
def _cap_is_excluded(item: str) -> bool:
    it = (item or "").strip()
    if it in CAPACITY_EXCLUDE_EXACT:
        return True
    if CAPACITY_EXCLUDE_REGEX.search(it):
        return True
    return False


def _row_values_by_excel_row(raw: pd.DataFrame, row_1idx: int, year_cols_idx: list[int], years: list[int]) -> pd.Series:
    r0 = row_1idx - 1
    if r0 < 0 or r0 >= len(raw):
        return pd.Series(dtype=float)

    data = {}
    for y, c in zip(years, year_cols_idx):
        if y <= MAX_YEAR:
            data[y] = pd.to_numeric(raw.iloc[r0, c], errors="coerce")
    return pd.Series(data, dtype=float)


def total_capacity_series_from_rows(raw: pd.DataFrame, year_cols_idx: list[int], years: list[int]) -> pd.DataFrame:
    s79 = _row_values_by_excel_row(raw, KG_BASE_ROW_1IDX, year_cols_idx, years)
    s102 = _row_values_by_excel_row(raw, KG_SUB_ROW_1IDX_1, year_cols_idx, years)
    s106 = _row_values_by_excel_row(raw, KG_SUB_ROW_1IDX_2, year_cols_idx, years)

    kg = s79 - (s102.fillna(0) + s106.fillna(0))
    out = pd.DataFrame({"year": kg.index.astype(int), "value": kg.values})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).sort_values("year")
    return out


def storage_series_capacity(installed_cap_df: pd.DataFrame) -> pd.DataFrame:
    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_cap_is_excluded)]
    s = get_series_from_block(df, TOTAL_STORAGE_LABEL)
    if not s.empty:
        s["category"] = "Total Storage"
        return s[["year", "category", "value"]]

    comps = df[df["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))]
    if comps.empty:
        return pd.DataFrame(columns=["year", "category", "value"])
    long = _to_long(comps, value_name="value")
    out = long.groupby("year", as_index=False)["value"].sum()
    out["category"] = "Total Storage"
    return out[["year", "category", "value"]]


def ptx_series_capacity(installed_cap_df: pd.DataFrame) -> pd.DataFrame:
    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_cap_is_excluded)]
    s = get_series_from_block(df, TOTAL_PTX_LABEL)
    if not s.empty:
        s["category"] = "Power to X"
        return s[["year", "category", "value"]]

    comps = df[df["item"].astype(str).apply(lambda x: bool(PTX_COMPONENT_REGEX.search(x.strip())))]
    if comps.empty:
        return pd.DataFrame(columns=["year", "category", "value"])
    long = _to_long(comps, value_name="value")
    out = long.groupby("year", as_index=False)["value"].sum()
    out["category"] = "Power to X"
    return out[["year", "category", "value"]]


def capacity_mix_excl_storage_ptx(installed_cap_df: pd.DataFrame, cap_total: pd.DataFrame, cap_storage: pd.DataFrame, cap_ptx: pd.DataFrame) -> pd.DataFrame:
    if installed_cap_df is None or installed_cap_df.empty or cap_total.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_cap_is_excluded)]
    df = df[df["item"].str.strip().ne(TOTAL_CAPACITY_LABEL)]

    # storage & ptx components out
    df = df[~df["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))]
    df = df[df["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    df = df[~df["item"].astype(str).apply(lambda x: bool(PTX_COMPONENT_REGEX.search(x.strip())))]
    df = df[df["item"].str.strip().ne(TOTAL_PTX_LABEL)]

    natgas_rows = df[df["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"
    natgas_series = natgas_series[["year", "group", "value"]]

    df_rest = df[~df["item"].apply(_is_natural_gas_item)].copy()

    long = _to_long(df_rest, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)
    long.loc[long["group"] == "Other Renewables", "group"] = "Other"

    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()
    mix = pd.concat([mix, natgas_series], ignore_index=True)

    total_map = cap_total.set_index("year")["value"].to_dict()
    storage_map = (cap_storage.set_index("year")["value"].to_dict() if not cap_storage.empty else {})
    ptx_map = (cap_ptx.set_index("year")["value"].to_dict() if not cap_ptx.empty else {})

    known = mix[mix["group"] != "Other"].groupby("year", as_index=False)["value"].sum().rename(columns={"value": "known_sum"})
    known["total_excl"] = known["year"].map(total_map) - known["year"].map(storage_map).fillna(0) - known["year"].map(ptx_map).fillna(0)
    known["residual_other"] = (known["total_excl"] - known["known_sum"]).clip(lower=0)

    other_rows = known[["year", "residual_other"]].rename(columns={"residual_other": "value"})
    other_rows["group"] = "Other"

    mix = mix[mix["group"] != "Other"]
    mix = pd.concat([mix, other_rows], ignore_index=True)

    return mix


# -----------------------------
# UI
# -----------------------------
st.title("Türkiye Ulusal Enerji Planı Modeli Arayüzü")

with st.sidebar:
    st.header("Dosyalar (çoklu senaryo)")
    uploaded_files = st.file_uploader(
        "Excel yükleyin (.xlsx) — en fazla 12 dosya",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    # Güvenlik notu (kısa ve net)
    show_security_note = st.checkbox("Güvenlik notunu göster", value=True)
    if show_security_note:
        st.info("Yüklediğiniz Excel dosyaları yalnızca bu uygulamanın çalışması için okunur; dosyalar uygulama kimseye olarak paylaşılmaz.")

    st.divider()
    st.header("Ayarlar")

    # ÜSTTE: İlk Senaryo Yılı
    start_year_options = [2018, 2020, 2025, 2030, 2035, 2040, 2045]
    start_year = st.selectbox(
        "İlk Senaryo Yılı",
        start_year_options,
        index=start_year_options.index(2025) if 2025 in start_year_options else 0,
    )

    # ALTTA: Son Senaryo Yılı (eski Maksimum yıl)
    max_year = st.selectbox("Son Senaryo Yılı", [2050, 2045, 2040, 2035], index=0)
    MAX_YEAR = int(max_year)

    st.divider()
    st.header("Karşılaştırma modu")
    compare_mode = st.radio(
        "Stacked grafikler",
        ["Small multiples (önerilen)", "Yıl içinde yan yana (clustered)", "2035/2050 snapshot", "2025/2035 snapshot"],
        index=0,
    )

    stacked_value_mode = st.select_slider(
        "Stacked gosterim",
        options=["Mutlak", "Pay (%)"],
        value="Mutlak",
        help="Stacked grafiklerde mutlak deger (GWh/GW) yerine her yil icin paylari (%) gostermek icin Pay (%) secin.",
    )

    st.divider()
    st.header("Grafik tipi")
    ts_chart_style = st.selectbox(
        "Zaman serisi grafikleri",
        ["Bar (Gruplu)", "Çizgi", "Bar (Stack)"],
        index=0,
        help="Nüfus, GSYH, kişi başına tüketim gibi tek-değer zaman serilerini bu seçenekle çizdirirsiniz.",
    )

if not uploaded_files:
    st.info("Başlamak için en az 1 Excel dosyası yükleyin.")
    st.stop()

if len(uploaded_files) > 12:
    st.warning("En fazla 12 dosya yükleyebilirsiniz. İlk 12 dosya alınacak.")
    uploaded_files = uploaded_files[:12]


def _derive_scenario_name(uploaded) -> str:
    name = getattr(uploaded, "name", "Scenario")
    stem = Path(name).stem
    if stem.lower().startswith("finalreport_"):
        stem = stem[len("FinalReport_") :]
    return stem or "Scenario"


scenario_names_all = [_derive_scenario_name(f) for f in uploaded_files]

seen = {}
scenario_names_unique = []
for s in scenario_names_all:
    if s not in seen:
        seen[s] = 1
        scenario_names_unique.append(s)
    else:
        seen[s] += 1
        scenario_names_unique.append(f"{s} ({seen[s]})")

scenario_to_file = dict(zip(scenario_names_unique, uploaded_files))

default_n = 3 if len(scenario_names_unique) >= 3 else len(scenario_names_unique)
default_selected = scenario_names_unique[:default_n]

selected_scenarios = st.multiselect(
    "Karşılaştırılacak senaryolar",
    options=scenario_names_unique,
    default=default_selected,
)

with st.sidebar:
    st.markdown("**Karşılaştırılan senaryolar (tam ad):**")
    for i, scn in enumerate(selected_scenarios, 1):
        st.markdown(f"{i}. {scn}")

if not selected_scenarios:
    st.info("En az 1 senaryo seçin.")
    st.stop()

if len(selected_scenarios) >= 4 and compare_mode not in {"2035/2050 snapshot", "2025/2035 snapshot"}:
    st.warning("4+ senaryoda okunabilirlik için snapshot modları önerilir. Şimdilik en fazla 3 senaryo gösterilecek.")
    selected_scenarios = selected_scenarios[:3]


if len(selected_scenarios) == 2:
    with st.sidebar:
        st.divider()
        st.header("2 Senaryo Fark Modu : Kullanmadan önce Karşılaştırma modu: Small multiples harici mod seçin!")
        diff_mode_enabled = st.checkbox(
            "Farkı göster (A - B)",
            value=False,
            help="Sadece 2 senaryo seçiliyken aktif olur. A - B farkını tek seri olarak çizer.",
        )
        diff_scn_a = st.selectbox("İlave önlem/politika Senaryoları", options=selected_scenarios, index=0)
        diff_scn_b = st.selectbox("Referans senaryo ", options=selected_scenarios, index=1)
        if diff_scn_a == diff_scn_b:
            diff_scn_b = selected_scenarios[1] if selected_scenarios[0] == diff_scn_a else selected_scenarios[0]
else:
    diff_mode_enabled = False
    diff_scn_a = None
    diff_scn_b = None


def _ncols_for_selected(n: int) -> int:
    if n <= 1:
        return 1
    if n == 2:
        return 2
    return 3


# -----------------------------
# Scenario read/compute
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_scenario_bundle(xlsx_file, scenario: str, start_year: int, max_year: int):
    blocks = read_power_generation(xlsx_file)
    balance = blocks["electricity_balance"]
    gross_gen = blocks["gross_generation"]
    installed_cap = blocks["installed_capacity"]

    pop = _filter_years(read_population_series(xlsx_file), start_year, max_year)
    gdp = _filter_years(read_gdp_series(xlsx_file), start_year, max_year)
    carbon_price = _filter_years(read_carbon_price_series(xlsx_file), start_year, max_year)
    co2 = _filter_years(read_co2_emissions_series(xlsx_file), start_year, max_year)

    # Sektorel enerji kaynakli emisyonlar (CO2e) — Summary&Indicators (CRF 2023 varsayimi ile CO2->CO2e)
    energy_em_sector_co2e = _filter_years(read_energy_emissions_sectoral_co2e(xlsx_file), start_year, max_year)
    energy_em_total_co2e = pd.DataFrame(columns=["year", "value", "series"])
    if energy_em_sector_co2e is not None and not energy_em_sector_co2e.empty:
        energy_em_total_co2e = energy_em_sector_co2e.groupby("year", as_index=False)["value"].sum()
        energy_em_total_co2e["series"] = "Enerji Kaynakli Emisyonlar (CO2e, ktn)"

    # Net-Zero icin toplam (tahmini) bileenleri: Enerji + Enerji Disi&SGE + LULUCF
    co2_nz_stack = pd.DataFrame(columns=["year", "category", "value", "scenario"])
    if energy_em_total_co2e is not None and not energy_em_total_co2e.empty:
        recs = []
        for _, r in energy_em_total_co2e.iterrows():
            y = int(r["year"])
            e = float(r["value"])
            share = float(energy_share_assumption(y))
            total_excl_lulucf = e / share if share > 0 else np.nan
            other = total_excl_lulucf - e
            lulucf = float(lulucf_assumption_ktn(y))
            if np.isfinite(e):
                recs.append({"year": y, "category": "Enerji Kaynakli (Model, CO2e)", "value": e, "scenario": scenario})
            if np.isfinite(other):
                recs.append({"year": y, "category": "Enerji Disi Emisyonlar ve Diger SGE (Tahmini)", "value": float(other), "scenario": scenario})
            if np.isfinite(lulucf):
                recs.append({"year": y, "category": "LULUCF (Net Yutak, Tahmini)", "value": lulucf, "scenario": scenario})
        co2_nz_stack = pd.DataFrame(recs)

    primary_energy_source = _filter_years(read_primary_energy_consumption_by_source(xlsx_file), start_year, max_year)
    final_energy_source = _filter_years(read_final_energy_consumption_by_source(xlsx_file), start_year, max_year)
    dependency_ratio = _filter_years(read_energy_import_dependency_ratio(xlsx_file), start_year, max_year)
    electrification_ratio = _filter_years(read_final_energy_electrification_ratio(xlsx_file), start_year, max_year)

    total_supply = _filter_years(get_series_from_block(balance, TOTAL_SUPPLY_LABEL), start_year, max_year)

    # Generation mix (GWh) — Total Storage HARİÇ (kritik)
    gen_mix = _filter_years(generation_mix_from_block(gross_gen), start_year, max_year)

    ye_total = _filter_years(share_series_from_mix(gen_mix, total_supply, RENEWABLE_GROUPS, "Toplam YE"), start_year, max_year)
    ye_int = _filter_years(share_series_from_mix(gen_mix, total_supply, INTERMITTENT_RE_GROUPS, "Kesintili YE"), start_year, max_year)
    ye_both = pd.concat([ye_total, ye_int], ignore_index=True) if (not ye_total.empty or not ye_int.empty) else pd.DataFrame(columns=["year", "series", "value"])

    cap_total = _filter_years(
        total_capacity_series_from_rows(
            raw=blocks["_raw"],
            year_cols_idx=blocks["_year_cols_idx"],
            years=blocks["_years"],
        ),
        start_year,
        max_year,
    )

    cap_storage = _filter_years(storage_series_capacity(installed_cap), start_year, max_year)
    cap_ptx = _filter_years(ptx_series_capacity(installed_cap), start_year, max_year)
    storage_ptx = pd.concat([cap_storage, cap_ptx], ignore_index=True).rename(columns={"category": "group"})

    cap_mix = _filter_years(
        capacity_mix_excl_storage_ptx(
            installed_cap,
            cap_total,
            cap_storage.rename(columns={"group": "category"}, errors="ignore"),
            cap_ptx.rename(columns={"group": "category"}, errors="ignore"),
        ),
        start_year,
        max_year,
    )

    # --- NEW: add extra rows (98-101) into capacity mix (counted in totals) ---
    extra_cap = _filter_years(_read_power_generation_fixed_rows_as_stack(xlsx_file, CAPACITY_EXTRA_ROWS_1IDX), start_year, max_year)
    if extra_cap is not None and not extra_cap.empty:
        cap_mix = pd.concat([cap_mix, extra_cap], ignore_index=True) if cap_mix is not None else extra_cap

    electricity_by_sector = _filter_years(read_electricity_consumption_by_sector(xlsx_file), start_year, max_year)

    per_capita = pd.DataFrame(columns=["year", "value"])
    if (not total_supply.empty) and (not pop.empty):
        ts = total_supply.copy()
        pp = pop.copy()
        ts["year"] = pd.to_numeric(ts["year"], errors="coerce").astype("Int64")
        pp["year"] = pd.to_numeric(pp["year"], errors="coerce").astype("Int64")
        ts = ts.dropna(subset=["year", "value"])
        pp = pp.dropna(subset=["year", "value"])
        ts["year"] = ts["year"].astype(int)
        pp["year"] = pp["year"].astype(int)

        pop_median = float(pp["value"].median()) if len(pp) else np.nan
        pop_multiplier = 1e6 if np.isfinite(pop_median) and pop_median < 1000 else 1.0
        pp["pop_person"] = pp["value"] * pop_multiplier

        merged = ts.merge(pp[["year", "pop_person"]], on="year", how="inner")
        merged = merged[(merged["pop_person"] > 0) & merged["value"].notna()]
        merged["value"] = (merged["value"] * 1_000_000.0) / merged["pop_person"]  # kWh/kişi
        per_capita = merged[["year", "value"]].sort_values("year")

    def _add_scn(df):
        if df is None:
            return df
        out = df.copy()
        out["scenario"] = scenario
        return out

    bundle = {
        "scenario": scenario,
        "pop": _add_scn(pop),
        "gdp": _add_scn(gdp),
        "carbon_price": _add_scn(carbon_price),
        "co2": _add_scn(co2),
        "energy_em_sector_co2e": _add_scn(energy_em_sector_co2e),
        "energy_em_total_co2e": _add_scn(energy_em_total_co2e),
        "co2_nz_stack": co2_nz_stack,
        "total_supply": _add_scn(total_supply),
        "gen_mix": _add_scn(gen_mix),
        "cap_total": _add_scn(cap_total),
        "cap_mix": _add_scn(cap_mix),
        "electricity_by_sector": _add_scn(electricity_by_sector),
        "primary_energy_source": _add_scn(primary_energy_source),
        "final_energy_source": _add_scn(final_energy_source),
        "dependency_ratio": _add_scn(dependency_ratio),
        "electrification_ratio": _add_scn(electrification_ratio),
        "ye_both": _add_scn(ye_both),
        "per_capita_el": _add_scn(per_capita),
        "storage_ptx": _add_scn(storage_ptx),
    }
    return bundle


bundles = []
for scn in selected_scenarios:
    f = scenario_to_file[scn]
    bundles.append(compute_scenario_bundle(f, scn, start_year, MAX_YEAR))


def _concat(key: str):
    dfs = [b.get(key) for b in bundles if b.get(key) is not None and not b.get(key).empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


df_pop = _concat("pop")
df_gdp = _concat("gdp")
df_co2 = _concat("co2")
df_energy_em_sector_co2e = _concat("energy_em_sector_co2e")
df_energy_em_total_co2e = _concat("energy_em_total_co2e")
df_co2_nz_stack = _concat("co2_nz_stack")
df_cp = _concat("carbon_price")
df_supply = _concat("total_supply")
df_genmix = _concat("gen_mix")
df_capmix = _concat("cap_mix")
df_primary = _concat("primary_energy_source")
df_sector_el = _concat("electricity_by_sector")
df_final = _concat("final_energy_source")
df_electrification = _concat("electrification_ratio")
df_storage_ptx = _concat("storage_ptx")


# -----------------------------
# Line charts (single axis, scenario colors)
# -----------------------------
def _line_chart(
    df,
    title: str,
    y_title: str,
    value_format: str = ",.2f",
    chart_style: str | None = None,
):
    if df is None or df.empty:
        st.subheader(title)
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce")
    dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
    dfp = dfp.dropna(subset=["year", "value", "scenario"])
    dfp["year"] = dfp["year"].astype(int)

    # --- 2 senaryo fark modu (A - B) ---
    diff_on = bool(globals().get("diff_mode_enabled", False))
    a = globals().get("diff_scn_a")
    b = globals().get("diff_scn_b")
    if diff_on and a and b:
        sub = dfp[dfp["scenario"].isin([a, b])]
        if not sub.empty:
            wide = sub.pivot_table(index="year", columns="scenario", values="value", aggfunc="mean")
            if (a in wide.columns) and (b in wide.columns):
                wide = wide[[a, b]].copy()
                wide["value"] = wide[a] - wide[b]
                out = wide.reset_index()[["year", "value"]]
                out["scenario"] = f"Fark: {a} - {b}"
                dfp = out
                title = f"{title} — Fark ({a} - {b})"

    st.subheader(title)

    year_vals = sorted(dfp["year"].unique().tolist())
    style = chart_style or globals().get("ts_chart_style", "Bar (Gruplu)")

    base = alt.Chart(dfp).encode(
        color=alt.Color("scenario:N", title="Senaryo", legend=alt.Legend(labelLimit=0, titleLimit=0)),
        tooltip=[
            alt.Tooltip("scenario:N", title="Senaryo"),
            alt.Tooltip("year:O", title="Yıl"),
            alt.Tooltip("value:Q", title=y_title, format=value_format),
        ],
    )

    if style == "Çizgi":
        chart = (
            base.mark_line(point=True)
            .encode(
                x=alt.X(
                    "year:Q",
                    title="Yıl",
                    scale=alt.Scale(domain=[min(year_vals), max(year_vals)]),
                    axis=alt.Axis(values=year_vals, format="d", labelAngle=0),
                ),
                y=alt.Y("value:Q", title=y_title),
            )
        )
    elif style == "Bar (Stack)":
        chart = (
            base.mark_bar()
            .encode(
                x=alt.X("year:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0)),
                y=alt.Y("value:Q", title=y_title, stack="zero"),
            )
        )
    else:
        chart = (
            base.mark_bar()
            .encode(
                x=alt.X("year:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0)),
                xOffset=alt.XOffset("scenario:N"),
                y=alt.Y("value:Q", title=y_title),
            )
        )

    st.altair_chart(chart.properties(height=320), use_container_width=True)


_line_chart(df_pop, "Türkiye Nüfus Gelişimi", "Nüfus (milyon)", value_format=",.3f")
_line_chart(df_gdp, "GSYH (Milyar ABD Doları, 2015 fiyatlarıyla)", "Milyar ABD Doları (2015)", value_format=",.2f")
st.divider()

df_pc = _concat("per_capita_el")
_line_chart(df_pc, "Kişi Başına Elektrik Tüketimi (kWh/kişi)", "kWh/kişi", value_format=",.0f")
_line_chart(df_electrification, "Nihai Enerjide Elektrifikasyon Oranı (%)", "%", value_format=",.1f")

# -----------------------------
# KPI row (per scenario)
# -----------------------------
st.subheader("Özet Bilgi Kartları (Seçili Senaryolar)")
ncols = _ncols_for_selected(len(selected_scenarios))
cols = st.columns(ncols)


def _kpi_for_bundle(b):
    scn = b["scenario"]
    supply = b["total_supply"]
    ye = b["ye_both"]
    cap_total = b["cap_total"]
    gdp = b["gdp"]

    latest_year = int(supply["year"].max()) if supply is not None and not supply.empty else None
    latest_total = float(supply.loc[supply["year"] == latest_year, "value"].iloc[0]) if latest_year else np.nan

    latest_ye_total = np.nan
    latest_ye_int = np.nan
    if latest_year and ye is not None and not ye.empty and (ye["year"] == latest_year).any():
        tmp = ye[ye["year"] == latest_year].set_index("series")["value"].to_dict()
        latest_ye_total = float(tmp.get("Toplam YE", np.nan))
        latest_ye_int = float(tmp.get("Kesintili YE", np.nan))

    latest_cap = np.nan
    if latest_year and cap_total is not None and not cap_total.empty and (cap_total["year"] == latest_year).any():
        latest_cap = float(cap_total.loc[cap_total["year"] == latest_year, "value"].iloc[0])

    gdp_cagr = np.nan
    if gdp is not None and not gdp.empty:
        g = gdp.sort_values("year")
        y0 = int(g.iloc[0]["year"])
        y1 = int(g.iloc[-1]["year"])
        v0 = float(g.iloc[0]["value"])
        v1 = float(g.iloc[-1]["value"])
        gdp_cagr = _cagr(v0, v1, int(y1 - y0))

    return {
        "scenario": scn,
        "latest_year": latest_year,
        "total_supply": latest_total,
        "ye_total": latest_ye_total,
        "ye_int": latest_ye_int,
        "cap_total": latest_cap,
        "gdp_cagr": gdp_cagr,
    }


kpis = [_kpi_for_bundle(b) for b in bundles]

for i, kpi in enumerate(kpis[:ncols]):
    with cols[i]:
        st.markdown(f"**{kpi['scenario']}**")
        st.metric("GSYH CAGR (%)", f"{kpi['gdp_cagr']*100:.2f}%" if np.isfinite(kpi["gdp_cagr"]) else "—")
        st.metric(f"Toplam Arz (GWh) – {kpi['latest_year'] or ''}", f"{kpi['total_supply']:,.0f}" if np.isfinite(kpi["total_supply"]) else "—")
        st.metric("YE Payı (%)", f"{kpi['ye_total']:.1f}% / {kpi['ye_int']:.1f}%" if np.isfinite(kpi["ye_total"]) else "—")
        st.metric("Elektrik Kurulu Gücü (GW)", f"{kpi['cap_total']:,.3f}" if np.isfinite(kpi["cap_total"]) else "—")

if len(kpis) > ncols:
    with st.expander("Diğer seçili senaryoların KPI’ları"):
        for kpi in kpis[ncols:]:
            st.markdown(
                f"**{kpi['scenario']}** — "
                f"GSYH CAGR: {(kpi['gdp_cagr']*100):.2f}% | "
                f"Toplam Arz: {kpi['total_supply']:,.0f} GWh | "
                f"YE Payı: {kpi['ye_total']:.1f}%/{kpi['ye_int']:.1f}% | "
                f"KG: {kpi['cap_total']:,.3f} GW"
            )

st.divider()

# -----------------------------
# Stacked charts helpers
# -----------------------------
def _normalize_stacked_to_percent(df: pd.DataFrame, stack_field: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    dfp = df.copy()
    dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce").astype("Int64")
    dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
    dfp = dfp.dropna(subset=["scenario", "year", stack_field, "value"])
    totals = dfp.groupby(["scenario", "year"], as_index=False)["value"].sum().rename(columns={"value": "total"})
    dfp = dfp.merge(totals, on=["scenario", "year"], how="left")
    dfp["value"] = np.where(dfp["total"] > 0, (dfp["value"] / dfp["total"]) * 100.0, np.nan)
    dfp = dfp.drop(columns=["total"])
    return dfp


def _stacked_small_multiples(
    df,
    title: str,
    x_field: str,
    stack_field: str,
    y_title: str,
    category_title: str,
    value_format: str,
    order=None,
    is_percent: bool = False,
):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = dfp["year"].astype(int)

    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["scenario", "year", stack_field])

    ymax = 100.0 if is_percent else float(dfp.groupby(["scenario", "year"])["value"].sum().max()) if len(dfp) else None
    yscale = alt.Scale(domain=[0, ymax]) if ymax and np.isfinite(ymax) else alt.Undefined

    n = len(selected_scenarios)
    ncols = _ncols_for_selected(n)
    cols = st.columns(ncols)

    for idx, scn in enumerate(selected_scenarios):
        sub = dfp[dfp["scenario"] == scn]
        if sub.empty:
            continue

        bars_src = alt.Chart(sub)
        if not is_percent:
            bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=[x_field])

        bars = (
            bars_src.mark_bar()
            .encode(
                x=alt.X(f"{x_field}:O", title="Yıl"),
                y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
                color=alt.Color(f"{stack_field}:N", title=category_title),
                tooltip=[
                    alt.Tooltip(f"{x_field}:O", title="Yıl"),
                    alt.Tooltip(f"{stack_field}:N", title=category_title),
                    alt.Tooltip("value:Q", title=y_title, format=value_format),
                    *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
                ],
            )
        )

        with cols[idx % ncols]:
            st.markdown(f"**{scn}**")
            st.altair_chart(bars.properties(height=380), use_container_width=True)


def _stacked_clustered(
    df,
    title: str,
    x_field: str,
    stack_field: str,
    y_title: str,
    category_title: str,
    value_format: str,
    order=None,
    is_percent: bool = False,
):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = dfp["year"].astype(int)
    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["year", "scenario", stack_field])

    yscale = alt.Scale(domain=[0, 100]) if is_percent else alt.Undefined

    bars_src = alt.Chart(dfp)
    if not is_percent:
        bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=["scenario", x_field])

    bars = (
        bars_src.mark_bar()
        .encode(
            x=alt.X(f"{x_field}:O", title="Yıl"),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
            color=alt.Color(f"{stack_field}:N", title=category_title),
            tooltip=[
                alt.Tooltip("scenario:N", title="Senaryo"),
                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                alt.Tooltip(f"{stack_field}:N", title=category_title),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
                *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
            ],
        )
    )
    st.altair_chart(bars.properties(height=420), use_container_width=True)


def _stacked_snapshot(
    df,
    title: str,
    x_field: str,
    stack_field: str,
    y_title: str,
    category_title: str,
    value_format: str,
    years=(2035, 2050),
    order=None,
    is_percent: bool = False,
):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return
    dfp = df.copy()
    dfp["year"] = dfp["year"].astype(int)
    dfp = dfp[dfp["year"].isin(list(years))]
    if dfp.empty:
        st.warning("Seçilen yıllar için veri yok (seçili snapshot yılları).")
        return

    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["year", "scenario", stack_field])

    yscale = alt.Scale(domain=[0, 100]) if is_percent else alt.Undefined

    bars_src = alt.Chart(dfp)
    if not is_percent:
        bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=["scenario", x_field])

    bars = (
        bars_src.mark_bar()
        .encode(
            x=alt.X(f"{x_field}:O", title="Yıl"),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
            color=alt.Color(f"{stack_field}:N", title=category_title),
            tooltip=[
                alt.Tooltip("scenario:N", title="Senaryo"),
                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                alt.Tooltip(f"{stack_field}:N", title=category_title),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
                *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
            ],
        )
    )
    st.altair_chart(bars.properties(height=420), use_container_width=True)


def _render_stacked(df, title, x_field, stack_field, y_title, category_title, value_format, order=None):
    df_use = df
    y_title_use = y_title
    value_format_use = value_format
    is_percent = False

    # --- 2 senaryo fark modu (A - B) — SADECE MUTLAK DEĞERLERDE ---
    diff_on = bool(globals().get("diff_mode_enabled", False))
    a = globals().get("diff_scn_a")
    b = globals().get("diff_scn_b")
    if diff_on and a and b and (globals().get("stacked_value_mode") != "Pay (%)"):
        d0 = df_use.copy() if df_use is not None else None
        if d0 is not None and (not d0.empty) and ("scenario" in d0.columns):
            sub = d0[d0["scenario"].isin([a, b])].copy()
            if not sub.empty and (x_field in sub.columns) and (stack_field in sub.columns):
                sub[x_field] = pd.to_numeric(sub[x_field], errors="coerce")
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub.dropna(subset=[x_field, stack_field, "value", "scenario"])
                sub[x_field] = sub[x_field].astype(int)
                wide = sub.pivot_table(index=[x_field, stack_field], columns="scenario", values="value", aggfunc="sum")
                if (a in wide.columns) and (b in wide.columns):
                    wide = wide[[a, b]].copy().fillna(0.0)
                    wide["value"] = wide[a] - wide[b]
                    out = wide.reset_index()[[x_field, stack_field, "value"]]
                    out["scenario"] = f"Fark: {a} - {b}"
                    df_use = out
                    title = f"{title} — Fark ({a} - {b})"

    if stacked_value_mode == "Pay (%)":
        df_use = _normalize_stacked_to_percent(df_use, stack_field=stack_field)
        y_title_use = "%"
        value_format_use = ".1f"
        is_percent = True

    title_use = title + (" (Pay %)" if is_percent else "")

    safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", f"show_total_{title}")
    show_total_panel = st.checkbox(
        "Sadece toplamı (Total) ayrı grafikte göster",
        key=safe_key,
        value=False,
        help="Stacked grafikte toplamı okumak zor olursa açın: altta sadece toplam çizgi grafiği gösterilir.",
        disabled=is_percent,
    )

    def _render_main():
        if compare_mode == "Small multiples (önerilen)":
            _stacked_small_multiples(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, order=order, is_percent=is_percent)
        elif compare_mode == "Yıl içinde yan yana (clustered)":
            _stacked_clustered(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, order=order, is_percent=is_percent)
        elif compare_mode == "2035/2050 snapshot":
            _stacked_snapshot(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, years=(2035, 2050), order=order, is_percent=is_percent)
        else:
            _stacked_snapshot(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, years=(2025, 2035), order=order, is_percent=is_percent)

    def _render_total():
        if df_use is None or df_use.empty:
            return
        totals = df_use.groupby(["scenario", x_field], as_index=False)["value"].sum().rename(columns={"value": "Total"})

        if compare_mode == "2035/2050 snapshot":
            totals = totals[totals[x_field].isin([2035, 2050])]
        elif compare_mode == "2025/2035 snapshot":
            totals = totals[totals[x_field].isin([2025, 2035])]

        if totals.empty:
            return

        st.markdown("**Toplam (Total) — ayrı grafik**")

        if compare_mode == "Small multiples (önerilen)":
            scenarios_to_show = list(dict.fromkeys(totals["scenario"].tolist()))
            n = len(scenarios_to_show)
            ncols = _ncols_for_selected(n)
            cols = st.columns(ncols)
            for idx, scn in enumerate(scenarios_to_show):
                sub = totals[totals["scenario"] == scn]
                if sub.empty:
                    continue
                with cols[idx % ncols]:
                    st.caption(scn)
                    ch = (
                        alt.Chart(sub)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(f"{x_field}:O", title="Yıl"),
                            y=alt.Y("Total:Q", title=y_title),
                            tooltip=[
                                alt.Tooltip("scenario:N", title="Senaryo"),
                                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                                alt.Tooltip("Total:Q", title="Total", format=value_format_use),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(ch, use_container_width=True)
        else:
            ch = (
                alt.Chart(totals)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{x_field}:O", title="Yıl"),
                    y=alt.Y("Total:Q", title=y_title),
                    color=alt.Color("scenario:N", title="Senaryo", legend=alt.Legend(labelLimit=0, titleLimit=0)),
                    tooltip=[
                        alt.Tooltip("scenario:N", title="Senaryo"),
                        alt.Tooltip(f"{x_field}:O", title="Yıl"),
                        alt.Tooltip("Total:Q", title="Total", format=value_format_use),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(ch, use_container_width=True)

    _render_main()
    if show_total_panel and (not is_percent):
        _render_total()


# =========================
# FIXED HELPERS: Waterfall preparation + render
# =========================
def prepare_yearly_transition_waterfall(
    df_mix: pd.DataFrame,
    scenario: str,
    start_year: int,
    end_year: int,
    value_col: str = "value",
    group_col: str = "category",
) -> pd.DataFrame:
    """
    Aynı senaryo içinde start_year -> end_year yakıt/teknoloji dönüşümü (Δ=end-start).
    Çıktı: step, delta, y0, y1 (waterfall için).
    """
    if df_mix is None or df_mix.empty:
        return pd.DataFrame()

    df = df_mix.copy()
    if "scenario" not in df.columns:
        return pd.DataFrame()

    # Güvenli tip dönüşümleri (2050 gibi boş/metin kolonlar için kritik)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["year", value_col, "scenario"])
    df["year"] = df["year"].astype(int)

    df = df[(df["scenario"] == scenario) & (df["year"].isin([start_year, end_year]))]
    if df.empty:
        return pd.DataFrame()

    df[value_col] = df[value_col].fillna(0.0)

    wide = (
        df.pivot_table(index=group_col, columns="year", values=value_col, aggfunc="sum")
        .fillna(0.0)
        .reset_index()
    )

    if start_year not in wide.columns or end_year not in wide.columns:
        return pd.DataFrame()

    wide[start_year] = pd.to_numeric(wide[start_year], errors="coerce").fillna(0.0)
    wide[end_year] = pd.to_numeric(wide[end_year], errors="coerce").fillna(0.0)

    wide["delta"] = wide[end_year] - wide[start_year]
    wide = wide[wide["delta"].abs() > 1e-9]
    if wide.empty:
        return pd.DataFrame()

    wide = wide.sort_values("delta")

    records = []
    cumulative = 0.0
    for _, r in wide.iterrows():
        y0 = cumulative
        y1 = cumulative + float(r["delta"])
        records.append({"step": str(r[group_col]), "delta": float(r["delta"]), "y0": y0, "y1": y1})
        cumulative = y1

    records.append({"step": "Net Değişim", "delta": cumulative, "y0": 0.0, "y1": cumulative})
    return pd.DataFrame(records)


def render_waterfall(df_wf: pd.DataFrame, title: str, y_title: str):
    if df_wf is None or df_wf.empty:
        st.info("Seçilen yıllar için dönüşüm verisi bulunamadı.")
        return

    d = df_wf.copy()
    d["color"] = np.where(
        d["step"] == "Net Değişim",
        "Net",
        np.where(d["delta"] >= 0, "Artış", "Azalış"),
    )

    ch = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("step:N", title="Yakıt / Teknoloji", sort=None),
            y=alt.Y("y0:Q", title=y_title),
            y2="y1:Q",
            color=alt.Color(
                "color:N",
                scale=alt.Scale(domain=["Artış", "Azalış", "Net"], range=["#2ca02c", "#d62728", "#1f77b4"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("step:N", title="Adım"),
                alt.Tooltip("delta:Q", title="Δ", format=",.2f"),
            ],
        )
        .properties(height=340)
    )

    st.markdown(f"**{title}**")
    st.altair_chart(ch, use_container_width=True)


# -----------------------------
# 1) Elektrik üretim karması (stacked) — TOTAL STORAGE HARİÇ
# -----------------------------
order_gen = [
    "Hydro",
    "Wind (RES)",
    "Solar (GES)",
    "Other Renewables",
    "Natural gas",
    "Coal",
    "Lignite",
    "Nuclear",
    "Other",
]
_render_stacked(
    df_genmix.rename(columns={"group": "category"}),
    title="Kaynaklarına Göre Elektrik Üretimi (GWh)",
    x_field="year",
    stack_field="category",
    y_title="GWh",
    category_title="Kaynak/Teknoloji",
    value_format=",.0f",
    order=order_gen,
)

st.divider()

# -----------------------------
# 2) Kurulu güç karması (stacked, storage & PTX hariç)
# -----------------------------
order_cap = ["Hydro", "Wind (RES)", "Solar (GES)", "Natural gas", "Coal", "Lignite", "Nuclear", "Other"]
_render_stacked(
    df_capmix.rename(columns={"group": "category"}),
    title="Elektrik Kurulu Gücü (GW) – Depolama & PTX Hariç",
    x_field="year",
    stack_field="category",
    y_title="GW",
    category_title="Teknoloji",
    value_format=",.2f",
    order=order_cap,
)

st.divider()

# -----------------------------
# 2.1) Sektörlere Göre Elektrik Tüketimi (stacked)
# -----------------------------
_render_stacked(
    df_sector_el.rename(columns={"sector": "category"}),
    title="Sektörlere Göre Elektrik Tüketimi (GWh)",
    x_field="year",
    stack_field="category",
    y_title="GWh",
    category_title="Sektör",
    value_format=", .0f".replace(" ", ""),
)

# -----------------------------
# UI PANELİ: Yakıt/Teknoloji Bazlı Enerji Dönüşümü (Δ)
# -----------------------------
if stacked_value_mode != "Pay (%)":  # yüzde modunda anlamsız, kapat
    st.markdown("### Yakıt/Teknoloji Bazlı Enerji Dönüşümü (Δ)")
    st.markdown(
        """
        <div style="display:flex; gap:18px; align-items:center; margin:6px 0 8px 0;">
          <span><span style="display:inline-block;width:10px;height:10px;background:#2ca02c;border-radius:50%;margin-right:6px;"></span><b>Artış</b></span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#d62728;border-radius:50%;margin-right:6px;"></span><b>Azalış</b></span>
          <span><span style="display:inline-block;width:10px;height:10px;background:#1f77b4;border-radius:50%;margin-right:6px;"></span><b>Net Değişim</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Grafikler, seçili senaryoda başlangıç ve bitiş yılları (ayarlardan seçiniz) arasındaki "
        "yakıt/teknoloji bazlı elektrik üretimi ve kurulu güç değişimlerini (Δ) ifade etmektedir. "
        "Kırmızılar azalışı, yeşiller artışı, mavi ise kümülatif net değişimi ifade eder."
    )

    if len(selected_scenarios) == 1:
        scn_tr = selected_scenarios[0]
    else:
        scn_tr = st.selectbox(
            "Dönüşüm analizi için senaryo seçin",
            options=selected_scenarios,
            index=0,
            key="transition_scn_select",
        )

    gen_for_wf = df_genmix.rename(columns={"group": "category"}).copy()
    cap_for_wf = df_capmix.rename(columns={"group": "category"}).copy()

    wf_gen = prepare_yearly_transition_waterfall(
        gen_for_wf,
        scenario=scn_tr,
        start_year=int(start_year),
        end_year=int(MAX_YEAR),
        value_col="value",
        group_col="category",
    )

    wf_cap = prepare_yearly_transition_waterfall(
        cap_for_wf,
        scenario=scn_tr,
        start_year=int(start_year),
        end_year=int(MAX_YEAR),
        value_col="value",
        group_col="category",
    )

    colA, colB = st.columns(2)
    with colA:
        render_waterfall(
            wf_gen,
            title=f"Elektrik Üretimi Dönüşümü (GWh) — {start_year} → {MAX_YEAR}",
            y_title="GWh",
        )
    with colB:
        render_waterfall(
            wf_cap,
            title=f"Kurulu Güç Dönüşümü (GW) — {start_year} → {MAX_YEAR}",
            y_title="GW",
        )

st.divider()

# -----------------------------
# 2.2) Depolama & PTX Kurulu Gücü (GW)
# -----------------------------
order_storage_ptx = ["Total Storage", "Power to X"]
_render_stacked(
    df_storage_ptx.rename(columns={"group": "category"}),
    title="Depolama & PTX Kurulu Gücü (GW)",
    x_field="year",
    stack_field="category",
    y_title="GW",
    category_title="Kategori",
    value_format=",.3f",
    order=order_storage_ptx,
)

st.divider()

# -----------------------------
# 3) Birincil enerji talebi (stacked)
# -----------------------------
_render_stacked(
    df_primary.rename(columns={"source": "category"}),
    title="Birincil Enerji Talebi (GWh)",
    x_field="year",
    stack_field="category",
    y_title="GWh",
    category_title="Kaynak",
    value_format=",.0f",
)

st.divider()

# -----------------------------
# 4) Nihai enerji tüketimi (stacked)
# -----------------------------
_render_stacked(
    df_final.rename(columns={"source": "category"}),
    title="Kaynaklarına Göre Nihai Enerji Tüketimi (GWh)",
    x_field="year",
    stack_field="category",
    y_title="GWh",
    category_title="Kaynak",
    value_format=",.0f",
)

st.divider()

# -----------------------------
# Remaining line charts
# -----------------------------
_line_chart(df_co2, "CO2 Emisyonları (ktn CO2)", "ktn CO2", value_format=",.0f")
_line_chart(df_cp, "Karbon Fiyatı (Varsayım) -$", "ABD Doları (2015) / tCO₂", value_format=",.2f")

# -----------------------------
# Net-Zero takibi icin: Emisyonlar (CO2e) — Enerji + Enerji Disi&SGE (+ LULUCF)
# - Mutlak mod: LULUCF dahil
# - Pay (%) mod: LULUCF otomatik haric
# -----------------------------
st.divider()
st.markdown("## Türkiye Seragazı Emisyonları — Net Zero Hedefi Takibi (CO₂e)")
st.caption(
    "Bu panel, Net Zero hedefine yaklasimi izlemek amaciyla kullanilir. "
    "Enerji disi emisyonlar/SGE ve LULUCF degerleri varsayimsaldir. "
    "CO₂→CO₂e donusumu (2023 CRF varsayimi): Elektrik 0.99, Ulastirma 0.94, Sanayi 0.97, Tarim 0.01, Diger 0.99 (CO₂/CO₂e)."
)

df_nz_plot = df_co2_nz_stack.copy() if df_co2_nz_stack is not None else pd.DataFrame()
if not df_nz_plot.empty and stacked_value_mode == "Pay (%)":
    df_nz_plot = df_nz_plot[df_nz_plot["category"] != "LULUCF (Net Yutak, Tahmini)"]

_render_stacked(
    df_nz_plot,
    title="CO₂e Emisyon Bileşenleri (ktn CO₂e)",
    x_field="year",
    stack_field="category",
    y_title="ktn CO₂e",
    category_title="Bileşen",
    value_format=",.0f",
    order=[
        "Enerji Kaynakli (Model, CO2e)",
        "Enerji Disi Emisyonlar ve Diger SGE (Tahmini)",
        "LULUCF (Net Yutak, Tahmini)",
    ],
)

with st.expander("Çalıştırma"):
    st.code("pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py", language="bash")
