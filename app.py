
# =======================
# Sabit Kaynak Renk Haritası (Energy Source Color Map)
# =======================
SOURCE_COLOR_MAP = {
    "Coal": "#3a3a3a",        # koyu gri
    "Lignite": "#9e9e9e",     # açık gri
    "Natural Gas": "#1f77b4", # mavi
    "Gas": "#1f77b4",
    "Hydro": "#0b3d91",       # lacivert
    "Hydropower": "#0b3d91",
    "Wind": "#2ca02c",        # yeşil
    "Solar": "#ffd60a",       # sarı
    "Nuclear": "#ff7f0e",     # turuncu
    # Alternatif/Türkçe etiketler (Excel / arayüz uyumu)
    "Kömür": "#3a3a3a",
    "Linyit": "#9e9e9e",
    "Doğal Gaz": "#1f77b4",
    "Hidro": "#0b3d91",
    "Rüzgar": "#2ca02c",
    "Güneş": "#ffd60a",
    "Nükleer": "#ff7f0e",
    # Modeldeki grup adları
    "Wind (RES)": "#2ca02c",
    "Solar (GES)": "#ffd60a",
}

# =======================
# Grafik-Türüne Özel Renk Haritaları (IEA/Bloomberg benzeri, yüksek kontrast)
# =======================
SECTOR_COLOR_MAP = {
    "Energy Branch & Other Uses": "#4e79a7",  # mavi
    "Industry": "#f28e2b",                   # turuncu
    "Residential": "#59a14f",                # yeşil
    "Tertiary": "#af7aa1",                   # mor
    "Transport": "#e15759",                  # kırmızı
}

EMISSION_COMPONENT_COLOR_MAP = {
    "Enerji Kaynakli (Model, CO2e)": "#4e79a7",                # mavi
    "Enerji Disi Emisyonlar ve Diger SGE (Tahmini)": "#b07aa1", # magenta/mor
    "LULUCF (Net Yutak, Tahmini)": "#59a14f",                  # yeşil
}


def get_source_color(name):
    if isinstance(name, str):
        for key, val in SOURCE_COLOR_MAP.items():
            if key.lower() in name.lower():
                return val
    return None

# Kaynak/teknoloji renk ölçeği (Altair) – önemli yakıtlar sabit, diğerleri deterministik
SOURCE_PRIORITY = [
    "Coal", "Kömür",
    "Lignite", "Linyit",
    "Natural Gas", "Doğal Gaz", "Gas",
    "Hydro", "Hidro", "Hydropower",
    "Wind", "Rüzgar", "Wind (RES)",
    "Solar", "Güneş", "Solar (GES)",
    "Nuclear", "Nükleer",
]

def _hash_color(name: str) -> str:
    """Bilinmeyen kategoriler için deterministik HEX renk."""
    if not isinstance(name, str):
        name = str(name)
    h = abs(hash(name)) % 360  # hue
    s = 60
    l = 50
    # HSL -> RGB
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def _ordered_domain(values: list[str]) -> list[str]:
    """Önce öncelikli yakıtlar, sonra kalanları alfabetik."""
    vals = [v for v in values if isinstance(v, str)]
    # benzersiz koru
    seen = set()
    uniq = []
    for v in vals:
        if v not in seen:
            seen.add(v); uniq.append(v)

    def prio_key(v):
        vv = v.lower()
        for i, p in enumerate(SOURCE_PRIORITY):
            if p.lower() in vv:
                return (0, i)
        return (1, v.lower())

    return sorted(uniq, key=prio_key)

def _source_color_encoding(df, field: str, title: str, order=None, color_map=None):
    """Stacked/bar grafiklerde kategorilerin rengini sabitle.

    - color_map verilirse (dict: {kategori: '#RRGGBB'}) önce onu kullanır.
    - Aksi halde enerji kaynak haritası + deterministik hash renk.
    """
    try:
        # normalize optional custom map (case-insensitive exact match)
        cmap = None
        if isinstance(color_map, dict) and len(color_map):
            cmap = {str(k).strip().lower(): str(v).strip() for k, v in color_map.items() if k is not None and v is not None}
        if order is not None and len(order):
            domain = _ordered_domain([str(x) for x in order])
        else:
            domain = _ordered_domain([str(x) for x in df[field].dropna().unique().tolist()])

        cmap = (color_map or SOURCE_COLOR_MAP)
        rng = []
        for d in domain:
            c = (cmap.get(str(d).strip().lower()) if cmap else None)
            if not c:
                c = get_source_color(d)
            rng.append(c if c else _hash_color(d))

        return alt.Color(f"{field}:N", title=title, sort=domain, scale=alt.Scale(domain=domain, range=rng))
    except Exception:
        # fallback: Altair default
        return alt.Color(f"{field}:N", title=title)


# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO

st.set_page_config(page_title="Power Generation Dashboard", layout="wide")

# -----------------------------
# Units
# -----------------------------
# Electricity / energy values are read as GWh.
# Optional display conversion: GWh -> thousand toe (ktoe, "bin TEP").
# 1 toe = 11.63 MWh  =>  1 GWh = 1000/11.63 = 85.9845 toe = 0.0859845 ktoe
GWH_TO_KTOE = 0.0859845

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
    """Yıl hücresini mümkün olduğunca sağlam parse et."""
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
    """Excel sheet içinden bir blok (tablo) çek.

    Eski yaklaşım "ilk boş satırda dur" idi; bazı dosyalarda ilk sütun hiç boş
    olmadığı için blok yanlış/boş dönebiliyordu.

    Yeni yaklaşım:
    - Başlığı (block_title_regex) bul
    - Başlığın altından itibaren, yıl sütunlarında en az bir sayı olan satırları al
    - Üst üste 2 satır boyunca hem item boş hem de tüm yıl hücreleri NaN ise bloğu bitir
    - "(in GW" gibi yeni alt-blok başlıklarına gelince de dur
    """
    if raw is None or raw.empty:
        return None

    c0 = raw.iloc[:, first_col_idx].astype(str)
    mask = c0.str.contains(block_title_regex, case=False, na=False, regex=True)
    if not mask.any():
        return None

    title_row = mask[mask].index[0]
    start = title_row + 1

    def _cell_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x).strip()
        return "" if s.lower() == "nan" else s

    empty_streak = 0
    end = start - 1

    for r in range(start, len(raw)):
        item = _cell_str(raw.iloc[r, first_col_idx])

        # Yeni bir alt-blok başlığına gelince (ör: "(in GW"), mevcut bloğu bitir.
        if item and re.search(r"\(in\s+gw", item, flags=re.IGNORECASE):
            break

        vals = raw.iloc[r, year_cols_idx]
        vals_num = pd.to_numeric(vals, errors="coerce")
        has_any = bool(np.isfinite(vals_num).any())

        is_blank_row = (item == "") and (not has_any)

        if is_blank_row:
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue

        empty_streak = 0

        # En az bir yıl değeri olan satırları bloğa dahil et.
        if item != "" or has_any:
            end = r

    if end < start:
        return None

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



def _interpolate_years_long(df: pd.DataFrame, start_year: int, end_year: int, value_col: str = "value") -> pd.DataFrame:
    """Fill intermediate *annual* years by linear interpolation within each series/group.

    Notes:
    - Expects a long dataframe with at least: year + value columns.
    - Grouping columns are inferred as all columns except year/value.
    """
    if df is None or df.empty:
        return df

    if "year" not in df.columns or value_col not in df.columns:
        return df

    out = df.copy()

    # Normalize dtypes
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    out = out.dropna(subset=["year"])
    if out.empty:
        return out

    out["year"] = out["year"].astype(int)

    group_cols = [c for c in out.columns if c not in {"year", value_col}]
    year_full = pd.Index(range(int(start_year), int(end_year) + 1), name="year")

    def _fill_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("year")
        s = g.set_index("year")[value_col]
        s = s.reindex(year_full)
        # Linear interpolation only inside the known range; keep edges as is (forward/backward).
        s = s.interpolate(method="linear", limit_direction="both")
        gg = s.reset_index().rename(columns={0: value_col})
        for c in group_cols:
            gg[c] = g.iloc[0][c]
        return gg

    if group_cols:
        # NOTE: keep grouping columns available inside _fill_one (pandas include_groups=True)
        filled = (
            out.groupby(group_cols, dropna=False, sort=False)
            .apply(_fill_one, include_groups=True)
            .reset_index(drop=True)
        )
    else:
        filled = _fill_one(out)

    # Keep column order stable
    cols = ["year"] + [c for c in out.columns if c not in {"year"}]
    filled = filled[cols]
    return filled


def _maybe_fill_years(df: pd.DataFrame, start_year: int, end_year: int, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    return _interpolate_years_long(df, start_year, end_year, value_col="value")

def _cagr(start_value: float, end_value: float, n_years: int) -> float:
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
# Display unit helpers
# -----------------------------
def _energy_unit_is_ktoe() -> bool:
    try:
        return str(globals().get("energy_unit", "GWh")).lower().startswith("bin")
    except Exception:
        return False


def _energy_unit_label() -> str:
    return "bin TEP (ktoe)" if _energy_unit_is_ktoe() else "GWh"


def _energy_value_format() -> str:
    # GWh is typically big integers; ktoe tends to benefit from 1 decimal.
    return ",.1f" if _energy_unit_is_ktoe() else ",.0f"


def _convert_energy_df(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Convert a df with energy values in GWh to display unit (optionally ktoe)."""
    if df is None or df.empty or value_col not in df.columns:
        return df
    if not _energy_unit_is_ktoe():
        return df
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce") * float(GWH_TO_KTOE)
    return out


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
            v_co2e = v_co2 / ratio if ratio and np.isfinite(ratio) else np.nan
            if not np.isfinite(v_co2e):
                continue
            records.append({"year": int(y), "sector": label, "value": float(v_co2e), "ratio_co2_over_co2e": ratio})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["year", "sector", "value", "series", "sheet", "ratio_co2_over_co2e"])

    df = df.sort_values(["year", "sector"])
    df["series"] = "Sektorel Enerji Emisyonlari (CO2e, ktn)"
    df["sheet"] = "Summary&Indicators"
    return df


def energy_share_assumption(year: int) -> float:
    if year <= 2025:
        return 0.70
    if year >= 2050:
        return 0.75
    return 0.70 + (year - 2025) * (0.75 - 0.70) / (2050 - 2025)


def lulucf_assumption_ktn(year: int) -> float:
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

    # Total Storage ve storage alt bileşenlerini tamamen çıkar
    df = df[~df["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))]
    df = df[df["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    natgas_rows = df[df["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"
    natgas_series = natgas_series[["year", "group", "value"]]

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

# -----------------------------
# Main (top) – File upload (professional, compact)
# -----------------------------
st.subheader("Senaryo dosyaları")
st.caption("Excel (.xlsx) • Çoklu senaryo • Maks. 12 dosya")

uploaded_files = st.file_uploader(
    "Excel dosyaları",
    type=["xlsx"],
    accept_multiple_files=True,
    help="Bir dosya = bir senaryo. Dosya adları senaryo adı olarak kullanılır.",
    label_visibility="collapsed",
)

st.divider()

with st.sidebar:
    st.header("Paneller")

    show_electric = st.checkbox("Elektrik", value=True)
    show_energy = st.checkbox("Enerji", value=True)
    show_emissions = st.checkbox("Sera Gazı Emisyonları", value=True)

    selected_panels = []
    if show_electric:
        selected_panels.append("Elektrik")
    if show_energy:
        selected_panels.append("Enerji")
    if show_emissions:
        selected_panels.append("Sera Gazı Emisyonları")
    st.divider()
    st.header("Ayarlar")

    # Year range slider (replaces start_year + max_year)
    year_min_default = 2018
    year_max_default = 2050
    YEAR_OPTIONS = [2018, 2020, 2025, 2030, 2035, 2040, 2045, 2050]

    # Preset buttons will drive the slider via session_state (no changes to calculations/plots)
    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (2025, 2050)

    year_range = st.select_slider(
        "Senaryo yıl aralığı",
        options=YEAR_OPTIONS,
        value=st.session_state["year_range"],
        key="year_range",
        help="Tüm grafikler bu yıl aralığına göre filtrelenir.",
    )


    # --- UI polish: smaller, stacked preset buttons (visual only) ---
    st.markdown(
        """
        <style>
        /* Make sidebar preset buttons compact */
        section[data-testid="stSidebar"] button[kind="secondary"] {
            padding: 0.25rem 0.5rem;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Hızlı seçim")

    # Preset buttons (safe via callbacks)
    def _preset_netzero():
        st.session_state["year_range"] = (2025, 2050)

    def _preset_tuep():
        st.session_state["year_range"] = (2025, 2035)

    st.button("Net Zero (2025–2050)", use_container_width=True, on_click=_preset_netzero)
    st.button("TUEP (2025–2035)", use_container_width=True, on_click=_preset_tuep)

    start_year, max_year = year_range


    # -----------------------------
    # Ara yılları doldurma (opsiyonel)
    # -----------------------------
    fill_years_enabled = st.checkbox(
        "Ara yılları doldur (tahmini)",
        value=False,
        help="Model çıktıları 5 yıllık zaman adımlarındadır. Okunabilirlik için ara yıllar lineer interpolasyonla tahmini doldurulur.",
    )
    fill_method = st.selectbox(
        "Doldurma yöntemi",
        options=["Lineer interpolasyon"],
        index=0,
        disabled=not fill_years_enabled,
    )
    MAX_YEAR = int(max_year)

    st.divider()
    st.header("Birim")
    energy_unit = st.select_slider(
        "Enerji birimi (GWh → bin TEP)",
        options=["GWh", "bin TEP (ktoe)"],
        value="GWh",
        help="Elektrik üretimi/tüketimi ve enerji talebi grafiklerinde birimi değiştirir. Model verisi GWh'dir.",
    )

    st.divider()
    st.header("Karşılaştırma")
    compare_mode = st.radio(
        label="",
        options=[
            "Küçük paneller (Ayrı Grafikler)",
            "Yan yana sütun — aynı yılda kıyas",
        ],
        index=0,
        help="Birden fazla senaryoyu farklı görünümlerle kıyaslayın. Okunabilirlik için çoğu durumda 'Küçük paneller' önerilir.",
    )

    stacked_value_mode = st.select_slider(
        "Stacked gösterim",
        options=["Mutlak", "Pay (%)"],
        value="Mutlak",
        help="Stacked grafiklerde mutlak değer yerine yıl içi pay (%) göstermek için Pay (%) seçin.",
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

if len(selected_scenarios) >= 4:
    st.warning("4+ senaryoda okunabilirlik için en fazla 3 senaryo gösterilecek.")
    selected_scenarios = selected_scenarios[:3]

if len(selected_scenarios) == 2:
    with st.sidebar:
        st.divider()
        st.caption("ℹ️ **2 Senaryo Fark Modu**, *Small multiples* dışındaki karşılaştırma modlarında çalışır.")
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
def compute_scenario_bundle(xlsx_file, scenario: str, start_year: int, max_year: int, interpolate_years: bool):
    blocks = read_power_generation(xlsx_file)
    balance = blocks["electricity_balance"]
    gross_gen = blocks["gross_generation"]
    installed_cap = blocks["installed_capacity"]

    pop = _filter_years(read_population_series(xlsx_file), start_year, max_year)
    gdp = _filter_years(read_gdp_series(xlsx_file), start_year, max_year)
    carbon_price = _filter_years(read_carbon_price_series(xlsx_file), start_year, max_year)
    co2 = _filter_years(read_co2_emissions_series(xlsx_file), start_year, max_year)

    energy_em_sector_co2e = _filter_years(read_energy_emissions_sectoral_co2e(xlsx_file), start_year, max_year)
    energy_em_total_co2e = pd.DataFrame(columns=["year", "value", "series"])
    if energy_em_sector_co2e is not None and not energy_em_sector_co2e.empty:
        energy_em_total_co2e = energy_em_sector_co2e.groupby("year", as_index=False)["value"].sum()
        energy_em_total_co2e["series"] = "Enerji Kaynakli Emisyonlar (CO2e, ktn)"

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


    def _add_scn_filled(df: pd.DataFrame) -> pd.DataFrame:
        df2 = _add_scn(df)
        return _maybe_fill_years(df2, start_year, max_year, interpolate_years)

    bundle = {
        "scenario": scenario,
        "pop": _add_scn_filled(pop),
        "gdp": _add_scn_filled(gdp),
        "carbon_price": _add_scn_filled(carbon_price),
        "co2": _add_scn_filled(co2),
        "energy_em_sector_co2e": _add_scn_filled(energy_em_sector_co2e),
        "energy_em_total_co2e": _add_scn_filled(energy_em_total_co2e),
        "co2_nz_stack": co2_nz_stack,
        "total_supply": _add_scn_filled(total_supply),
        "gen_mix": _add_scn_filled(gen_mix),
        "cap_total": _add_scn_filled(cap_total),
        "cap_mix": _add_scn_filled(cap_mix),
        "electricity_by_sector": _add_scn_filled(electricity_by_sector),
        "primary_energy_source": _add_scn_filled(primary_energy_source),
        "final_energy_source": _add_scn_filled(final_energy_source),
        "dependency_ratio": _add_scn_filled(dependency_ratio),
        "electrification_ratio": _add_scn_filled(electrification_ratio),
        "ye_both": _add_scn_filled(ye_both),
        "per_capita_el": _add_scn_filled(per_capita),
        "storage_ptx": _add_scn_filled(storage_ptx),
    }
    return bundle


bundles = []
for scn in selected_scenarios:
    f = scenario_to_file[scn]
    bundles.append(compute_scenario_bundle(f, scn, start_year, MAX_YEAR, fill_years_enabled))


def _concat(key: str):
    dfs = [b.get(key) for b in bundles if b.get(key) is not None and not b.get(key).empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


df_pop = _concat("pop")
df_gdp = _concat("gdp")
df_co2 = _concat("co2")
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
# Export: chart data to Excel
# -----------------------------
def _normalize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column order/types for export."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # common cleanups
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out = out.dropna(subset=["year"])
        out["year"] = out["year"].astype(int)
    for c in ["value"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _export_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            if df is None or df.empty:
                continue
            safe = re.sub(r"[^A-Za-z0-9_\- ]+", "", str(sheet))[:31] or "Sheet"
            df.to_excel(writer, sheet_name=safe, index=False)
    bio.seek(0)
    return bio.read()

def _build_export_workbook() -> bytes:
    """Create a single Excel containing the *displayed* chart datasets."""
    frames: dict[str, pd.DataFrame] = {}

    # Parameters / metadata
    frames["params"] = pd.DataFrame(
        {
            "key": [
                "start_year",
                "end_year",
                "energy_unit",
                "fill_years_enabled",
                "stacked_value_mode",
                "scenarios",
            ],
            "value": [
                int(start_year),
                int(MAX_YEAR),
                str(energy_unit),
                bool(fill_years_enabled),
                str(stacked_value_mode),
                ", ".join(list(selected_scenarios)),
            ],
        }
    )

    # Helpers for unit conversion (export what user is seeing on screen)
    def _maybe_energy(df: pd.DataFrame) -> pd.DataFrame:
        return _convert_energy_df(df, value_col="value")

    # Time series
    frames["population"] = _normalize_for_export(df_pop)
    frames["gdp"] = _normalize_for_export(df_gdp)
    frames["carbon_price"] = _normalize_for_export(df_cp)
    frames["co2"] = _normalize_for_export(df_co2)

    # Electricity
    frames["total_supply"] = _normalize_for_export(_maybe_energy(df_supply))
    frames["gen_mix"] = _normalize_for_export(_maybe_energy(df_genmix))
    frames["electricity_by_sector"] = _normalize_for_export(_maybe_energy(df_sector_el))

    # Capacity & storage / ptx (GW; no energy conversion)
    frames["capacity_mix"] = _normalize_for_export(df_capmix)
    frames["storage_ptx"] = _normalize_for_export(df_storage_ptx)

    # Energy
    frames["primary_energy_by_source"] = _normalize_for_export(_maybe_energy(df_primary))
    frames["final_energy_by_source"] = _normalize_for_export(_maybe_energy(df_final))
    frames["electrification_ratio"] = _normalize_for_export(df_electrification)

    # Emissions (some are stacked)
    frames["co2_nz_stack"] = _normalize_for_export(df_co2_nz_stack)

    # Optional bundles (if present later in the script)
    try:
        frames["ye_shares"] = _normalize_for_export(_concat("ye_both"))
    except Exception:
        pass
    try:
        frames["per_capita_electricity"] = _normalize_for_export(_concat("per_capita_el"))
    except Exception:
        pass

    # Drop empties
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    return _export_excel_bytes(frames)

# Download UI
with st.expander("📥 Grafik verilerini indir", expanded=False):
    st.caption("Seçili senaryolar ve yıl aralığına göre ekranda gördüğünüz veriler tek bir Excel dosyası olarak indirilir.")
    xlsx_bytes = _build_export_workbook()
    st.download_button(
        label="Excel indir (.xlsx)",
        data=xlsx_bytes,
        file_name=f"dashboard_chart_data_{int(start_year)}-{int(MAX_YEAR)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# -----------------------------
# Line charts (single axis, scenario colors)
# -----------------------------
def _line_chart(df, title: str, y_title: str, value_format: str = ",.2f", chart_style: str | None = None):
    if df is None or df.empty:
        st.subheader(title)
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce")
    dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
    dfp = dfp.dropna(subset=["year", "value", "scenario"])
    dfp["year"] = dfp["year"].astype(int)

    diff_on = bool(globals().get("diff_mode_enabled", False)) and (globals().get("compare_mode") != "Küçük paneller (Ayrı Grafikler)") and (globals().get("compare_mode") != "Küçük paneller (Ayrı Grafikler)")
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

    # --- Y-axis domain override for selected KPI time-series ---
    # For these indicators, starting the Y-axis at 0 reduces readability; we start near the series minimum instead.
    _NONZERO_AXIS_KEYS = [
        "Türkiye Nüfus Gelişimi",
        "GSYH (Milyar ABD Doları",
        "Kişi Başına Elektrik Tüketimi",
        "Nihai Enerjide Elektrifikasyon Oranı",
        "CO2 Emisyonları (ktn CO2)",
    ]
    _use_nonzero_axis = any(k in str(title) for k in _NONZERO_AXIS_KEYS)

    # --- Y ekseni: bazı tek-seri metriklerde 0'dan başlamak okunabilirliği düşürüyor.
    # Bu metriklerde ekseni serinin minimumuna yakın bir yerden başlatıyoruz ve
    # ayrıca ilk (minimum) değeri eksen üzerinde bir tick olarak özellikle gösteriyoruz.
    _use_nonzero_axis = any(
        k in str(title).lower()
        for k in [
            "türkiye nüfus gelişimi",
            "nüfus",
            "gsyh",
            "kişi başına elektrik tüketimi",
            "kişi basina elektrik tüketimi",
            "nihai enerjide elektrifikasyon oranı",
            "nihai enerjide elektrifikasyon orani",
            "co2 emisyonları",
            "co2 emisyonlari",
        ]
    )

    y_scale = None
    y_axis = None
    if _use_nonzero_axis:
        y_min = float(dfp["value"].min())
        y_max = float(dfp["value"].max())
        span = y_max - y_min

        if np.isfinite(span) and span > 0:
            pad = span * 0.03
        else:
            pad = max(abs(y_min) * 0.05, 1.0)

        lo = y_min - pad
        hi = y_max + pad

        # Yüzde serilerinde (özellikle elektrifikasyon) 0–100 bandı mantıklı
        if "%" in str(y_title):
            lo = max(0.0, lo)
            hi = min(100.0, hi)

        y_scale = alt.Scale(domain=[lo, hi], zero=False)

        # Tick değerleri: min değeri mutlaka göster + birkaç "nice" ara tick
        tick_count = 6
        ticks = []
        if np.isfinite(y_min):
            ticks.append(float(y_min))

        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            lin = np.linspace(lo, hi, tick_count)
            ticks.extend([float(v) for v in lin])

        # unique + sorted (float hassasiyeti için yuvarla)
        ticks = sorted({round(t, 6) for t in ticks})

        # Çok fazla tick olursa sadeleştir
        if len(ticks) > 10 and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ticks = [round(float(v), 6) for v in np.linspace(lo, hi, 7)]
            ticks = sorted({round(t, 6) for t in ticks})

        y_axis = alt.Axis(values=ticks, format=value_format)

    def _y_enc():
        if y_scale is not None:
            return alt.Y("value:Q", title=y_title, scale=y_scale, axis=y_axis)
        return alt.Y("value:Q", title=y_title)
    style = "Çizgi"  # zaman serilerinde bar kapatıldı

    base = alt.Chart(dfp).encode(
        color=alt.Color("scenario:N", title="Senaryo", legend=alt.Legend(orient='right', direction='vertical', labelLimit=180, titleLimit=0)),
        tooltip=[
            alt.Tooltip("scenario:N", title="Senaryo"),
            alt.Tooltip("year:O", title="Yıl"),
            alt.Tooltip("value:Q", title=y_title, format=value_format),
        ],
    )
    if style == "Çizgi":
        hover = alt.selection_point(
            fields=["scenario"],
            on="mouseover",
            nearest=True,
            clear="mouseout",
            empty="all",
        )

        base_h = base.add_params(hover)

        lines = base_h.mark_line(interpolate="monotone").encode(
            x=alt.X(
                "year:Q",
                title="Yıl",
                scale=alt.Scale(domain=[min(year_vals), max(year_vals)]),
                axis=alt.Axis(values=year_vals, format="d", labelAngle=0),
            ),
            y=_y_enc(),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.25)),
            strokeWidth=alt.condition(hover, alt.value(3), alt.value(2)),
        )

        points = base_h.mark_circle(size=70).encode(
            x=alt.X(
                "year:Q",
                title="Yıl",
                scale=alt.Scale(domain=[min(year_vals), max(year_vals)]),
                axis=alt.Axis(values=year_vals, format="d", labelAngle=0),
            ),
            y=_y_enc(),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.0)),
        ).transform_filter(hover)

        chart = (lines + points).configure_axis(grid=True, gridOpacity=0.15)

    elif style == "Bar (Stack)":
        chart = base.mark_bar().encode(
            x=alt.X("year:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0)),
            y=alt.Y("value:Q", title=y_title, stack="zero"),
        )

    else:
        chart = base.mark_bar().encode(
            x=alt.X("year:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0)),
            xOffset=alt.XOffset("scenario:N"),
            y=_y_enc(),
        )

    st.altair_chart(chart.properties(height=320), use_container_width=True)



# -----------------------------
# Donut charts (KPI mini-pies)
# -----------------------------

def _donut_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str, value_format: str = ",.0f"):
    """KPI donut (uyumlu/sabit).

    Not: Bazı Streamlit/Altair sürümlerinde padAngle/cornerRadius/radius gibi
    gelişmiş özellikler chart'ı boş gösterebiliyor. Bu sürüm en uyumlu şekilde:
    - 4 kategori (Fossil fuels / Renewables / Nuclear / Other)
    - Sabit renkler + sabit legend sırası
    - Dilime tıklayınca büyür
    """
    if df is None or df.empty:
        st.caption(f"{title}: veri yok")
        return

    d = df.copy()
    d[category_col] = d[category_col].astype(str)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[category_col, value_col])
    d = d[d[value_col] != 0]
    if d.empty:
        st.caption(f"{title}: veri yok")
        return

    total = float(d[value_col].sum())
    if not np.isfinite(total) or total == 0:
        st.caption(f"{title}: veri yok")
        return

    DONUT_DOMAIN = ["Fossil fuels", "Renewables", "Nuclear", "Other"]
    DONUT_RANGE = ["#F39C12", "#2ECC71", "#9B59B6", "#95A5A6"]

    sel = alt.selection_point(fields=[category_col], on="click", empty="all")

    base = (
        alt.Chart(d)
        .add_params(sel)
        .encode(
            theta=alt.Theta(f"{value_col}:Q", stack=True),
            color=alt.Color(
                f"{category_col}:N",
                sort=DONUT_DOMAIN,
                scale=alt.Scale(domain=DONUT_DOMAIN, range=DONUT_RANGE),
                legend=alt.Legend(
                    orient="right",
                    direction="vertical",
                    columns=1,
                    labelLimit=180,
                    titleLimit=0,
                    symbolType="circle",
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title="Kategori"),
                alt.Tooltip(f"{value_col}:Q", title="Değer", format=value_format),
                alt.Tooltip("pct:Q", title="Pay (%)", format=".1f"),
            ],
        )
    )

    arcs = base.mark_arc(innerRadius=62, outerRadius=98)
    arcs_hi = base.transform_filter(sel).mark_arc(innerRadius=60, outerRadius=112)

    st.caption(title)
    st.altair_chart((arcs + arcs_hi).properties(height=260), use_container_width=True)

    # NOTE: Bazı Streamlit/Altair sürümlerinde donut'un üzerine yazı (mark_text + radius)
    # chart'ı tamamen boş gösterebiliyor. Bu yüzden yüzdeleri "donut üstüne" değil,
    # grafiğin hemen altında kompakt şekilde veriyoruz (bozulma riski yok).
    if "pct" in d.columns:
        order = ["Fossil fuels", "Renewables", "Nuclear", "Other"]
        dd = d.copy()
        dd[category_col] = dd[category_col].astype(str)
        dd["_cat"] = pd.Categorical(dd[category_col], categories=order, ordered=True)
        dd = dd.sort_values("_cat")
        parts = [f"{row[category_col]}: {row['pct']:.0f}%" for _, row in dd.iterrows()]
        if parts:
            st.caption("Pay (%) — " + " • ".join(parts))


# -----------------------------
# KPI row (per scenario)
# -----------------------------
st.subheader("Özet Bilgi Kartları (Seçili Senaryolar)")
ncols = _ncols_for_selected(len(selected_scenarios))
cols = st.columns(ncols)



def _kpi_gen_bucket(cat: str) -> str:
    """KPI donut için 4'lü sınıflama."""
    s = (cat or "").strip().lower()

    # Fossil fuels
    if ("coal" in s and "lignite" not in s) or ("lignite" in s) or ("natural gas" in s) or (s == "gas"):
        return "Fossil fuels"

    # Renewables
    if "solar" in s or "wind" in s or "hydro" in s:
        return "Renewables"

    # Nuclear
    if "nuclear" in s:
        return "Nuclear"

    return "Other"



def _kpi_for_bundle(b):
    scn = b["scenario"]
    supply = b["total_supply"]
    ye = b["ye_both"]
    gen_mix = b.get("gen_mix")
    cap_mix = b.get("cap_mix")
    cap_total = b["cap_total"]
    gdp = b["gdp"]

    available_max_year = int(supply["year"].max()) if supply is not None and not supply.empty else None
    latest_year = int(min(int(MAX_YEAR), available_max_year)) if available_max_year is not None else None
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



    # KPI donut (elektrik üretimi dağılımı) — seçili yıl
    donut_y = globals().get("donut_year", None)
    try:
        donut_y = int(donut_y) if donut_y is not None else None
    except Exception:
        donut_y = None
    if donut_y is None:
        donut_y = latest_year

    pie_gen = pd.DataFrame(columns=["category", "value"])
    if donut_y and gen_mix is not None and (not gen_mix.empty):
        gm = gen_mix.copy()
        # gen_mix kolon adı bazen group olabilir
        if "group" in gm.columns and "category" not in gm.columns:
            gm = gm.rename(columns={"group": "category"})
        gm["year"] = pd.to_numeric(gm["year"], errors="coerce")
        gm["value"] = pd.to_numeric(gm["value"], errors="coerce")
        gm = gm.dropna(subset=["year", "category", "value"])
        gm["year"] = gm["year"].astype(int)

        # Eğer donut_y dosyada yoksa, en yakın (maks) yılı kullan
        if int(donut_y) not in set(gm["year"].unique().tolist()):
            donut_y = int(gm["year"].max())

        gm = gm[gm["year"] == int(donut_y)]
        if _energy_unit_is_ktoe():
            gm["value"] = gm["value"] * float(GWH_TO_KTOE)

        if not gm.empty:
            gm["bucket"] = gm["category"].apply(_kpi_gen_bucket)
            pie_gen = gm.groupby("bucket", as_index=False)["value"].sum().rename(columns={"bucket": "category"})
            # Sabit sırayla göster
            order = ["Fossil fuels", "Renewables", "Nuclear", "Other"]
            pie_gen["category"] = pd.Categorical(pie_gen["category"], categories=order, ordered=True)
            pie_gen = pie_gen.sort_values("category")

            # Donut için yüzde (Pay %) hesapla (gösterim için; hesaplamaları etkilemez)
            _tot = float(pd.to_numeric(pie_gen["value"], errors="coerce").sum())
            if np.isfinite(_tot) and _tot != 0:
                pie_gen["pct"] = (pd.to_numeric(pie_gen["value"], errors="coerce") / _tot) * 100.0
            else:
                pie_gen["pct"] = np.nan

            # Yüzde (pay) değerini donut üstüne yazmak bazı ortamlarda grafiği boş gösterebildiği için
            # yüzdeyi tooltip + grafiğin altında kompakt metin olarak gösteriyoruz.

    return {
        "scenario": scn,
        "latest_year": latest_year,
        "total_supply": latest_total,
        "ye_total": latest_ye_total,
        "ye_int": latest_ye_int,
        "cap_total": latest_cap,
        "gdp_cagr": gdp_cagr,
        "donut_year": donut_y,
        "pie_gen": pie_gen,
        
    }


kpis = [_kpi_for_bundle(b) for b in bundles]


for i, kpi in enumerate(kpis[:ncols]):
    with cols[i]:
        st.markdown(f"**{kpi['scenario']}**")
        st.metric("GSYH CAGR (%)", f"{kpi['gdp_cagr']*100:.2f}%" if np.isfinite(kpi["gdp_cagr"]) else "—")

        supply_display = (kpi["total_supply"] * GWH_TO_KTOE) if (_energy_unit_is_ktoe() and np.isfinite(kpi["total_supply"])) else kpi["total_supply"]
        st.metric(
            f"Toplam Arz ({_energy_unit_label()}) – {kpi['latest_year'] or ''}",
            f"{supply_display:{_energy_value_format()}}" if np.isfinite(supply_display) else "—",
        )

        st.metric("YE Payı (%)", f"{kpi['ye_total']:.1f}% / {kpi['ye_int']:.1f}%" if np.isfinite(kpi["ye_total"]) else "—")
        st.metric("Elektrik Kurulu Gücü (GW)", f"{kpi['cap_total']:,.3f}" if np.isfinite(kpi["cap_total"]) else "—")

        # Mini dağılım grafiği (donut) — dilime tıklayınca büyür
        _donut_chart(
            kpi.get("pie_gen"),
            category_col="category",
            value_col="value",
            title=f"Elektrik üretimi dağılımı ({kpi.get('donut_year')}) — {_energy_unit_label()}",
            value_format=_energy_value_format(),
        )

if len(kpis) > ncols:
    with st.expander("Diğer seçili senaryoların KPI’ları"):
        for kpi in kpis[ncols:]:
            supply_display = (kpi["total_supply"] * GWH_TO_KTOE) if (_energy_unit_is_ktoe() and np.isfinite(kpi["total_supply"])) else kpi["total_supply"]
            st.markdown(
                f"**{kpi['scenario']}** — "
                f"GSYH CAGR: {(kpi['gdp_cagr']*100):.2f}% | "
                f"Toplam Arz: {supply_display:{_energy_value_format()}} {_energy_unit_label()} | "
                f"YE Payı: {kpi['ye_total']:.1f}%/{kpi['ye_int']:.1f}% | "
                f"KG: {kpi['cap_total']:,.3f} GW"
            )


st.divider()

# -----------------------------
# Stacked charts helpers (with legend-click filter)
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


def _legend_filter_params(stack_field: str, sel_name: str | None = None):
    # Legend tıklamasıyla seçili kategori "tek başına sıfırdan" görünsün diye
    # selection ile filtreliyoruz; empty='all' sayesinde seçim yokken tüm seriler görünür.
    # clear="dblclick": legend üzerinde çift tıkla seçim temizlenir.
    if sel_name is None:
        sel_name = f"legend_sel_{abs(hash(stack_field))}"
    sel = alt.selection_point(
        fields=[stack_field],
        bind="legend",
        name=sel_name,
        clear="dblclick",
        empty="all",
    )
    return sel, sel_name


def _stacked_small_multiples(df, title: str, x_field: str, stack_field: str, y_title: str, category_title: str, value_format: str, order=None, is_percent: bool = False, color_map=None):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = dfp["year"].astype(int)
    year_vals = sorted(pd.to_numeric(dfp[x_field], errors="coerce").dropna().astype(int).unique().tolist())

    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["scenario", "year", stack_field])

    # --- FIX: Stacked bar y-domain'i (toplamın altında kalmasın) ---
    if is_percent:
        yscale = alt.Scale(domain=[0, 100])
    else:
        dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")

        # Pozitif üst sınır: yıl bazında toplam pozitif katkı (stack toplamı)
        pos_tot = (
            dfp.assign(_v=np.where(dfp["value"] > 0, dfp["value"], 0.0))
            .groupby(["scenario", x_field], as_index=False)["_v"]
            .sum()["_v"]
        )
        ymax = float(pos_tot.max()) if len(pos_tot) else None

        # Negatif alt sınır: yıl bazında toplam negatif katkı (LULUCF gibi)
        neg_tot = (
            dfp.assign(_v=np.where(dfp["value"] < 0, dfp["value"], 0.0))
            .groupby(["scenario", x_field], as_index=False)["_v"]
            .sum()["_v"]
        )
        ymin = float(neg_tot.min()) if len(neg_tot) else 0.0

        if ymax is not None and np.isfinite(ymax) and np.isfinite(ymin):
            # Daha fazla headroom: toplamın üstünde kesilmesin
            span = (ymax - ymin) if (ymax - ymin) > 0 else max(abs(ymax), 1.0)
            pad = 0.10 * span
            yscale = alt.Scale(domain=[ymin - pad, ymax + pad])
        else:
            yscale = alt.Undefined

    n = len(selected_scenarios)
    ncols = _ncols_for_selected(n)
    cols = st.columns(ncols)

    for idx, scn in enumerate(selected_scenarios):
        sub = dfp[dfp["scenario"] == scn]
        if sub.empty:
            continue

        safe_sf = re.sub(r"[^0-9a-zA-Z_]+", "_", str(stack_field))
        sel, sel_name = _legend_filter_params(stack_field, sel_name=f"legend_{safe_sf}_{idx}")

        # Legend seçimi: Legend’e tıklayınca sadece seçili kategori kalsın (stack tabandan başlasın).
        # empty='all' sayesinde hiçbir seçim yokken tüm seriler görünür; çift tık (dblclick) ile sıfırlanır.
        bars_src = alt.Chart(sub).add_params(sel).transform_filter(sel)
        if not is_percent:
            bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=[x_field])

        bars = (
            bars_src.mark_bar()
            .encode(
                x=alt.X(f"{x_field}:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0, labelPadding=14, titlePadding=10)),
                y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
color=_source_color_encoding(sub, stack_field, category_title, order=order, color_map=color_map),
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
            st.altair_chart(bars.properties(height=380, padding={"bottom": 28}), use_container_width=True)


def _stacked_clustered(df, title: str, x_field: str, stack_field: str, y_title: str, category_title: str, value_format: str, order=None, is_percent: bool = False, color_map=None):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return

    dfp = df.copy()
    dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce")
    dfp = dfp.dropna(subset=["year"])
    dfp["year"] = dfp["year"].astype(int)
    year_vals = sorted(pd.to_numeric(dfp[x_field], errors="coerce").dropna().astype(int).unique().tolist())
    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["year", "scenario", stack_field])

    if is_percent:
        yscale = alt.Scale(domain=[0, 100])
    else:
        # Stacked bar'larda y ekseni, tekil parçaların max'ına değil yıl bazında TOPLAM'a göre ayarlanmalı.
        dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
        # Pozitif ve negatif yığınları ayrı topla (negatifler aşağı doğru)
        pos_tot = (
            dfp.assign(_v=np.where(dfp["value"] > 0, dfp["value"], 0.0))
            .groupby(["scenario", x_field], as_index=False)["_v"]
            .sum()["_v"]
        )
        neg_tot = (
            dfp.assign(_v=np.where(dfp["value"] < 0, dfp["value"], 0.0))
            .groupby(["scenario", x_field], as_index=False)["_v"]
            .sum()["_v"]
        )
        ymax = float(pos_tot.max()) if len(pos_tot) else None
        ymin = float(neg_tot.min()) if len(neg_tot) else 0.0

        if ymax is not None and np.isfinite(ymax) and np.isfinite(ymin):
            # Daha fazla headroom: toplamın üstünde kesilmesin (özellikle 110-120 gibi sınırda)
            span = (ymax - ymin) if (ymax - ymin) > 0 else max(abs(ymax), 1.0)
            pad = 0.10 * span
            yscale = alt.Scale(domain=[ymin - pad, ymax + pad])
        else:
            yscale = alt.Undefined

    sel, _ = _legend_filter_params(stack_field)
    bars_src = alt.Chart(dfp)
        if not is_percent:
            bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=["scenario", x_field])

    # 1) Base layer: normal stacked bars, but non-selected categories fade when a selection exists.
    base = (
        bars_src.mark_bar()
        .encode(
            x=alt.X(f"{x_field}:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0, labelPadding=14, titlePadding=10)),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
            color=_source_color_encoding(dfp, stack_field, category_title, order=order, color_map=color_map),
            opacity=alt.condition(sel, alt.value(0.18), alt.value(1.0)),
            tooltip=[
                alt.Tooltip("scenario:N", title="Senaryo"),
                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                alt.Tooltip(f"{stack_field}:N", title=category_title),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
                *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
            ],
        )
    )

    # 2) Overlay layer: re-stacked selected category only => starts from zero (no 'floating' segment).
    focus = (
        bars_src.transform_filter(sel)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_field}:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals, labelAngle=0, labelPadding=14, titlePadding=10)),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
            color=_source_color_encoding(dfp, stack_field, category_title, order=order, color_map=color_map),
            tooltip=[
                alt.Tooltip("scenario:N", title="Senaryo"),
                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                alt.Tooltip(f"{stack_field}:N", title=category_title),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
                *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
            ],
        )
    )

    chart = (base + focus).add_params(sel)
    st.altair_chart(chart.properties(height=420, padding={"bottom": 28}), use_container_width=True)


def _stacked_snapshot(df, title: str, x_field: str, stack_field: str, y_title: str, category_title: str, value_format: str, years=(2035, 2050), order=None, is_percent: bool = False, color_map=None):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("Veri bulunamadı.")
        return
    dfp = df.copy()
    dfp["year"] = pd.to_numeric(dfp["year"], errors="coerce")
    dfp = dfp.dropna(subset=["year"])
    dfp["year"] = dfp["year"].astype(int)
    dfp = dfp[dfp["year"].isin(list(years))]
    if dfp.empty:
        st.warning("Seçilen yıllar için veri yok (seçili snapshot yılları).")
        return

    if order is not None:
        dfp[stack_field] = pd.Categorical(dfp[stack_field], categories=order, ordered=True)
        dfp = dfp.sort_values(["year", "scenario", stack_field])

    yscale = alt.Scale(domain=[0, 100]) if is_percent else alt.Undefined

    sel, _ = _legend_filter_params(stack_field)
bars_src = alt.Chart(dfp)
    if not is_percent:
        bars_src = bars_src.transform_joinaggregate(total="sum(value)", groupby=["scenario", x_field])

    bars = (
        bars_src.transform_filter(sel).mark_bar()
        .encode(
            x=alt.X(f"{x_field}:O", title="Yıl"),
            xOffset=alt.XOffset("scenario:N"),
            y=alt.Y("value:Q", title=y_title, stack=True, scale=yscale),
            color=_source_color_encoding(dfp, stack_field, category_title, order=order, color_map=color_map),
            tooltip=[
                alt.Tooltip("scenario:N", title="Senaryo"),
                alt.Tooltip(f"{x_field}:O", title="Yıl"),
                alt.Tooltip(f"{stack_field}:N", title=category_title),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
                *([] if is_percent else [alt.Tooltip("total:Q", title="Total", format=value_format)]),
            ],
        )
        .add_params(sel)
    )
    st.altair_chart(bars.properties(height=420), use_container_width=True)


def _render_stacked(df, title, x_field, stack_field, y_title, category_title, value_format, order=None, color_map=None):
    df_use = df
    y_title_use = y_title
    value_format_use = value_format
    is_percent = False

    diff_on = bool(globals().get("diff_mode_enabled", False)) and (globals().get("compare_mode") != "Küçük paneller (Ayrı Grafikler)") and (globals().get("compare_mode") != "Küçük paneller (Ayrı Grafikler)")
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
    def _render_main():
        if compare_mode == "Küçük paneller (Ayrı Grafikler)":
            _stacked_small_multiples(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, order=order, is_percent=is_percent, color_map=color_map)
        else:
            _stacked_clustered(df_use, title_use, x_field, stack_field, y_title_use, category_title, value_format_use, order=order, is_percent=is_percent)
    _render_main()
# =========================
# Waterfall helpers (unchanged)
# =========================
def prepare_yearly_transition_waterfall(df_mix: pd.DataFrame, scenario: str, start_year: int, end_year: int, value_col: str = "value", group_col: str = "category") -> pd.DataFrame:
    if df_mix is None or df_mix.empty:
        return pd.DataFrame()
    df = df_mix.copy()
    if "scenario" not in df.columns:
        return pd.DataFrame()

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
    d["color"] = np.where(d["step"] == "Net Değişim", "Net", np.where(d["delta"] >= 0, "Artış", "Azalış"))

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
# PANELS
# -----------------------------
# ELECTRICITY PANEL
if "Elektrik" in selected_panels:
    st.markdown("## Elektrik")

    _line_chart(df_pop, "Türkiye Nüfus Gelişimi", "Nüfus (milyon)", value_format=",.3f")
    _line_chart(df_gdp, "GSYH (Milyar ABD Doları, 2015 fiyatlarıyla)", "Milyar ABD Doları (2015)", value_format=",.2f")

    st.divider()

    df_pc = _concat("per_capita_el")
    _line_chart(df_pc, "Kişi Başına Elektrik Tüketimi (kWh/kişi)", "kWh/kişi", value_format=",.0f")
    _line_chart(df_electrification, "Nihai Enerjide Elektrifikasyon Oranı (%)", "%", value_format=",.1f")

    st.divider()

    order_gen = ["Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables", "Natural gas", "Coal", "Lignite", "Nuclear", "Other"]
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

    _render_stacked(
        _convert_energy_df(df_sector_el).rename(columns={"sector": "category"}),
        title=f"Sektörlere Göre Elektrik Tüketimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Sektör",
        value_format=_energy_value_format(),
        order=["Energy Branch & Other Uses", "Industry", "Residential", "Tertiary", "Transport"],
        color_map=SECTOR_COLOR_MAP,
    )

    st.divider()

    if stacked_value_mode != "Pay (%)":
        st.markdown("### Yakıt/Teknoloji Bazlı Enerji Dönüşümü (Δ)")
        st.caption(
            "Seçili senaryoda başlangıç ve bitiş yılları arasındaki elektrik üretimi ve kurulu güç değişimlerini (Δ) gösterir."
        )

        if len(selected_scenarios) == 1:
            scn_tr = selected_scenarios[0]
        else:
            scn_tr = st.selectbox("Dönüşüm analizi için senaryo seçin", options=selected_scenarios, index=0, key="transition_scn_select")

        gen_for_wf = df_genmix.rename(columns={"group": "category"}).copy()
        cap_for_wf = df_capmix.rename(columns={"group": "category"}).copy()

        wf_gen = prepare_yearly_transition_waterfall(gen_for_wf, scenario=scn_tr, start_year=int(start_year), end_year=int(MAX_YEAR), value_col="value", group_col="category")
        wf_cap = prepare_yearly_transition_waterfall(cap_for_wf, scenario=scn_tr, start_year=int(start_year), end_year=int(MAX_YEAR), value_col="value", group_col="category")

        wf_gen = _convert_energy_df(wf_gen, value_col="delta")
        wf_gen = _convert_energy_df(wf_gen, value_col="y0")
        wf_gen = _convert_energy_df(wf_gen, value_col="y1")

        colA, colB = st.columns(2)
        with colA:
            render_waterfall(wf_gen, title=f"Elektrik Üretimi Dönüşümü ({_energy_unit_label()}) — {start_year} → {MAX_YEAR}", y_title=_energy_unit_label())
        with colB:
            render_waterfall(wf_cap, title=f"Kurulu Güç Dönüşümü (GW) — {start_year} → {MAX_YEAR}", y_title="GW")

    st.divider()

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

# ENERGY PANEL
if "Enerji" in selected_panels:
    st.markdown("## Enerji")

    _render_stacked(
        _convert_energy_df(df_primary).rename(columns={"source": "category"}),
        title=f"Birincil Enerji Talebi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Kaynak",
        value_format=_energy_value_format(),
    )

    st.divider()

    _render_stacked(
        _convert_energy_df(df_final).rename(columns={"source": "category"}),
        title=f"Kaynaklarına Göre Nihai Enerji Tüketimi ({_energy_unit_label()})",
        x_field="year",
        stack_field="category",
        y_title=_energy_unit_label(),
        category_title="Kaynak",
        value_format=_energy_value_format(),
    )

    st.divider()

# EMISSIONS PANEL
if "Sera Gazı Emisyonları" in selected_panels:
    st.markdown("## Sera Gazı Emisyonları")

    _line_chart(df_co2, "CO2 Emisyonları (ktn CO2)", "ktn CO2", value_format=",.0f")
    _line_chart(df_cp, "Karbon Fiyatı (Varsayım) -$", "ABD Doları (2015) / tCO₂", value_format=",.2f")

    st.divider()

    st.markdown("### Türkiye Seragazı Emisyonları — Net Zero Hedefi Takibi (CO₂e)")
    st.caption(
        "Enerji dışı emisyonlar/SGE ve LULUCF değerleri varsayımsaldır. "
        "CO₂→CO₂e dönüşümü (2023 CRF varsayımı): Elektrik 0.99, Ulaştırma 0.94, Sanayi 0.97, Tarım 0.01, Diğer 0.99 (CO₂/CO₂e)."
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
        color_map=EMISSION_COMPONENT_COLOR_MAP,
    )

    st.divider()

with st.expander("Çalıştırma"):
    st.code("pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py", language="bash")



# ============================================================
# Plotly Bar Chart Race – Elektrik Üretimi (Kaynaklara Göre)
# Canva-benzeri zamanla akan grafik
# ============================================================

import plotly.express as px

def _prepare_genmix_for_plotly(df, scenario):
    d = df.copy()
    d = d[d["scenario"] == scenario]
    d = d.rename(columns={"group": "source"})
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["year", "source", "value"])
    return d

def _plot_generation_bar_race(df, unit_label):
    max_val = df["value"].max()

    fig = px.bar(
        df,
        x="value",
        y="source",
        orientation="h",
        animation_frame="year",
        animation_group="source",
        range_x=[0, max_val * 1.1],
        color="source",
        labels={
            "value": unit_label,
            "source": "Kaynak",
            "year": "Yıl",
        },
    )

    fig.update_layout(
        height=520,
        showlegend=False,
        title="Elektrik Üretimi – Kaynaklara Göre Dağılım (Zamana Göre)",
        transition={"duration": 600, "easing": "cubic-in-out"},
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )

    fig.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(color="black", size=13),
        cliponaxis=False,
    )

    return fig


# -----------------------------
# Plotly panel (render)
# -----------------------------
st.divider()
st.subheader(" Elektrik Üretimi – Kaynaklara Göre Zaman İçinde Değişim ")

for scn in selected_scenarios:
    st.markdown(f"**Senaryo: {scn}**")
    d_plotly = _prepare_genmix_for_plotly(df_genmix, scn)

    if d_plotly.empty:
        st.warning("Bu senaryo için veri bulunamadı.")
    else:
        fig = _plot_generation_bar_race(d_plotly, _energy_unit_label())
        st.plotly_chart(fig, use_container_width=True)
