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

# --- GDP RULE (Excel 1-indexed row in Scenario assumption sheet) ---
GDP_ROW_1IDX = 6  # Scenario_Assumptions sekmesinde 6. satır (GDP)
POP_ROW_1IDX = 5  # Scenario_Assumptions sekmesinde 5. satır (Nüfus)
SCENARIO_ASSUMP_YEARS_ROW_1IDX = 3  # Scenario_Assumptions sekmesinde 3. satır (Yıllar)

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
    "Other Renewables": [r"\bgeothermal\b", r"\bbiomass\b", r"\bbiogas\b", r"\bwaste\b", r"\bwave\b", r"\btidal\b"],
}

RENEWABLE_GROUPS = {"Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables"}
INTERMITTENT_RE_GROUPS = {"Wind (RES)", "Solar (GES)"}


# -----------------------------
# Helpers
# -----------------------------
def _as_int_year(x):
    try:
        v = int(float(x))
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
# Reading: Scenario assumption (GDP)
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_scenario_assumptions_row(xlsx_file, value_row_1idx: int, series_name: str) -> pd.DataFrame:
    """
    Scenario_Assumptions sekmesinden tek bir satırı seri olarak okur.

    - Yıllar: 3. satır (SCENARIO_ASSUMP_YEARS_ROW_1IDX)
    - Değerler: value_row_1idx (Excel 1-indexed)
    """
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

    # years from fixed row
    year_r0 = SCENARIO_ASSUMP_YEARS_ROW_1IDX - 1
    if year_r0 < 0 or year_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series"])

    years, year_cols_idx = _extract_years(raw, year_r0)
    if not years:
        return pd.DataFrame(columns=["year", "value", "series"])

    # values from fixed row
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
    """
    Belirli bir sekmeden sabit satır/yıl satırı ile seri okur.
    - Yıllar: years_row_1idx (Excel 1-indexed)
    - Değerler: value_row_1idx (Excel 1-indexed)
    """
    try:
        raw = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    yr_r0 = years_row_1idx - 1
    val_r0 = value_row_1idx - 1
    if yr_r0 < 0 or val_r0 < 0 or yr_r0 >= len(raw) or val_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    # years are detected by scanning the whole row
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
    """PowerGeneration-Indicators: CO2 Emissions (in ktn CO2) – 30. satır."""
    return _read_fixed_row_sheet(
        xlsx_file,
        sheet_name="PowerGeneration-Indicators",
        years_row_1idx=3,
        value_row_1idx=30,
        series_name="CO2 Emissions (ktn CO2)",
    )


def read_primary_energy_production_series(xlsx_file) -> pd.DataFrame:
    """Summary&Indicators: Primary Energy Production (in GWh) – C14 satırından yatay seri."""
    try:
        raw = pd.read_excel(xlsx_file, sheet_name="Summary&Indicators", header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    YEARS_ROW_1IDX = 3   # years are in row 3
    VALUE_ROW_1IDX = 14  # values start at C14 row
    START_COL_IDX = 2    # column C

    yr_r0 = YEARS_ROW_1IDX - 1
    val_r0 = VALUE_ROW_1IDX - 1

    if yr_r0 < 0 or val_r0 < 0 or yr_r0 >= len(raw) or val_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series", "sheet"])

    years_row = raw.iloc[yr_r0, START_COL_IDX:].tolist()
    vals_row = raw.iloc[val_r0, START_COL_IDX:].tolist()

    out_years, out_vals = [], []
    for y_cell, v_cell in zip(years_row, vals_row):
        y = _as_int_year(y_cell)
        if y is None:
            continue
        y = int(y)
        if y <= MAX_YEAR:
            v = pd.to_numeric(v_cell, errors="coerce")
            if not pd.isna(v):
                out_years.append(y)
                out_vals.append(float(v))

    df = pd.DataFrame({"year": out_years, "value": out_vals})
    df = df.dropna(subset=["value"]).sort_values("year")
    df["series"] = "Primary Energy Production (GWh)"
    df["sheet"] = "Summary&Indicators"
    return df


def read_primary_energy_consumption_by_source(xlsx_file) -> pd.DataFrame:
    """Summary&Indicators: Gross Inland Consumption – satır 15-23 (yığılmış)."""
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
    df["series"] = "Gross Inland Consumption"
    df["sheet"] = "Summary&Indicators"
    return df

def read_gdp_series(xlsx_file) -> pd.DataFrame:
    """GDP serisi: Scenario_Assumptions 6. satır."""
    return _read_scenario_assumptions_row(xlsx_file, GDP_ROW_1IDX, "GDP")


def read_population_series(xlsx_file) -> pd.DataFrame:
    """Nüfus serisi: Scenario_Assumptions 5. satır."""
    return _read_scenario_assumptions_row(xlsx_file, POP_ROW_1IDX, "Nüfus")



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


def _series_total_storage_from_block(df_block: pd.DataFrame) -> pd.DataFrame:
    s = get_series_from_block(df_block, TOTAL_STORAGE_LABEL)
    if not s.empty:
        s["group"] = "Total Storage"
        return s[["year", "group", "value"]]

    comps = df_block[df_block["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))]
    if comps.empty:
        return pd.DataFrame(columns=["year", "group", "value"])
    long = _to_long(comps, value_name="value")
    out = long.groupby("year", as_index=False)["value"].sum()
    out["group"] = "Total Storage"
    return out[["year", "group", "value"]]


def generation_mix_from_block(gross_gen_df: pd.DataFrame) -> pd.DataFrame:
    if gross_gen_df is None or gross_gen_df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = gross_gen_df.copy()
    df = df[~df["item"].apply(_gen_is_excluded)]

    storage_bucket = _series_total_storage_from_block(df)

    df_no_storage_components = df[~df["item"].astype(str).apply(lambda x: bool(STORAGE_COMPONENT_REGEX.search(x)))].copy()
    if not storage_bucket.empty:
        df_no_storage_components = df_no_storage_components[df_no_storage_components["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    natgas_rows = df_no_storage_components[df_no_storage_components["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"
    natgas_series = natgas_series[["year", "group", "value"]]

    df_rest = df_no_storage_components[~df_no_storage_components["item"].apply(_is_natural_gas_item)].copy()

    long = _to_long(df_rest, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)
    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()

    mix = pd.concat([mix, natgas_series], ignore_index=True)
    if not storage_bucket.empty:
        mix = pd.concat([mix, storage_bucket], ignore_index=True)

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
st.title("Power_Generation Dashboard")

with st.sidebar:
    st.header("Dosya")
    uploaded = st.file_uploader("Excel yükleyin (.xlsx)", type=["xlsx"])

    default_scenario = "Scenario"
    if uploaded is not None and getattr(uploaded, "name", None):
        stem = Path(uploaded.name).stem
        if stem.lower().startswith("finalreport_"):
            stem = stem[len("FinalReport_") :]
        default_scenario = stem
    scenario_name = st.text_input("Senaryo adı", value=default_scenario)

    st.divider()
    st.header("Ayarlar")
    max_year = st.selectbox("Maksimum yıl", [2050, 2045, 2040, 2035], index=0)
    MAX_YEAR = int(max_year)

    start_year = st.selectbox("Başlangıç yılı", [2018, 2020, 2025, 2030, 2035, 2040, 2045], index=0)

if not uploaded:
    st.info("Başlamak için Excel dosyanızı yükleyin.")
    st.stop()

# Read blocks
blocks = read_power_generation(uploaded)
balance = blocks["electricity_balance"]
gross_gen = blocks["gross_generation"]
installed_cap = blocks["installed_capacity"]

# Scenario_Assumptions (Nüfus & GDP)
pop = read_population_series(uploaded)
pop = _filter_years(pop, start_year, MAX_YEAR)

gdp = read_gdp_series(uploaded)
gdp = _filter_years(gdp, start_year, MAX_YEAR)

co2 = read_co2_emissions_series(uploaded)
co2 = _filter_years(co2, start_year, MAX_YEAR)

primary_energy = read_primary_energy_production_series(uploaded)
primary_energy = _filter_years(primary_energy, start_year, MAX_YEAR)

primary_energy_source = read_primary_energy_consumption_by_source(uploaded)
primary_energy_source = _filter_years(primary_energy_source, start_year, MAX_YEAR)

# GDP CAGR between start_year and MAX_YEAR (based on filtered years)
gdp_cagr = np.nan
gdp_start_val = np.nan
gdp_end_val = np.nan
if not gdp.empty:
    # Prefer exact boundary years if present; else use nearest inside range
    gdp_sorted = gdp.sort_values("year")
    # start
    if (gdp_sorted["year"] == start_year).any():
        gdp_start_val = float(gdp_sorted.loc[gdp_sorted["year"] == start_year, "value"].iloc[0])
        y0 = start_year
    else:
        gdp_start_val = float(gdp_sorted.iloc[0]["value"])
        y0 = int(gdp_sorted.iloc[0]["year"])
    # end
    if (gdp_sorted["year"] == MAX_YEAR).any():
        gdp_end_val = float(gdp_sorted.loc[gdp_sorted["year"] == MAX_YEAR, "value"].iloc[0])
        y1 = MAX_YEAR
    else:
        gdp_end_val = float(gdp_sorted.iloc[-1]["value"])
        y1 = int(gdp_sorted.iloc[-1]["year"])

    gdp_cagr = _cagr(gdp_start_val, gdp_end_val, int(y1 - y0))


# ---- FIRST DISPLAY GRAPH: POPULATION ----
st.subheader("Türkiye Nüfus Gelişimi")
if pop.empty:
    st.warning("Nüfus serisi okunamadı: Scenario_Assumptions sekmesi veya 5. satır/yıl kolonları bulunamadı.")
else:
    # Yıllar sadece Scenario_Assumptions (3. satır) üzerinden gelsin:
    pop = pop.copy()
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce")
    pop["value"] = pd.to_numeric(pop["value"], errors="coerce")
    pop = pop.dropna(subset=["year", "value"]).sort_values("year")
    pop["year"] = pop["year"].astype(int)

    year_values = sorted(pop["year"].unique().tolist())

    # Çizgiyi net göstermek için nokta + çizgi (üst üste binme yok, tek chart)
    pop_line = (
        alt.Chart(pop)
        .mark_line()
        .encode(
            x=alt.X("year:O", title="Yıl", sort=year_values, axis=alt.Axis(values=year_values)),
            y=alt.Y("value:Q", title="Nüfus (milyon)"),
            tooltip=[
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("value:Q", title="Nüfus (milyon)", format=",.3f"),
            ],
        )
    )

    pop_points = (
        alt.Chart(pop)
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X("year:O", sort=year_values, axis=alt.Axis(values=year_values)),
            y=alt.Y("value:Q"),
            tooltip=[
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("value:Q", title="Nüfus (milyon)", format=",.3f"),
            ],
        )
    )

    st.altair_chart((pop_line + pop_points).properties(height=300), use_container_width=True)

st.divider()

# Electricity supply (GWh)
total_supply = get_series_from_block(balance, TOTAL_SUPPLY_LABEL)
total_supply = _filter_years(total_supply, start_year, MAX_YEAR)

# Generation mix (GWh)
gen_mix = generation_mix_from_block(gross_gen)
gen_mix = _filter_years(gen_mix, start_year, MAX_YEAR)

# YE share: total RE + intermittent (RES+GES)
ye_total = share_series_from_mix(gen_mix, total_supply, RENEWABLE_GROUPS, "YE Payı (Toplam)")
ye_int = share_series_from_mix(gen_mix, total_supply, INTERMITTENT_RE_GROUPS, "YE Payı (RES+GES)")
ye_both = pd.concat([ye_total, ye_int], ignore_index=True)
ye_both = _filter_years(ye_both, start_year, MAX_YEAR)

# Installed capacity total (GW) – Row79 - (Row102 + Row106)
cap_total = total_capacity_series_from_rows(
    raw=blocks["_raw"],
    year_cols_idx=blocks["_year_cols_idx"],
    years=blocks["_years"],
)
cap_total = _filter_years(cap_total, start_year, MAX_YEAR)

# Storage & PTX (GW) – separate chart
cap_storage = storage_series_capacity(installed_cap)
cap_ptx = ptx_series_capacity(installed_cap)
cap_storage = _filter_years(cap_storage, start_year, MAX_YEAR)
cap_ptx = _filter_years(cap_ptx, start_year, MAX_YEAR)

storage_ptx = pd.concat([cap_storage, cap_ptx], ignore_index=True)
storage_ptx = storage_ptx.rename(columns={"category": "group"})

# Capacity mix excluding storage & PTX (GW)
cap_mix = capacity_mix_excl_storage_ptx(
    installed_cap,
    cap_total,
    cap_storage.rename(columns={"group": "category"}, errors="ignore"),
    cap_ptx.rename(columns={"group": "category"}, errors="ignore"),
)
cap_mix = _filter_years(cap_mix, start_year, MAX_YEAR)

# -----------------------------
# KPI row (GDP CAGR FIRST)
# -----------------------------
st.subheader("Özet KPI’lar")
k0, k1, k2, k3, k4, k5 = st.columns(6)

latest_year = int(total_supply["year"].max()) if not total_supply.empty else None
latest_total = float(total_supply.loc[total_supply["year"] == latest_year, "value"].iloc[0]) if latest_year else np.nan

latest_ye_total = np.nan
latest_ye_int = np.nan
if latest_year and not ye_both.empty and (ye_both["year"] == latest_year).any():
    tmp = ye_both[ye_both["year"] == latest_year].set_index("series")["value"].to_dict()
    latest_ye_total = float(tmp.get("YE Payı (Toplam)", np.nan))
    latest_ye_int = float(tmp.get("YE Payı (RES+GES)", np.nan))

latest_ren = np.nan
if latest_year and not gen_mix.empty:
    latest_ren = float(gen_mix[(gen_mix["year"] == latest_year) & (gen_mix["group"].isin(RENEWABLE_GROUPS))]["value"].sum())

latest_cap = np.nan
if latest_year and not cap_total.empty and (cap_total["year"] == latest_year).any():
    latest_cap = float(cap_total.loc[cap_total["year"] == latest_year, "value"].iloc[0])

# KPI: GDP CAGR between start_year and MAX_YEAR (or nearest available inside range)
cagr_label = f"GDP CAGR (%) – {start_year}–{MAX_YEAR}"
k0.metric(cagr_label, f"{(gdp_cagr*100):.2f}%" if np.isfinite(gdp_cagr) else "—")

k1.metric("Senaryo", scenario_name)
k2.metric(f"Toplam Arz (GWh) – {latest_year if latest_year else ''}", f"{latest_total:,.0f}" if np.isfinite(latest_total) else "—")
k3.metric(f"YE Üretimi (GWh) – {latest_year if latest_year else ''}", f"{latest_ren:,.0f}" if np.isfinite(latest_ren) else "—")
k4.metric(
    f"YE Payı (%) – {latest_year if latest_year else ''}",
    f"{latest_ye_total:,.1f}% / {latest_ye_int:,.1f}%" if np.isfinite(latest_ye_total) else "—"
)
k5.metric(f"Kurulu Güç (GW) – {latest_year if latest_year else ''}", f"{latest_cap:,.3f}" if np.isfinite(latest_cap) else "—")

st.divider()


# -----------------------------
# Gross Inland Consumption (GWh) – Summary&Indicators (rows 15–23)
# -----------------------------
st.subheader("Gross Inland Consumption (in GWh)")

if primary_energy_source.empty:
    st.warning("Gross Inland Consumption verisi bulunamadı (Summary&Indicators: yıllar 3. satır, satırlar 15–23).")
else:
    pes = primary_energy_source.copy()
    pes["year"] = pes["year"].astype(int)
    year_vals = sorted(pes["year"].unique().tolist())

    # Keep a stable legend order by total over all years
    src_order = (
        pes.groupby("source", as_index=False)["value"].sum()
        .sort_values("value", ascending=False)["source"]
        .tolist()
    )

    pes["source"] = pd.Categorical(pes["source"], categories=src_order, ordered=True)
    pes = pes.sort_values(["year", "source"])

    pes_chart = (
        alt.Chart(pes)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl", sort=year_vals, axis=alt.Axis(values=year_vals)),
            y=alt.Y("value:Q", title="GWh", stack=True),
            color=alt.Color("source:N", title="Kaynak"),
            tooltip=[
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("source:N", title="Kaynak"),
                alt.Tooltip("value:Q", title="GWh", format=",.0f"),
            ],
        )
        .properties(height=420)
    )

    st.altair_chart(pes_chart, use_container_width=True)

# -----------------------------
# Charts: Supply + YE share
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Toplam Elektrik Arzı Trend (GWh)")
    if total_supply.empty:
        st.error(f"'{TOTAL_SUPPLY_LABEL}' satırı bulunamadı (Electricity Balance bloğunda).")
    else:
        st.altair_chart(
            alt.Chart(total_supply)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="GWh"))
            .properties(height=320),
            use_container_width=True,
        )

with right:
    st.subheader("YE Payı (%) – Toplam ve RES+GES")
    if ye_both.empty:
        st.warning("YE payı hesaplanamadı (mix veya total boş).")
    else:
        dash = alt.condition(
            alt.datum.series == "YE Payı (RES+GES)",
            alt.value([6, 4]),
            alt.value([1, 0]),
        )
        st.altair_chart(
            alt.Chart(ye_both)
            .mark_line()
            .encode(
                x=alt.X("year:O", title="Yıl"),
                y=alt.Y("value:Q", title="%"),
                color=alt.Color("series:N", title="Seri"),
                strokeDash=dash,
            )
            .properties(height=320),
            use_container_width=True,
        )

st.divider()

# -----------------------------
# Charts: Generation mix (GWh)
# -----------------------------
st.subheader("Üretim Karması (Gross, GWh) – Teknoloji Bazında")
st.caption("Not: Renewables/Combustion Plants ara-toplamları hariç. Depolama tek kalem: 'Total Storage'. Gas = 'Natural gas' (5 satır toplamı).")

if gen_mix.empty:
    st.warning("Gross generation bloğundan teknoloji karması çıkarılamadı.")
else:
    order_gen = [
        "Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables",
        "Natural gas", "Coal", "Lignite", "Nuclear",
        "Total Storage", "Other"
    ]
    gen_mix["group"] = pd.Categorical(gen_mix["group"], categories=order_gen, ordered=True)
    gen_mix = gen_mix.sort_values(["year", "group"])

    st.altair_chart(
        alt.Chart(gen_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GWh", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["year:O", "group:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=420),
        use_container_width=True,
    )

st.divider()

# -----------------------------
# Capacity section
# -----------------------------
st.subheader("Kurulu Güç (GW)")
st.caption("KG toplamı: Row79 − (Row102 + Row106). KG karması depolama & PTX hariç; depolama ve PTX ayrı grafikte verilir.")

c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("### Toplam Kurulu Güç Trend (GW)")
    if cap_total.empty:
        st.warning("Toplam kurulu güç serisi üretilemedi (satır bazlı okuma başarısız).")
    else:
        st.altair_chart(
            alt.Chart(cap_total)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="GW"))
            .properties(height=320),
            use_container_width=True,
        )

with c_right:
    st.markdown("### KG – Son Yıl (GW)")
    if latest_year and np.isfinite(latest_cap):
        st.metric(str(latest_year), f"{latest_cap:,.3f} GW")
    else:
        st.metric("—", "—")

st.markdown("### Kurulu Güç Karması (GW) – Depolama & PTX Hariç")

if cap_mix.empty:
    st.warning("Kurulu güç karması çıkarılamadı.")
else:
    order_cap = ["Hydro", "Wind (RES)", "Solar (GES)", "Natural gas", "Coal", "Lignite", "Nuclear", "Other"]
    cap_mix["group"] = pd.Categorical(cap_mix["group"], categories=order_cap, ordered=True)
    cap_mix = cap_mix.sort_values(["year", "group"])

    st.altair_chart(
        alt.Chart(cap_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["year:O", "group:N", alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(height=420),
        use_container_width=True,
    )

st.markdown("### Depolama ve Power-to-X (GW)")

if storage_ptx.empty:
    st.warning("Depolama/PTX serileri bulunamadı (Total Storage / Total Power to X yok ve bileşenler de bulunamadı).")
else:
    st.altair_chart(
        alt.Chart(storage_ptx)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW", stack=True),
            color=alt.Color("group:N", title="Kategori"),
            tooltip=["year:O", "group:N", alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(height=320),
        use_container_width=True,
    )

st.divider()

# -----------------------------
# Data tabs (debug/control)
# -----------------------------
st.divider()

# -----------------------------
# Data tabs (debug/control)
# -----------------------------

st.divider()

st.subheader("CO2 Emissions (ktn CO2)")
if co2.empty:
    st.warning("CO2 serisi okunamadı: PowerGeneration-Indicators sekmesi veya 30. satır/yıl kolonları bulunamadı.")
else:
    # X ekseninde sadece dosyadan gelen yılları göster (2018, 2020, 2025, ... gibi)
    co2_plot = co2.copy()
    co2_plot["year"] = co2_plot["year"].astype(int)
    co2_plot = co2_plot.sort_values("year")
    st.altair_chart(
        alt.Chart(co2_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="CO2 Emissions (ktn CO2)"),
            tooltip=["year:O", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=300),
        use_container_width=True,
    )
st.subheader("Veri Kontrolü")
tab_pop, tab_gdp, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_co2, tab_pe, tab_pes = st.tabs(
    ["Nüfus", "GDP", "Electricity Balance", "Gross Generation", "Mix (generation)", "Installed Capacity", "KG Total (rows)", "KG Mix excl. S&PTX", "Storage+PTX", "CO2 Emissions", "Primary Energy", "Primary Energy Cons. Source"]
)
with tab_pop:
    st.dataframe(pop, use_container_width=True)
with tab_gdp:
    st.dataframe(gdp, use_container_width=True)
with tab2:
    st.dataframe(balance, use_container_width=True)
with tab3:
    st.dataframe(gross_gen, use_container_width=True)
with tab4:
    st.dataframe(gen_mix, use_container_width=True)
with tab5:
    st.dataframe(installed_cap, use_container_width=True)
with tab6:
    st.dataframe(cap_total, use_container_width=True)
with tab7:
    st.dataframe(cap_mix, use_container_width=True)
with tab8:
    st.dataframe(storage_ptx, use_container_width=True)
with tab_co2:
    st.dataframe(co2, use_container_width=True)


with tab_pe:
    st.dataframe(primary_energy, use_container_width=True)


with tab_pes:
    st.dataframe(primary_energy_source, use_container_width=True)


with st.expander("Çalıştırma"):

    st.code("pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py", language="bash")
