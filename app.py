# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Power Generation Dashboard", layout="wide")

# -----------------------------
# Defaults / labels
# -----------------------------
DEFAULT_MAX_YEAR = 2050

TOTAL_SUPPLY_LABEL = "Total Domestic Power Generation - Gross"
TOTAL_CAPACITY_LABEL = "Gross Installed Capacity (in GWe)"
TOTAL_STORAGE_LABEL = "Total Storage"
TOTAL_PTX_LABEL = "Total Power to X"

# Power_Generation sheet: KG total rule (Excel 1-indexed rows)
KG_BASE_ROW_1IDX = 79
KG_SUB_ROW_1IDX_1 = 102
KG_SUB_ROW_1IDX_2 = 106

# Scenario_Assumptions sheet: years and values (Excel 1-indexed rows)
SCENARIO_SHEET_NAME = "Scenario_Assumptions"
SCENARIO_YEAR_ROW_1IDX = 3
POP_VALUE_ROW_1IDX = 5
GDP_VALUE_ROW_1IDX = 6

# Exclude subtotal-like rows
CAPACITY_EXCLUDE_EXACT = {"Renewables", "Combustion Plants"}
CAPACITY_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)(?!Power to X\b)", flags=re.IGNORECASE)

GEN_EXCLUDE_EXACT = {"Renewables", "Combustion Plants"}
GEN_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)(?!Power to X\b)", flags=re.IGNORECASE)

# Storage components (avoid double count if Total Storage exists)
STORAGE_COMPONENT_REGEX = re.compile(r"(pumped\s+storage|\bbattery\b|demand\s+response)", flags=re.IGNORECASE)

# PTX components (fallback if Total Power to X not present)
PTX_COMPONENT_REGEX = re.compile(r"^power\s+to\s+(hydrogen|gas|liquids)\b|power\s+to\s+x", flags=re.IGNORECASE)

# Natural gas is sum of these items only (avoid overcount)
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
        score = sum(1 for y in years if y is not None)
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


def _extract_block(raw: pd.DataFrame, first_col_idx: int, year_cols_idx: list[int], years: list[int], block_title_regex: str, max_year: int):
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

    keep_years = [y for y in years if y <= max_year]
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
    if n_years <= 0:
        return np.nan
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        return np.nan
    if start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (1.0 / n_years) - 1.0


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


def _gen_is_excluded(item: str) -> bool:
    it = (item or "").strip()
    return (it in GEN_EXCLUDE_EXACT) or bool(GEN_EXCLUDE_REGEX.search(it))


def _cap_is_excluded(item: str) -> bool:
    it = (item or "").strip()
    return (it in CAPACITY_EXCLUDE_EXACT) or bool(CAPACITY_EXCLUDE_REGEX.search(it))


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


@st.cache_data(show_spinner=False)
def read_power_generation_bytes(file_bytes: bytes, max_year: int):
    raw = pd.read_excel(file_bytes, sheet_name="Power_Generation", header=None)
    year_row = _find_year_row(raw)
    years, year_cols_idx = _extract_years(raw, year_row)

    first_col_idx = 0
    return {
        "electricity_balance": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Electricity\s+Balance", max_year),
        "gross_generation": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Gross\s+Electricity\s+Generation\s+by\s+plant\s+type", max_year),
        "installed_capacity": _extract_block(raw, first_col_idx, year_cols_idx, years, r"Gross\s+Installed\s+Capacity", max_year),
        "_raw": raw,
        "_years": years,
        "_year_cols_idx": year_cols_idx,
    }


@st.cache_data(show_spinner=False)
def read_scenario_series_bytes(file_bytes: bytes, value_row_1idx: int, max_year: int, series_name: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(file_bytes, sheet_name=SCENARIO_SHEET_NAME, header=None)
    except Exception:
        return pd.DataFrame(columns=["year", "value", "series"])

    year_r0 = SCENARIO_YEAR_ROW_1IDX - 1
    val_r0 = value_row_1idx - 1
    if year_r0 < 0 or val_r0 < 0 or year_r0 >= len(raw) or val_r0 >= len(raw):
        return pd.DataFrame(columns=["year", "value", "series"])

    row_year = raw.iloc[year_r0, :].tolist()
    row_val = raw.iloc[val_r0, :].tolist()

    years, vals = [], []
    for y_cell, v_cell in zip(row_year, row_val):
        y = _as_int_year(y_cell)
        if y is None:
            continue
        if y <= max_year:
            years.append(int(y))
            vals.append(pd.to_numeric(v_cell, errors="coerce"))

    df = pd.DataFrame({"year": years, "value": vals}).dropna(subset=["value"]).sort_values("year")
    df["series"] = series_name
    return df


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


def _row_values_by_excel_row(raw: pd.DataFrame, row_1idx: int, year_cols_idx: list[int], years: list[int], max_year: int) -> pd.Series:
    r0 = row_1idx - 1
    if r0 < 0 or r0 >= len(raw):
        return pd.Series(dtype=float)

    data = {}
    for y, c in zip(years, year_cols_idx):
        if y <= max_year:
            data[y] = pd.to_numeric(raw.iloc[r0, c], errors="coerce")
    return pd.Series(data, dtype=float)


def total_capacity_series_from_rows(raw: pd.DataFrame, year_cols_idx: list[int], years: list[int], max_year: int) -> pd.DataFrame:
    s79 = _row_values_by_excel_row(raw, KG_BASE_ROW_1IDX, year_cols_idx, years, max_year)
    s102 = _row_values_by_excel_row(raw, KG_SUB_ROW_1IDX_1, year_cols_idx, years, max_year)
    s106 = _row_values_by_excel_row(raw, KG_SUB_ROW_1IDX_2, year_cols_idx, years, max_year)

    kg = s79 - (s102.fillna(0) + s106.fillna(0))
    out = pd.DataFrame({"year": kg.index.astype(int), "value": kg.values}).dropna(subset=["value"]).sort_values("year")
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


def scenario_name_from_filename(name: str) -> str:
    if not name:
        return "Scenario"
    stem = Path(name).stem
    if stem.lower().startswith("finalreport_"):
        stem = stem[len("FinalReport_") :]
    return stem


def compute_gdp_cagr(gdp_df: pd.DataFrame, start_year: int, end_year: int) -> float:
    if gdp_df is None or gdp_df.empty:
        return np.nan
    g = gdp_df.sort_values("year")
    g = g[(g["year"] >= start_year) & (g["year"] <= end_year)]
    if g.empty:
        return np.nan

    # start
    if (g["year"] == start_year).any():
        v0 = float(g.loc[g["year"] == start_year, "value"].iloc[0])
        y0 = start_year
    else:
        v0 = float(g.iloc[0]["value"])
        y0 = int(g.iloc[0]["year"])

    # end
    if (g["year"] == end_year).any():
        v1 = float(g.loc[g["year"] == end_year, "value"].iloc[0])
        y1 = end_year
    else:
        v1 = float(g.iloc[-1]["value"])
        y1 = int(g.iloc[-1]["year"])

    return _cagr(v0, v1, int(y1 - y0))


def process_one_scenario(file_bytes: bytes, scenario: str, start_year: int, max_year: int):
    blocks = read_power_generation_bytes(file_bytes, max_year=max_year)
    balance = blocks["electricity_balance"]
    gross_gen = blocks["gross_generation"]
    installed_cap = blocks["installed_capacity"]

    # Scenario_Assumptions
    pop = read_scenario_series_bytes(file_bytes, POP_VALUE_ROW_1IDX, max_year=max_year, series_name="Türkiye Nüfus Gelişimi")
    gdp = read_scenario_series_bytes(file_bytes, GDP_VALUE_ROW_1IDX, max_year=max_year, series_name="GDP")

    pop = _filter_years(pop, start_year, max_year)
    gdp = _filter_years(gdp, start_year, max_year)

    # Supply
    total_supply = get_series_from_block(balance, TOTAL_SUPPLY_LABEL)
    total_supply = _filter_years(total_supply, start_year, max_year)

    # Generation mix
    gen_mix = generation_mix_from_block(gross_gen)
    gen_mix = _filter_years(gen_mix, start_year, max_year)

    # YE shares (two series)
    ye_total = share_series_from_mix(gen_mix, total_supply, RENEWABLE_GROUPS, "YE Payı (Toplam)")
    ye_int = share_series_from_mix(gen_mix, total_supply, INTERMITTENT_RE_GROUPS, "YE Payı (RES+GES)")
    ye_both = pd.concat([ye_total, ye_int], ignore_index=True)
    ye_both = _filter_years(ye_both, start_year, max_year)

    # Installed capacity total (rows formula)
    cap_total = total_capacity_series_from_rows(
        raw=blocks["_raw"],
        year_cols_idx=blocks["_year_cols_idx"],
        years=blocks["_years"],
        max_year=max_year,
    )
    cap_total = _filter_years(cap_total, start_year, max_year)

    # Storage & PTX
    cap_storage = storage_series_capacity(installed_cap)
    cap_ptx = ptx_series_capacity(installed_cap)
    cap_storage = _filter_years(cap_storage, start_year, max_year)
    cap_ptx = _filter_years(cap_ptx, start_year, max_year)
    storage_ptx = pd.concat([cap_storage, cap_ptx], ignore_index=True).rename(columns={"category": "group"})

    # Capacity mix excluding storage & PTX
    cap_mix = capacity_mix_excl_storage_ptx(
        installed_cap,
        cap_total,
        cap_storage.rename(columns={"group": "category"}, errors="ignore"),
        cap_ptx.rename(columns={"group": "category"}, errors="ignore"),
    )
    cap_mix = _filter_years(cap_mix, start_year, max_year)

    # Add scenario column everywhere
    for df in (pop, gdp):
        if not df.empty:
            df["scenario"] = scenario
    for df in (total_supply, cap_total):
        if not df.empty:
            df["scenario"] = scenario
    for df in (gen_mix, cap_mix, storage_ptx):
        if not df.empty:
            df["scenario"] = scenario
    if not ye_both.empty:
        ye_both["scenario"] = scenario

    # KPIs
    latest_year = int(total_supply["year"].max()) if not total_supply.empty else np.nan
    latest_total = float(total_supply.loc[total_supply["year"] == latest_year, "value"].iloc[0]) if np.isfinite(latest_year) else np.nan

    latest_ye_total = np.nan
    latest_ye_int = np.nan
    if np.isfinite(latest_year) and not ye_both.empty and (ye_both["year"] == latest_year).any():
        tmp = ye_both[ye_both["year"] == latest_year].set_index("series")["value"].to_dict()
        latest_ye_total = float(tmp.get("YE Payı (Toplam)", np.nan))
        latest_ye_int = float(tmp.get("YE Payı (RES+GES)", np.nan))

    latest_ye_gwh = np.nan
    if np.isfinite(latest_year) and not gen_mix.empty:
        latest_ye_gwh = float(gen_mix[(gen_mix["year"] == latest_year) & (gen_mix["group"].isin(RENEWABLE_GROUPS))]["value"].sum())

    latest_cap = np.nan
    if np.isfinite(latest_year) and not cap_total.empty and (cap_total["year"] == latest_year).any():
        latest_cap = float(cap_total.loc[cap_total["year"] == latest_year, "value"].iloc[0])

    gdp_cagr = compute_gdp_cagr(gdp[["year", "value"]] if not gdp.empty else gdp, start_year, max_year)

    kpi = {
        "scenario": scenario,
        "latest_year": int(latest_year) if np.isfinite(latest_year) else np.nan,
        "GDP_CAGR_%": (gdp_cagr * 100.0) if np.isfinite(gdp_cagr) else np.nan,
        "Total_Supply_GWh": latest_total,
        "YE_Generation_GWh": latest_ye_gwh,
        "YE_Share_Total_%": latest_ye_total,
        "YE_Share_RES+GES_%": latest_ye_int,
        "Installed_Capacity_GW": latest_cap,
    }

    return {
        "population": pop,
        "gdp": gdp,
        "total_supply": total_supply,
        "ye_both": ye_both,
        "gen_mix": gen_mix,
        "cap_total": cap_total,
        "cap_mix": cap_mix,
        "storage_ptx": storage_ptx,
        "kpi": kpi,
    }


# -----------------------------
# UI - Sidebar
# -----------------------------
st.title("Power_Generation Dashboard")

with st.sidebar:
    st.header("Dosyalar (Senaryolar)")
    uploaded_files = st.file_uploader("Excel yükleyin (.xlsx) – çoklu seçim", type=["xlsx"], accept_multiple_files=True)

    st.divider()
    st.header("Ayarlar")
    max_year = st.selectbox("Maksimum yıl", [2050, 2045, 2040, 2035], index=0)
    max_year = int(max_year)

    start_year = st.selectbox("Başlangıç yılı", [2018, 2020, 2025, 2030, 2035, 2040, 2045], index=0)

    st.divider()
    st.header("Senaryo Seçimi")

if not uploaded_files:
    st.info("Başlamak için en az 1 Excel dosyası yükleyin.")
    st.stop()

# Build scenario list from filenames
scenario_map = {}
for f in uploaded_files:
    scen = scenario_name_from_filename(getattr(f, "name", "Scenario"))
    # ensure unique
    base = scen
    i = 2
    while scen in scenario_map:
        scen = f"{base} ({i})"
        i += 1
    scenario_map[scen] = f

all_scenarios = list(scenario_map.keys())

# Selection controls
with st.sidebar:
    colA, colB = st.columns([1, 1])
    with colA:
        select_all = st.checkbox("Tümünü seç", value=True)
    with colB:
        n_slider = st.slider("Gösterilecek senaryo sayısı", min_value=1, max_value=len(all_scenarios), value=min(len(all_scenarios), 4))

    default_selected = all_scenarios[:n_slider] if select_all else all_scenarios[:n_slider]
    selected = st.multiselect("Grafikte/KPI'da göster", options=all_scenarios, default=default_selected)

if not selected:
    st.warning("En az 1 senaryo seçin.")
    st.stop()

# -----------------------------
# Process selected scenarios
# -----------------------------
pop_all = []
gdp_all = []
supply_all = []
ye_all = []
genmix_all = []
cap_total_all = []
cap_mix_all = []
storage_ptx_all = []
kpi_rows = []

errors = []

for scen in selected:
    f = scenario_map[scen]
    try:
        file_bytes = f.getvalue()
        out = process_one_scenario(file_bytes, scen, start_year=start_year, max_year=max_year)

        if not out["population"].empty:
            pop_all.append(out["population"])
        if not out["gdp"].empty:
            gdp_all.append(out["gdp"])
        if not out["total_supply"].empty:
            supply_all.append(out["total_supply"])
        if not out["ye_both"].empty:
            ye_all.append(out["ye_both"])
        if not out["gen_mix"].empty:
            genmix_all.append(out["gen_mix"])
        if not out["cap_total"].empty:
            cap_total_all.append(out["cap_total"])
        if not out["cap_mix"].empty:
            cap_mix_all.append(out["cap_mix"])
        if not out["storage_ptx"].empty:
            storage_ptx_all.append(out["storage_ptx"])

        kpi_rows.append(out["kpi"])
    except Exception as e:
        errors.append(f"{scen}: {e}")

if errors:
    st.error("Bazı senaryolarda okuma/işleme hatası oluştu:")
    for msg in errors:
        st.write(f"- {msg}")

population = pd.concat(pop_all, ignore_index=True) if pop_all else pd.DataFrame(columns=["year", "value", "series", "scenario"])
gdp = pd.concat(gdp_all, ignore_index=True) if gdp_all else pd.DataFrame(columns=["year", "value", "series", "scenario"])
total_supply = pd.concat(supply_all, ignore_index=True) if supply_all else pd.DataFrame(columns=["year", "value", "scenario"])
ye_both = pd.concat(ye_all, ignore_index=True) if ye_all else pd.DataFrame(columns=["year", "series", "value", "scenario"])
gen_mix = pd.concat(genmix_all, ignore_index=True) if genmix_all else pd.DataFrame(columns=["year", "group", "value", "scenario"])
cap_total = pd.concat(cap_total_all, ignore_index=True) if cap_total_all else pd.DataFrame(columns=["year", "value", "scenario"])
cap_mix = pd.concat(cap_mix_all, ignore_index=True) if cap_mix_all else pd.DataFrame(columns=["year", "group", "value", "scenario"])
storage_ptx = pd.concat(storage_ptx_all, ignore_index=True) if storage_ptx_all else pd.DataFrame(columns=["year", "group", "value", "scenario"])

kpi_df = pd.DataFrame(kpi_rows)
if not kpi_df.empty:
    kpi_df = kpi_df.sort_values(["latest_year", "scenario"], ascending=[False, True])

# -----------------------------
# FIRST GRAPH: Population (multi-scenario)
# -----------------------------
st.subheader("Türkiye Nüfus Gelişimi")
if population.empty:
    st.warning("Nüfus serisi bulunamadı (Scenario_Assumptions: yıl=3. satır, nüfus=5. satır).")
else:
    st.altair_chart(
        alt.Chart(population)
        .mark_line()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="Nüfus"),
            color=alt.Color("scenario:N", title="Senaryo"),
            tooltip=["scenario:N", "year:O", alt.Tooltip("value:Q", format=",.3f")],
        )
        .properties(height=280),
        use_container_width=True,
    )

# SECOND GRAPH: GDP (multi-scenario)
st.subheader("GDP (Scenario_Assumptions) – Trend")
if gdp.empty:
    st.warning("GDP serisi bulunamadı (Scenario_Assumptions: yıl=3. satır, GDP=6. satır).")
else:
    st.altair_chart(
        alt.Chart(gdp)
        .mark_line()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GDP"),
            color=alt.Color("scenario:N", title="Senaryo"),
            tooltip=["scenario:N", "year:O", alt.Tooltip("value:Q", format=",.3f")],
        )
        .properties(height=280),
        use_container_width=True,
    )

st.divider()

# -----------------------------
# KPI Table (scenario-by-scenario)
# -----------------------------
st.subheader("Özet KPI’lar (Senaryo Bazında)")
if kpi_df.empty:
    st.warning("KPI üretilemedi.")
else:
    show_cols = [
        "scenario",
        "latest_year",
        "GDP_CAGR_%",
        "Total_Supply_GWh",
        "YE_Generation_GWh",
        "YE_Share_Total_%",
        "YE_Share_RES+GES_%",
        "Installed_Capacity_GW",
    ]
    fmt = kpi_df.copy()
    fmt["GDP_CAGR_%"] = fmt["GDP_CAGR_%"].map(lambda x: f"{x:.2f}%" if np.isfinite(x) else "—")
    fmt["Total_Supply_GWh"] = fmt["Total_Supply_GWh"].map(lambda x: f"{x:,.0f}" if np.isfinite(x) else "—")
    fmt["YE_Generation_GWh"] = fmt["YE_Generation_GWh"].map(lambda x: f"{x:,.0f}" if np.isfinite(x) else "—")
    fmt["YE_Share_Total_%"] = fmt["YE_Share_Total_%"].map(lambda x: f"{x:.1f}%" if np.isfinite(x) else "—")
    fmt["YE_Share_RES+GES_%"] = fmt["YE_Share_RES+GES_%"].map(lambda x: f"{x:.1f}%" if np.isfinite(x) else "—")
    fmt["Installed_Capacity_GW"] = fmt["Installed_Capacity_GW"].map(lambda x: f"{x:,.3f}" if np.isfinite(x) else "—")

    st.dataframe(fmt[show_cols], use_container_width=True)

st.divider()

# -----------------------------
# Supply + YE share charts
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Toplam Elektrik Arzı Trend (GWh)")
    if total_supply.empty:
        st.warning(f"'{TOTAL_SUPPLY_LABEL}' serisi bulunamadı.")
    else:
        st.altair_chart(
            alt.Chart(total_supply)
            .mark_line()
            .encode(
                x=alt.X("year:O", title="Yıl"),
                y=alt.Y("value:Q", title="GWh"),
                color=alt.Color("scenario:N", title="Senaryo"),
                tooltip=["scenario:N", "year:O", alt.Tooltip("value:Q", format=",.0f")],
            )
            .properties(height=320),
            use_container_width=True,
        )

with right:
    st.subheader("YE Payı (%) – Toplam ve RES+GES")
    if ye_both.empty:
        st.warning("YE payı hesaplanamadı.")
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
                color=alt.Color("scenario:N", title="Senaryo"),
                strokeDash=dash,
                tooltip=["scenario:N", "series:N", "year:O", alt.Tooltip("value:Q", format=",.1f")],
            )
            .properties(height=320),
            use_container_width=True,
        )

st.divider()

# -----------------------------
# Generation mix (stacked) – facet by scenario
# -----------------------------
st.subheader("Üretim Karması (Gross, GWh) – Teknoloji Bazında (Senaryo Karşılaştırma)")
st.caption("Stacked grafikte aynı yıl için senaryoları yan yana görmek için senaryo bazında facet kullanılır.")

if gen_mix.empty:
    st.warning("Üretim karması çıkarılamadı.")
else:
    order_gen = [
        "Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables",
        "Natural gas", "Coal", "Lignite", "Nuclear",
        "Total Storage", "Other"
    ]
    gen_mix = gen_mix.copy()
    gen_mix["group"] = pd.Categorical(gen_mix["group"], categories=order_gen, ordered=True)

    base = (
        alt.Chart(gen_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GWh", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["scenario:N", "year:O", "group:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=340)
    )

    st.altair_chart(
        base.facet(
            column=alt.Column("scenario:N", title="Senaryo", sort=selected),
        ).resolve_scale(y="independent"),
        use_container_width=True,
    )

st.divider()

# -----------------------------
# Installed Capacity: total trend (multi-scenario)
# -----------------------------
st.subheader("Toplam Kurulu Güç Trend (GW) – Senaryo Karşılaştırma")
if cap_total.empty:
    st.warning("Kurulu güç toplam serisi üretilemedi.")
else:
    st.altair_chart(
        alt.Chart(cap_total)
        .mark_line()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW"),
            color=alt.Color("scenario:N", title="Senaryo"),
            tooltip=["scenario:N", "year:O", alt.Tooltip("value:Q", format=",.3f")],
        )
        .properties(height=320),
        use_container_width=True,
    )

# Capacity mix stacked – facet
st.subheader("Kurulu Güç Karması (GW) – Depolama & PTX Hariç (Senaryo Karşılaştırma)")
if cap_mix.empty:
    st.warning("Kurulu güç karması çıkarılamadı.")
else:
    order_cap = ["Hydro", "Wind (RES)", "Solar (GES)", "Natural gas", "Coal", "Lignite", "Nuclear", "Other"]
    cap_mix = cap_mix.copy()
    cap_mix["group"] = pd.Categorical(cap_mix["group"], categories=order_cap, ordered=True)

    base2 = (
        alt.Chart(cap_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["scenario:N", "year:O", "group:N", alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(height=340)
    )
    st.altair_chart(
        base2.facet(column=alt.Column("scenario:N", title="Senaryo", sort=selected)).resolve_scale(y="independent"),
        use_container_width=True,
    )

# Storage + PTX stacked – facet
st.subheader("Depolama ve Power-to-X (GW) – Senaryo Karşılaştırma")
if storage_ptx.empty:
    st.warning("Depolama/PTX serileri bulunamadı.")
else:
    base3 = (
        alt.Chart(storage_ptx)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW", stack=True),
            color=alt.Color("group:N", title="Kategori"),
            tooltip=["scenario:N", "year:O", "group:N", alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(
        base3.facet(column=alt.Column("scenario:N", title="Senaryo", sort=selected)).resolve_scale(y="independent"),
        use_container_width=True,
    )

# -----------------------------
# Optional: Debug tabs
# -----------------------------
with st.expander("Veri Kontrolü (Debug)"):
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
        ["Nüfus", "GDP", "Arz", "YE Payı", "Gen Mix", "KG Total", "KG Mix", "Storage+PTX"]
    )
    with t1:
        st.dataframe(population, use_container_width=True)
    with t2:
        st.dataframe(gdp, use_container_width=True)
    with t3:
        st.dataframe(total_supply, use_container_width=True)
    with t4:
        st.dataframe(ye_both, use_container_width=True)
    with t5:
        st.dataframe(gen_mix, use_container_width=True)
    with t6:
        st.dataframe(cap_total, use_container_width=True)
    with t7:
        st.dataframe(cap_mix, use_container_width=True)
    with t8:
        st.dataframe(storage_ptx, use_container_width=True)

with st.expander("Çalıştırma"):
    st.code("pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py", language="bash")
