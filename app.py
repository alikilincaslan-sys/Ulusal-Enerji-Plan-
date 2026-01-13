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

# Installed capacity total row (GW)
TOTAL_CAPACITY_LABEL = "Gross Installed Capacity (in GWe)"

# Storage total rows (both generation/capacity blocks)
TOTAL_STORAGE_LABEL = "Total Storage"

# Exclude headers/subtotals (do not count) – installed capacity
# NOTE: "Total Storage" is NOT excluded anymore; we will use it as the storage total.
CAPACITY_EXCLUDE_EXACT = {
    "Renewables",
    "Combustion Plants",
    "Total Power to X",
}
CAPACITY_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)", flags=re.IGNORECASE)  # exclude Total * except Total Storage

# Exclude headers/subtotals (do not count) – generation mix block
GEN_EXCLUDE_EXACT = {
    "Renewables",
    "Combustion Plants",
    "Total Power to X",
}
GEN_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+(?!Storage\b)", flags=re.IGNORECASE)

# Components that should NOT be counted if Total Storage exists (avoid double count)
STORAGE_COMPONENT_REGEX = re.compile(
    r"(pumped\s+storage|\bbattery\b|demand\s+response)",
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
# fallback fuzzy match (in case of minor spelling differences)
NATURAL_GAS_REGEX = re.compile(
    r"^(industrial\s+chp\s+plant\s+solid\s+fuels|ccgt\s+without\s+ccs|ccgt\s+with\s+ccs|open\s+cycle,\s*ic\s+and\s+gt|industrial\s+chp\s+plants?\s+oil/gas)$",
    flags=re.IGNORECASE,
)

# Technology mapping (used for generation/capacity mixes)
# Gas is renamed to Natural gas, but we do NOT use broad gas regex to avoid over-counting.
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
    return out[out["year"] <= MAX_YEAR].sort_values("year")


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
    """
    Üretim karması (GWh):
    - Renewables / Combustion Plants / Total Power to X gibi ara toplamlar hariç
    - Pumped Storage / Battery / Demand Response bileşenleri hariç
    - Depolama: doğrudan 'Total Storage' satırından (varsa) alınır
    - Natural gas: sadece verilen 5 satırın toplamı
    """
    if gross_gen_df is None or gross_gen_df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = gross_gen_df.copy()
    df = df[~df["item"].apply(_gen_is_excluded)]

    # --- Total Storage series (preferred) + remove storage components to avoid double count
    storage_series = get_series_from_block(df, TOTAL_STORAGE_LABEL)
    df_wo_storage_components = df[~df["item"].astype(str).apply(lambda s: bool(STORAGE_COMPONENT_REGEX.search(s)))].copy()

    # If Total Storage exists, keep it for later and remove from df used for grouping
    if not storage_series.empty:
        df_wo_storage_components = df_wo_storage_components[df_wo_storage_components["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    # --- Natural gas series: sum of specific items, then remove them from other grouping
    natgas_rows = df_wo_storage_components[df_wo_storage_components["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"

    df_rest = df_wo_storage_components[~df_wo_storage_components["item"].apply(_is_natural_gas_item)].copy()

    # --- Group remaining items
    long = _to_long(df_rest, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)
    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()

    # add Natural gas
    mix = pd.concat([mix, natgas_series[["year", "group", "value"]]], ignore_index=True)

    # add Total Storage (as single bucket)
    if not storage_series.empty:
        ts = storage_series.rename(columns={"value": "value"}).copy()
        ts["group"] = "Total Storage"
        mix = pd.concat([mix, ts[["year", "group", "value"]]], ignore_index=True)
    else:
        # fallback: if Total Storage row not present, sum components
        comps = df[df["item"].astype(str).apply(lambda s: bool(STORAGE_COMPONENT_REGEX.search(s)))]
        comps_long = _to_long(comps, value_name="value")
        ts2 = comps_long.groupby("year", as_index=False)["value"].sum()
        ts2["group"] = "Total Storage"
        mix = pd.concat([mix, ts2[["year", "group", "value"]]], ignore_index=True)

    mix = mix[mix["year"] <= MAX_YEAR]
    return mix


def renewable_share(mix_df: pd.DataFrame, total_series_df: pd.DataFrame) -> pd.DataFrame:
    if mix_df.empty or total_series_df.empty:
        return pd.DataFrame(columns=["year", "value"])

    ren = mix_df[mix_df["group"].isin(RENEWABLE_GROUPS)].groupby("year", as_index=False)["value"].sum()
    tot = total_series_df.rename(columns={"value": "total"})
    out = ren.merge(tot, on="year", how="inner")
    out["value"] = (out["value"] / out["total"]) * 100.0
    return out[["year", "value"]].sort_values("year")


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


def total_capacity_series(installed_cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Toplam kurulu güç (GW): TOTAL_CAPACITY_LABEL satırından.
    Bulunamazsa: ara toplamları hariç tutarak topla.
    """
    if installed_cap_df is None or installed_cap_df.empty:
        return pd.DataFrame(columns=["year", "value"])

    s = get_series_from_block(installed_cap_df, TOTAL_CAPACITY_LABEL)
    if not s.empty:
        return s

    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_cap_is_excluded)]
    df = df[df["item"].str.strip().ne(TOTAL_CAPACITY_LABEL)]

    year_cols = [c for c in df.columns if isinstance(c, int)]
    summed = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    out = pd.DataFrame({"year": year_cols, "value": summed.values}).dropna()
    return out[out["year"] <= MAX_YEAR].sort_values("year")


def capacity_mix_from_block(installed_cap_df: pd.DataFrame, cap_total: pd.DataFrame) -> pd.DataFrame:
    """
    Kurulu güç karması (GW):
    - Renewables / Combustion Plants / Total Power to X gibi ara toplamlar hariç
    - Depolama: doğrudan 'Total Storage' satırından (varsa), bileşenler sayılmaz
    - Other Renewables -> Other içinde gösterilir
    - Natural gas: sadece verilen 5 satırın toplamı
    - Other: residual (Total - diğer gruplar - Total Storage)
    """
    if installed_cap_df is None or installed_cap_df.empty or cap_total is None or cap_total.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_cap_is_excluded)]
    df = df[df["item"].str.strip().ne(TOTAL_CAPACITY_LABEL)]

    # Total Storage from row + remove components
    storage_series = get_series_from_block(df, TOTAL_STORAGE_LABEL)
    df = df[~df["item"].astype(str).apply(lambda s: bool(STORAGE_COMPONENT_REGEX.search(s)))].copy()
    if not storage_series.empty:
        df = df[df["item"].str.strip().ne(TOTAL_STORAGE_LABEL)]

    # Natural gas from specific items
    natgas_rows = df[df["item"].apply(_is_natural_gas_item)]
    natgas_long = _to_long(natgas_rows, value_name="value")
    natgas_series = natgas_long.groupby("year", as_index=False)["value"].sum()
    natgas_series["group"] = "Natural gas"

    df_rest = df[~df["item"].apply(_is_natural_gas_item)].copy()

    # Group remaining
    long = _to_long(df_rest, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)

    # Fold Other Renewables into Other (capacity only)
    long.loc[long["group"] == "Other Renewables", "group"] = "Other"

    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()

    # Add Natural gas
    mix = pd.concat([mix, natgas_series[["year", "group", "value"]]], ignore_index=True)

    # Add Total Storage
    if not storage_series.empty:
        ts = storage_series.copy()
        ts["group"] = "Total Storage"
        mix = pd.concat([mix, ts[["year", "group", "value"]]], ignore_index=True)
    else:
        # fallback: sum components if row not present
        comps = installed_cap_df[installed_cap_df["item"].astype(str).apply(lambda s: bool(STORAGE_COMPONENT_REGEX.search(s)))]
        comps_long = _to_long(comps, value_name="value")
        ts2 = comps_long.groupby("year", as_index=False)["value"].sum()
        ts2["group"] = "Total Storage"
        mix = pd.concat([mix, ts2[["year", "group", "value"]]], ignore_index=True)

    mix = mix[mix["year"] <= MAX_YEAR]

    # Residual Other to match TOTAL_CAPACITY_LABEL each year
    total_map = cap_total.set_index("year")["value"].to_dict()

    known = mix[mix["group"] != "Other"].groupby("year", as_index=False)["value"].sum().rename(columns={"value": "known_sum"})
    known["total"] = known["year"].map(total_map)
    known["residual_other"] = (known["total"] - known["known_sum"]).clip(lower=0)

    other_rows = known[["year", "residual_other"]].rename(columns={"residual_other": "value"})
    other_rows["group"] = "Other"

    mix = mix[mix["group"] != "Other"]
    mix = pd.concat([mix, other_rows], ignore_index=True)

    # Safety: ensure all years exist
    for y, tot in total_map.items():
        if y <= MAX_YEAR and y not in set(mix["year"]):
            mix = pd.concat([mix, pd.DataFrame([{"year": y, "group": "Other", "value": tot}])], ignore_index=True)

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

if not uploaded:
    st.info("Başlamak için Excel dosyanızı yükleyin.")
    st.stop()

blocks = read_power_generation(uploaded)
balance = blocks["electricity_balance"]
gross_gen = blocks["gross_generation"]
installed_cap = blocks["installed_capacity"]

# Electricity supply (GWh)
total_supply = get_series_from_block(balance, TOTAL_SUPPLY_LABEL)

# Generation mix (GWh) + RE share
gen_mix = generation_mix_from_block(gross_gen)
ye = renewable_share(gen_mix, total_supply)

# Installed capacity (GW) total + mix
cap_total = total_capacity_series(installed_cap)
cap_mix = capacity_mix_from_block(installed_cap, cap_total)

# -----------------------------
# KPI row
# -----------------------------
st.subheader("Özet KPI’lar")
k1, k2, k3, k4, k5 = st.columns(5)

latest_year = int(total_supply["year"].max()) if not total_supply.empty else None

latest_total = float(total_supply.loc[total_supply["year"] == latest_year, "value"].iloc[0]) if latest_year else np.nan

latest_ye = (
    float(ye.loc[ye["year"] == latest_year, "value"].iloc[0])
    if (latest_year and not ye.empty and (ye["year"] == latest_year).any())
    else np.nan
)

latest_ren = np.nan
if latest_year and not gen_mix.empty:
    latest_ren = float(gen_mix[(gen_mix["year"] == latest_year) & (gen_mix["group"].isin(RENEWABLE_GROUPS))]["value"].sum())

latest_cap = np.nan
if latest_year and not cap_total.empty and (cap_total["year"] == latest_year).any():
    latest_cap = float(cap_total.loc[cap_total["year"] == latest_year, "value"].iloc[0])

k1.metric("Senaryo", scenario_name)
k2.metric(f"Toplam Arz (GWh) – {latest_year if latest_year else ''}", f"{latest_total:,.0f}" if np.isfinite(latest_total) else "—")
k3.metric(f"YE Üretimi (GWh) – {latest_year if latest_year else ''}", f"{latest_ren:,.0f}" if np.isfinite(latest_ren) else "—")
k4.metric(f"YE Payı (%) – {latest_year if latest_year else ''}", f"{latest_ye:,.1f}%" if np.isfinite(latest_ye) else "—")
k5.metric(f"Kurulu Güç (GW) – {latest_year if latest_year else ''}", f"{latest_cap:,.1f}" if np.isfinite(latest_cap) else "—")

st.divider()

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
    st.subheader("YE Payı (%)")
    if ye.empty:
        st.warning("YE payı hesaplanamadı (mix veya total boş).")
    else:
        st.altair_chart(
            alt.Chart(ye)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="%"))
            .properties(height=320),
            use_container_width=True,
        )

st.divider()

# -----------------------------
# Charts: Generation mix (GWh)
# -----------------------------
st.subheader("Üretim Karması (Gross, GWh) – Teknoloji Bazında")
st.caption("Not: Renewables/Combustion Plants ara-toplamları hariç. Pumped storage/battery/demand response sayılmaz; 'Total Storage' satırı kullanılır. Gas = 'Natural gas' (5 satır toplamı).")

if gen_mix.empty:
    st.warning("Gross generation bloğundan teknoloji karması çıkarılamadı.")
else:
    order_gen = ["Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables", "Natural gas", "Coal", "Lignite", "Nuclear", "Total Storage", "Other"]
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
# Charts: Installed Capacity (GW)
# -----------------------------
st.subheader("Kurulu Güç (Gross Installed Capacity, GW)")
st.caption("Not: Renewables/Combustion Plants ara-toplamları hariç. Depolama 'Total Storage' satırından. Natural gas = 5 satır toplamı. Other Renewables capacity tarafında Other içine alınır; Other residual (Toplam satıra eşitlenir).")

c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("### Toplam Kurulu Güç Trend (GW)")
    if cap_total.empty:
        st.warning("Toplam kurulu güç serisi üretilemedi (toplam satırı bulunamadı).")
    else:
        st.altair_chart(
            alt.Chart(cap_total)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="GW"))
            .properties(height=320),
            use_container_width=True,
        )

with c_right:
    st.markdown("### Kurulu Güç (GW) – Son Yıl")
    if latest_year and np.isfinite(latest_cap):
        st.metric(str(latest_year), f"{latest_cap:,.1f} GW")
    else:
        st.metric("—", "—")

st.markdown("### Kurulu Güç Karması (GW) – Teknoloji Bazında")

if cap_mix.empty:
    st.warning("Kurulu güç karması çıkarılamadı.")
else:
    order_cap = ["Hydro", "Wind (RES)", "Solar (GES)", "Natural gas", "Coal", "Lignite", "Nuclear", "Total Storage", "Other"]
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

st.divider()

# -----------------------------
# Data tabs (debug/control)
# -----------------------------
st.subheader("Veri Kontrolü")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Electricity Balance", "Gross Generation", "Mix (generation)", "Installed Capacity", "Mix (capacity)"]
)
with tab1:
    st.dataframe(balance, use_container_width=True)
with tab2:
    st.dataframe(gross_gen, use_container_width=True)
with tab3:
    st.dataframe(gen_mix, use_container_width=True)
with tab4:
    st.dataframe(installed_cap, use_container_width=True)
with tab5:
    st.dataframe(cap_mix, use_container_width=True)

with st.expander("Çalıştırma"):
    st.code("pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py", language="bash")
