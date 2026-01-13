# app.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Power Generation Dashboard", layout="wide")

# -----------------------------
# Config (your rules)
# -----------------------------
MAX_YEAR = 2050

TOTAL_SUPPLY_LABEL = "Total Domestic Power Generation - Gross"

# Installed capacity total satırı (ekrandaki en üst satır)
TOTAL_CAPACITY_LABEL = "Gross Installed Capacity (in GWe)"

# Kurulu güç bloğunda ara-toplamlar / üst başlıklar: HESABA DAHİL ETME
CAPACITY_EXCLUDE_EXACT = {
    "Renewables",
    "Combustion Plants",
    "Total Storage",
    "Total Power to X",
}
# Daha genel güvenlik: "Total " ile başlayan alt toplamlar da hariç
CAPACITY_EXCLUDE_REGEX = re.compile(r"^\s*Total\s+", flags=re.IGNORECASE)

# Teknoloji kırılımı: Coal/Lignite ayrı, Hydro tek, RES ve GES ayrı
TECH_GROUPS = {
    "Hydro": [r"\bhydro\b"],
    "Wind (RES)": [r"\bwind\b", r"\bres\b"],
    "Solar (GES)": [r"\bsolar\b", r"\bges\b", r"\bpv\b"],
    "Coal": [r"\bcoal\b(?!.*lignite)"],
    "Lignite": [r"\blignite\b"],
    "Gas": [r"\bgas\b", r"\bccgt\b", r"\bocgt\b"],
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


def _extract_block(
    raw: pd.DataFrame,
    first_col_idx: int,
    year_cols_idx: list[int],
    years: list[int],
    block_title_regex: str,
):
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
    long = long.dropna(subset=[value_name])
    return long


@st.cache_data(show_spinner=False)
def read_power_generation(xlsx_file):
    raw = pd.read_excel(xlsx_file, sheet_name="Power_Generation", header=None)

    year_row = _find_year_row(raw)
    years, year_cols_idx = _extract_years(raw, year_row)

    first_col_idx = 0

    blocks = {
        "electricity_balance": _extract_block(
            raw, first_col_idx, year_cols_idx, years, r"Electricity\s+Balance"
        ),
        "gross_generation": _extract_block(
            raw, first_col_idx, year_cols_idx, years, r"Gross\s+Electricity\s+Generation\s+by\s+plant\s+type"
        ),
        "net_generation": _extract_block(
            raw, first_col_idx, year_cols_idx, years, r"Net\s+Electricity\s+Generation\s+by\s+plant\s+type"
        ),
        "installed_capacity": _extract_block(
            raw, first_col_idx, year_cols_idx, years, r"Gross\s+Installed\s+Capacity"
        ),
    }
    return blocks


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
    out = out[out["year"] <= MAX_YEAR].sort_values("year")
    return out


def _strict_match_group(item: str) -> str:
    s = item.lower().strip()
    for grp, pats in TECH_GROUPS.items():
        for p in pats:
            if re.search(p, s, flags=re.IGNORECASE):
                return grp
    return "Other"


def generation_mix_from_block(gross_gen_df: pd.DataFrame) -> pd.DataFrame:
    if gross_gen_df is None or gross_gen_df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    long = _to_long(gross_gen_df, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)

    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()
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
# Installed Capacity (GW) helpers
# -----------------------------
def _capacity_is_excluded(item: str) -> bool:
    it = (item or "").strip()
    if it in CAPACITY_EXCLUDE_EXACT:
        return True
    if CAPACITY_EXCLUDE_REGEX.search(it):
        # "Total Storage", "Total Power to X" vb. zaten yakalanır
        return True
    return False


def capacity_mix_from_block(installed_cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gross Installed Capacity bloğundan teknoloji bazlı kurulu güç karması (GW).
    Ara toplam satırları hariç.
    """
    if installed_cap_df is None or installed_cap_df.empty:
        return pd.DataFrame(columns=["year", "group", "value"])

    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_capacity_is_excluded)]

    # Toplam satırını karma grafiğine dahil etmeyelim
    df = df[df["item"].str.strip().ne(TOTAL_CAPACITY_LABEL)]

    long = _to_long(df, value_name="value")
    long["group"] = long["item"].apply(_strict_match_group)

    mix = long.groupby(["year", "group"], as_index=False)["value"].sum()
    mix = mix[mix["year"] <= MAX_YEAR]
    return mix


def total_capacity_series(installed_cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Toplam kurulu güç serisi (GW):
    Öncelik: doğrudan toplam satırı 'Gross Installed Capacity (in GWe)'.
    Bulunamazsa: ara toplamları hariç tutarak tüm kalemleri toplar.
    """
    if installed_cap_df is None or installed_cap_df.empty:
        return pd.DataFrame(columns=["year", "value"])

    # 1) Doğrudan toplam satırı
    s = get_series_from_block(installed_cap_df, TOTAL_CAPACITY_LABEL)
    if not s.empty:
        return s

    # 2) Yedek: "Gross Installed Capacity" içeren satır
    cand = installed_cap_df[installed_cap_df["item"].str.contains(r"Gross\s+Installed\s+Capacity", case=False, na=False)]
    if not cand.empty:
        tmp = installed_cap_df.copy()
        tmp["item"] = tmp["item"].astype(str).str.strip()
        tmp = tmp[tmp.index.isin(cand.index)]
        # ilk eşleşen
        label = tmp.iloc[0]["item"]
        s2 = get_series_from_block(installed_cap_df, label)
        if not s2.empty:
            return s2

    # 3) Son çare: ara toplamları hariç tutarak topla
    df = installed_cap_df.copy()
    df = df[~df["item"].apply(_capacity_is_excluded)]
    # varsa toplam satırını da hariç tut (çift saymasın)
    df = df[df["item"].str.strip().ne(TOTAL_CAPACITY_LABEL)]

    year_cols = [c for c in df.columns if isinstance(c, int)]
    summed = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    out = pd.DataFrame({"year": year_cols, "value": summed.values}).dropna()
    return out[out["year"] <= MAX_YEAR].sort_values("year")


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

# Read blocks
blocks = read_power_generation(uploaded)
balance = blocks["electricity_balance"]
gross_gen = blocks["gross_generation"]
installed_cap = blocks["installed_capacity"]

# Supply series (GWh)
total_supply = get_series_from_block(balance, TOTAL_SUPPLY_LABEL)

# Generation mix (GWh)
gen_mix = generation_mix_from_block(gross_gen)

# Renewables share (%)
ye = renewable_share(gen_mix, total_supply)

# Installed capacity (GW)
cap_mix = capacity_mix_from_block(installed_cap)
cap_total = total_capacity_series(installed_cap)

# -----------------------------
# KPI row
# -----------------------------
st.subheader("Özet KPI’lar")

k1, k2, k3, k4, k5 = st.columns(5)

latest_year = int(total_supply["year"].max()) if not total_supply.empty else None

latest_total = (
    float(total_supply.loc[total_supply["year"] == latest_year, "value"].iloc[0])
    if latest_year
    else np.nan
)

latest_ye = (
    float(ye.loc[ye["year"] == latest_year, "value"].iloc[0])
    if (latest_year and not ye.empty and (ye["year"] == latest_year).any())
    else np.nan
)

latest_ren = np.nan
if latest_year and not gen_mix.empty:
    latest_ren = float(
        gen_mix[(gen_mix["year"] == latest_year) & (gen_mix["group"].isin(RENEWABLE_GROUPS))]["value"].sum()
    )

latest_cap = np.nan
if latest_year and not cap_total.empty and (cap_total["year"] == latest_year).any():
    latest_cap = float(cap_total.loc[cap_total["year"] == latest_year, "value"].iloc[0])

k1.metric("Senaryo", scenario_name)
k2.metric(
    f"Toplam Arz (GWh) – {latest_year if latest_year else ''}",
    f"{latest_total:,.0f}" if np.isfinite(latest_total) else "—",
)
k3.metric(
    f"YE Üretimi (GWh) – {latest_year if latest_year else ''}",
    f"{latest_ren:,.0f}" if np.isfinite(latest_ren) else "—",
)
k4.metric(
    f"YE Payı (%) – {latest_year if latest_year else ''}",
    f"{latest_ye:,.1f}%" if np.isfinite(latest_ye) else "—",
)
k5.metric(
    f"Kurulu Güç (GW) – {latest_year if latest_year else ''}",
    f"{latest_cap:,.1f}" if np.isfinite(latest_cap) else "—",
)

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
        c = (
            alt.Chart(total_supply)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="GWh"))
            .properties(height=320)
        )
        st.altair_chart(c, use_container_width=True)

with right:
    st.subheader("YE Payı (%)")
    if ye.empty:
        st.warning("YE payı hesaplanamadı (mix veya total boş).")
    else:
        c = (
            alt.Chart(ye)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="%"))
            .properties(height=320)
        )
        st.altair_chart(c, use_container_width=True)

st.divider()

# -----------------------------
# Charts: Generation mix (GWh)
# -----------------------------
st.subheader("Üretim Karması (Gross, GWh) – Teknoloji Bazında")

if gen_mix.empty:
    st.warning("Gross generation bloğundan teknoloji karması çıkarılamadı.")
else:
    order = ["Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables", "Gas", "Coal", "Lignite", "Nuclear", "Other"]
    gen_mix["group"] = pd.Categorical(gen_mix["group"], categories=order, ordered=True)
    gen_mix = gen_mix.sort_values(["year", "group"])

    stacked = (
        alt.Chart(gen_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GWh", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["year:O", "group:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=420)
    )
    st.altair_chart(stacked, use_container_width=True)

st.divider()

# -----------------------------
# Charts: Installed Capacity (GW)
# -----------------------------
st.subheader("Kurulu Güç (Gross Installed Capacity, GW)")

c_left, c_right = st.columns([2, 1])

with c_left:
    st.markdown("### Toplam Kurulu Güç Trend (GW)")
    if cap_total.empty:
        st.warning("Toplam kurulu güç serisi üretilemedi (Installed Capacity bloğu / toplam satırı bulunamadı).")
    else:
        chart_cap = (
            alt.Chart(cap_total)
            .mark_line()
            .encode(x=alt.X("year:O", title="Yıl"), y=alt.Y("value:Q", title="GW"))
            .properties(height=320)
        )
        st.altair_chart(chart_cap, use_container_width=True)

with c_right:
    st.markdown("### Kurulu Güç (GW) – Son Yıl")
    if latest_year and np.isfinite(latest_cap):
        st.metric(str(latest_year), f"{latest_cap:,.1f} GW")
    else:
        st.metric("—", "—")

st.markdown("### Kurulu Güç Karması (GW) – Teknoloji Bazında (Ara toplamlar hariç)")

if cap_mix.empty:
    st.warning("Kurulu güç karması çıkarılamadı.")
else:
    order = ["Hydro", "Wind (RES)", "Solar (GES)", "Other Renewables", "Gas", "Coal", "Lignite", "Nuclear", "Other"]
    cap_mix["group"] = pd.Categorical(cap_mix["group"], categories=order, ordered=True)
    cap_mix = cap_mix.sort_values(["year", "group"])

    cap_stacked = (
        alt.Chart(cap_mix)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="GW", stack=True),
            color=alt.Color("group:N", title="Teknoloji"),
            tooltip=["year:O", "group:N", alt.Tooltip("value:Q", format=",.2f")],
        )
        .properties(height=420)
    )
    st.altair_chart(cap_stacked, use_container_width=True)

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
    st.code(
        "pip install streamlit pandas openpyxl altair numpy\nstreamlit run app.py",
        language="bash",
    )
