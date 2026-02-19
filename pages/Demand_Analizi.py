import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Soğutma ve Veri Merkezleri Analizi", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _scenario_from_filename(name: str) -> str:
    """
    Rules:
    - If filename starts with Demand_ or FinalReport_, scenario is the substring AFTER that prefix
      and BEFORE _tria (if present).
    - Keep "b_" if it's part of the scenario (do NOT drop).
    """
    base = name
    # drop extension
    base = re.sub(r"\.[^.]+$", "", base)

    # remove known prefixes
    for pref in ("Demand_", "FinalReport_"):
        if base.startswith(pref):
            base = base[len(pref):]
            break

    # cut at _tria (case-insensitive) if present
    m = re.search(r"(_tria.*)$", base, flags=re.IGNORECASE)
    if m:
        base = base[:m.start()]

    return base.strip() or name


def _read_final_energy_matrix(file) -> Tuple[List[int], pd.DataFrame]:
    """
    Reads sheet FINAL_ENERGY without headers.
    Years are in row 1 (Excel) starting from column C.
    Returns:
      years: list[int]
      mat: DataFrame of numeric values (same shape as original) for convenience.
    """
    df = pd.read_excel(file, sheet_name="FINAL_ENERGY", header=None, engine=None)

    # Years: Excel row 1 -> df.iloc[0], start at col C -> index 2
    years_raw = df.iloc[0, 2:].tolist()
    years: List[int] = []
    for y in years_raw:
        if pd.isna(y):
            continue
        try:
            years.append(int(float(y)))
        except Exception:
            s = str(y).strip()
            if s.isdigit():
                years.append(int(s))
            else:
                break  # stop at first non-year token

    # Keep only the year columns we detected
    n_years = len(years)

    # Numeric matrix for year columns
    mat = (
        df.iloc[:, 2 : 2 + n_years]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return years, mat


def _rows_sum(mat: pd.DataFrame, excel_rows_1idx: List[int]) -> List[float]:
    """
    Sum specified Excel row numbers (1-indexed) across year columns.
    """
    idxs = [r - 1 for r in excel_rows_1idx]  # convert to 0-index for pandas iloc
    idxs = [i for i in idxs if 0 <= i < len(mat)]  # guard against out-of-range

    if not idxs:
        return [0.0] * mat.shape[1]

    series = mat.iloc[idxs, :].sum(axis=0)
    return series.tolist()


def _stacked_area(
    years: List[int],
    series_map: Dict[str, List[float]],
    title: str,
    percent: bool = False,
) -> go.Figure:
    fig = go.Figure()

    # consistent order as provided
    for label, values in series_map.items():
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines",
                name=label,
                stackgroup="one",
            )
        )

    if percent:
        # Plotly groupnorm percent
        for tr in fig.data:
            tr.update(groupnorm="percent")
        fig.update_layout(yaxis=dict(title="%", ticksuffix="%", range=[0, 100]))
    else:
        fig.update_layout(yaxis=dict(title="GWh"))

    fig.update_layout(
        title=title,
        legend_title_text="Kullanım",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_household_series(years: List[int], mat: pd.DataFrame) -> Dict[str, List[float]]:
    # Konut
    return {
        "Ev Aletleri": _rows_sum(mat, [235, 253, 241]),
        "Alan Isıtma": _rows_sum(mat, [245, 248]),
        "Su Isıtma": _rows_sum(mat, [260, 262]),
        "Pişirme": _rows_sum(mat, [236]),
        "Soğutma": _rows_sum(mat, [234]),
    }


def _build_services_series(years: List[int], mat: pd.DataFrame) -> Dict[str, List[float]]:
    # Hizmet
    return {
        "Veri Merkezleri": _rows_sum(mat, [578]),
        "Soğutma": _rows_sum(mat, [577]),
        "Aydınlatma": _rows_sum(mat, [579]),
        "Alan Isıtma": _rows_sum(mat, [588, 591]),
        "Su Isıtma": _rows_sum(mat, [602, 604]),
    }


def _total(series_map: Dict[str, List[float]]) -> List[float]:
    labels = list(series_map.keys())
    n = len(series_map[labels[0]]) if labels else 0
    out = [0.0] * n
    for v in series_map.values():
        out = [a + b for a, b in zip(out, v)]
    return out


def _to_percent(series_map: Dict[str, List[float]]) -> Dict[str, List[float]]:
    tot = _total(series_map)
    out: Dict[str, List[float]] = {}
    for k, v in series_map.items():
        out[k] = [(vi / ti * 100.0) if ti else 0.0 for vi, ti in zip(v, tot)]
    return out


# ----------------------------
# UI
# ----------------------------
st.title("Soğutma ve Veri Merkezleri Analizi")
st.caption("Demand Excel dosyalarını (1–3 senaryo) yükle. Grafikler FINAL_ENERGY sekmesinden okunur.")

uploaded = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Devam etmek için en az 1 Demand Excel yükle.")
    st.stop()

if len(uploaded) > 3:
    st.warning("En fazla 3 dosya seçilebilir. İlk 3 dosya kullanılacak.")
    uploaded = uploaded[:3]

# Layout: 1->2 cols, 2->2 cols, 3->3 cols
n = len(uploaded)
n_cols = 3 if n == 3 else 2
cols = st.columns(n_cols, gap="large")

for i, file in enumerate(uploaded):
    col = cols[i] if n_cols == 3 else cols[i % 2]
    with col:
        scenario = _scenario_from_filename(file.name)
        st.subheader(scenario)

        try:
            years, mat = _read_final_energy_matrix(file)
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
            continue

        if not years:
            st.error("Yıl satırı bulunamadı (FINAL_ENERGY: 1. satır, C sütunundan başlamalı).")
            continue

        # Household
        hh_abs = _build_household_series(years, mat)
        hh_pct = _to_percent(hh_abs)

        st.plotly_chart(
            _stacked_area(years, hh_abs, "Konutlarda Elektrik Tüketimi (GWh) – Mutlak", percent=False),
            use_container_width=True,
        )
        st.plotly_chart(
            _stacked_area(years, hh_pct, "Konutlarda Elektrik Tüketimi (%) – Dağılım", percent=True),
            use_container_width=True,
        )

        # Services
        sv_abs = _build_services_series(years, mat)
        sv_pct = _to_percent(sv_abs)

        st.plotly_chart(
            _stacked_area(years, sv_abs, "Hizmet Sektörü Elektrik Tüketimi (GWh) – Mutlak", percent=False),
            use_container_width=True,
        )
        st.plotly_chart(
            _stacked_area(years, sv_pct, "Hizmet Sektörü Elektrik Tüketimi (%) – Dağılım", percent=True),
            use_container_width=True,
        )

st.divider()
st.caption(
    "Not: Senaryo adı dosya isminden `Demand_..._tria..` veya `FinalReport_..._tria..` kuralıyla çıkarılır; `b_` korunur."
)
