import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Soğutma ve Veri Merkezleri Analizi", layout="wide")

# ============================================================
# 1) TRANSPORT ELECTRIC CODE MAP (only *_ELE used)
# ============================================================
TRANSPORT_ELE_TR = {
    # Passenger (private/public)
    "PSCAR_ELE": "Özel yolcu (Binek araç) – Elektrik",
    "PS2WL_ELE": "Özel yolcu (2 teker) – Elektrik",
    "PSPRD_ELE": "Toplu yolcu (Karayolu) – Elektrik",

    # Rail passenger
    "PSRLM_ELE": "Metro/Tramvay – Elektrik",
    "PSRLL_ELE": "Demiryolu yolcu (Yavaş) – Elektrik",
    "PSRLF_ELE": "Demiryolu yolcu (Hızlı) – Elektrik",

    # Water / Air passenger
    "PSWTR_ELE": "İç su yolları yolcu – Elektrik",
    "PSAIR_ELE": "Havayolu yolcu – Elektrik",

    # Freight
    "FRHDT_ELE": "Karayolu yük (Ağır ticari) – Elektrik",
    "FRLDT_ELE": "Karayolu yük (Hafif ticari) – Elektrik",
    "FRRLS_ELE": "Demiryolu yük – Elektrik",
    "FRWTR_ELE": "İç su yolları yük – Elektrik",
}

TRANSPORT_ELE_CODES = list(TRANSPORT_ELE_TR.keys())


# ============================================================
# Helpers
# ============================================================
def _scenario_from_filename(name: str) -> str:
    base = re.sub(r"\.[^.]+$", "", name)
    for pref in ("Demand_", "FinalReport_"):
        if base.startswith(pref):
            base = base[len(pref):]
            break
    m = re.search(r"(_tria.*)$", base, flags=re.IGNORECASE)
    if m:
        base = base[:m.start()]
    return base.strip() or name


def _read_final_energy(filelike) -> Tuple[List[int], pd.DataFrame, List[str], pd.DataFrame]:
    """
    Reads sheet FINAL_ENERGY without headers.
    Years are in row 1 (Excel) starting from column C.
    Returns:
      years: list[int]
      mat: numeric matrix for year columns (C..)
      codes: column A strings for each row (same row count as mat)
      df_raw: raw dataframe (needed for B-column ELC filter totals)
    """
    df = pd.read_excel(filelike, sheet_name="FINAL_ENERGY", header=None, engine=None)

    # Years: Excel row 1 -> df.iloc[0], from col C -> index 2
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
                break

    n_years = len(years)
    mat = (
        df.iloc[:, 2:2 + n_years]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    codes = df.iloc[:, 0].astype(str).fillna("").tolist()
    return years, mat, codes, df


def _filter_years(
    years: List[int],
    series_map: Dict[str, List[float]],
    y0: int,
    y1: int,
) -> Tuple[List[int], Dict[str, List[float]]]:
    idx = [i for i, y in enumerate(years) if y0 <= y <= y1]
    years_f = [years[i] for i in idx]
    out = {k: [v[i] for i in idx] for k, v in series_map.items()}
    return years_f, out


def _stacked_bar(
    years: List[int],
    series_map: Dict[str, List[float]],
    title: str,
    *,
    y_title: str,
    y_tickformat: str,
    use_percent_suffix: bool = False,
    barnorm_percent: bool = False,
) -> go.Figure:
    """
    General stacked bar.
    - barnorm_percent=True -> 100% normalized stacked (distribution).
    - barnorm_percent=False -> normal stacked (values add up; not forced to 100%).
    """
    fig = go.Figure()
    for label, values in series_map.items():
        fig.add_trace(go.Bar(x=years, y=values, name=label))

    layout_kwargs = dict(
        template="plotly_dark",
        title=dict(text=title, x=0.0, xanchor="left"),
        barmode="stack",
        bargap=0.35,
        bargroupgap=0.05,
        hovermode="x unified",
        legend=dict(
            title="Kullanım",
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=40, r=190, t=60, b=40),
        font=dict(size=13),
        xaxis=dict(
            type="category",
            tickmode="array",
            tickvals=years,
            ticktext=[str(y) for y in years],
            showgrid=False,
        ),
        yaxis=dict(
            title=y_title,
            tickformat=y_tickformat,
            ticksuffix="%" if use_percent_suffix else "",
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        ),
    )

    if barnorm_percent:
        layout_kwargs["barnorm"] = "percent"
        layout_kwargs["yaxis"] = dict(
            title=y_title,
            range=[0, 100],
            tickformat=y_tickformat,
            ticksuffix="%" if use_percent_suffix else "",
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        )

    fig.update_layout(**layout_kwargs)

    # Hover formatting
    if use_percent_suffix:
        for tr in fig.data:
            tr.update(hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:.0f}%<extra></extra>")
    else:
        for tr in fig.data:
            tr.update(hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:,.0f}<extra></extra>")

    return fig


def _total_elc_from_B_filter(df_raw: pd.DataFrame, n_years: int) -> List[float]:
    """
    Total electricity demand = sum of all rows where column B == 'ELC'
    across year columns (C..).
    """
    if df_raw is None or df_raw.empty:
        return [0.0] * n_years

    b = df_raw.iloc[:, 1].astype(str).str.strip().str.upper()  # column B
    mask = b.eq("ELC")

    mat_years = (
        df_raw.iloc[:, 2:2 + n_years]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    if mask.sum() == 0:
        return [0.0] * n_years

    return mat_years.loc[mask].sum(axis=0).tolist()


def _to_share_percent(numer: List[float], denom: List[float]) -> List[float]:
    return [(n / d * 100.0) if d else 0.0 for n, d in zip(numer, denom)]


def _sum_by_code(mat: pd.DataFrame, codes: List[str], code: str) -> List[float]:
    """
    Sum all rows where column A exactly equals `code` (case-insensitive, trimmed).
    Returns a year-series list.
    """
    if mat is None or mat.empty or not codes:
        return [0.0] * (mat.shape[1] if mat is not None else 0)

    codes_s = pd.Series(codes).astype(str).str.strip().str.upper()
    target = code.strip().upper()
    idxs = codes_s[codes_s == target].index.tolist()
    if not idxs:
        return [0.0] * mat.shape[1]
    return mat.iloc[idxs, :].sum(axis=0).tolist()


def _build_household_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    # (senin daha önce verdiğin kural)
    return {
        "Ev Aletleri": (mat.iloc[[235 - 1, 253 - 1, 241 - 1], :].sum(axis=0)).tolist(),
        "Alan Isıtma": (mat.iloc[[245 - 1, 248 - 1], :].sum(axis=0)).tolist(),
        "Su Isıtma": (mat.iloc[[260 - 1, 262 - 1], :].sum(axis=0)).tolist(),
        "Pişirme": (mat.iloc[[236 - 1], :].sum(axis=0)).tolist(),
        "Soğutma": (mat.iloc[[234 - 1], :].sum(axis=0)).tolist(),
    }


def _build_services_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    # (senin daha önce verdiğin kural)
    return {
        "Veri Merkezleri": (mat.iloc[[578 - 1], :].sum(axis=0)).tolist(),
        "Soğutma": (mat.iloc[[577 - 1], :].sum(axis=0)).tolist(),
        "Aydınlatma": (mat.iloc[[579 - 1], :].sum(axis=0)).tolist(),
        "Alan Isıtma": (mat.iloc[[588 - 1, 591 - 1], :].sum(axis=0)).tolist(),
        "Su Isıtma": (mat.iloc[[602 - 1, 604 - 1], :].sum(axis=0)).tolist(),
    }


def _build_transport_electric_series(mat: pd.DataFrame, codes: List[str]) -> Dict[str, List[float]]:
    """
    Transport electricity series from A-column code matching, only *_ELE codes in TRANSPORT_ELE_CODES.
    Labels are Turkish (TRANSPORT_ELE_TR).
    """
    out: Dict[str, List[float]] = {}
    for code in TRANSPORT_ELE_CODES:
        label = TRANSPORT_ELE_TR.get(code, code)
        out[label] = _sum_by_code(mat, codes, code)
    return out


def _sum_series_map(series_map: Dict[str, List[float]]) -> List[float]:
    if not series_map:
        return []
    keys = list(series_map.keys())
    n = len(series_map[keys[0]])
    tot = [0.0] * n
    for v in series_map.values():
        tot = [a + b for a, b in zip(tot, v)]
    return tot


# ============================================================
# UI
# ============================================================
st.title("Soğutma ve Veri Merkezleri Analizi")
st.caption("Demand Excel dosyalarını (1–3 senaryo) yükle. Grafikler FINAL_ENERGY sekmesinden okunur.")

STATE_KEY = "demand_files_v_codebased"
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = []

new_uploads = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
    key="demand_uploader_v_codebased",
)

if new_uploads:
    st.session_state[STATE_KEY] = [{"name": f.name, "bytes": f.getvalue()} for f in new_uploads[:3]]

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Yüklenen Excel'leri temizle", use_container_width=True):
        st.session_state[STATE_KEY] = []
        st.rerun()
with c2:
    if st.session_state[STATE_KEY]:
        st.caption("Kayıtlı Demand dosyaları: " + ", ".join(x["name"] for x in st.session_state[STATE_KEY]))
    else:
        st.caption("Kayıtlı Demand dosyası yok. Yükleyince sayfa geçişlerinde kaybolmaz.")

files = st.session_state[STATE_KEY]
if not files:
    st.info("Devam etmek için en az 1 Demand Excel yükle.")
    st.stop()

# Pre-read
all_years: List[int] = []
pre_read: List[Tuple[str, List[int], pd.DataFrame, List[str], pd.DataFrame, str]] = []

for item in files:
    try:
        years, mat, codes, df_raw = _read_final_energy(BytesIO(item["bytes"]))
        pre_read.append((item["name"], years, mat, codes, df_raw, ""))
        all_years.extend(years)
    except Exception as e:
        pre_read.append((item["name"], [], None, [], None, str(e)))

all_years = sorted(set(all_years))
if not all_years:
    st.error("FINAL_ENERGY yıl satırı bulunamadı (1. satır, C sütunundan başlamalı).")
    st.stop()

# Sidebar year range
st.sidebar.markdown("### Ayarlar")
ymin, ymax = int(all_years[0]), int(all_years[-1])
y0, y1 = st.sidebar.slider(
    "Senaryo yıl aralığı",
    min_value=ymin,
    max_value=ymax,
    value=(ymin, ymax),
    step=1,
)

cols = st.columns(2, gap="large")
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

for i, (fname, years, mat, codes, df_raw, err) in enumerate(pre_read):
    with cols[i % 2]:
        scenario = _scenario_from_filename(fname)
        st.subheader(scenario)

        if err or not years or mat is None or df_raw is None:
            st.error(f"Dosya: {fname}\n\nHata: {err or 'Dosya okunamadı veya yıl satırı bulunamadı.'}")
            continue

        n_years = len(years)

        # ============================================================
        # A) SUMMARY (NOT 100% normalized)
        # Values are contribution % to total electricity demand
        # ============================================================
        total_elc = _total_elc_from_B_filter(df_raw, n_years=n_years)

        # Electric transport total (sum of all transport *_ELE codes you provided)
        tr_ele_map = _build_transport_electric_series(mat, codes)
        transport_ele_total_abs = _sum_series_map(tr_ele_map)

        # Household cooling (row 234) and services DC (row 578)
        household_cooling_abs = (mat.iloc[[234 - 1], :].sum(axis=0)).tolist()
        services_dc_abs = (mat.iloc[[578 - 1], :].sum(axis=0)).tolist()

        summary_pct = {
            "Veri Merkezleri": _to_share_percent(services_dc_abs, total_elc),
            "Soğutma": _to_share_percent(household_cooling_abs, total_elc),
            "Elektrikli Araçlar": _to_share_percent(transport_ele_total_abs, total_elc),
        }
        years_sum, summary_pct_f = _filter_years(years, summary_pct, y0, y1)

        st.plotly_chart(
            _stacked_bar(
                years_sum,
                summary_pct_f,
                "Veri merkezi, Soğutma ve Elektrikli Araçların Nihai Elektrik Talebine Katkısı (%)",
                y_title="%",
                y_tickformat=".0f",
                use_percent_suffix=True,
                barnorm_percent=False,  # IMPORTANT: not normalized to 100
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_summary_contrib_pct",
        )

        # ============================================================
        # B) Household (abs + % distribution)
        # ============================================================
        hh_abs = _build_household_series(mat)
        years_hh, hh_abs_f = _filter_years(years, hh_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar(
                years_hh, hh_abs_f,
                "Konutlarda Elektrik Tüketimi (GWh) – Mutlak",
                y_title="GWh",
                y_tickformat=",.0f",
                use_percent_suffix=False,
                barnorm_percent=False,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_hh_abs",
        )
        st.plotly_chart(
            _stacked_bar(
                years_hh, hh_abs_f,
                "Konutlarda Elektrik Tüketimi (%) – Dağılım",
                y_title="%",
                y_tickformat=".0f",
                use_percent_suffix=True,
                barnorm_percent=True,  # 100% distribution
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_hh_pct",
        )

        # ============================================================
        # C) Services (abs + % distribution)
        # ============================================================
        sv_abs = _build_services_series(mat)
        years_sv, sv_abs_f = _filter_years(years, sv_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar(
                years_sv, sv_abs_f,
                "Hizmet Sektörü Elektrik Tüketimi (GWh) – Mutlak",
                y_title="GWh",
                y_tickformat=",.0f",
                use_percent_suffix=False,
                barnorm_percent=False,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_sv_abs",
        )
        st.plotly_chart(
            _stacked_bar(
                years_sv, sv_abs_f,
                "Hizmet Sektörü Elektrik Tüketimi (%) – Dağılım",
                y_title="%",
                y_tickformat=".0f",
                use_percent_suffix=True,
                barnorm_percent=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_sv_pct",
        )

        # ============================================================
        # D) Transport ELECTRIC only (abs + % distribution)
        # Now legend is Turkish via TRANSPORT_ELE_TR (code-based; robust to row shifts)
        # ============================================================
        years_tr, tr_abs_f = _filter_years(years, tr_ele_map, y0, y1)

        st.plotly_chart(
            _stacked_bar(
                years_tr, tr_abs_f,
                "Ulaştırma Elektrik Tüketimi (GWh) – Mutlak",
                y_title="GWh",
                y_tickformat=",.0f",
                use_percent_suffix=False,
                barnorm_percent=False,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_tr_abs",
        )
        st.plotly_chart(
            _stacked_bar(
                years_tr, tr_abs_f,
                "Ulaştırma Elektrik Tüketimi (%) – Dağılım",
                y_title="%",
                y_tickformat=".0f",
                use_percent_suffix=True,
                barnorm_percent=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_tr_pct",
        )

st.divider()
st.caption(
    "Not: Özet grafikte yüzdeler normalize edilmez; her seri toplam nihai elektrik talebine katkı (%) olarak üst üste toplanır. "
    "Toplam elektrik talebi, FINAL_ENERGY içinde B sütunu 'ELC' olan tüm satırların toplamıdır. "
    "Ulaştırma grafikleri A sütunundaki kodlardan (yalnız *_ELE) okunur; satır kayması sorun olmaz."
)
