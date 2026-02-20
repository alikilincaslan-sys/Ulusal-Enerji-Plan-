import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="TUEP Demand Analizi", layout="wide")

# ============================================================
# TRANSPORT ELECTRIC CODE MAP (only *_ELE used)
# ============================================================
TRANSPORT_ELE_TR = {
    "PSCAR_ELE": "Özel yolcu (Binek araç) – Elektrik",
    "PS2WL_ELE": "Özel yolcu (2 teker) – Elektrik",
    "PSPRD_ELE": "Toplu yolcu (Karayolu) – Elektrik",
    "PSRLM_ELE": "Metro/Tramvay – Elektrik",
    "PSRLL_ELE": "Demiryolu yolcu (Yavaş) – Elektrik",
    "PSRLF_ELE": "Demiryolu yolcu (Hızlı) – Elektrik",
    "PSWTR_ELE": "İç su yolları yolcu – Elektrik",
    "PSAIR_ELE": "Havayolu yolcu – Elektrik",
    "FRHDT_ELE": "Karayolu yük (Ağır ticari) – Elektrik",
    "FRLDT_ELE": "Karayolu yük (Hafif ticari) – Elektrik",
    "FRRLS_ELE": "Demiryolu yük – Elektrik",
    "FRWTR_ELE": "İç su yolları yük – Elektrik",
}
TRANSPORT_ELE_CODES = list(TRANSPORT_ELE_TR.keys())

# ============================================================
# TRANSPORT (TOP LEVEL) TR MAP by ROW INDEX (Excel 1-indexed)
# ============================================================
TRANSPORT_ROW_TR = {
    220: "Özel yolcu taşımacılığı",
    224: "Toplu yolcu taşımacılığı",
    229: "Demiryolu yolcu taşımacılığı",
    230: "İç su yollarında yolcu taşımacılığı",
    541: "Havayolu yolcu taşımacılığı",
    546: "Bunker yakıtları",
    551: "Karayolu yük taşımacılığı",
    558: "Demiryolu yük taşımacılığı",
    560: "İç su yollarında yük taşımacılığı",
    563: "Diğer Ulaştırma (563)",
    564: "Diğer Ulaştırma (564)",
    565: "Diğer Ulaştırma (565)",
}

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
    df = pd.read_excel(filelike, sheet_name="FINAL_ENERGY", header=None, engine=None)

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


def _rows_sum(mat: pd.DataFrame, excel_rows_1idx: List[int]) -> List[float]:
    idxs = [r - 1 for r in excel_rows_1idx]
    idxs = [i for i in idxs if 0 <= i < len(mat)]
    if not idxs:
        return [0.0] * mat.shape[1]
    return mat.iloc[idxs, :].sum(axis=0).tolist()


def _sum_series_map(series_map: Dict[str, List[float]]) -> List[float]:
    if not series_map:
        return []
    n = len(next(iter(series_map.values())))
    out = [0.0] * n
    for v in series_map.values():
        out = [a + b for a, b in zip(out, v)]
    return out


def _to_share_percent(numer: List[float], denom: List[float]) -> List[float]:
    out = []
    for a, b in zip(numer, denom):
        out.append((a / b * 100.0) if b else 0.0)
    return out


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
    y_title: str,
    y_tickformat: str,
    use_percent_suffix: bool,
    barnorm_percent: bool,
) -> go.Figure:
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
        margin=dict(l=40, r=160, t=60, b=40),
        font=dict(size=13),
        xaxis=dict(
            title="",
            type="category",
            tickmode="array",
            tickvals=years,
            ticktext=[str(y) for y in years],
            showgrid=False,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
            tickformat=y_tickformat,
        ),
    )

    if barnorm_percent:
        layout_kwargs["barnorm"] = "percent"
        layout_kwargs["yaxis"] = dict(
            title="%",
            range=[0, 100],
            ticksuffix="%",
            tickformat=".0f",
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        )

    fig.update_layout(**layout_kwargs)

    # hover formats
    if barnorm_percent:
        for tr in fig.data:
            tr.update(hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:.0f}%<extra></extra>")
    else:
        if use_percent_suffix:
            for tr in fig.data:
                tr.update(hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:.0f}%<extra></extra>")
        else:
            for tr in fig.data:
                tr.update(hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:,0f} GWh<extra></extra>")

    return fig


def _build_household_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    return {
        "Ev Aletleri": _rows_sum(mat, [235, 253, 241]),
        "Alan Isıtma": _rows_sum(mat, [245, 248]),
        "Su Isıtma": _rows_sum(mat, [260, 262]),
        "Pişirme": _rows_sum(mat, [236]),
        "Soğutma": _rows_sum(mat, [234]),
    }


def _build_services_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    return {
        "Veri Merkezleri": _rows_sum(mat, [578]),
        "Soğutma": _rows_sum(mat, [577]),
        "Aydınlatma": _rows_sum(mat, [579]),
        "Alan Isıtma": _rows_sum(mat, [588, 591]),
        "Su Isıtma": _rows_sum(mat, [602, 604]),
    }


def _build_transport_top_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    # Excel row numbers → Turkish labels
    out: Dict[str, List[float]] = {}
    for r, label in TRANSPORT_ROW_TR.items():
        out[label] = _rows_sum(mat, [r])
    return out


def _build_transport_electric_series(mat: pd.DataFrame, codes: List[str]) -> Dict[str, List[float]]:
    # codes list is column A values (strings) aligned with mat rows
    idx_by_code: Dict[str, int] = {}
    for i, c in enumerate(codes):
        idx_by_code[str(c).strip()] = i

    out: Dict[str, List[float]] = {}
    n_years = mat.shape[1]
    for code in TRANSPORT_ELE_CODES:
        i = idx_by_code.get(code, None)
        if i is None or i < 0 or i >= len(mat):
            out[TRANSPORT_ELE_TR[code]] = [0.0] * n_years
        else:
            out[TRANSPORT_ELE_TR[code]] = mat.iloc[i, :].tolist()
    return out


def _total_elc_from_B_filter(df_raw: pd.DataFrame, n_years: int) -> List[float]:
    # B sütunu "ELC" olanların toplamı (C..)
    try:
        bcol = df_raw.iloc[:, 1].astype(str).str.upper().fillna("")
        mask = bcol.eq("ELC")
        mat = (
            df_raw.loc[mask, df_raw.columns[2:2 + n_years]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        return mat.sum(axis=0).tolist()
    except Exception:
        return [0.0] * n_years


# ============================================================
# UI
# ============================================================
st.title("Soğutma ve Veri Merkezleri Analizi")
st.caption("Demand Excel dosyalarını (1–3 senaryo) yükle. Grafikler FINAL_ENERGY sekmesinden okunur.")

# ---- Persisted upload store (page navigation safe) ----
STATE_KEY = "demand_files_v1"  # versioned key to avoid old state conflicts
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = []  # list[{"name": str, "bytes": bytes}]

# Uploader (new files overwrite stored set)
new_uploads = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
    key="demand_uploader_v1",
)

if new_uploads:
    st.session_state[STATE_KEY] = [{"name": f.name, "bytes": f.getvalue()} for f in new_uploads[:3]]

# Controls + info
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

# Plotly config
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": False,
}

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

# ----------------------------
# Sidebar: Ayarlar + Hızlı seçim
# ----------------------------
st.sidebar.markdown("### Ayarlar")

ymin, ymax = int(all_years[0]), int(all_years[-1])

# Persisted year range state
YR_KEY = "demand_year_range_v1"
if YR_KEY not in st.session_state:
    st.session_state[YR_KEY] = (ymin, ymax)

st.sidebar.markdown("**Hızlı seçim**")
b1, b2 = st.sidebar.columns(2)
with b1:
    if st.button("Net Zero (2025–2050)", use_container_width=True, key="demand_quick_netzero"):
        st.session_state[YR_KEY] = (max(2025, ymin), min(2050, ymax))
with b2:
    if st.button("TUEP (2025–2035)", use_container_width=True, key="demand_quick_tuep"):
        st.session_state[YR_KEY] = (max(2025, ymin), min(2035, ymax))

# slider: step=5
y0, y1 = st.sidebar.slider(
    "Senaryo yıl aralığı",
    min_value=ymin,
    max_value=ymax,
    value=st.session_state[YR_KEY],
    step=5,
    key="demand_year_slider_v1",
)
st.session_state[YR_KEY] = (y0, y1)

# ----------------------------
# Layout: Tek senaryoda tek kolon, 2+ senaryoda 2 kolon
# ----------------------------
n_cards = len(pre_read)
n_cols = 1 if n_cards == 1 else 2
cols = st.columns(n_cols, gap="large")

for i, (fname, years, mat, codes, df_raw, err) in enumerate(pre_read):
    with cols[i % n_cols]:
        scenario = _scenario_from_filename(fname)
        st.subheader(scenario)

        if err or not years or mat is None or df_raw is None:
            st.error(f"Dosya: {fname}\n\nHata: {err or 'Dosya okunamadı veya yıl satırı bulunamadı.'}")
            continue

        n_years = len(years)

        # ============================================================
        # A) SUMMARY (NOT 100% normalized)  -> % of TOTAL ELC
        # ============================================================
        total_elc = _total_elc_from_B_filter(df_raw, n_years=n_years)

        tr_ele_map = _build_transport_electric_series(mat, codes)
        transport_ele_total_abs = _sum_series_map(tr_ele_map)

        household_cooling_abs = mat.iloc[[234 - 1], :].sum(axis=0).tolist() if len(mat) >= 234 else [0.0] * n_years
        services_dc_abs = mat.iloc[[578 - 1], :].sum(axis=0).tolist() if len(mat) >= 578 else [0.0] * n_years

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
                barnorm_percent=False,  # IMPORTANT: not 100% normalized
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
                y_tickformat=",0f",
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
                barnorm_percent=True,  # distribution (100%)
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
                y_tickformat=",0f",
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
                barnorm_percent=True,  # distribution (100%)
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_sv_pct",
        )

        # ============================================================
        # D) Transport (top-level) abs + % distribution (using row numbers)
        # ============================================================
        tr_top_abs = _build_transport_top_series(mat)
        years_tr, tr_top_abs_f = _filter_years(years, tr_top_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar(
                years_tr, tr_top_abs_f,
                "Ulaştırma Elektrik Tüketimi (GWh) – Mutlak",
                y_title="GWh",
                y_tickformat=",0f",
                use_percent_suffix=False,
                barnorm_percent=False,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_tr_abs",
        )

        st.plotly_chart(
            _stacked_bar(
                years_tr, tr_top_abs_f,
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
    "Not: Yüklenen dosyalar sayfa geçişlerinde kaybolmaması için oturum hafızasında tutulur. "
    "Senaryo adı dosya isminden `Demand_..._tria..` veya `FinalReport_..._tria..` kuralıyla çıkarılır; `b_` korunur."
)
