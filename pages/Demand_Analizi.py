import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Soğutma ve Veri Merkezleri Analizi", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
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


def _read_final_energy_matrix(filelike) -> Tuple[List[int], pd.DataFrame]:
    df = pd.read_excel(filelike, sheet_name="FINAL_ENERGY", header=None, engine=None)

    years_raw = df.iloc[0, 2:].tolist()  # row 1, from col C
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
        df.iloc[:, 2 : 2 + n_years]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return years, mat


def _rows_sum(mat: pd.DataFrame, excel_rows_1idx: List[int]) -> List[float]:
    idxs = [r - 1 for r in excel_rows_1idx]
    idxs = [i for i in idxs if 0 <= i < len(mat)]
    if not idxs:
        return [0.0] * mat.shape[1]
    return mat.iloc[idxs, :].sum(axis=0).tolist()


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


def _stacked_bar_iea_like(
    years: List[int],
    series_map: Dict[str, List[float]],
    title: str,
    percent: bool = False,
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
            title="%" if percent else "GWh",
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        ),
    )

    if percent:
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
        for tr in fig.data:
            tr.update(
                hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:.0f}%<extra></extra>"
            )
    else:
        layout_kwargs["yaxis"] = dict(
            title="GWh",
            tickformat=",.0f",
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        )
        fig.update_layout(**layout_kwargs)
        for tr in fig.data:
            tr.update(
                hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:,.0f} GWh<extra></extra>"
            )

    return fig


# ----------------------------
# UI
# ----------------------------
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
    # limit to 3, and persist bytes (so it survives page switches)
    persisted = [{"name": f.name, "bytes": f.getvalue()} for f in new_uploads[:3]]
    st.session_state[STATE_KEY] = persisted

# Controls + info
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Yüklenen Excel Dosyalarını temizle", use_container_width=True):
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

# read once to get year range + matrices
all_years: List[int] = []
pre_read = []
for item in files:
    try:
        years, mat = _read_final_energy_matrix(BytesIO(item["bytes"]))
        pre_read.append((item["name"], years, mat))
        all_years.extend(years)
    except Exception:
        pre_read.append((item["name"], [], None))

all_years = sorted(set(all_years))
if not all_years:
    st.error("FINAL_ENERGY yıl satırı bulunamadı (1. satır, C sütunundan başlamalı).")
    st.stop()

# Sidebar year range slider
st.sidebar.markdown("### Ayarlar")
ymin, ymax = all_years[0], all_years[-1]
y0, y1 = st.sidebar.slider(
    "Senaryo yıl aralığı",
    min_value=int(ymin),
    max_value=int(ymax),
    value=(int(ymin), int(ymax)),
    step=1,
)

# Layout
n = len(files)
n_cols = 3 if n == 3 else 2
cols = st.columns(n_cols, gap="large")

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

for i, (fname, years, mat) in enumerate(pre_read):
    col = cols[i] if n_cols == 3 else cols[i % 2]
    with col:
        scenario = _scenario_from_filename(fname)
        st.subheader(scenario)

        if not years or mat is None:
            st.error("Dosya okunamadı veya yıl satırı bulunamadı.")
            continue

        # Household (abs & %)
        hh_abs = _build_household_series(mat)
        years_hh, hh_abs_f = _filter_years(years, hh_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(years_hh, hh_abs_f, "Konutlarda Elektrik Tüketimi (GWh) – Mutlak", percent=False),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.plotly_chart(
            _stacked_bar_iea_like(years_hh, hh_abs_f, "Konutlarda Elektrik Tüketimi (%) – Dağılım", percent=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        # Services (abs & %)
        sv_abs = _build_services_series(mat)
        years_sv, sv_abs_f = _filter_years(years, sv_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(years_sv, sv_abs_f, "Hizmet Sektörü Elektrik Tüketimi (GWh) – Mutlak", percent=False),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        st.plotly_chart(
            _stacked_bar_iea_like(years_sv, sv_abs_f, "Hizmet Sektörü Elektrik Tüketimi (%) – Dağılım", percent=True),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

st.divider()
st.caption(
    "Not: Yüklenen dosyalar sayfa geçişlerinde kaybolmaması için oturum hafızasında tutulur. "
    "Senaryo adı dosya isminden `Demand_..._tria..` veya `FinalReport_..._tria..` kuralıyla çıkarılır; `b_` korunur."
)
