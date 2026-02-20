import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Soğutma ve Veri Merkezleri Analizi", layout="wide")

# ----------------------------
# Transport labels (ROW-based; safest)
# ----------------------------
TRANSPORT_ROWS_1IDX = [220, 224, 229, 230, 541, 546, 551, 558, 560, 563, 564, 565]

TRANSPORT_ROW_LABEL_TR = {
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


def _read_final_energy(filelike) -> Tuple[List[int], pd.DataFrame, List[str], pd.DataFrame]:
    """
    Reads sheet FINAL_ENERGY without headers.
    Years are in row 1 (Excel) starting from column C.
    Returns:
      years: list[int]
      mat: numeric matrix for year columns (C..)
      codes: column A strings for each row
      df_raw: raw dataframe (needed for B-column ELC filter totals)
    """
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
    *,
    y_title: str,
    y_tickformat: str,
    use_percent_suffix: bool = False,
    barnorm_percent: bool = False,
) -> go.Figure:
    """
    General stacked bar.
    - If barnorm_percent=True -> Plotly normalizes to 100% (NOT what we want for summary now).
    - Otherwise, uses given y values as-is.
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
            tickformat=y_tickformat,
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        ),
    )

    if barnorm_percent:
        layout_kwargs["barnorm"] = "percent"
        layout_kwargs["yaxis"] = dict(
            title=y_title,
            range=[0, 100],
            ticksuffix="%" if use_percent_suffix else "",
            tickformat=y_tickformat,
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        )

    fig.update_layout(**layout_kwargs)

    # Hover
    if use_percent_suffix:
        for tr in fig.data:
            tr.update(
                hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:.0f}%<extra></extra>"
            )
    else:
        for tr in fig.data:
            tr.update(
                hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:,.0f}<extra></extra>"
            )

    return fig


def _series_from_rows_with_rowlabels(
    mat: pd.DataFrame,
    excel_rows_1idx: List[int],
    row_label_map: Dict[int, str],
) -> Dict[str, List[float]]:
    """
    Build series map from exact Excel rows, labels taken from row_label_map (row-index based).
    """
    out: Dict[str, List[float]] = {}
    for r in excel_rows_1idx:
        i = r - 1
        if i < 0 or i >= len(mat):
            continue
        label = row_label_map.get(r, f"Satır {r}")
        # avoid duplicates
        if label in out:
            label = f"{label} ({r})"
        out[label] = mat.iloc[i, :].tolist()
    return out


def _total_elc_from_B_filter(df_raw: pd.DataFrame, n_years: int) -> List[float]:
    """
    Total electricity demand = sum of all rows where column B == 'ELC'
    across year columns (C..).
    """
    if df_raw is None or df_raw.empty:
        return [0.0] * n_years

    b = df_raw.iloc[:, 1].astype(str).str.strip().str.upper()
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


# ----------------------------
# UI
# ----------------------------
st.title("Soğutma ve Veri Merkezleri Analizi")
st.caption("Demand Excel dosyalarını (1–3 senaryo) yükle. Grafikler FINAL_ENERGY sekmesinden okunur.")

STATE_KEY = "demand_files_v_final_fix"
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = []

new_uploads = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
    key="demand_uploader_v_final_fix",
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

        # ----------------------------
        # 1) SUMMARY (NOT 100% normalized)
        # y values already percent contribution to total electricity
        # ----------------------------
        total_elc = _total_elc_from_B_filter(df_raw, n_years=n_years)

        transport_abs_total = _rows_sum(mat, TRANSPORT_ROWS_1IDX)
        household_cooling_abs = _rows_sum(mat, [234])
        services_dc_abs = _rows_sum(mat, [578])

        summary_pct_values = {
            "Veri Merkezleri": _to_share_percent(services_dc_abs, total_elc),
            "Soğutma": _to_share_percent(household_cooling_abs, total_elc),
            "Elektrikli Araçlar": _to_share_percent(transport_abs_total, total_elc),
        }
        years_sum, summary_pct_f = _filter_years(years, summary_pct_values, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
                years_sum,
                summary_pct_f,
                "Veri merkezi, Soğutma ve Elektrikli Araçların Nihai Elektrik Talebine Katkısı (%)",
                y_title="%",
                y_tickformat=".0f",
                use_percent_suffix=True,
                barnorm_percent=False,  # IMPORTANT: no 100% normalization
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_summary_contrib_pct",
        )

        # ----------------------------
        # 2) Household (abs + % normalized)
        # ----------------------------
        hh_abs = _build_household_series(mat)
        years_hh, hh_abs_f = _filter_years(years, hh_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
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
            _stacked_bar_iea_like(
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

        # ----------------------------
        # 3) Services (abs + % normalized)
        # ----------------------------
        sv_abs = _build_services_series(mat)
        years_sv, sv_abs_f = _filter_years(years, sv_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
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
            _stacked_bar_iea_like(
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

        # ----------------------------
        # 4) Transport (ROW-labeled; abs + % normalized)
        # ----------------------------
        tr_abs = _series_from_rows_with_rowlabels(mat, TRANSPORT_ROWS_1IDX, TRANSPORT_ROW_LABEL_TR)
        years_tr, tr_abs_f = _filter_years(years, tr_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
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
            _stacked_bar_iea_like(
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
    "Ulaştırma grafikleri, belirtilen satırlardan okunur ve legend satır numarasına göre Türkçe isimlendirilir."
)
