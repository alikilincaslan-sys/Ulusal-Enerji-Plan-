import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Soğutma ve Veri Merkezleri Analizi", layout="wide")

# ----------------------------
# Label dictionary (transport)
# ----------------------------
TRANSPORT_TR = {
    "PSPRV": "Özel yolcu taşımacılığı",
    "PSPBL": "Toplu yolcu taşımacılığı",
    "PSRLS": "Demiryolu yolcu taşımacılığı",
    "PSWTR": "İç su yollarında yolcu taşımacılığı",
    "PSAIR": "Havayolu yolcu taşımacılığı",
    "FRBNK": "Bunker yakıtları",
    "FRTRK": "Karayolu yük taşımacılığı",
    "FRRLS": "Demiryolu yük taşımacılığı",
    "FRWTR": "İç su yollarında yük taşımacılığı",
}

# Excel row numbers (1-indexed) for transport electricity series
TRANSPORT_ROWS_1IDX = [220, 224, 229, 230, 541, 546, 551, 558, 560, 563, 564, 565]


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
            tickformat=".0f",  # ondalık yok
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
            tickformat=",.0f",  # k yok, ondalık yok
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        )
        fig.update_layout(**layout_kwargs)
        for tr in fig.data:
            tr.update(
                hovertemplate="%{x}<br>" + f"{tr.name}: " + "%{y:,.0f} GWh<extra></extra>"
            )

    return fig


def _series_from_rows_with_labels(
    mat: pd.DataFrame,
    codes: List[str],
    excel_rows_1idx: List[int],
    mapping: Dict[str, str],
) -> Dict[str, List[float]]:
    """
    Build series map from exact Excel rows.
    Label comes from column A, translated with mapping.
    mapping uses base code before '_' (e.g., FRRLS_ELE -> FRRLS).
    """
    out: Dict[str, List[float]] = {}
    for r in excel_rows_1idx:
        i = r - 1
        if i < 0 or i >= len(codes) or i >= len(mat):
            continue

        raw = (codes[i] or "").strip()
        base = raw.split("_")[0] if raw else f"ROW_{r}"
        label = mapping.get(base, raw if raw else base)

        if label in out:
            label = f"{label} ({base})"

        out[label] = mat.iloc[i, :].tolist()
    return out


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


# ----------------------------
# UI
# ----------------------------
st.title("Soğutma ve Veri Merkezleri Analizi")
st.caption("Demand Excel dosyalarını (1–3 senaryo) yükle. Grafikler FINAL_ENERGY sekmesinden okunur.")

STATE_KEY = "demand_files_v_final"
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = []  # list[{"name": str, "bytes": bytes}]

new_uploads = st.file_uploader(
    "Demand Excel dosyaları (en az 1, en fazla 3)",
    type=["xlsx", "xlsm", "xls"],
    accept_multiple_files=True,
    key="demand_uploader_v_final",
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

# Pre-read for year universe + matrices
all_years: List[int] = []
pre_read: List[Tuple[str, List[int], pd.DataFrame, List[str], pd.DataFrame, str]] = []  # (name, years, mat, codes, df_raw, err)

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

# Layout: always 2 columns (3. senaryo altta görünür)
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
        # NEW SUMMARY: stacked % shares vs total electricity demand
        # Denominator: sum of all rows where column B == ELC
        # ----------------------------
        total_elc = _total_elc_from_B_filter(df_raw, n_years=n_years)

        transport_abs_total = _rows_sum(mat, TRANSPORT_ROWS_1IDX)
        household_cooling_abs = _rows_sum(mat, [234])
        services_dc_abs = _rows_sum(mat, [578])

        summary_pct = {
            "Ulaştırma / Toplam Elektrik": _to_share_percent(transport_abs_total, total_elc),
            "Konut Soğutma / Toplam Elektrik": _to_share_percent(household_cooling_abs, total_elc),
            "Hizmet Veri Merkezleri / Toplam Elektrik": _to_share_percent(services_dc_abs, total_elc),
        }
        years_sum, summary_pct_f = _filter_years(years, summary_pct, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
                years_sum,
                summary_pct_f,
                "Özet Paylar (% of Toplam Elektrik Talebi) – Yığılmış",
                percent=True,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_summary_pct",
        )

        # 1) Household
        hh_abs = _build_household_series(mat)
        years_hh, hh_abs_f = _filter_years(years, hh_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
                years_hh, hh_abs_f,
                "Konutlarda Elektrik Tüketimi (GWh) – Mutlak",
                percent=False
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_hh_abs",
        )
        st.plotly_chart(
            _stacked_bar_iea_like(
                years_hh, hh_abs_f,
                "Konutlarda Elektrik Tüketimi (%) – Dağılım",
                percent=True
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_hh_pct",
        )

        # 2) Services
        sv_abs = _build_services_series(mat)
        years_sv, sv_abs_f = _filter_years(years, sv_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
                years_sv, sv_abs_f,
                "Hizmet Sektörü Elektrik Tüketimi (GWh) – Mutlak",
                percent=False
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_sv_abs",
        )
        st.plotly_chart(
            _stacked_bar_iea_like(
                years_sv, sv_abs_f,
                "Hizmet Sektörü Elektrik Tüketimi (%) – Dağılım",
                percent=True
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_sv_pct",
        )

        # 3) Transport (rows) – abs and %
        tr_abs = _series_from_rows_with_labels(mat, codes, TRANSPORT_ROWS_1IDX, TRANSPORT_TR)
        years_tr, tr_abs_f = _filter_years(years, tr_abs, y0, y1)

        st.plotly_chart(
            _stacked_bar_iea_like(
                years_tr, tr_abs_f,
                "Ulaştırma Elektrik Tüketimi (GWh) – Mutlak",
                percent=False
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_tr_abs",
        )
        st.plotly_chart(
            _stacked_bar_iea_like(
                years_tr, tr_abs_f,
                "Ulaştırma Elektrik Tüketimi (%) – Dağılım",
                percent=True
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_tr_pct",
        )

st.divider()
st.caption(
    "Not: Yüklenen dosyalar sayfa geçişlerinde kaybolmaması için oturum hafızasında tutulur. "
    "Senaryo adı dosya isminden `Demand_..._tria..` veya `FinalReport_..._tria..` kuralıyla çıkarılır; `b_` korunur. "
    "Özet grafikte toplam elektrik talebi, FINAL_ENERGY içinde B sütunu 'ELC' olan tüm satırların toplamıdır. "
    "Ulaştırma grafikleri FINAL_ENERGY içinde verilen satırlardan (220,224,229,230,541,546,551,558,560,563,564,565) okunur."
)
