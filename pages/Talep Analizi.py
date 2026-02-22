import re
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Elektrik Talebinin Yeni Dinamikleri", layout="wide")
st.caption("Soğutma • Veri Merkezleri • Elektrikli Araçlar")

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
# Helpers
# ============================================================
def _scenario_from_filename(name: str) -> str:
    base = re.sub(r"\.[^.]+$", "", name)
    for pref in ("Demand_", "FinalReport_"):
        if base.startswith(pref):
            base = base[len(pref) :]
            break
    m = re.search(r"(_tria.*)$", base, flags=re.IGNORECASE)
    if m:
        base = base[: m.start()]
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
        df.iloc[:, 2 : 2 + n_years]
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

def _stacked_bar_with_total_line(
    years: List[int],
    series_map: Dict[str, List[float]],
    total_line: List[float],
    title: str,
    *,
    y_title: str,
    y_tickformat: str,
) -> go.Figure:
    """Stacked bar (components) + total line on the same y-axis."""
    fig = _stacked_bar(
        years,
        series_map,
        title,
        y_title=y_title,
        y_tickformat=y_tickformat,
        use_percent_suffix=False,
        barnorm_percent=False,
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=total_line,
            name="Toplam Nihai Elektrik Talebi",
            mode="lines+markers",
        )
    )

    # Hover: keep bars as-is; customize total line for clarity
    fig.data[-1].update(hovertemplate="%{x}<br>Toplam: %{y:,.0f}<extra></extra>")

    # Ensure the line appears on top
    fig.update_layout(barmode="stack", hovermode="x unified")
    return fig


def _total_elc_from_B_filter(df_raw: pd.DataFrame, n_years: int) -> List[float]:
    if df_raw is None or df_raw.empty:
        return [0.0] * n_years
    b = df_raw.iloc[:, 1].astype(str).str.strip().str.upper()
    mask = b.eq("ELC")
    mat_years = (
        df_raw.iloc[:, 2 : 2 + n_years]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    if mask.sum() == 0:
        return [0.0] * n_years
    return mat_years.loc[mask].sum(axis=0).tolist()


def _to_share_percent(numer: List[float], denom: List[float]) -> List[float]:
    return [(n / d * 100.0) if d else 0.0 for n, d in zip(numer, denom)]


def _sum_by_code(mat: pd.DataFrame, codes: List[str], code: str) -> List[float]:
    if mat is None or mat.empty or not codes:
        return [0.0] * (mat.shape[1] if mat is not None else 0)
    codes_s = pd.Series(codes).astype(str).str.strip().str.upper()
    target = code.strip().upper()
    idxs = codes_s[codes_s == target].index.tolist()
    if not idxs:
        return [0.0] * mat.shape[1]
    return mat.iloc[idxs, :].sum(axis=0).tolist()


def _build_household_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    def safe_sum(rows_1idx: List[int]) -> List[float]:
        idx = [r - 1 for r in rows_1idx if 0 <= (r - 1) < len(mat)]
        if not idx:
            return [0.0] * mat.shape[1]
        return mat.iloc[idx, :].sum(axis=0).tolist()

    return {
        "Ev Aletleri": safe_sum([235, 253, 241]),
        "Alan Isıtma": safe_sum([245, 248]),
        "Su Isıtma": safe_sum([260, 262]),
        "Pişirme": safe_sum([236]),
        "Soğutma": safe_sum([234]),
    }


def _build_services_series(mat: pd.DataFrame) -> Dict[str, List[float]]:
    def safe_sum(rows_1idx: List[int]) -> List[float]:
        idx = [r - 1 for r in rows_1idx if 0 <= (r - 1) < len(mat)]
        if not idx:
            return [0.0] * mat.shape[1]
        return mat.iloc[idx, :].sum(axis=0).tolist()

    return {
        "Veri Merkezleri": safe_sum([578]),
        "Soğutma": safe_sum([577]),
        "Aydınlatma": safe_sum([579]),
        "Alan Isıtma": safe_sum([588, 591]),
        "Su Isıtma": safe_sum([602, 604]),
    }


def _build_transport_electric_series(mat: pd.DataFrame, codes: List[str]) -> Dict[str, List[float]]:
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
st.title("Elektrik Talebinin Yeni Dinamikleri")
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

# Plotly: ZOOM / FULLSCREEN aktif
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": False,   # mouse wheel zoom
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

# Sidebar year range
st.sidebar.markdown("### Ayarlar")
ymin, ymax = int(all_years[0]), int(all_years[-1])

# 5 yıllık adım: mümkünse sadece 5'in katları (örn. 2025, 2030, 2035...)
_year_opts = [y for y in all_years if int(y) % 5 == 0]
if len(_year_opts) < 2:
    _year_opts = all_years[:]  # fallback

def _snap_year(target: int) -> int:
    return min(_year_opts, key=lambda y: abs(int(y) - int(target)))

STATE_YR = "demand_year_range"
if STATE_YR not in st.session_state:
    st.session_state[STATE_YR] = (_year_opts[0], _year_opts[-1])

st.sidebar.markdown("#### Hızlı seçim")
b1, b2 = st.sidebar.columns(2)
if b1.button("Net Zero (2025–2050)", use_container_width=True):
    st.session_state[STATE_YR] = (_snap_year(2025), _snap_year(2050))
    st.rerun()
if b2.button("TUEP (2025–2035)", use_container_width=True):
    st.session_state[STATE_YR] = (_snap_year(2025), _snap_year(2035))
    st.rerun()

# Yıl aralığı (seçim değiştikçe dosyalar silinmez; dosyalar session_state'te tutuluyor)
y0, y1 = st.sidebar.select_slider(
    "Senaryo yıl aralığı",
    options=_year_opts,
    value=st.session_state[STATE_YR],
    key=STATE_YR,
)

# --- FIX: Tek senaryoda boşluk kalmasın
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
        # A) SUMMARY (NOT 100% normalized)
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
                barnorm_percent=False,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_summary_contrib_pct",
        )

        
        # ============================================================
        # A2) SUMMARY (ABS components + TOTAL as line)
        # ============================================================
        summary_abs = {
            "Veri Merkezleri": services_dc_abs,
            "Soğutma": household_cooling_abs,
            "Elektrikli Araçlar": transport_ele_total_abs,
        }
        years_abs, summary_abs_f = _filter_years(years, summary_abs, y0, y1)
        years_tot, tot_map_f = _filter_years(years, {"Toplam": total_elc}, y0, y1)
        total_elc_f = tot_map_f.get("Toplam", [0.0] * len(years_abs))

        st.plotly_chart(
            _stacked_bar_with_total_line(
                years_abs,
                summary_abs_f,
                total_elc_f,
                "Veri Merkezi, Soğutma ve Elektrikli Araçlar (GWh) + Toplam Nihai Elektrik Talebi",
                y_title="GWh",
                y_tickformat=",.0f",
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key=f"demand_{i}_summary_contrib_abs_total",
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
                barnorm_percent=True,
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
    "Ulaştırma grafikleri A sütunundaki kodlardan (yalnız *_ELE) okunur; satır kayması sorun olmaz. "
    "Zoom için grafik üstünde fare tekerleği (scroll) aktif; modebar açık."
)
