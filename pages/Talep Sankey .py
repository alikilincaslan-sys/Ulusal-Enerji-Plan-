import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Talep Akışı (Sankey)")

SHEET = "SECTORAL_ENERGY"

# =========================
# 1) SEKTÖR SÖZLÜĞÜ (senin verdiğin)
# =========================
SECTOR_MAP = {
    "FERRO": "Demir Çelik",
    "NONFER": "Demir Dışı Metaller",
    "CHEM": "Kimya",
    "NMETM": "Metal olmayan Mineral Ürünleri",
    "PAPP": "Kağıt",
    "FDDRTB": "Yemek Sektörü",
    "ENGNR": "Mühendislik",
    "TEXTL": "Tekstil",
    "OTHR": "Diğer Sanayi",
    "REFIN": "Rafineri",
    "HOU": "Konutlar",
    "TER": "Tarım ve Hizmetler",
    "PSTRA": "Yolcu Taşımacılığı",
    "AIRTRA": "Havayolu Taşımacılığı",
    "FRTRA": "Yük Taşımacılığı",
}

# =========================
# 2) 5 ANA GRUP (senin mantığın)
# =========================
SECTOR_GROUP_MAP = {
    # Sanayi
    "Demir Çelik": "Sanayi",
    "Demir Dışı Metaller": "Sanayi",
    "Kimya": "Sanayi",
    "Metal olmayan Mineral Ürünleri": "Sanayi",
    "Kağıt": "Sanayi",
    "Yemek Sektörü": "Sanayi",
    "Mühendislik": "Sanayi",
    "Tekstil": "Sanayi",
    "Diğer Sanayi": "Sanayi",
    "Rafineri": "Sanayi",

    # Konut
    "Konutlar": "Konutlar",

    # Ulaştırma
    "Yolcu Taşımacılığı": "Ulaştırma",
    "Havayolu Taşımacılığı": "Ulaştırma",
    "Yük Taşımacılığı": "Ulaştırma",

    # Tarım+Hizmetler (dosyada birleşik)
    "Tarım ve Hizmetler": "Tarım & Hizmetler",
}

# Yakıt isimleri Excel’de nasıl geliyorsa ona göre yakıt gruplama
def fuel_group(fuel: str) -> str:
    f = str(fuel).strip().lower()

    if "electric" in f:
        return "Elektrik"
    if "liquid" in f:
        return "Sıvı Yakıtlar"
    if "solid" in f or "coal" in f or "lignite" in f:
        return "Katı Yakıtlar"
    if "gas" in f:
        return "Gazlar"
    if "biomass" in f or "waste" in f:
        return "Biyokütle & Atık"
    if "steam" in f or "heat" in f:
        return "Isı / Buhar"
    if f == "res" or "renew" in f:
        return "Diğer"
    return "Diğer"


st.title("Enerji Talebi Sankey Diagramı TEST")

# =========================
# 3) Excel yükleme + Görselleştirme ayarları
# =========================
with st.sidebar:
    st.header("📁 Veri Kaynağı")
    uploaded = st.file_uploader(
        "TUEP Excel dosyasını yükle (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=False
    )
    st.divider()

    st.subheader("⚙️ Görselleştirme")
    grouped_mode = st.toggle("Özet (Gruplu Sankey) önerilen", value=True)
    min_share = st.slider("Küçük akış eşiği (Toplamın %'si)", 0.0, 5.0, 1.0, 0.25)
    st.caption("Eşik, çok küçük linkleri 'Diğer'leştirir ve diyagramı temizler.")

    st.divider()
    st.subheader("✨ Tableau tarzı vurgu")
    highlight_mode = st.selectbox("Highlight tipi", ["Yok", "Sektör Grubu", "Yakıt Grubu"], index=0)
    highlight_value = None
    if highlight_mode != "Yok":
        highlight_value = st.selectbox(
            "Vurgulanacak değer",
            # seçenekler veri gelince main'de güncellenecek; şimdilik placeholder
            ["(veri yüklenince gelecek)"],
            index=0
        )


def apply_maps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sector"] = out["Sector"].map(SECTOR_MAP).fillna(out["Sector"])
    out["SectorGroup"] = out["Sector"].map(SECTOR_GROUP_MAP).fillna("Diğer")
    out["FuelGroup"] = out["Fuel"].apply(fuel_group)
    return out


@st.cache_data
def load_sectoral_energy_from_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    xls = pd.ExcelFile(file_bytes)
    if SHEET not in xls.sheet_names:
        raise ValueError(f"'{SHEET}' sekmesi bulunamadı. Mevcut sekmeler: {xls.sheet_names}")

    df = pd.read_excel(xls, sheet_name=SHEET)

    # İlk iki kolon bazen isimsiz gelir: Sector/Fuel
    if "Sector" not in df.columns or "Fuel" not in df.columns:
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: "Sector", cols[1]: "Fuel"})

    # Yıl kolonlarını yakala + standardize et
    year_cols = []
    rename = {}
    for c in df.columns:
        s = str(c).strip().replace(".0", "")
        if s.isdigit():
            rename[c] = s
            year_cols.append(s)

    if rename:
        df = df.rename(columns=rename)

    for y in year_cols:
        df[y] = pd.to_numeric(df[y], errors="coerce")

    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Fuel"] = df["Fuel"].astype(str).str.strip()

    return df, year_cols


def _apply_threshold_to_links(links: pd.DataFrame, total_value: float, min_share_pct: float) -> pd.DataFrame:
    if total_value <= 0 or min_share_pct <= 0:
        return links

    thr = (min_share_pct / 100.0) * total_value
    small = links["value"] < thr
    if not small.any():
        return links

    links2 = links.copy()
    links2.loc[small, "target_label"] = "Diğer"
    links2 = links2.groupby(["source_label", "target_label"], as_index=False)["value"].sum()
    return links2


# =========================
# 4) Tableau-look Sankey (Gruplu)
# =========================
def build_plotly_sankey_grouped_tableau(df: pd.DataFrame, year: str, min_share_pct: float,
                                       highlight_mode: str = "Yok", highlight_value: str | None = None):
    import plotly.graph_objects as go

    df_tot = df[df["Fuel"].str.lower() == "total"].copy()
    df_fuels = df[df["Fuel"].str.lower() != "total"].copy()

    df_tot[year] = df_tot[year].fillna(0).clip(lower=0)
    df_fuels[year] = df_fuels[year].fillna(0).clip(lower=0)

    total_value = float(df_tot[year].sum())
    if total_value <= 0:
        raise ValueError("Seçili yıl için toplam değer 0 görünüyor (Total satırlarını kontrol et).")

    root = "TOPLAM NİHAİ ENERJİ"

    # Root -> SectorGroup
    g1 = (
        df_tot.groupby("SectorGroup", as_index=False)[year]
        .sum()
        .rename(columns={"SectorGroup": "target_label", year: "value"})
    )
    g1["source_label"] = root
    links_1 = g1[["source_label", "target_label", "value"]]
    links_1 = links_1[links_1["value"] > 0]
    links_1 = _apply_threshold_to_links(links_1, total_value, min_share_pct)

    # SectorGroup -> FuelGroup
    g2 = (
        df_fuels.groupby(["SectorGroup", "FuelGroup"], as_index=False)[year]
        .sum()
        .rename(columns={"SectorGroup": "source_label", "FuelGroup": "target_label", year: "value"})
    )
    links_2 = g2[["source_label", "target_label", "value"]]
    links_2 = links_2[links_2["value"] > 0]
    links_2 = _apply_threshold_to_links(links_2, total_value, min_share_pct)

    # Düğümler (sıralı)
    sector_order = ["Sanayi", "Konutlar", "Tarım & Hizmetler", "Ulaştırma", "Diğer"]
    fuel_order = ["Elektrik", "Gazlar", "Sıvı Yakıtlar", "Katı Yakıtlar", "Biyokütle & Atık", "Isı / Buhar", "Diğer"]

    sectors = [s for s in sector_order if s in set(links_1["target_label"]) or s in set(links_2["source_label"])]
    fuels = [f for f in fuel_order if f in set(links_2["target_label"])]

    labels = [root] + sectors + fuels
    idx = {name: i for i, name in enumerate(labels)}

    # Tableau hissi: node renkleri + linkler source rengine göre şeffaf
    node_color_map = {
        root: "rgba(120,120,120,0.90)",

        "Sanayi": "rgba(31,119,180,0.95)",
        "Konutlar": "rgba(255,127,14,0.95)",
        "Tarım & Hizmetler": "rgba(44,160,44,0.95)",
        "Ulaştırma": "rgba(214,39,40,0.95)",
        "Diğer": "rgba(148,103,189,0.95)",

        "Elektrik": "rgba(23,190,207,0.95)",
        "Gazlar": "rgba(127,127,127,0.95)",
        "Sıvı Yakıtlar": "rgba(255,187,120,0.95)",
        "Katı Yakıtlar": "rgba(174,199,232,0.95)",
        "Biyokütle & Atık": "rgba(152,223,138,0.95)",
        "Isı / Buhar": "rgba(199,199,199,0.95)",
    }
    node_colors = [node_color_map.get(l, "rgba(160,160,160,0.90)") for l in labels]

    def link_rgba_from_source(src_label: str, alpha: float) -> str:
        c = node_color_map.get(src_label, "rgba(160,160,160,0.95)")
        try:
            head = c.split("rgba(")[1].split(")")[0]
            r, g, b, _a = [x.strip() for x in head.split(",")]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(160,160,160,{alpha})"

    def is_highlighted(src_label: str, tgt_label: str) -> bool:
        if highlight_mode == "Yok" or not highlight_value:
            return True
        hv = highlight_value
        # Gruplu modda hem sektör grubu hem yakıt grubu için aynı mantık: node'a değiyorsa açık
        return (src_label == hv) or (tgt_label == hv)

    sources, targets, values, link_colors, hover = [], [], [], [], []

    def add_links(links_df: pd.DataFrame):
        for _, r in links_df.iterrows():
            s = r["source_label"]
            t = r["target_label"]
            v = float(r["value"])
            if v <= 0:
                continue

            # ensure in idx
            if s not in idx:
                idx[s] = len(labels)
                labels.append(s)
                node_colors.append(node_color_map.get(s, "rgba(160,160,160,0.90)"))
            if t not in idx:
                idx[t] = len(labels)
                labels.append(t)
                node_colors.append(node_color_map.get(t, "rgba(160,160,160,0.90)"))

            sources.append(idx[s])
            targets.append(idx[t])
            values.append(v)

            on = is_highlighted(s, t)
            link_colors.append(link_rgba_from_source(s, 0.55 if on else 0.10))
            hover.append(f"{s} → {t}<br><b>{v:,.0f}</b>")

    add_links(links_1)
    add_links(links_2)

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=16, thickness=16,
                label=labels,
                color=node_colors,
                line=dict(color="rgba(0,0,0,0)", width=0)  # border yok
            ),
            link=dict(
                source=sources, target=targets, value=values,
                color=link_colors,
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>"
            ),
        )]
    )

    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=60, b=10),
        title_text=f"{year} — Talep Akışı (Toplam → 5 Grup → Yakıt Grupları)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Georgia, Times New Roman, serif", size=13),
    )

    totals = (
        df_tot.groupby("SectorGroup", as_index=False)[year].sum()
        .sort_values(year, ascending=False)
        .rename(columns={"SectorGroup": "Sektör Grubu", year: "GWh"})
        .reset_index(drop=True)
    )
    return fig, totals


# =========================
# 5) Detay Sankey (eski) + basit highlight
# =========================
def build_plotly_sankey_detail(df: pd.DataFrame, year: str, highlight_value: str | None = None):
    import plotly.graph_objects as go

    df_tot = df[df["Fuel"].str.lower() == "total"].copy()
    df_fuels = df[df["Fuel"].str.lower() != "total"].copy()

    df_tot[year] = df_tot[year].fillna(0).clip(lower=0)
    df_fuels[year] = df_fuels[year].fillna(0).clip(lower=0)

    df_tot = df_tot[df_tot[year] > 0]
    df_fuels = df_fuels[df_fuels[year] > 0]

    root = "TOPLAM NİHAİ ENERJİ"
    sectors = sorted(df_tot["Sector"].unique().tolist())

    fuel_nodes = (df_fuels["Sector"] + " | " + df_fuels["Fuel"]).unique().tolist()

    labels = [root] + sectors + fuel_nodes
    idx = {name: i for i, name in enumerate(labels)}

    def on_link(src_label: str, tgt_label: str) -> bool:
        if not highlight_value:
            return True
        hv = highlight_value
        return (hv in src_label) or (hv in tgt_label)

    sources, targets, values, link_colors, hover = [], [], [], [], []

    def link_color(alpha: float) -> str:
        return f"rgba(120,120,120,{alpha})"

    for _, r in df_tot.iterrows():
        s = r["Sector"]
        v = float(r[year])
        if v > 0 and s in idx:
            sources.append(idx[root])
            targets.append(idx[s])
            values.append(v)
            ok = on_link(root, s)
            link_colors.append(link_color(0.45 if ok else 0.10))
            hover.append(f"{root} → {s}<br><b>{v:,.0f}</b>")

    for _, r in df_fuels.iterrows():
        s = r["Sector"]
        fnode = f"{r['Sector']} | {r['Fuel']}"
        v = float(r[year])
        if v > 0 and s in idx and fnode in idx:
            sources.append(idx[s])
            targets.append(idx[fnode])
            values.append(v)
            ok = on_link(s, fnode)
            link_colors.append(link_color(0.45 if ok else 0.10))
            hover.append(f"{s} → {fnode}<br><b>{v:,.0f}</b>")

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=12, thickness=14, label=labels, line=dict(color="rgba(0,0,0,0)", width=0)),
            link=dict(
                source=sources, target=targets, value=values,
                color=link_colors,
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>"
            ),
        )]
    )
    fig.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=55, b=10),
        title_text=f"{year} — Talep Akışı (Detay: Toplam → Sektör → Yakıt)",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Georgia, Times New Roman, serif", size=12),
    )

    totals = df_tot[["Sector", year]].sort_values(year, ascending=False).reset_index(drop=True)
    return fig, totals


def fallback_view(df: pd.DataFrame, year: str):
    st.warning("Plotly bulunamadı / Sankey çizilemedi. Aynı akışı fallback görünümle gösteriyorum.")

    df_tot = df[df["Fuel"].str.lower() == "total"].copy()
    df_fuels = df[df["Fuel"].str.lower() != "total"].copy()

    df_tot[year] = df_tot[year].fillna(0).clip(lower=0)
    df_fuels[year] = df_fuels[year].fillna(0).clip(lower=0)

    df_tot = df_tot[df_tot[year] > 0]
    df_fuels = df_fuels[df_fuels[year] > 0]

    df_tot_g = df_tot.groupby("SectorGroup", as_index=False)[year].sum().sort_values(year, ascending=False)
    df_fuels_g = df_fuels.groupby(["SectorGroup", "FuelGroup"], as_index=False)[year].sum()

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Toplam → 5 Grup")
        st.dataframe(df_tot_g.rename(columns={"SectorGroup": "Grup", year: "GWh"}), use_container_width=True, height=560)

    with c2:
        st.subheader("Grup → Yakıt Grupları")
        grp = st.selectbox("Grup seç", df_tot_g["SectorGroup"].tolist())
        sub = df_fuels_g[df_fuels_g["SectorGroup"] == grp].sort_values(year, ascending=False)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(sub["FuelGroup"], sub[year].values)
        ax.set_ylabel("GWh")
        ax.set_title(f"{grp} — Yakıt Dağılımı ({year})")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig, use_container_width=True)


# =========================
# 6) ÇALIŞTIR
# =========================
if uploaded is None:
    st.info("Sol menüden bir .xlsx yükle. (TUEP senaryo dosyan)")
    st.stop()

try:
    file_bytes = uploaded.getvalue()
    df_raw, years = load_sectoral_energy_from_bytes(file_bytes)
except Exception as e:
    st.error(f"Excel okunamadı: {e}")
    st.stop()

if not years:
    st.error("SECTORAL_ENERGY sekmesinde yıl kolonları bulunamadı.")
    st.stop()

df = apply_maps(df_raw)

default_year = "2035" if "2035" in years else years[-1]
year = st.selectbox("Yıl", years, index=years.index(default_year))

# Highlight seçeneklerini veri geldikten sonra sidebar'da güncellemek için:
# (Streamlit'te sidebar elemanları üstte çiziliyor; burada değer listesini sağ tarafta da sunuyoruz.)
if grouped_mode:
    # veri tabanlı seçenek listesi
    sector_opts = ["Sanayi", "Konutlar", "Tarım & Hizmetler", "Ulaştırma", "Diğer"]
    fuel_opts = ["Elektrik", "Gazlar", "Sıvı Yakıtlar", "Katı Yakıtlar", "Biyokütle & Atık", "Isı / Buhar", "Diğer"]
    st.caption("💡 Vurgu için önerilen değerler: " + ", ".join([o for o in sector_opts + fuel_opts if o]))
else:
    st.caption("💡 Detay modda vurgu: Sektör adı veya Fuel adı yaz (örn: Konutlar / Electricity / Gas).")

# Sidebar selectbox placeholder nedeniyle highlight_value'yu burada daha sağlam bir şekilde alalım:
with st.sidebar:
    if highlight_mode != "Yok":
        if grouped_mode and highlight_mode in ["Sektör Grubu", "Yakıt Grubu"]:
            options = [""] + (["Sanayi", "Konutlar", "Tarım & Hizmetler", "Ulaştırma", "Diğer"] if highlight_mode == "Sektör Grubu"
                              else ["Elektrik", "Gazlar", "Sıvı Yakıtlar", "Katı Yakıtlar", "Biyokütle & Atık", "Isı / Buhar", "Diğer"])
            highlight_value = st.selectbox("Vurgulanacak değer", options, index=0)
            highlight_value = highlight_value.strip() or None
        else:
            highlight_value = st.text_input("Vurgu metni (detay)", "")
            highlight_value = highlight_value.strip() or None

try:
    import plotly  # var mı?
    if grouped_mode:
        fig, totals = build_plotly_sankey_grouped_tableau(
            df, year, min_share,
            highlight_mode=highlight_mode,
            highlight_value=highlight_value
        )
    else:
        fig, totals = build_plotly_sankey_detail(df, year, highlight_value=highlight_value)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Toplamlar (kontrol)"):
        st.dataframe(totals, use_container_width=True)

except Exception:
    fallback_view(df, year)