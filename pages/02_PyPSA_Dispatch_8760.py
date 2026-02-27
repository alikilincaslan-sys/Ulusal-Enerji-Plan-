from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import pypsa

st.set_page_config(page_title="PyPSA Dispatch 8760", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

st.title("PyPSA Dispatch (Türkiye Tek-Node, 8760)")

st.markdown("""
Bu sayfa, DataPrep'ten üretilen `profiles_YYYY.parquet` dosyasını okuyup
tek node dispatch optimizasyonu yapar (HiGHS).
""")

# -----------------------------
# Load profiles
# -----------------------------
st.subheader("1) Profil seçimi")

available = sorted(DATA_DIR.glob("profiles_*.parquet"))
if not available:
    st.error("profiles_YYYY.parquet bulunamadı. Önce DataPrep sayfasını çalıştır.")
    st.stop()

profile_file = st.selectbox(
    "Profiles dosyası",
    options=available,
    format_func=lambda p: p.name
)

df = pd.read_parquet(profile_file)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

required_cols = {"load_gross", "load_net", "solar_shape", "wind_shape", "hydro_shape"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Profiles içinde eksik kolonlar var: {sorted(missing)}")
    st.stop()

load_mode = st.radio("Load modu", ["Gross load", "Net load"], horizontal=True, index=0)

load = df["load_gross"] if load_mode == "Gross load" else df["load_net"]
solar_shape = df["solar_shape"].clip(0, 1)
wind_shape  = df["wind_shape"].clip(0, 1)
hydro_shape = df["hydro_shape"].clip(0, 1)

st.caption(f"Seçili dosya: {profile_file.name} | Saat sayısı: {len(df):,}")

# -----------------------------
# Scenario inputs (manual for now)
# -----------------------------
st.subheader("2) Senaryo girdileri (şimdilik manuel)")

with st.expander("Kurulu güçler (MW) ve marjinal maliyetler", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        cap_coal = st.number_input("Coal kapasite (MW)", min_value=0.0, value=20000.0, step=500.0)
        cap_lignite = st.number_input("Lignite kapasite (MW)", min_value=0.0, value=10000.0, step=500.0)
        cap_gas = st.number_input("Natural gas kapasite (MW)", min_value=0.0, value=25000.0, step=500.0)

    with c2:
        cap_hydro = st.number_input("Hydro kapasite (MW)", min_value=0.0, value=30000.0, step=500.0)
        cap_wind  = st.number_input("Wind (RES) kapasite (MW)", min_value=0.0, value=12000.0, step=500.0)
        cap_solar = st.number_input("Solar (GES) kapasite (MW)", min_value=0.0, value=15000.0, step=500.0)

    with c3:
        cap_nuclear = st.number_input("Nuclear kapasite (MW)", min_value=0.0, value=0.0, step=500.0)
        cap_other = st.number_input("Other (dispatchable) kapasite (MW)", min_value=0.0, value=0.0, step=500.0)

    st.markdown("**Marjinal maliyetler** (ör. TL/MWh veya €/MWh — hangisini seçersen tutarlı kal)")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        mc_coal = st.number_input("Coal MC", min_value=0.0, value=55.0, step=1.0)
        mc_lignite = st.number_input("Lignite MC", min_value=0.0, value=45.0, step=1.0)
    with d2:
        mc_gas = st.number_input("Natural gas MC", min_value=0.0, value=75.0, step=1.0)
        mc_other = st.number_input("Other MC", min_value=0.0, value=90.0, step=1.0)
    with d3:
        mc_nuclear = st.number_input("Nuclear MC", min_value=0.0, value=10.0, step=1.0)
        mc_hydro = st.number_input("Hydro MC (opsiyonel)", min_value=0.0, value=5.0, step=1.0)
    with d4:
        mc_wind = st.number_input("Wind MC", min_value=0.0, value=0.0, step=1.0)
        mc_solar = st.number_input("Solar MC", min_value=0.0, value=0.0, step=1.0)

with st.expander("Güvenlik: Load shedding (VOLL)", expanded=False):
    use_voll = st.checkbox("Load shedding kullan (önerilir)", value=True)
    voll = st.number_input("VOLL (çok yüksek maliyet)", min_value=0.0, value=10000.0, step=100.0)

run = st.button("Optimize et (8760)", type="primary")

# -----------------------------
# Build & solve PyPSA
# -----------------------------
if run:
    with st.spinner("Network kuruluyor..."):
        n = pypsa.Network()
        n.set_snapshots(df.index)

        n.add("Bus", "TR")

        # Load (MWh/hour = MW average for that hour)
        n.add("Load", "Load", bus="TR", p_set=load.values)

        # Dispatchable thermal
        if cap_coal > 0:
            n.add("Generator", "Coal", bus="TR", p_nom=cap_coal, marginal_cost=mc_coal)
        if cap_lignite > 0:
            n.add("Generator", "Lignite", bus="TR", p_nom=cap_lignite, marginal_cost=mc_lignite)
        if cap_gas > 0:
            n.add("Generator", "Natural gas", bus="TR", p_nom=cap_gas, marginal_cost=mc_gas)
        if cap_nuclear > 0:
            n.add("Generator", "Nuclear", bus="TR", p_nom=cap_nuclear, marginal_cost=mc_nuclear)
        if cap_other > 0:
            n.add("Generator", "Other", bus="TR", p_nom=cap_other, marginal_cost=mc_other)

        # VRE (availability shapes)
        if cap_wind > 0:
            n.add("Generator", "Wind (RES)", bus="TR", p_nom=cap_wind, marginal_cost=mc_wind, p_max_pu=wind_shape.values)
        if cap_solar > 0:
            n.add("Generator", "Solar (GES)", bus="TR", p_nom=cap_solar, marginal_cost=mc_solar, p_max_pu=solar_shape.values)

        # Hydro: for now treat as "profile-limited" generator (like run-of-river)
        if cap_hydro > 0:
            n.add("Generator", "Hydro", bus="TR", p_nom=cap_hydro, marginal_cost=mc_hydro, p_max_pu=hydro_shape.values)

        # Load shedding (ensures feasibility)
        if use_voll:
            n.add("Generator", "Load shedding", bus="TR", p_nom=1e9, marginal_cost=voll)

    with st.spinner("Optimize ediliyor (HiGHS)..."):
        n.optimize(solver_name="highs")

    st.success("Çözüm tamam!")

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Sonuçlar")

    gen = n.generators_t.p.copy()
    # Remove load shedding from main stack unless it's used
    cols = [c for c in gen.columns if c != "Load shedding"]
    gen_main = gen[cols]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Objective (toplam maliyet)", f"{n.objective:,.0f}")
    with c2:
        st.metric("Toplam üretim (MWh)", f"{gen.sum().sum():,.0f}")
    with c3:
        if "Load shedding" in gen.columns:
            st.metric("Unserved energy (MWh)", f"{gen['Load shedding'].sum():,.0f}")

    # Annual totals by tech
    totals = gen.sum().sort_values(ascending=False).rename("MWh").to_frame()
    st.write("Yıllık toplam üretim (teknoloji bazında)")
    st.dataframe(totals)

    # Curtailment for VRE
    cur = []
    if "Wind (RES)" in n.generators.index:
        pot = (cap_wind * wind_shape).sum()
        act = gen["Wind (RES)"].sum()
        cur.append(("Wind (RES)", float(pot - act)))
    if "Solar (GES)" in n.generators.index:
        pot = (cap_solar * solar_shape).sum()
        act = gen["Solar (GES)"].sum()
        cur.append(("Solar (GES)", float(pot - act)))
    if cur:
        st.write("Curtailment (potansiyel - gerçekleşen) [MWh]")
        st.dataframe(pd.DataFrame(cur, columns=["tech", "curtailment_mwh"]))

    # Quick chart (stacked area)
    st.write("Saatlik üretim (ilk 168 saat örnek)")
    st.area_chart(gen_main.head(168))

    # Export
    st.download_button(
        "Saatlik üretimi CSV indir",
        data=gen.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="dispatch_hourly.csv",
        mime="text/csv"
    )
