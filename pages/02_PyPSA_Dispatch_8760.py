from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pypsa

st.set_page_config(page_title="PyPSA Dispatch 8760", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

st.title("PyPSA Dispatch (Türkiye Tek-Node, 8760)")

# Costs (same as before; kept short)
COSTS = pd.DataFrame([
    ("Coal",        16.77, 0.43, 0.00400, 0.90),
    ("Lignite",     13.00, 0.33, 0.00400, 1.10),
    ("Natural gas", 27.00, 0.60, 0.00200, 0.40),
    ("Hydro_Res",    0.00, 1.00, 0.00032,0.00),
    ("Hydro_RoR",    0.00, 1.00, 0.00032,0.00),
    ("Wind (RES)",   0.00, 1.00, 0.00043,0.00),
    ("Solar (GES)",  0.00, 1.00, 0.00000,0.00),
    ("Other",       35.00, 0.40, 0.00400, 0.70),
], columns=["tech", "fuel_cost_usd_per_mwh_th", "eff", "vom_usd_per_kwh", "ef_tco2_per_mwh_e"]).set_index("tech")

RAMP_LIMITS = {
    "Coal": 0.0386,
    "Lignite": 0.0386,
    "Natural gas": 0.0754,
}

def compute_marginal_costs(co2_price_usd_per_t: float) -> pd.Series:
    dfc = COSTS.copy()
    fuel = dfc["fuel_cost_usd_per_mwh_th"].fillna(0)
    eff = dfc["eff"].replace(0, np.nan)
    fuel_mc = (fuel / eff).replace([np.inf, -np.inf], np.nan).fillna(0)
    vom = dfc["vom_usd_per_kwh"].fillna(0) * 1000.0
    ef = dfc["ef_tco2_per_mwh_e"].fillna(0)
    co2 = ef * float(co2_price_usd_per_t)
    return fuel_mc + vom + co2

def make_shape(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    tot = float(s.sum())
    if tot <= 0:
        return s * 0.0
    return s / tot

st.subheader("1) Profiles seçimi")

available = sorted(DATA_DIR.glob("profiles_*.parquet"))
if not available:
    st.error("profiles_YYYY.parquet bulunamadı. Önce DataPrep çalıştır.")
    st.stop()

profile_file = st.selectbox("Profiles dosyası", options=available, format_func=lambda p: p.name)
prof = pd.read_parquet(profile_file)

if "timestamp" in prof.columns:
    prof["timestamp"] = pd.to_datetime(prof["timestamp"], errors="coerce")
    prof = prof.dropna(subset=["timestamp"]).set_index("timestamp")

# Compatibility for load columns
rename_map = {
    "load_base": "load_gross",
    "net_load_base": "load_net",
    "load_gross_mwh": "load_gross",
    "load_net_mwh": "load_net",
}
for src, dst in rename_map.items():
    if src in prof.columns and dst not in prof.columns:
        prof[dst] = prof[src]

needed = {"load_gross", "load_net", "solar_shape", "wind_shape"}
missing = needed - set(prof.columns)
if missing:
    st.error(f"Profiles içinde eksik kolon var: {sorted(missing)}")
    st.stop()

# ✅ Hydro shapes: prefer split, fallback to old total hydro_shape
if "hydro_res_shape" in prof.columns and "hydro_ror_shape" in prof.columns:
    hydro_res_pmax = pd.to_numeric(prof["hydro_res_shape"], errors="coerce").fillna(0).clip(0, 1)
    hydro_ror_pmax = pd.to_numeric(prof["hydro_ror_shape"], errors="coerce").fillna(0).clip(0, 1)
elif "hydro_shape" in prof.columns:
    base = pd.to_numeric(prof["hydro_shape"], errors="coerce").fillna(0).clip(0, 1)
    hydro_res_pmax = base
    hydro_ror_pmax = base
else:
    st.error("Profiles içinde hidro shape yok. DataPrep’i güncelleyip yeniden üret.")
    st.stop()

snapshots = prof.index
solar_shape = pd.to_numeric(prof["solar_shape"], errors="coerce").fillna(0).clip(0, 1)
wind_shape  = pd.to_numeric(prof["wind_shape"], errors="coerce").fillna(0).clip(0, 1)

st.caption(f"{profile_file.name} | Saat: {len(prof):,}")

st.subheader("2) Talep (yıllık hedef + saatlik shape)")

load_mode = st.radio("Baz load kaynağı", ["Gross load", "Net load"], horizontal=True, index=0)
base_hourly = prof["load_gross"] if load_mode == "Gross load" else prof["load_net"]
shape = make_shape(base_hourly)

base_twh = float(pd.to_numeric(base_hourly, errors="coerce").fillna(0).clip(lower=0).sum()) / 1e6
c1, c2 = st.columns(2)
with c1:
    target_twh = st.number_input("Yıllık hedef (TWh)", min_value=0.0, value=550.0, step=10.0)
with c2:
    st.metric("Baz yıl toplamı (TWh)", f"{base_twh:,.1f}")

load = shape * (target_twh * 1e6)

st.subheader("3) Kapasiteler (MW)")

cA, cB, cC = st.columns([1.2, 1.2, 1.0])
with cA:
    cap_coal = st.number_input("Coal (MW)", min_value=0.0, value=20000.0, step=500.0)
    cap_lignite = st.number_input("Lignite (MW)", min_value=0.0, value=10000.0, step=500.0)
    cap_gas = st.number_input("Natural gas (MW)", min_value=0.0, value=25000.0, step=500.0)
with cB:
    cap_hydro_res = st.number_input("Hydro_Res (Hidro) (MW)", min_value=0.0, value=23000.0, step=500.0)
    cap_hydro_ror = st.number_input("Hydro_RoR (Akarsu) (MW)", min_value=0.0, value=12000.0, step=500.0)
    cap_wind  = st.number_input("Wind (RES) (MW)", min_value=0.0, value=12000.0, step=500.0)
with cC:
    cap_solar = st.number_input("Solar (GES) (MW)", min_value=0.0, value=15000.0, step=500.0)
    cap_other = st.number_input("Other (MW)", min_value=0.0, value=0.0, step=500.0)

st.subheader("4) ETS / CO₂")
co2_price = st.slider("CO₂ fiyatı ($/tCO₂)", min_value=0.0, max_value=250.0, value=50.0, step=5.0)
mc = compute_marginal_costs(co2_price)

with st.expander("Güvenlik: Load shedding (VOLL)", expanded=False):
    use_voll = st.checkbox("Load shedding kullan", value=True)
    voll = st.number_input("VOLL ($/MWh)", min_value=0.0, value=10000.0, step=100.0)

run = st.button("Optimize et (8760)", type="primary")

if run:
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    n.add("Bus", "TR")
    n.add("Load", "Load", bus="TR", p_set=load.values)

    # Dispatchables
    if cap_coal > 0:
        n.add("Generator", "Coal", bus="TR", p_nom=cap_coal, marginal_cost=float(mc["Coal"]),
              ramp_limit_up=RAMP_LIMITS["Coal"], ramp_limit_down=RAMP_LIMITS["Coal"])
    if cap_lignite > 0:
        n.add("Generator", "Lignite", bus="TR", p_nom=cap_lignite, marginal_cost=float(mc["Lignite"]),
              ramp_limit_up=RAMP_LIMITS["Lignite"], ramp_limit_down=RAMP_LIMITS["Lignite"])
    if cap_gas > 0:
        n.add("Generator", "Natural gas", bus="TR", p_nom=cap_gas, marginal_cost=float(mc["Natural gas"]),
              ramp_limit_up=RAMP_LIMITS["Natural gas"], ramp_limit_down=RAMP_LIMITS["Natural gas"])
    if cap_other > 0:
        n.add("Generator", "Other", bus="TR", p_nom=cap_other, marginal_cost=float(mc["Other"]))

    # VRE
    if cap_wind > 0:
        n.add("Generator", "Wind (RES)", bus="TR", p_nom=cap_wind, marginal_cost=float(mc["Wind (RES)"]),
              p_max_pu=wind_shape.values)
    if cap_solar > 0:
        n.add("Generator", "Solar (GES)", bus="TR", p_nom=cap_solar, marginal_cost=float(mc["Solar (GES)"]),
              p_max_pu=solar_shape.values)

    # ✅ Hydro split
    if cap_hydro_res > 0:
        n.add("Generator", "Hydro_Res", bus="TR", p_nom=cap_hydro_res, marginal_cost=float(mc["Hydro_Res"]),
              p_max_pu=hydro_res_pmax.values)
    if cap_hydro_ror > 0:
        n.add("Generator", "Hydro_RoR", bus="TR", p_nom=cap_hydro_ror, marginal_cost=float(mc["Hydro_RoR"]),
              p_max_pu=hydro_ror_pmax.values)

    if use_voll:
        n.add("Generator", "Load shedding", bus="TR", p_nom=1e9, marginal_cost=float(voll))

    with st.spinner("Optimize ediliyor (HiGHS)..."):
        n.optimize(solver_name="highs")

    st.success("Bitti!")

    gen = n.generators_t.p.copy()

    # Summary
    totals = (gen.sum() / 1e6).sort_values(ascending=False).rename("TWh").to_frame()
    st.subheader("Yıllık üretim (TWh)")
    st.dataframe(totals)

    # Hydro check
    if "Hydro_Res" in gen.columns or "Hydro_RoR" in gen.columns:
        hr = float(gen["Hydro_Res"].sum())/1e6 if "Hydro_Res" in gen.columns else 0.0
        hh = float(gen["Hydro_RoR"].sum())/1e6 if "Hydro_RoR" in gen.columns else 0.0
        st.info(f"Hidro_Res (Hidro): {hr:,.1f} TWh | Hydro_RoR (Akarsu): {hh:,.1f} TWh | Toplam: {(hr+hh):,.1f} TWh")

    # Window plot
    st.subheader("Saatlik üretim – seçilebilir pencere (168 saat)")
    non_shed_cols = [c for c in gen.columns if c != "Load shedding"]
    gen_main = gen[non_shed_cols]
    start_idx = st.slider("Başlangıç saati", 0, len(snapshots) - 168, 0, step=24)
    st.area_chart(gen_main.iloc[start_idx:start_idx + 168])

    st.download_button(
        "dispatch_hourly.csv indir",
        data=gen.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="dispatch_hourly.csv",
        mime="text/csv"
    )
