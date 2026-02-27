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

**Talep yaklaşımı (LEAP-style):**
- Saatlik baz seri → **shape** (yüzde/pay) üretilir (toplamı 1).
- Sen **yıllık hedef (TWh)** girersin.
- Model her saat için yükü: `Load_t = shape_t × Annual_MWh` şeklinde kurar.

Maliyetler sabittir ve **model içinde hesaplanır**: yakıt/verim + VOM + CO₂.

**Teknik kısıt (Ramp):**
- Nuclear: 0.1356 pu/saat
- Coal & Lignite: 0.0386 pu/saat
- Natural gas: 0.0754 pu/saat
(ramp up = ramp down)
""")

# -----------------------------
# Cost block (from your table, with sensible defaults)
# Units:
# - Fuel Cost: $/MWh_th (thermal)  -> converted to $/MWh_e by dividing by efficiency
# - VOM: $/kWh -> converted to $/MWh by * 1000
# - CO2 price: $/tCO2
# - Emission factor: tCO2/MWh_e
# -----------------------------
COSTS = pd.DataFrame([
    # Tech, FuelCost, Efficiency, VOM($/kWh), EmFactor (tCO2/MWh_e)
    ("Coal",        16.77, 0.43, 0.00400, 0.90),
    ("Lignite",     13.00, 0.33, 0.00400, 1.10),
    ("Natural gas", 27.00, 0.60, 0.00200, 0.40),
    ("Nuclear",      0.00, 0.38, 0.00900, 0.00),
    ("Hydro",        0.00, 1.00, 0.00032, 0.00),
    ("Wind (RES)",   0.00, 1.00, 0.00043, 0.00),
    ("Solar (GES)",  0.00, 1.00, 0.00000, 0.00),
    # Filled defaults for Other (adjust anytime)
    ("Other",       35.00, 0.40, 0.00400, 0.70),
], columns=["tech", "fuel_cost_usd_per_mwh_th", "eff", "vom_usd_per_kwh", "ef_tco2_per_mwh_e"]).set_index("tech")

# --- Fixed ramp limits (per unit of p_nom per hour) ---
# Given by user; ramp_up = ramp_down.
RAMP_LIMITS = {
    "Coal": 0.0386,
    "Lignite": 0.0386,
    "Natural gas": 0.0754,
    "Nuclear": 0.1356,
}

def compute_marginal_costs(co2_price_usd_per_t: float) -> pd.Series:
    dfc = COSTS.copy()
    # Fuel component ($/MWh_e)
    fuel = dfc["fuel_cost_usd_per_mwh_th"].fillna(0)
    eff = dfc["eff"].replace(0, np.nan)
    fuel_mc = (fuel / eff).replace([np.inf, -np.inf], np.nan).fillna(0)

    # VOM ($/MWh_e)
    vom = dfc["vom_usd_per_kwh"].fillna(0) * 1000.0

    # CO2 ($/MWh_e)
    ef = dfc["ef_tco2_per_mwh_e"].fillna(0)
    co2 = ef * float(co2_price_usd_per_t)

    mc = fuel_mc + vom + co2
    return mc

def _make_shape_from_hourly(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    total = float(s.sum())
    if total <= 0:
        return s * 0.0
    return s / total  # sums to 1

# -----------------------------
# Load profiles
# -----------------------------
st.subheader("1) Profil seçimi")

available = sorted(DATA_DIR.glob("profiles_*.parquet"))
if not available:
    st.error("profiles_YYYY.parquet bulunamadı. Önce DataPrep sayfasını çalıştır.")
    st.stop()

profile_file = st.selectbox("Profiles dosyası", options=available, format_func=lambda p: p.name)
df = pd.read_parquet(profile_file)

# timestamp -> index
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")

# Compatibility layer (DataPrep versions)
rename_map = {
    "load_base": "load_gross",
    "net_load_base": "load_net",
    "load_gross_mwh": "load_gross",
    "load_net_mwh": "load_net",
}
for src, dst in rename_map.items():
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]

required_cols = {"load_gross", "load_net", "solar_shape", "wind_shape", "hydro_shape"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Profiles içinde eksik kolonlar var: {sorted(missing)}")
    st.stop()

solar_shape = pd.to_numeric(df["solar_shape"], errors="coerce").fillna(0).clip(0, 1)
wind_shape  = pd.to_numeric(df["wind_shape"], errors="coerce").fillna(0).clip(0, 1)
hydro_shape = pd.to_numeric(df["hydro_shape"], errors="coerce").fillna(0).clip(0, 1)

st.caption(f"Seçili dosya: {profile_file.name} | Saat sayısı: {len(df):,}")

# -----------------------------
# Scenario inputs
# -----------------------------
st.subheader("2) Talep (LEAP-style: yıllık hedef + saatlik shape)")

load_mode = st.radio("Baz load kaynağı (shape bunun üzerinden üretilecek)", ["Gross load", "Net load"], horizontal=True, index=0)
base_hourly = df["load_gross"] if load_mode == "Gross load" else df["load_net"]

base_twh = float(pd.to_numeric(base_hourly, errors="coerce").fillna(0).clip(lower=0).sum()) / 1e6
shape = _make_shape_from_hourly(base_hourly)

cL1, cL2, cL3 = st.columns([1.2, 1.0, 1.0])
with cL1:
    target_twh = st.number_input("Yıllık hedef talep / gross üretim (TWh)", min_value=0.0, value=550.0, step=10.0)
with cL2:
    st.metric("Baz yıl toplamı (TWh)", f"{base_twh:,.1f}")
with cL3:
    if base_twh > 0:
        st.metric("Ölçek çarpanı", f"{(target_twh/base_twh):.3f}")
    else:
        st.metric("Ölçek çarpanı", "—")

annual_mwh = target_twh * 1e6
load = shape * annual_mwh  # sums to target

st.caption(f"Shape normalize edildi (toplam=1). Model load toplamı = {float(load.sum())/1e6:.1f} TWh (hedef={target_twh:.1f}).")

st.subheader("3) Kapasite girdileri (MW)")

cA, cB, cC = st.columns([1.2, 1.2, 1.0])

with cA:
    cap_coal = st.number_input("Coal kapasite (MW)", min_value=0.0, value=20000.0, step=500.0)
    cap_lignite = st.number_input("Lignite kapasite (MW)", min_value=0.0, value=10000.0, step=500.0)
    cap_gas = st.number_input("Natural gas kapasite (MW)", min_value=0.0, value=25000.0, step=500.0)

with cB:
    cap_hydro = st.number_input("Hydro kapasite (MW)", min_value=0.0, value=30000.0, step=500.0)
    cap_wind  = st.number_input("Wind (RES) kapasite (MW)", min_value=0.0, value=12000.0, step=500.0)
    cap_solar = st.number_input("Solar (GES) kapasite (MW)", min_value=0.0, value=15000.0, step=500.0)

with cC:
    cap_nuclear = st.number_input("Nuclear kapasite (MW)", min_value=0.0, value=0.0, step=500.0)
    cap_other = st.number_input("Other kapasite (MW)", min_value=0.0, value=0.0, step=500.0)

st.subheader("4) ETS / CO₂")
co2_price = st.slider("CO₂ fiyatı ($/tCO₂)", min_value=0.0, max_value=250.0, value=50.0, step=5.0)

with st.expander("Güvenlik: Load shedding (VOLL)", expanded=False):
    use_voll = st.checkbox("Load shedding kullan (önerilir)", value=True)
    voll = st.number_input("VOLL ($/MWh)", min_value=0.0, value=10000.0, step=100.0)

mc = compute_marginal_costs(co2_price)
st.write("Hesaplanan marjinal maliyetler ($/MWh_e)")
st.dataframe(mc.rename("MC").to_frame())

with st.expander("Teknik kısıtlar: Ramp limitleri (sabit)", expanded=False):
    st.write({k: f"{v:.4f} pu/saat" for k, v in RAMP_LIMITS.items()})

run = st.button("Optimize et (8760)", type="primary")

# -----------------------------
# Build & solve PyPSA
# -----------------------------
if run:
    with st.spinner("Network kuruluyor..."):
        n = pypsa.Network()
        n.set_snapshots(df.index)
        n.add("Bus", "TR")
        # Load (1 saatlik adımda MWh ~= MW)
        n.add("Load", "Load", bus="TR", p_set=load.values)

        # Dispatchable (ramp limits applied where provided)
        if cap_coal > 0:
            n.add(
                "Generator", "Coal", bus="TR",
                p_nom=cap_coal, marginal_cost=float(mc["Coal"]),
                ramp_limit_up=RAMP_LIMITS["Coal"], ramp_limit_down=RAMP_LIMITS["Coal"]
            )
        if cap_lignite > 0:
            n.add(
                "Generator", "Lignite", bus="TR",
                p_nom=cap_lignite, marginal_cost=float(mc["Lignite"]),
                ramp_limit_up=RAMP_LIMITS["Lignite"], ramp_limit_down=RAMP_LIMITS["Lignite"]
            )
        if cap_gas > 0:
            n.add(
                "Generator", "Natural gas", bus="TR",
                p_nom=cap_gas, marginal_cost=float(mc["Natural gas"]),
                ramp_limit_up=RAMP_LIMITS["Natural gas"], ramp_limit_down=RAMP_LIMITS["Natural gas"]
            )
        if cap_nuclear > 0:
            n.add(
                "Generator", "Nuclear", bus="TR",
                p_nom=cap_nuclear, marginal_cost=float(mc["Nuclear"]),
                ramp_limit_up=RAMP_LIMITS["Nuclear"], ramp_limit_down=RAMP_LIMITS["Nuclear"]
            )
        if cap_other > 0:
            n.add("Generator", "Other", bus="TR", p_nom=cap_other, marginal_cost=float(mc["Other"]))

        # VRE (availability shapes)
        if cap_wind > 0:
            n.add("Generator", "Wind (RES)", bus="TR", p_nom=cap_wind,
                  marginal_cost=float(mc["Wind (RES)"]), p_max_pu=wind_shape.values)
        if cap_solar > 0:
            n.add("Generator", "Solar (GES)", bus="TR", p_nom=cap_solar,
                  marginal_cost=float(mc["Solar (GES)"]), p_max_pu=solar_shape.values)

        # Hydro (profile-limited for now)
        if cap_hydro > 0:
            n.add("Generator", "Hydro", bus="TR", p_nom=cap_hydro,
                  marginal_cost=float(mc["Hydro"]), p_max_pu=hydro_shape.values)

        # Feasibility
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
    cols = [c for c in gen.columns if c != "Load shedding"]
    gen_main = gen[cols]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Objective (toplam maliyet)", f"{n.objective:,.0f}")
    with c2:
        st.metric("Toplam üretim (TWh)", f"{gen.sum().sum()/1e6:,.1f}")
    with c3:
        if "Load shedding" in gen.columns:
            st.metric("Unserved energy (TWh)", f"{gen['Load shedding'].sum()/1e6:,.3f}")

    totals = gen.sum().sort_values(ascending=False).rename("MWh").to_frame()
    totals["TWh"] = totals["MWh"] / 1e6
    st.write("Yıllık toplam üretim (teknoloji bazında)")
    st.dataframe(totals[["TWh"]])

    # Curtailment (TWh)
    cur = []
    if "Wind (RES)" in gen.columns:
        pot = float((cap_wind * wind_shape).sum())
        act = float(gen["Wind (RES)"].sum())
        cur.append(("Wind (RES)", (pot - act) / 1e6))
    if "Solar (GES)" in gen.columns:
        pot = float((cap_solar * solar_shape).sum())
        act = float(gen["Solar (GES)"].sum())
        cur.append(("Solar (GES)", (pot - act) / 1e6))
    if cur:
        st.write("Curtailment (potansiyel - gerçekleşen) [TWh]")
        st.dataframe(pd.DataFrame(cur, columns=["tech", "curtailment_twh"]))

    st.write("Saatlik üretim (ilk 168 saat örnek)")
    st.area_chart(gen_main.head(168))

    st.download_button(
        "Saatlik üretimi CSV indir",
        data=gen.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="dispatch_hourly.csv",
        mime="text/csv"
    )
