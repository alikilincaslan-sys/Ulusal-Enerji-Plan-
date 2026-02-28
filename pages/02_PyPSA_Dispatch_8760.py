# pages/02_PyPSA_Dispatch_8760.py
# ------------------------------------------------------------
# Türkiye tek-node PyPSA dispatch (8760)
# - Kapasite girişleri: GW (içeride MW'a çevrilir)
# - Toplam kapasite KPI
# - Yıllık üretim tablosu: TWh + Capacity Factor (CF)
# - 8760 gösterimler: Aylık TWh (mevsimsellik), Load Duration Curve, Heatmap (Load shedding / Balance)
# - Pencere grafiği: stack üretim + load çizgisi (Altair layer)
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pypsa
import altair as alt

st.set_page_config(page_title="PyPSA Dispatch 8760", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# -----------------------------
# Helpers
# -----------------------------
def gw_to_mw(x_gw: float) -> float:
    return float(x_gw) * 1000.0

def make_shape(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    tot = float(s.sum())
    if tot <= 0:
        return s * 0.0
    return s / tot

def compute_marginal_costs(costs_df: pd.DataFrame, co2_price_usd_per_t: float) -> pd.Series:
    dfc = costs_df.copy()
    fuel = dfc["fuel_cost_usd_per_mwh_th"].fillna(0)
    eff = dfc["eff"].replace(0, np.nan)
    fuel_mc = (fuel / eff).replace([np.inf, -np.inf], np.nan).fillna(0)
    vom = dfc["vom_usd_per_kwh"].fillna(0) * 1000.0
    ef = dfc["ef_tco2_per_mwh_e"].fillna(0)
    co2 = ef * float(co2_price_usd_per_t)
    return (fuel_mc + vom + co2).rename("marginal_cost_usd_per_mwh")

def safe_num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# Page header
# -----------------------------
st.title("PyPSA Dispatch (Türkiye Tek-Node, 8760)")

st.caption(
    "Bu sayfa DataPrep'in ürettiği `profiles_YYYY.parquet` dosyasını okur. "
    "Hidro split (Barajlı=Hydro_Res, Akarsu=Hydro_RoR) shape'leri parquet içinde olmalı."
)

# -----------------------------
# Costs (OPEX only: fuel/eff + VOM + CO2)
# Capital cost yok (kapasiteyi sen giriyorsun)
# -----------------------------
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

# Ramps (pu per hour)
RAMP_LIMITS = {
    "Coal": 0.0386,
    "Lignite": 0.0386,
    "Natural gas": 0.0754,
}

# -----------------------------
# Profiles selection
# -----------------------------
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

# Back-compat load columns
rename_map = {
    "load_base": "load_gross",
    "net_load_base": "load_net",
    "load_gross_mwh": "load_gross",
    "load_net_mwh": "load_net",
}
for src, dst in rename_map.items():
    if src in prof.columns and dst not in prof.columns:
        prof[dst] = prof[src]

need = {"load_gross", "load_net", "solar_shape", "wind_shape"}
missing = need - set(prof.columns)
if missing:
    st.error(f"Profiles içinde eksik kolon var: {sorted(missing)}")
    st.stop()

# Hydro shapes: prefer split, fallback to old hydro_shape
if "hydro_res_shape" in prof.columns and "hydro_ror_shape" in prof.columns:
    hydro_res_pmax = pd.to_numeric(prof["hydro_res_shape"], errors="coerce").fillna(0).clip(0, 1)
    hydro_ror_pmax = pd.to_numeric(prof["hydro_ror_shape"], errors="coerce").fillna(0).clip(0, 1)
elif "hydro_shape" in prof.columns:
    base = pd.to_numeric(prof["hydro_shape"], errors="coerce").fillna(0).clip(0, 1)
    hydro_res_pmax = base
    hydro_ror_pmax = base
else:
    st.error("Profiles içinde hidro shape yok. DataPrep’te hydro_res_shape/hydro_ror_shape üret.")
    st.stop()

snapshots = prof.index
solar_pmax = pd.to_numeric(prof["solar_shape"], errors="coerce").fillna(0).clip(0, 1)
wind_pmax  = pd.to_numeric(prof["wind_shape"], errors="coerce").fillna(0).clip(0, 1)

st.caption(f"{profile_file.name} | Saat sayısı: {len(prof):,}")

# -----------------------------
# Demand (annual target + hourly shape)
# -----------------------------
st.subheader("2) Talep (yıllık hedef + saatlik shape)")

load_mode = st.radio("Baz load kaynağı (shape bunun üzerinden)", ["Gross load", "Net load"], horizontal=True, index=0)
base_hourly = prof["load_gross"] if load_mode == "Gross load" else prof["load_net"]

shape = make_shape(base_hourly)
base_twh = float(pd.to_numeric(base_hourly, errors="coerce").fillna(0).clip(lower=0).sum()) / 1e6

cL1, cL2, cL3 = st.columns([1.2, 1.0, 1.0])
with cL1:
    target_twh = st.number_input("Yıllık hedef talep/gross üretim (TWh)", min_value=0.0, value=550.0, step=10.0)
with cL2:
    st.metric("Baz yıl toplamı (TWh)", f"{base_twh:,.1f}")
with cL3:
    st.metric("Ölçek çarpanı", f"{(target_twh/base_twh):.3f}" if base_twh > 0 else "—")

# Load as MW (since 1h timestep, MWh/h == MW)
load_s = (shape * (target_twh * 1e6)).rename("Load_MW")
load_s.index = snapshots

# -----------------------------
# Capacities (GW inputs)
# -----------------------------
st.subheader("3) Kapasiteler (GW)")

cA, cB, cC = st.columns([1.2, 1.2, 1.0])
with cA:
    cap_coal_gw    = st.number_input("Coal (GW)", min_value=0.0, value=20.0, step=0.5)
    cap_lignite_gw = st.number_input("Lignite (GW)", min_value=0.0, value=10.0, step=0.5)
    cap_gas_gw     = st.number_input("Natural gas (GW)", min_value=0.0, value=25.0, step=0.5)
with cB:
    cap_hres_gw = st.number_input("Hydro_Res (Barajlı) (GW)", min_value=0.0, value=23.0, step=0.5)
    cap_hror_gw = st.number_input("Hydro_RoR (Akarsu) (GW)", min_value=0.0, value=12.0, step=0.5)
    cap_wind_gw = st.number_input("Wind (RES) (GW)", min_value=0.0, value=12.0, step=0.5)
with cC:
    cap_solar_gw = st.number_input("Solar (GES) (GW)", min_value=0.0, value=15.0, step=0.5)
    cap_other_gw = st.number_input("Other (GW)", min_value=0.0, value=0.0, step=0.5)

total_cap_gw = float(cap_coal_gw + cap_lignite_gw + cap_gas_gw + cap_hres_gw + cap_hror_gw + cap_wind_gw + cap_solar_gw + cap_other_gw)
peak_load_gw = float(load_s.max()) / 1000.0

k1, k2, k3 = st.columns(3)
k1.metric("Toplam kurulu güç (GW)", f"{total_cap_gw:,.1f}")
k2.metric("Peak load (GW)", f"{peak_load_gw:,.1f}")
k3.metric("Kapasite / Peak", f"{(total_cap_gw/peak_load_gw):.2f}" if peak_load_gw > 0 else "—")

cap_coal    = gw_to_mw(cap_coal_gw)
cap_lignite = gw_to_mw(cap_lignite_gw)
cap_gas     = gw_to_mw(cap_gas_gw)
cap_hres    = gw_to_mw(cap_hres_gw)
cap_hror    = gw_to_mw(cap_hror_gw)
cap_wind    = gw_to_mw(cap_wind_gw)
cap_solar   = gw_to_mw(cap_solar_gw)
cap_other   = gw_to_mw(cap_other_gw)

# -----------------------------
# CO2
# -----------------------------
st.subheader("4) ETS / CO₂")
co2_price = st.slider("CO₂ fiyatı ($/tCO₂)", min_value=0.0, max_value=250.0, value=50.0, step=5.0)
mc = compute_marginal_costs(COSTS, co2_price)

with st.expander("Marjinal maliyetler ($/MWh)", expanded=False):
    st.dataframe(mc.to_frame())

# -----------------------------
# Storage (optional, bounded)
# -----------------------------
st.subheader("5) Depolama (opsiyonel)")

enable_storage = st.checkbox("Depolamayı modele dahil et", value=False)

storage_optimize_size = False
storage_p_nom_max_mw = 0.0
storage_p_nom_fixed_mw = 0.0
storage_max_hours = 4.0
storage_eff_roundtrip = 0.88
storage_standing_loss = 0.0
storage_capital_cost_per_mw_yr = 0.0
storage_marginal_cost = 0.0

if enable_storage:
    s1, s2, s3 = st.columns(3)
    with s1:
        storage_optimize_size = st.checkbox("Depolama gücünü optimize et (üst limitli)", value=True)
        storage_p_nom_max_mw = st.number_input("Güç üst sınırı (MW)", min_value=0.0, value=5000.0, step=250.0)
        if not storage_optimize_size:
            storage_p_nom_fixed_mw = st.number_input("Sabit güç (MW)", min_value=0.0, value=2000.0, step=250.0)
            storage_p_nom_max_mw = storage_p_nom_fixed_mw
    with s2:
        storage_max_hours = st.number_input("Max hours (h)", min_value=0.5, value=4.0, step=0.5)
        storage_eff_roundtrip = st.number_input("Round-trip verim", min_value=0.50, max_value=0.99, value=0.88, step=0.01)
    with s3:
        storage_standing_loss = st.number_input("Standing loss (pu/saat)", min_value=0.0, max_value=0.01, value=0.0, step=0.0005)
        storage_capital_cost_per_mw_yr = st.number_input("Yatırım maliyeti ($/MW-yıl)", min_value=0.0, value=0.0, step=1000.0)
        storage_marginal_cost = st.number_input("Degradasyon vb. ($/MWh)", min_value=0.0, value=0.0, step=1.0)

    st.caption(f"Depolama enerji üst sınırı ≈ {storage_p_nom_max_mw * storage_max_hours:,.0f} MWh (p_nom_max × max_hours)")

# -----------------------------
# Feasibility (load shedding)
# -----------------------------
with st.expander("6) Güvenlik: Load shedding (önerilir)", expanded=False):
    use_voll = st.checkbox("Load shedding açık", value=True)
    voll = st.number_input("VOLL ($/MWh)", min_value=0.0, value=10000.0, step=100.0)

# -----------------------------
# Run
# -----------------------------
run = st.button("Optimize et (8760)", type="primary")

if run:
    with st.spinner("Network kuruluyor..."):
        n = pypsa.Network()
        n.set_snapshots(snapshots)
        n.add("Bus", "TR")

        # Load
        n.add("Load", "Load", bus="TR", p_set=load_s.values)

        # Dispatchables
        if cap_coal > 0:
            n.add("Generator", "Coal", bus="TR", p_nom=cap_coal, marginal_cost=safe_num(mc.get("Coal", 0)),
                  ramp_limit_up=RAMP_LIMITS["Coal"], ramp_limit_down=RAMP_LIMITS["Coal"])
        if cap_lignite > 0:
            n.add("Generator", "Lignite", bus="TR", p_nom=cap_lignite, marginal_cost=safe_num(mc.get("Lignite", 0)),
                  ramp_limit_up=RAMP_LIMITS["Lignite"], ramp_limit_down=RAMP_LIMITS["Lignite"])
        if cap_gas > 0:
            n.add("Generator", "Natural gas", bus="TR", p_nom=cap_gas, marginal_cost=safe_num(mc.get("Natural gas", 0)),
                  ramp_limit_up=RAMP_LIMITS["Natural gas"], ramp_limit_down=RAMP_LIMITS["Natural gas"])
        if cap_other > 0:
            n.add("Generator", "Other", bus="TR", p_nom=cap_other, marginal_cost=safe_num(mc.get("Other", 0)))

        # VRE
        if cap_wind > 0:
            n.add("Generator", "Wind (RES)", bus="TR", p_nom=cap_wind, marginal_cost=safe_num(mc.get("Wind (RES)", 0)),
                  p_max_pu=wind_pmax.values)
        if cap_solar > 0:
            n.add("Generator", "Solar (GES)", bus="TR", p_nom=cap_solar, marginal_cost=safe_num(mc.get("Solar (GES)", 0)),
                  p_max_pu=solar_pmax.values)

        # Hydro split
        if cap_hres > 0:
            n.add("Generator", "Hydro_Res", bus="TR", p_nom=cap_hres, marginal_cost=safe_num(mc.get("Hydro_Res", 0)),
                  p_max_pu=hydro_res_pmax.values)
        if cap_hror > 0:
            n.add("Generator", "Hydro_RoR", bus="TR", p_nom=cap_hror, marginal_cost=safe_num(mc.get("Hydro_RoR", 0)),
                  p_max_pu=hydro_ror_pmax.values)

        # Storage
        if enable_storage and storage_p_nom_max_mw > 0:
            eff_sd = float(np.sqrt(storage_eff_roundtrip))
            if storage_optimize_size:
                n.add(
                    "StorageUnit", "Battery",
                    bus="TR",
                    p_nom_extendable=True,
                    p_nom_max=float(storage_p_nom_max_mw),
                    max_hours=float(storage_max_hours),
                    efficiency_store=eff_sd,
                    efficiency_dispatch=eff_sd,
                    standing_loss=float(storage_standing_loss),
                    capital_cost=float(storage_capital_cost_per_mw_yr),
                    marginal_cost=float(storage_marginal_cost),
                    cyclic_state_of_charge=True,
                )
            else:
                n.add(
                    "StorageUnit", "Battery",
                    bus="TR",
                    p_nom=float(storage_p_nom_fixed_mw),
                    max_hours=float(storage_max_hours),
                    efficiency_store=eff_sd,
                    efficiency_dispatch=eff_sd,
                    standing_loss=float(storage_standing_loss),
                    marginal_cost=float(storage_marginal_cost),
                    cyclic_state_of_charge=True,
                )

        # Feasibility
        if use_voll:
            n.add("Generator", "Load shedding", bus="TR", p_nom=1e9, marginal_cost=float(voll))

    with st.spinner("Optimize ediliyor (HiGHS)..."):
        n.optimize(solver_name="highs")

    st.success("Çözüm tamam!")

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Sonuçlar")

    gen = n.generators_t.p.copy()  # MW time series
    gen.index = pd.to_datetime(gen.index)

    # Key metrics
    total_twh = float(gen.sum().sum()) / 1e6
    shed_twh = float(gen["Load shedding"].sum()) / 1e6 if "Load shedding" in gen.columns else 0.0
    shed_hours = int((gen["Load shedding"] > 1e-6).sum()) if "Load shedding" in gen.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objective", f"{n.objective:,.0f}")
    c2.metric("Toplam üretim (TWh)", f"{total_twh:,.1f}")
    c3.metric("Load shedding (TWh)", f"{shed_twh:,.3f}")
    c4.metric("Load shedding saat", f"{shed_hours:,}")

    # -----------------------------
    # Annual table: TWh + CF
    # -----------------------------
    annual_mwh = gen.sum()  # by generator name
    twh = (annual_mwh / 1e6).rename("TWh")

    cap_mw = n.generators["p_nom"].copy()
    cap_mw = cap_mw.reindex(twh.index)

    cf = (annual_mwh / (cap_mw * len(snapshots))).rename("CF").replace([np.inf, -np.inf], np.nan)

    annual_tbl = pd.concat([twh, cf], axis=1).sort_values("TWh", ascending=False)

    st.subheader("Yıllık üretim ve kapasite faktörü")
    st.dataframe(annual_tbl.style.format({"TWh": "{:.2f}", "CF": "{:.2%}"}))

    # Storage summary
    if hasattr(n, "storage_units") and "Battery" in n.storage_units.index:
        st.subheader("Depolama özeti")
        p_nom_inst = float(n.storage_units.loc["Battery", "p_nom"])
        max_h_inst = float(n.storage_units.loc["Battery", "max_hours"])
        e_cap = p_nom_inst * max_h_inst
        s1, s2, s3 = st.columns(3)
        s1.metric("Battery power (MW)", f"{p_nom_inst:,.0f}")
        s2.metric("max_hours (h)", f"{max_h_inst:,.1f}")
        s3.metric("Energy cap (MWh)", f"{e_cap:,.0f}")

    # -----------------------------
    # 8760 diagnostics
    # -----------------------------
    st.subheader("8760 Tanı Paneli (mevsimsellik + sorun tespiti)")

    # A) Monthly energy (TWh)
    show_cols = [c for c in gen.columns if c != "Load shedding"]
    monthly_twh = (gen[show_cols].resample("M").sum() / 1e6)

    st.markdown("**Aylık üretim (TWh) – mevsimsellik**")
    st.bar_chart(monthly_twh)

    # B) Load duration curve (8760)
    ldc = load_s.sort_values(ascending=False).reset_index(drop=True)
    st.markdown("**Load Duration Curve (8760)**")
    st.line_chart(ldc)

    # C) Balance (TotalGen - Load)
    total_gen = gen[show_cols].sum(axis=1)
    balance = (total_gen - load_s).rename("Balance_MW")
    st.markdown("**Arz - Talep (MW)** (negatif saat varsa -> ya load shedding ya da kısıt)")
    st.line_chart(balance)

    # D) Heatmap: choose series
    st.markdown("**Heatmap (Ay × Saat)**")
    hm_mode = st.selectbox("Heatmap metriği", ["Load shedding (MW)", "Balance (MW)", "Marginal price ($/MWh)"], index=0)

    if hm_mode == "Load shedding (MW)":
        if "Load shedding" in gen.columns:
            hm_s = gen["Load shedding"].copy()
        else:
            hm_s = pd.Series(0.0, index=gen.index)
    elif hm_mode == "Balance (MW)":
        hm_s = balance.copy()
    else:
        # Marginal price
        try:
            hm_s = n.buses_t.marginal_price["TR"].copy()
            hm_s.index = pd.to_datetime(hm_s.index)
        except Exception:
            hm_s = pd.Series(0.0, index=gen.index)

    df_hm = pd.DataFrame({
        "ts": hm_s.index,
        "month": hm_s.index.month,
        "hour": hm_s.index.hour,
        "value": pd.to_numeric(hm_s.values, errors="coerce")
    }).dropna()

    hm = alt.Chart(df_hm).mark_rect().encode(
        x=alt.X("hour:O", title="Saat"),
        y=alt.Y("month:O", title="Ay"),
        color=alt.Color("value:Q", title=hm_mode),
        tooltip=["month:O", "hour:O", alt.Tooltip("value:Q", title=hm_mode)]
    ).properties(height=300)

    st.altair_chart(hm, use_container_width=True)

    # Worst hours table: by load shedding then by price
    st.subheader("En kötü saatler (otomatik tespit)")
    if "Load shedding" in gen.columns and (gen["Load shedding"] > 1e-6).any():
        worst_shed = gen["Load shedding"].sort_values(ascending=False).head(50).rename("unserved_MW").to_frame()
        st.write("Load shedding en yüksek 50 saat")
        st.dataframe(worst_shed)
    else:
        st.info("Load shedding yok (veya sıfıra yakın). Fiyat/balance ile kontrol edebilirsin.")

    try:
        price = n.buses_t.marginal_price["TR"].copy()
        price.index = pd.to_datetime(price.index)
        worst_price = price.sort_values(ascending=False).head(50).rename("price_$perMWh").to_frame()
        st.write("Marjinal fiyat en yüksek 50 saat")
        st.dataframe(worst_price)
    except Exception:
        st.caption("Marjinal fiyat serisi okunamadı (PyPSA sürüm farkı olabilir).")

    # -----------------------------
    # Window chart: stacked generation + load line (Altair)
    # -----------------------------
    st.subheader("Saatlik pencere: üretim stack + Load çizgisi")

    start_idx = st.slider("Başlangıç saati", 0, len(snapshots) - 168, 0, step=24)
    win = pd.to_datetime(snapshots[start_idx:start_idx + 168])

    gen_w = gen.loc[win, show_cols].copy()

    df_area = gen_w.reset_index().rename(columns={"index": "timestamp"})
    df_area = df_area.melt(id_vars=["timestamp"], var_name="tech", value_name="mw")

    df_line = pd.DataFrame({"timestamp": win, "mw": load_s.loc[win].values})

    area = alt.Chart(df_area).mark_area().encode(
        x=alt.X("timestamp:T", title="Zaman"),
        y=alt.Y("mw:Q", stack=True, title="MW"),
        color=alt.Color("tech:N", title="Üretim"),
        tooltip=["timestamp:T", "tech:N", alt.Tooltip("mw:Q", format=",.0f")]
    )

    line = alt.Chart(df_line).mark_line(strokeWidth=2).encode(
        x="timestamp:T",
        y=alt.Y("mw:Q", title=""),
        tooltip=["timestamp:T", alt.Tooltip("mw:Q", title="Load (MW)", format=",.0f")]
    )

    st.altair_chart(area + line, use_container_width=True)

    # -----------------------------
    # Export
    # -----------------------------
    st.subheader("Çıktı indir")
    st.download_button(
        "dispatch_hourly.csv indir",
        data=gen.reset_index().rename(columns={"index": "timestamp"}).to_csv(index=False).encode("utf-8"),
        file_name="dispatch_hourly.csv",
        mime="text/csv"
    )
