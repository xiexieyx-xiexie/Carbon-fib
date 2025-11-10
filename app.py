# app.py — fib Carbonation (B1.2-8 W(t) with t0) • one-column UI
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="fib Carbonation – Reliability", layout="wide")

st.markdown("""
<style>
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{ -webkit-appearance:none; margin:0; }
input[type=number]{ -moz-appearance:textfield; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def beta_from_pf(Pf: np.ndarray) -> np.ndarray:
    Pf = np.clip(Pf, 1e-12, 1-1e-12)
    return -norm.ppf(Pf)

def lognorm_from_mu_sd(rng, n, mu, sd):
    mu = max(mu, 1e-12); sd = max(sd, 1e-12)
    s2 = math.log(1 + (sd**2)/(mu**2))
    mu_log = math.log(mu) - 0.5*s2
    sigma = math.sqrt(s2)
    return rng.lognormal(mu_log, sigma, n)

def beta01_shapes_from_mean_sd(mu01, sd01):
    mu01 = float(np.clip(mu01, 1e-9, 1-1e-9))
    var01 = max(sd01**2, 1e-12)
    t = mu01*(1-mu01)/var01 - 1
    alpha = max(mu01*t, 1e-6)
    beta  = max((1-mu01)*t, 1e-6)
    return alpha, beta

def beta_interval_from_mean_sd(rng, n, mu, sd, L, U):
    if U <= L:
        raise ValueError("Upper bound must be greater than lower bound.")
    mu = max(min(mu, U - 1e-12), L + 1e-12)
    sd = max(sd, 1e-12)
    mu01 = (mu - L) / (U - L)
    sd01 = sd / (U - L)
    a, b = beta01_shapes_from_mean_sd(mu01, sd01)
    return L + (U - L) * rng.beta(a, b, n)

# ---------- core model (fib B1.2-8 for W(t)) ----------
# x_c(t) = sqrt(2 ke kc (kt*Racc_inv + eps) Cs) * sqrt(t) * W(t)
# W(t)   = (t0 / t) ^ ( (pSR * ToW_years) ^ bw / 2 )
def run_fib_carbonation(params, N=100000, seed=42,
                        t_start=0.0, t_end=120.0, t_points=240):
    rng = np.random.default_rng(seed)
    t_min = 1e-6
    t_years = np.linspace(float(max(t_start, t_min)), float(t_end), int(t_points))

    # Cover a ~ LogNormal (mm)
    a_mm = lognorm_from_mu_sd(rng, N, params["a_mu_mm"], params["a_sd_mm"])

    # ke from RH (Beta on [L,U] via mean/sd)
    RH_real = beta_interval_from_mean_sd(
        rng, N, mu=params["RH_mu"], sd=params["RH_sd"], L=params["RH_L"], U=params["RH_U"]
    )
    RH_ref  = float(params["RH_ref"])
    fe, ge  = float(params["fe"]), float(params["ge"])
    ke = ((1.0 - (RH_real/100.0)**fe) / (1.0 - (RH_ref/100.0)**fe))**ge

    # kc = (tc/7)^bc (Normal)
    tc = float(params["tc_days"])
    bc = rng.normal(params["bc_mu"], params["bc_sd"], N)
    kc = (tc/7.0)**bc

    # R_ACC,0^{-1} ~ Normal in (m^2/s)/(kg/m^3); SD = 0.69 * mu**0.78
    Racc_mu_m2s = float(params["Racc_mu_m2s"])
    Racc_sd_m2s = 0.69 * (Racc_mu_m2s ** 0.78)
    Racc_inv_m2s = rng.normal(Racc_mu_m2s, Racc_sd_m2s, N)
    Racc_inv_m2s = np.maximum(Racc_inv_m2s, 1e-20)  # avoid negatives

    # convert to (mm^2/yr)/(kg/m^3) to match the rest of the model units
    SEC_PER_YR = 365.25 * 24 * 3600.0
    M2_TO_MM2 = 1e6
    Racc_inv = Racc_inv_m2s * M2_TO_MM2 * SEC_PER_YR  # (mm^2/yr)/(kg/m^3)

    # ACC→NAC regression & error (Normal) – already in (mm^2/yr)/(kg/m^3)
    kt  = rng.normal(params["kt_mu"],  params["kt_sd"],  N)
    eps = rng.normal(params["eps_mu"], params["eps_sd"], N)
    term = np.maximum(kt * Racc_inv + eps, 1e-12)

    # Surface CO2 (Normal)
    Cs_atm = rng.normal(params["Cs_atm_mu"], params["Cs_atm_sd"], N)
    Cs_emi = rng.normal(params["Cs_emi_mu"], params["Cs_emi_sd"], N)
    Cs = np.maximum(Cs_atm + Cs_emi, 0.0)

    # Weather exponent w_sample = ((pSR * ToW_years)^bw) / 2  (Normal for bw)
    pSR = float(params["pSR"])
    ToW_years = float(params["ToW_days"]) / 365.0
    bw  = rng.normal(params["bw_mu"], params["bw_sd"], N)
    w_sample = ((pSR * ToW_years) ** bw) / 2.0

    # Base multiplier per sample
    K_base = np.sqrt(2.0 * ke * kc * term * Cs)

    # Time loop with W(t) = (t0/t)^w_sample
    Pf = []
    t0_year = float(params["t0_year"])
    for t in t_years:
        t_safe = max(t, t_min)
        W_vec = np.power(t0_year / t_safe, w_sample)    # (N,)
        xc = K_base * math.sqrt(t_safe) * W_vec         # mm
        Pf.append(np.mean(xc >= a_mm))
    Pf = np.array(Pf)
    beta = beta_from_pf(Pf)
    return pd.DataFrame({"t_years": t_years, "Pf": Pf, "beta": beta}), Racc_sd_m2s

# ---------- plotting ----------
def plot_beta(df_window, t_end, axes_cfg=None,
              show_pf=True, beta_target=None, show_beta_target=False,
              pf_mode="overlay"):
    x_abs = df_window["t_years"].to_numpy()
    y_beta = df_window["beta"].to_numpy()
    y_pf   = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_abs, y_beta, lw=2)
    ax1.set_xlabel("Time (yr)")
    ax1.set_ylabel("Reliability index β(-)")
    ax1.grid(True)

    ax2 = None
    if show_pf and pf_mode == "overlay":
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, linestyle="--", lw=1.6)
        ax2.set_ylabel("Failure probability Pf(t)")

    axes_cfg = axes_cfg or {}
    ax1.set_xlim(0, float(t_end))
    if axes_cfg.get("x_tick"):
        ax1.set_xticks(np.arange(0, float(t_end)+1e-12, axes_cfg["x_tick"]))
    if axes_cfg.get("y1_min") is not None and axes_cfg.get("y1_max") is not None:
        ax1.set_ylim(axes_cfg["y1_min"], axes_cfg["y1_max"])
    if axes_cfg.get("y1_tick"):
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax+1e-12, axes_cfg["y1_tick"]))
    if ax2 is not None:
        if axes_cfg.get("y2_min") is not None and axes_cfg.get("y2_max") is not None:
            ax2.set_ylim(axes_cfg["y2_min"], axes_cfg["y2_max"])
        if axes_cfg.get("y2_tick"):
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_yticks(np.arange(ymin2, ymax2+1e-12, axes_cfg["y2_tick"]))

    ax1.axvline(float(t_end), linestyle=":", lw=1.5)
    if show_beta_target and (beta_target is not None):
        ax1.axhline(beta_target, color="red", linestyle="--", lw=1.6)
        crossing = None
        for i in range(len(y_beta)-1):
            if (y_beta[i]-beta_target)*(y_beta[i+1]-beta_target) <= 0:
                t1,t2 = x_abs[i], x_abs[i+1]
                b1,b2 = y_beta[i], y_beta[i+1]
                if b2 != b1:
                    crossing = t1 + (beta_target-b1)*(t2-t1)/(b2-b1)
                break
        msg = f"Year reached: {crossing:.2f} yr" if crossing is not None else "Year reached: —"
        txt = ax1.text(0.98, 0.98, msg, transform=ax1.transAxes, fontsize=10,
                       va='top', ha='right',
                       bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#333', alpha=0.98))
        txt.set_path_effects([pe.withSimplePatchShadow(offset=(2,-2), shadow_rgbFace=(0,0,0), alpha=0.18)])
        if crossing is not None:
            ax1.axvline(crossing, color='red', linestyle=':', lw=1.0, alpha=0.7)

    fig.tight_layout()
    return fig, (x_abs, y_pf)

def plot_pf_only(x_years, y_pf):
    fig_pf, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x_years, y_pf, lw=2)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Failure probability Pf(t)")
    ax.grid(True)
    fig_pf.tight_layout()
    return fig_pf

# ---------- one-column UI (labels include distribution types) ----------
st.title("fib carbonation – one column (B1.2-8 W(t))")

# Cover — LogNormal
a_mu_mm  = st.number_input("Cover μ (mm) — LogNormal", value=44.96)
a_sd_mm  = st.number_input("Cover σ (mm) — LogNormal", value=8.09)

# RH_real — Beta on [L,U] via mean/sd
RH_mu    = st.number_input("RH_real mean (%) — Beta[L,U]", value=70.0)
RH_sd    = st.number_input("RH_real SD (%) — Beta[L,U]", value=5.0)
RH_L     = st.number_input("RH lower bound L (%) — Beta[L,U]", value=40.0)
RH_U     = st.number_input("RH upper bound U (%) — Beta[L,U]", value=100.0)
RH_ref   = st.number_input("RH_ref (%) — Constant", value=65.0)
fe       = st.number_input("f_e — Constant", value=5.0)
ge       = st.number_input("g_e — Constant", value=2.5)

# kc — Normal on b_c, then kc=(tc/7)^b_c
tc_days  = st.number_input("t_c (days) — Constant", value=3.0)
bc_mu    = st.number_input("b_c μ — Normal", value=-0.567, format="%.3f")
bc_sd    = st.number_input("b_c σ — Normal", value=0.02, format="%.3f")

# R_ACC,0^{-1} — Normal (m^2/s)/(kg/m^3); SD auto from μ
Racc_mu_m2s = st.number_input("R_ACC,0^{-1} μ (m²/s)/(kg/m³) — Normal", value=8.67e-11, format="%.2e")
# display computed SD; locked (comes from 0.69 * μ^0.78)
Racc_sd_m2s_display = 0.69 * (Racc_mu_m2s ** 0.78)
st.number_input("R_ACC,0^{-1} σ (m²/s)/(kg/m³) — Normal (auto: 0.69·μ^0.78)",
                value=float(Racc_sd_m2s_display), format="%.2e", disabled=True)

# ACC→NAC regression & error — Normal, units (mm^2/yr)/(kg/m^3)
kt_mu       = st.number_input("k_t μ — Normal", value=1.25)
kt_sd       = st.number_input("k_t σ — Normal", value=0.35)
eps_mu      = st.number_input("ε_t μ (mm²/yr)/(kg/m³) — Normal", value=315.5, format="%.3f")
eps_sd      = st.number_input("ε_t σ (mm²/yr)/(kg/m³) — Normal", value=48.0, format="%.3f")

# CO2 — Normal
Cs_atm_mu = st.number_input("C_s,atm μ (kg/m³) — Normal", value=1.63e-3, format="%.6f")
Cs_atm_sd = st.number_input("C_s,atm σ (kg/m³) — Normal", value=1.0e-6, format="%.6f")
Cs_emi_mu = st.number_input("C_s,emi μ (kg/m³) — Normal", value=0.0, format="%.6f")
Cs_emi_sd = st.number_input("C_s,emi σ (kg/m³) — Normal", value=1.0e-6, format="%.6f")

# Weather + t0 — Normal on b_w
pSR      = st.number_input("pSR — Constant", value=0.090, format="%.3f")
ToW_days = st.number_input("ToW (days ≥2.5 mm) — Constant", value=22.0, format="%.1f")
bw_mu    = st.number_input("b_w μ — Normal", value=0.446, format="%.3f")
bw_sd    = st.number_input("b_w σ — Normal", value=0.163, format="%.3f")
t0_year  = st.number_input("t₀ (years, 28 d = 0.0767) — Constant", value=0.0767, format="%.4f")

# Window/MC/axes
t_start_disp = st.number_input("Plot start time (yr)", value=0.0)
t_end        = st.number_input("Plot end / target yr", value=120.0, min_value=0.0)
t_points     = st.number_input("Number of time points", value=240, min_value=20, step=10)
N            = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, step=1000)
seed         = st.number_input("Random seed", value=42, step=1)
beta_target      = st.number_input("Target β", value=1.30)
show_beta_target = st.checkbox("Show target β line", value=True)
pf_mode = st.radio("Pf display mode", ["Overlay on β figure", "Separate Pf figure"], index=0, horizontal=True)
x_tick  = st.number_input("X tick step (yr)", value=10.0)
y1_tick = st.number_input("Y₁ (β) tick step", value=0.5)
y1_min  = st.number_input("Y₁ (β) min", value=-2.0)
y1_max  = st.number_input("Y₁ (β) max", value=6.0)
y2_tick = st.number_input("Y₂ (Pf) tick step", value=0.1)
y2_min  = st.number_input("Y₂ (Pf) min", value=0.0)
y2_max  = st.number_input("Y₂ (Pf) max", value=1.0)

if st.button("Run Simulation", type="primary"):
    try:
        params = dict(
            a_mu_mm=float(a_mu_mm), a_sd_mm=float(a_sd_mm),
            RH_mu=float(RH_mu), RH_sd=float(RH_sd), RH_L=float(RH_L), RH_U=float(RH_U),
            RH_ref=float(RH_ref), fe=float(fe), ge=float(ge),
            tc_days=float(tc_days), bc_mu=float(bc_mu), bc_sd=float(bc_sd),
            Racc_mu_m2s=float(Racc_mu_m2s),
            kt_mu=float(kt_mu), kt_sd=float(kt_sd),
            eps_mu=float(eps_mu), eps_sd=float(eps_sd),
            Cs_atm_mu=float(Cs_atm_mu), Cs_atm_sd=float(Cs_atm_sd),
            Cs_emi_mu=float(Cs_emi_mu), Cs_emi_sd=float(Cs_emi_sd),
            pSR=float(pSR), ToW_days=float(ToW_days),
            bw_mu=float(bw_mu), bw_sd=float(bw_sd),
            t0_year=float(t0_year),
        )

        df_full, _r_sd = run_fib_carbonation(
            params, N=int(N), seed=int(seed),
            t_start=0.0, t_end=float(t_end), t_points=int(t_points)
        )
        df_window = df_full[(df_full["t_years"] >= float(max(t_start_disp, 1e-6))) &
                            (df_full["t_years"] <= float(t_end))].copy()
        if df_window.empty:
            st.error("No points in display window; increase time points or adjust times.")
        else:
            axes_cfg = dict(
                x_tick=float(x_tick),
                y1_min=float(y1_min), y1_max=float(y1_max), y1_tick=float(y1_tick),
                y2_min=float(y2_min), y2_max=float(y2_max), y2_tick=float(y2_tick),
            )
            fig, (x_pf, y_pf) = plot_beta(
                df_window, t_end=float(t_end), axes_cfg=axes_cfg, show_pf=True,
                beta_target=float(beta_target), show_beta_target=bool(show_beta_target),
                pf_mode=("overlay" if pf_mode == "Overlay on β figure" else "separate"),
            )
            st.pyplot(fig)

            if pf_mode == "Separate Pf figure":
                fig_pf = plot_pf_only(x_pf, y_pf)
                st.pyplot(fig_pf)

            st.download_button(
                "Download window data (CSV)",
                data=df_window.to_csv(index=False).encode("utf-8"),
                file_name="fib_carbonation_output_window.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Invalid input or computation error: {e}")
