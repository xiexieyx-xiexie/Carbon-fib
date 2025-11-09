# app.py — fib Carbonation (ToW form) • Excel-aligned reliability tool
# UI & plotting style mirrors your chloride app

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="fib Carbonation – Reliability (ToW form)", layout="wide")

# Hide +/- spinners on number_inputs
st.markdown("""
<style>
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button{ -webkit-appearance:none; margin:0; }
input[type=number]{ -moz-appearance:textfield; }
</style>
""", unsafe_allow_html=True)

# --------------- Helpers ---------------
def beta_from_pf(Pf: np.ndarray) -> np.ndarray:
    Pf = np.clip(Pf, 1e-12, 1-1e-12)
    return -norm.ppf(Pf)

def lognorm_from_mu_sd(rng, n, mu, sd):
    """Sample lognormal using linear-space mean, sd."""
    mu = max(mu, 1e-12); sd = max(sd, 1e-12)
    sigma2 = math.log(1 + (sd**2)/(mu**2))
    mu_log = math.log(mu) - 0.5*sigma2
    sigma = math.sqrt(sigma2)
    return rng.lognormal(mu_log, sigma, n)

def truncnorm_pos(rng, n, mu, sd, xmin=0.0):
    """Normal then clip to be strictly > xmin."""
    sd = max(sd, 1e-12)
    return np.maximum(rng.normal(mu, sd, n), xmin + 1e-12)

def beta_interval(rng, n, a, b, alpha, beta):
    """Beta on [a,b] using shapes alpha,beta."""
    return a + (b-a) * rng.beta(max(alpha,1e-6), max(beta,1e-6), n)

# --------------- Core model (Excel-aligned) ---------------
# xc(t) = sqrt(2 ke kc * (kt * Racc_inv + eps) * Cs) * sqrt(t) * W
# with W = ((pSR * ToW_years) ** bw) / 2 , ToW_years = ToW_days / 365
def run_fib_carbonation_excel(params, N=100000, seed=42,
                              t_start=0.0, t_end=120.0, t_points=200):
    """
    params:
      a_mu_mm, a_sd_mm
      RH_alpha, RH_beta, RH_L, RH_U, RH_ref, fe, ge
      tc_days, bc_mu, bc_sd
      Racc_inv_mu, Racc_inv_sd
      kt_mu, kt_sd, eps_mu, eps_sd
      Cs_atm_mu, Cs_atm_sd, Cs_emi_mu, Cs_emi_sd
      pSR, ToW_days, bw_mu, bw_sd
    """
    rng = np.random.default_rng(seed)
    t_years = np.linspace(float(t_start), float(t_end), int(t_points))

    # 1) Cover a ~ LogNormal (mm)
    a_mm = lognorm_from_mu_sd(rng, N, params["a_mu_mm"], params["a_sd_mm"])

    # 2) ke via RH (Beta on [L,U])
    RH_real = beta_interval(rng, N, params["RH_L"], params["RH_U"],
                            params["RH_alpha"], params["RH_beta"])
    RH_ref  = float(params["RH_ref"])
    fe, ge  = float(params["fe"]), float(params["ge"])
    # ke = ((1-(RH_real/100)^fe) / (1-(RH_ref/100)^fe))^ge
    ke = ((1.0 - (RH_real/100.0)**fe) / (1.0 - (RH_ref/100.0)**fe))**ge

    # 3) kc = (tc/7)^bc
    tc = float(params["tc_days"])
    bc = truncnorm_pos(rng, N, params["bc_mu"], params["bc_sd"], xmin=-10.0)  # allow negative mean but avoid NaN
    kc = (tc/7.0)**bc

    # 4) Resistance term = kt * Racc_inv + eps   (mm^2/yr)/(kg/m^3)
    Racc_inv = lognorm_from_mu_sd(rng, N, params["Racc_inv_mu"], params["Racc_inv_sd"])
    kt  = truncnorm_pos(rng, N, params["kt_mu"],  params["kt_sd"], xmin=1e-6)
    eps = rng.normal(params["eps_mu"], params["eps_sd"], N)
    term = np.maximum(kt*Racc_inv + eps, 1e-12)

    # 5) Surface CO2 (kg/m^3)
    Cs_atm = rng.normal(params["Cs_atm_mu"], params["Cs_atm_sd"], N)
    Cs_emi = rng.normal(params["Cs_emi_mu"], params["Cs_emi_sd"], N)
    Cs = np.maximum(Cs_atm + Cs_emi, 0.0)

    # 6) Weather W = ((pSR * ToW_years)^bw)/2
    pSR = float(params["pSR"])
    ToW_years = float(params["ToW_days"]) / 365.0
    bw  = rng.normal(params["bw_mu"], params["bw_sd"], N)
    W_const = (pSR * ToW_years)**bw / 2.0  # vector (bw random)

    # 7) Precompute K (vector), xc = K * sqrt(t)
    K = np.sqrt(2.0 * ke * kc * term * Cs) * W_const  # units consistent => xc in mm

    Pf = []
    for t in t_years:
        xc = K * math.sqrt(max(t, 0.0))  # mm
        Pf.append(np.mean(xc >= a_mm))   # failure when carbonation >= cover
    Pf = np.array(Pf)
    beta = beta_from_pf(Pf)
    return pd.DataFrame({"t_years": t_years, "Pf": Pf, "beta": beta})

# --------------- Plotting ---------------
def plot_beta(df_window, t_end, axes_cfg=None,
              show_pf=True, beta_target=None, show_beta_target=False,
              pf_mode="overlay"):
    x_abs = df_window["t_years"].to_numpy()
    y_beta = df_window["beta"].to_numpy()
    y_pf   = df_window["Pf"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x_abs, y_beta, lw=2, label="β(t)")
    ax1.set_xlabel("Time (yr)")
    ax1.set_ylabel("Reliability index β(-)")
    ax1.grid(True)

    # Pf overlay
    overlay_pf = (pf_mode == "overlay")
    ax2 = None
    if show_pf and overlay_pf:
        ax2 = ax1.twinx()
        ax2.plot(x_abs, y_pf, linestyle="--", lw=1.6, label="Pf(t)")
        ax2.set_ylabel("Failure probability Pf(t)")

    axes_cfg = axes_cfg or {}
    ax1.set_xlim(0, float(t_end))
    x_tick = axes_cfg.get("x_tick", None)
    if x_tick and x_tick > 0:
        ax1.set_xticks(np.arange(0, float(t_end) + 1e-12, x_tick))

    y1_min = axes_cfg.get("y1_min", None)
    y1_max = axes_cfg.get("y1_max", None)
    y1_tick = axes_cfg.get("y1_tick", None)
    if y1_min is not None and y1_max is not None and y1_max > y1_min:
        ax1.set_ylim(y1_min, y1_max)
    if y1_tick and y1_tick > 0:
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(ymin, ymax + 1e-12, y1_tick))

    if ax2 is not None:
        y2_min = axes_cfg.get("y2_min", None)
        y2_max = axes_cfg.get("y2_max", None)
        y2_tick = axes_cfg.get("y2_tick", None)
        if y2_min is not None and y2_max is not None and y2_max > y2_min:
            ax2.set_ylim(y2_min, y2_max)
        if y2_tick and y2_tick > 0:
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_yticks(np.arange(ymin2, ymax2 + 1e-12, y2_tick))

    ax1.axvline(float(t_end), linestyle=":", lw=1.5)

    # Target β line + year annotation
    if show_beta_target and (beta_target is not None):
        ax1.axhline(beta_target, color="red", linestyle="--", lw=1.6)
        crossing_year = None
        for i in range(len(y_beta) - 1):
            if (y_beta[i] - beta_target) * (y_beta[i+1] - beta_target) <= 0:
                t1, t2 = x_abs[i], x_abs[i+1]
                b1, b2 = y_beta[i], y_beta[i+1]
                if (b2 - b1) != 0:
                    crossing_year = t1 + (beta_target - b1) * (t2 - t1) / (b2 - b1)
                break
        msg = f"Year reached: {crossing_year:.2f} yr" if crossing_year is not None else "Year reached: —"
        txt = ax1.text(
            0.98, 0.98, msg, transform=ax1.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#333333', alpha=0.98)
        )
        txt.set_path_effects([pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0), alpha=0.18)])
        if crossing_year is not None:
            ax1.axvline(crossing_year, color='red', linestyle=':', lw=1.0, alpha=0.7)

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

# --------------- Tiny UI helpers ---------------
def help_badge(title_key: str, default_text: str = ""):
    if hasattr(st, "popover"):
        with st.popover("?", use_container_width=False):
            st.caption(f"Help – {title_key}")
            st.write(default_text)
    else:
        with st.expander("Help"):
            st.caption(f"Help – {title_key}")
            st.write(default_text)

# --------------- Page layout ---------------
st.title("fib carbonation (ToW) – reliability index vs time")

left_col, right_col = st.columns([1.15, 1.0], vertical_alignment="top")

# ===== LEFT: Inputs (Excel-aligned) =====
with left_col:
    st.subheader("Cover (LogNormal, mm)")
    a_mu_mm = st.number_input("Cover μ (mm)", value=44.96)
    a_sd_mm = st.number_input("Cover σ (mm)", value=8.09)

    st.divider()
    st.subheader("Environmental function ke via RH (Beta on [L,U])")
    c1, c2, c3, c4 = st.columns(4)
    with c1: RH_L   = st.number_input("RH lower bound L (%)", value=40.0)
    with c2: RH_U   = st.number_input("RH upper bound U (%)", value=100.0)
    with c3: RH_a   = st.number_input("Beta α", value=17.50)
    with c4: RH_b   = st.number_input("Beta β", value=17.50)
    c1, c2, c3 = st.columns(3)
    with c1: RH_ref = st.number_input("RH_ref (%)", value=65.0)
    with c2: fe     = st.number_input("f_e", value=5.0)
    with c3: ge     = st.number_input("g_e", value=2.5)

    st.divider()
    st.subheader("Execution / curing kc = (tc/7)^{bc}")
    c1, c2 = st.columns(2)
    with c1: tc_days = st.number_input("tc (days)", value=3.0)
    with c2: bc_mu   = st.number_input("b_c μ", value=-0.567, format="%.3f")
    bc_sd = st.number_input("b_c σ", value=0.02, format="%.3f")

    st.divider()
    st.subheader("ACC resistance & conversion (mm²/yr)/(kg/m³)")
    c1, c2 = st.columns(2)
    with c1: Racc_mu = st.number_input("R_ACC,0^{-1} μ", value=2737.0, format="%.3f")
    with c2: Racc_sd = st.number_input("R_ACC,0^{-1} σ", value=1174.0, format="%.3f")
    c1, c2, c3 = st.columns(3)
    with c1: kt_mu = st.number_input("k_t μ", value=1.25)
    with c2: kt_sd = st.number_input("k_t σ", value=0.35)
    with c3: st.write("")  # spacer
    c1, c2 = st.columns(2)
    with c1: eps_mu = st.number_input("ε_t μ", value=315.5, format="%.3f")
    with c2: eps_sd = st.number_input("ε_t σ", value=48.0, format="%.3f")

    st.divider()
    st.subheader("Surface CO₂ (kg/m³)")
    c1, c2 = st.columns(2)
    with c1: Cs_atm_mu = st.number_input("C_s,atm μ", value=1.63e-3, format="%.6f")
    with c2: Cs_atm_sd = st.number_input("C_s,atm σ", value=1.00e-6, format="%.6f")
    c1, c2 = st.columns(2)
    with c1: Cs_emi_mu = st.number_input("C_s,emi μ", value=0.0, format="%.6f")
    with c2: Cs_emi_sd = st.number_input("C_s,emi σ", value=1.00e-6, format="%.6f")

    st.divider()
    st.subheader("Weather function W = ((pSR * ToW_years)^bw) / 2")
    c1, c2 = st.columns(2)
    with c1: pSR = st.number_input("pSR (prob. of driving rain)", value=0.090, format="%.3f")
    with c2: ToW_days = st.number_input("ToW (days ≥2.5 mm)", value=22.0, format="%.1f")
    c1, c2 = st.columns(2)
    with c1: bw_mu = st.number_input("b_w μ", value=0.446, format="%.3f")
    with c2: bw_sd = st.number_input("b_w σ", value=0.163, format="%.3f")

# ===== RIGHT: Time window, MC, β target, axes, Run =====
with right_col:
    st.subheader("Time window & Monte Carlo")
    tw1, tw2 = st.columns(2)
    with tw1:
        t_start_disp = st.number_input("Plot start time (yr)", value=0.0)
        t_end        = st.number_input("Plot end time / target yr", value=120.0, min_value=0.0)
        t_points     = st.number_input("Number of time points", value=240, min_value=20, step=10)
    with tw2:
        N    = st.number_input("Monte Carlo samples N", value=100000, min_value=1000, step=1000)
        seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.subheader("Target reliability")
    beta_target      = st.number_input("Target β", value=1.30)
    show_beta_target = st.checkbox("Show target β line", value=True)

    st.divider()
    pf_mode = st.radio("Pf display mode",
        options=["Overlay on β figure", "Separate Pf figure"],
        index=0, horizontal=True)

    st.subheader("Plot axes controls")
    x_tick  = st.number_input("X tick step (yr)", value=10.0)
    y1_tick = st.number_input("Y₁ (β) tick step", value=0.5)
    r3c1, r3c2 = st.columns(2)
    with r3c1: y1_min = st.number_input("Y₁ (β) min", value=-2.0)
    with r3c2: y1_max = st.number_input("Y₁ (β) max", value=6.0)
    y2_tick = st.number_input("Y₂ (Pf) tick step", value=0.1)
    r5c1, r5c2 = st.columns(2)
    with r5c1: y2_min = st.number_input("Y₂ (Pf) min", value=0.0)
    with r5c2: y2_max = st.number_input("Y₂ (Pf) max", value=1.0)

    st.divider()
    run_btn = st.button("Run Simulation", type="primary")

    if run_btn:
        if t_end <= t_start_disp:
            st.error("Plot end time must be greater than plot start time.")
        else:
            try:
                params = {
                    "a_mu_mm": float(a_mu_mm), "a_sd_mm": float(a_sd_mm),
                    "RH_alpha": float(RH_a), "RH_beta": float(RH_b),
                    "RH_L": float(RH_L), "RH_U": float(RH_U),
                    "RH_ref": float(RH_ref), "fe": float(fe), "ge": float(ge),
                    "tc_days": float(tc_days), "bc_mu": float(bc_mu), "bc_sd": float(bc_sd),
                    "Racc_inv_mu": float(Racc_mu), "Racc_inv_sd": float(Racc_sd),
                    "kt_mu": float(kt_mu), "kt_sd": float(kt_sd),
                    "eps_mu": float(eps_mu), "eps_sd": float(eps_sd),
                    "Cs_atm_mu": float(Cs_atm_mu), "Cs_atm_sd": float(Cs_atm_sd),
                    "Cs_emi_mu": float(Cs_emi_mu), "Cs_emi_sd": float(Cs_emi_sd),
                    "pSR": float(pSR), "ToW_days": float(ToW_days),
                    "bw_mu": float(bw_mu), "bw_sd": float(bw_sd),
                }

                df_full = run_fib_carbonation_excel(
                    params, N=int(N), seed=int(seed),
                    t_start=0.0, t_end=float(t_end), t_points=int(t_points)
                )
                df_window = df_full[(df_full["t_years"] >= float(t_start_disp)) &
                                    (df_full["t_years"] <= float(t_end))].copy()
                if df_window.empty:
                    st.error("No points in display window; increase time points or adjust times.")
                else:
                    axes_cfg = {
                        "x_tick":  float(x_tick),
                        "y1_min":  float(y1_min),
                        "y1_max":  float(y1_max),
                        "y1_tick": float(y1_tick),
                        "y2_min":  float(y2_min),
                        "y2_max":  float(y2_max),
                        "y2_tick": float(y2_tick),
                    }
                    fig, (x_pf, y_pf) = plot_beta(
                        df_window,
                        t_end=float(t_end),
                        axes_cfg=axes_cfg,
                        show_pf=True,
                        beta_target=float(beta_target),
                        show_beta_target=bool(show_beta_target),
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
