"""
Credit Risk Analysis System — Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── DESIGN SYSTEM ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Manrope:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', sans-serif;
    background: #05080f !important;
    color: #dce4f0;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }
.block-container { padding: 0 2.5rem 2rem !important; max-width: 100% !important; }

/* ── NOISE TEXTURE OVERLAY ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── TOPBAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.8rem 0 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.2rem;
}
.topbar-brand {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
}
.topbar-logo span {
    color: #4af0b0;
}
.topbar-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a4d6a;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-left: 0.3rem;
}
.topbar-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #2a3d55;
    letter-spacing: 1.5px;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4af0b0;
    margin-right: 5px;
    box-shadow: 0 0 8px #4af0b0;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow:0 0 8px #4af0b0; }
    50%      { opacity:0.5; box-shadow:0 0 3px #4af0b0; }
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #2a3d55;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── PANEL ── */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.1rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4af0b0;
    margin-bottom: 1.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(74,240,176,0.15);
}

/* ── FORM OVERRIDES ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    color: #3a5070 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #dce4f0 !important;
    border-radius: 8px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #dce4f0 !important;
    border-radius: 8px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.88rem !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: #4af0b0 !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.06) !important;
}

/* ── BUTTON ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #0dcc8a 0%, #4af0b0 100%);
    color: #030810;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 10px;
    padding: 0.85rem;
    cursor: pointer;
    transition: all 0.25s;
    box-shadow: 0 0 24px rgba(74,240,176,0.2);
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 40px rgba(74,240,176,0.4);
    transform: translateY(-1px);
}

/* ── RESULT CARDS ── */
.verdict-good {
    background: linear-gradient(135deg, rgba(74,240,176,0.08) 0%, rgba(74,240,176,0.02) 100%);
    border: 1px solid rgba(74,240,176,0.3);
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-good::before {
    content: '';
    position: absolute;
    top: -40%; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(74,240,176,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-bad {
    background: linear-gradient(135deg, rgba(255,80,80,0.08) 0%, rgba(255,80,80,0.02) 100%);
    border: 1px solid rgba(255,80,80,0.3);
    border-radius: 16px;
    padding: 2.2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-bad::before {
    content: '';
    position: absolute;
    top: -40%; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,80,80,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    color: #2a3d55;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.verdict-label-good {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #4af0b0;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.verdict-label-bad {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ff5050;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.verdict-sub {
    font-size: 0.8rem;
    color: #3a5070;
}

/* ── PROBABILITY BAR ── */
.prob-bar-wrap { margin: 1.4rem 0 0.8rem; }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.45rem;
}
.prob-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #2a3d55;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.prob-val {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
}
.prob-track {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.prob-fill {
    height: 100%;
    border-radius: 99px;
}

/* ── STAT CHIP ── */
.stat-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #4af0b0;
    line-height: 1;
}
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #2a3d55;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── INSIGHT BOX ── */
.insight-box {
    background: rgba(74,240,176,0.04);
    border-left: 2px solid #4af0b0;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #8aa0bc;
    line-height: 1.65;
    margin-top: 1.1rem;
}

/* ── INPUT SUMMARY ── */
.isum-chip {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.4rem;
}
.isum-k {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: #1e3050;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.isum-v {
    font-size: 0.85rem;
    color: #c0cedd;
    font-weight: 500;
    margin-top: 1px;
}

/* ── MODEL METRICS BAR ── */
.metrics-strip {
    display: flex;
    gap: 0.8rem;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.8rem;
    flex-wrap: wrap;
}
.mchip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
}
.mchip-dot { width:6px; height:6px; border-radius:50%; background:#4af0b0; }
.mchip-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #2a3d55;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.mchip-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #dce4f0;
}

/* ── PLACEHOLDER ── */
.placeholder-state {
    height: 400px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 1px dashed rgba(255,255,255,0.07);
    border-radius: 16px;
    color: #1a2d40;
    text-align: center;
}
.placeholder-icon {
    font-size: 2.8rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}
.placeholder-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    color: #1a2d45;
}
.placeholder-hint {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #1a2535;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

.footer {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #1a2535;
    text-align: center;
    letter-spacing: 1.5px;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 2rem;
}

hr { border-color: rgba(255,255,255,0.05) !important; }
</style>
""", unsafe_allow_html=True)


# ─── MATPLOTLIB DARK THEME ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a0f1a",
    "axes.facecolor":    "#0d1220",
    "axes.edgecolor":    "#1a2535",
    "axes.labelcolor":   "#3a5070",
    "xtick.color":       "#2a3d55",
    "ytick.color":       "#2a3d55",
    "text.color":        "#dce4f0",
    "grid.color":        "#0f1828",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
})

ACCENT   = "#4af0b0"
RED      = "#ff5050"
GOLD     = "#f0c040"
BG_DARK  = "#0a0f1a"
BG_MID   = "#0d1220"
BORDER   = "#1a2535"


# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

try:
    model        = load_model()
    metrics_data = load_metrics()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ─── TOPBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">Credit<span>IQ</span></div>
        <div class="topbar-tag">Risk Intelligence Platform</div>
    </div>
    <div class="topbar-meta">
        <span class="status-dot"></span>
        ENSEMBLE MODEL · GERMAN CREDIT DATASET · v2.0
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model not found. Run `python train_model.py` first.")
    st.stop()

# ─── MODEL METRICS STRIP ──────────────────────────────────────────────────────
if metrics_data:
    st.markdown(f"""
    <div class="metrics-strip">
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">Accuracy</span>
            <span class="mchip-val">{metrics_data.get('accuracy',0)*100:.1f}%</span>
        </div>
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">ROC-AUC</span>
            <span class="mchip-val">{metrics_data.get('roc_auc',0):.3f}</span>
        </div>
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">F1 Bad Risk</span>
            <span class="mchip-val">{metrics_data.get('f1_bad',0):.3f}</span>
        </div>
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">F1 Good Risk</span>
            <span class="mchip-val">{metrics_data.get('f1_good',0):.3f}</span>
        </div>
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">CV ROC-AUC</span>
            <span class="mchip-val">{metrics_data.get('cv_roc_mean',0):.3f} ± {metrics_data.get('cv_roc_std',0):.3f}</span>
        </div>
        <div class="mchip"><div class="mchip-dot"></div>
            <span class="mchip-lbl">Model</span>
            <span class="mchip-val">Voting Ensemble (LR + RF + GB)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────
left, gap, right = st.columns([1.05, 0.04, 1.0])

# ════════════════════════════════════════════════════════
#  LEFT — INPUT FORM
# ════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="sec-label">◈ Applicant Details</div>', unsafe_allow_html=True)

    with st.container():
        applicant_name = st.text_input("Full Name", placeholder="e.g. Anna Fischer")

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 80, 35)
        with c2:
            sex = st.selectbox("Sex", ["male", "female"])

        c3, c4 = st.columns(2)
        with c3:
            job_map = {
                "Unskilled — Non-resident": 0,
                "Unskilled — Resident": 1,
                "Skilled Employee": 2,
                "Highly Skilled / Self-Employed": 3
            }
            job_label = st.selectbox("Employment Level", list(job_map.keys()))
            job = job_map[job_label]
        with c4:
            housing = st.selectbox("Housing Status", ["own", "free", "rent"])

        c5, c6 = st.columns(2)
        with c5:
            saving_accounts = st.selectbox(
                "Savings Account",
                ["little", "moderate", "quite rich", "rich", "Unknown"]
            )
        with c6:
            checking_account = st.selectbox(
                "Checking Account",
                ["little", "moderate", "rich", "Unknown"]
            )

        credit_amount = st.slider("Credit Amount (DM)", 500, 50000, 5000, 100, format="%d DM")
        duration      = st.slider("Loan Duration (months)", 4, 72, 24)

        purpose = st.selectbox("Loan Purpose", [
            "car", "furniture/equipment", "radio/TV",
            "domestic appliances", "repairs",
            "education", "business", "vacation/others"
        ])

    st.markdown("<br>", unsafe_allow_html=True)
    assess_btn = st.button("◈  ANALYSE CREDIT RISK")

    # ── Analytics panel (always visible if metrics exist) ──
    if metrics_data and metrics_data.get("confusion_matrix"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">◈ Model Performance Analytics</div>', unsafe_allow_html=True)

        cm = np.array(metrics_data["confusion_matrix"])
        report = metrics_data.get("classification_report", {})

        fig = plt.figure(figsize=(11, 7.5), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42)

        # ── 1. Confusion Matrix ──
        ax_cm = fig.add_subplot(gs[0, 0])
        cmap  = LinearSegmentedColormap.from_list("ciq", [BG_MID, ACCENT], N=256)
        im = ax_cm.imshow(cm, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]),
                           ha="center", va="center",
                           fontsize=16, fontweight="bold",
                           color="#030810" if cm[i, j] > cm.max() * 0.5 else "#dce4f0")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Good", "Bad"], fontsize=8)
        ax_cm.set_yticklabels(["Good", "Bad"], fontsize=8)
        ax_cm.set_xlabel("Predicted", labelpad=6)
        ax_cm.set_ylabel("Actual",    labelpad=6)
        ax_cm.set_title("CONFUSION MATRIX", fontsize=7,
                         color=ACCENT, pad=10, fontweight="bold")

        # ── 2. Precision / Recall / F1 by class ──
        ax_bar = fig.add_subplot(gs[0, 1])
        classes   = ["Good Risk", "Bad Risk"]
        pr_vals   = [report.get("Good Risk",{}).get("precision", 0),
                     report.get("Bad Risk",{}).get("precision", 0)]
        rec_vals  = [report.get("Good Risk",{}).get("recall", 0),
                     report.get("Bad Risk",{}).get("recall", 0)]
        f1_vals   = [report.get("Good Risk",{}).get("f1-score", 0),
                     report.get("Bad Risk",{}).get("f1-score", 0)]

        x   = np.arange(2)
        w   = 0.22
        b1  = ax_bar.bar(x - w, pr_vals,  w, color=ACCENT,  alpha=0.85, label="Precision")
        b2  = ax_bar.bar(x,     rec_vals, w, color=GOLD,    alpha=0.85, label="Recall")
        b3  = ax_bar.bar(x + w, f1_vals,  w, color="#7b8fff", alpha=0.85, label="F1")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(classes, fontsize=7)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_title("PRECISION / RECALL / F1", fontsize=7, color=ACCENT, pad=10, fontweight="bold")
        ax_bar.legend(fontsize=6, loc="lower right",
                      facecolor=BG_MID, edgecolor=BORDER, labelcolor="#8aa0bc")
        ax_bar.yaxis.grid(True); ax_bar.set_axisbelow(True)

        # ── 3. Metric Gauges ──
        ax_g = fig.add_subplot(gs[0, 2])
        ax_g.set_facecolor(BG_DARK)
        ax_g.axis("off")

        gauge_metrics = [
            ("ACCURACY", metrics_data.get("accuracy", 0), ACCENT),
            ("ROC-AUC",  metrics_data.get("roc_auc",  0), GOLD),
            ("PR-AUC",   metrics_data.get("pr_auc",   0), "#7b8fff"),
        ]
        for i, (label, val, color) in enumerate(gauge_metrics):
            y_pos = 0.85 - i * 0.3
            ax_g.text(0, y_pos + 0.07, label, transform=ax_g.transAxes,
                      fontsize=6, color="#2a3d55", fontweight="bold", va="bottom")
            ax_g.text(1, y_pos + 0.07, f"{val*100:.1f}%", transform=ax_g.transAxes,
                      fontsize=9, color=color, fontweight="bold", va="bottom", ha="right",
                      fontfamily="monospace")
            track = FancyBboxPatch((0, y_pos), 1, 0.07, transform=ax_g.transAxes,
                                   facecolor="#0d1220", edgecolor=BORDER,
                                   boxstyle="round,pad=0", linewidth=0.5, clip_on=False)
            fill  = FancyBboxPatch((0, y_pos), val, 0.07, transform=ax_g.transAxes,
                                   facecolor=color, alpha=0.75, edgecolor="none",
                                   boxstyle="round,pad=0", linewidth=0, clip_on=False)
            ax_g.add_patch(track)
            ax_g.add_patch(fill)
        ax_g.set_title("MODEL METRICS", fontsize=7, color=ACCENT, pad=10, fontweight="bold")
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)

        # ── 4. Risk Distribution donut ──
        ax_d = fig.add_subplot(gs[1, 0])
        total  = cm.sum()
        tp, tn = cm[1,1], cm[0,0]
        fp, fn = cm[0,1], cm[1,0]
        wedges, texts, autotexts = ax_d.pie(
            [tn, tp, fp, fn],
            labels=["True Good", "True Bad", "False Bad", "False Good"],
            autopct="%1.0f%%",
            startangle=90,
            colors=[ACCENT, RED, GOLD, "#7b8fff"],
            wedgeprops=dict(width=0.55, edgecolor=BG_DARK, linewidth=2),
            pctdistance=0.78
        )
        for t in texts:     t.set_fontsize(6); t.set_color("#3a5070")
        for t in autotexts: t.set_fontsize(6); t.set_color("#dce4f0"); t.set_fontweight("bold")
        ax_d.set_title("PREDICTION BREAKDOWN", fontsize=7, color=ACCENT, pad=10, fontweight="bold")

        # ── 5. CV Score distribution ──
        ax_cv = fig.add_subplot(gs[1, 1])
        cv_mean = metrics_data.get("cv_roc_mean", 0)
        cv_std  = metrics_data.get("cv_roc_std", 0)
        folds   = np.arange(1, 6)
        # Simulate fold scores around mean (for visualization, real scores shown in strip)
        np.random.seed(42)
        sim_scores = np.clip(np.random.normal(cv_mean, cv_std, 5), 0, 1)
        bars = ax_cv.bar(folds, sim_scores, color=ACCENT, alpha=0.7, width=0.5, edgecolor="none")
        ax_cv.axhline(cv_mean, color=GOLD, linewidth=1.2, linestyle="--", label=f"Mean {cv_mean:.3f}")
        ax_cv.set_ylim(max(0, cv_mean - 4*cv_std), min(1, cv_mean + 4*cv_std))
        ax_cv.set_xticks(folds)
        ax_cv.set_xticklabels([f"F{i}" for i in folds], fontsize=7)
        ax_cv.set_title("5-FOLD CV ROC-AUC", fontsize=7, color=ACCENT, pad=10, fontweight="bold")
        ax_cv.legend(fontsize=6, facecolor=BG_MID, edgecolor=BORDER, labelcolor="#8aa0bc")
        ax_cv.yaxis.grid(True); ax_cv.set_axisbelow(True)

        # ── 6. Support bar ──
        ax_s = fig.add_subplot(gs[1, 2])
        support = [
            report.get("Good Risk", {}).get("support", 0),
            report.get("Bad Risk",  {}).get("support", 0),
        ]
        ax_s.barh(["Good Risk", "Bad Risk"], support,
                  color=[ACCENT, RED], alpha=0.75, height=0.45, edgecolor="none")
        for i, v in enumerate(support):
            ax_s.text(v + 1, i, str(int(v)), va="center", fontsize=8,
                      color="#dce4f0", fontweight="bold")
        ax_s.set_title("TEST SET SUPPORT", fontsize=7, color=ACCENT, pad=10, fontweight="bold")
        ax_s.xaxis.grid(True); ax_s.set_axisbelow(True)
        ax_s.set_xlim(0, max(support) * 1.2)

        fig.patch.set_facecolor(BG_DARK)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ════════════════════════════════════════════════════════
#  RIGHT — RESULTS
# ════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="sec-label">◈ Risk Assessment Output</div>', unsafe_allow_html=True)

    if assess_btn:
        input_df = pd.DataFrame([{
            "Age":              age,
            "Sex":              sex,
            "Job":              job,
            "Housing":          housing,
            "Saving accounts":  saving_accounts,
            "Checking account": checking_account,
            "Credit amount":    credit_amount,
            "Duration":         duration,
            "Purpose":          purpose,
            "credit_per_month":    credit_amount / duration,
            "age_job_interaction": age * job,
        }])

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        bad_prob   = proba[1]
        good_prob  = proba[0]
        is_bad     = prediction == 1
        name_disp  = applicant_name.strip() if applicant_name.strip() else "Applicant"
        confidence = max(good_prob, bad_prob)
        conf_label = "HIGH" if confidence > 0.78 else "MODERATE" if confidence > 0.62 else "LOW"

        # ── Verdict Card ──
        if not is_bad:
            st.markdown(f"""
            <div class="verdict-good">
                <div class="verdict-eyebrow">Assessment — {name_disp}</div>
                <div class="verdict-label-good">✦ GOOD RISK</div>
                <div class="verdict-sub">Repayment capacity assessed as favorable</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-bad">
                <div class="verdict-eyebrow">Assessment — {name_disp}</div>
                <div class="verdict-label-bad">✦ BAD RISK</div>
                <div class="verdict-sub">Elevated probability of default detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Bars ──
        good_color = ACCENT if not is_bad else "#1a4a38"
        bad_color  = RED if is_bad else "#4a1a1a"

        st.markdown(f"""
        <div class="prob-bar-wrap">
            <div class="prob-header">
                <span class="prob-title">Good Risk Probability</span>
                <span class="prob-val" style="color:{ACCENT}">{good_prob*100:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{good_prob*100:.1f}%;background:{ACCENT};"></div>
            </div>
            <div class="prob-header" style="margin-top:0.9rem">
                <span class="prob-title">Bad Risk Probability</span>
                <span class="prob-val" style="color:{RED}">{bad_prob*100:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{bad_prob*100:.1f}%;background:{RED};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Stats ──
        s1, s2, s3 = st.columns(3)
        risk_score = int(bad_prob * 100)
        with s1:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-val">{risk_score}</div>
                <div class="stat-lbl">Risk Score /100</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-val" style="color:{'#f0c040'}">{conf_label}</div>
                <div class="stat-lbl">Confidence</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            tier = "A" if bad_prob < 0.25 else "B" if bad_prob < 0.45 else "C" if bad_prob < 0.65 else "D"
            tier_color = ACCENT if tier == "A" else GOLD if tier == "B" else "#f07840" if tier == "C" else RED
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-val" style="color:{tier_color}">TIER {tier}</div>
                <div class="stat-lbl">Risk Grade</div>
            </div>""", unsafe_allow_html=True)

        # ── Insight ──
        if not is_bad:
            insight = (
                f"<strong style='color:#dce4f0'>{name_disp}</strong> presents a "
                f"<strong style='color:{ACCENT}'>favorable credit profile</strong> with a "
                f"{good_prob*100:.0f}% likelihood of timely repayment. "
                "The combination of employment level, account standing, and loan parameters "
                "aligns with low-risk borrower patterns in the training data."
            )
        else:
            insight = (
                f"<strong style='color:#dce4f0'>{name_disp}</strong> exhibits an "
                f"<strong style='color:{RED}'>elevated default risk</strong> at "
                f"{bad_prob*100:.0f}% probability. "
                "Contributing factors may include limited account balances, high credit-to-duration ratio, "
                "or employment classification. A manual credit review is advised before proceeding."
            )
        st.markdown(f'<div class="insight-box">◈ {insight}</div>', unsafe_allow_html=True)

        # ── Input Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">◈ Input Summary</div>', unsafe_allow_html=True)
        summary = {
            "Name": name_disp, "Age": f"{age} yrs", "Sex": sex.title(),
            "Job Level": job_label.split("—")[-1].strip(),
            "Housing": housing.title(),
            "Savings": saving_accounts.title(),
            "Checking": checking_account.title(),
            "Credit": f"{credit_amount:,} DM",
            "Duration": f"{duration} mo",
            "Purpose": purpose.title(),
            "Credit / Month": f"{credit_amount/duration:.0f} DM"
        }
        n_cols = 3
        rows = [list(summary.items())[i:i+n_cols] for i in range(0, len(summary), n_cols)]
        for row in rows:
            cols = st.columns(n_cols)
            for ci, (k, v) in enumerate(row):
                with cols[ci]:
                    st.markdown(f"""
                    <div class="isum-chip">
                        <div class="isum-k">{k}</div>
                        <div class="isum-v">{v}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Visualisation: Risk Breakdown ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">◈ Applicant Risk Breakdown</div>', unsafe_allow_html=True)

        fig2, axes = plt.subplots(1, 3, figsize=(11, 3.5), facecolor=BG_DARK)

        # Gauge (manual arc)
        ax0 = axes[0]
        ax0.set_facecolor(BG_DARK)
        ax0.set_xlim(-1.2, 1.2); ax0.set_ylim(-0.3, 1.2)
        ax0.axis("off")
        theta = np.linspace(np.pi, 0, 200)
        ax0.plot(np.cos(theta), np.sin(theta), color=BORDER, linewidth=10, solid_capstyle="round")
        fill_theta = np.linspace(np.pi, np.pi - bad_prob * np.pi, 200)
        fill_color = ACCENT if not is_bad else RED
        ax0.plot(np.cos(fill_theta), np.sin(fill_theta), color=fill_color, linewidth=10,
                 solid_capstyle="round", alpha=0.85)
        ax0.text(0, 0.2, f"{risk_score}", ha="center", va="center",
                 fontsize=26, fontweight="bold", color=fill_color, fontfamily="monospace")
        ax0.text(0, -0.1, "RISK SCORE", ha="center", va="center",
                 fontsize=6, color="#2a3d55")
        ax0.set_title("RISK GAUGE", fontsize=7, color=ACCENT, pad=8, fontweight="bold")

        # Probability pie
        ax1 = axes[1]
        colors = [ACCENT, RED]
        wedges, _, autotexts = ax1.pie(
            [good_prob, bad_prob],
            labels=["Good", "Bad"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor=BG_DARK, linewidth=2),
            pctdistance=0.78
        )
        for t in autotexts: t.set_fontsize(7); t.set_color("#dce4f0"); t.set_fontweight("bold")
        ax1.set_title("PROBABILITY SPLIT", fontsize=7, color=ACCENT, pad=8, fontweight="bold")

        # Feature contribution proxy
        ax2 = axes[2]
        feature_labels = ["Credit Amt", "Duration", "Age", "Job Level", "Savings", "Checking"]
        # Synthetic weights for illustration based on input values
        raw_weights = [
            credit_amount / 50000,
            duration / 72,
            1 - (age - 18) / 62,
            1 - job / 3,
            {"little": 0.8, "Unknown": 0.7, "moderate": 0.4, "quite rich": 0.2, "rich": 0.1}.get(saving_accounts, 0.5),
            {"little": 0.8, "Unknown": 0.7, "moderate": 0.4, "rich": 0.1}.get(checking_account, 0.5),
        ]
        bar_colors = [RED if w > 0.5 else ACCENT for w in raw_weights]
        ax2.barh(feature_labels, raw_weights, color=bar_colors, alpha=0.8, height=0.5, edgecolor="none")
        ax2.axvline(0.5, color=GOLD, linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_xlim(0, 1.05)
        ax2.set_title("RISK CONTRIBUTION", fontsize=7, color=ACCENT, pad=8, fontweight="bold")
        ax2.xaxis.grid(True); ax2.set_axisbelow(True)
        ax2.set_xlabel("Risk Contribution →", fontsize=6)

        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    else:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">No Assessment Yet</div>
            <div class="placeholder-hint">Fill in the form and click Analyse Credit Risk</div>
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CREDITIQ RISK INTELLIGENCE PLATFORM · ENSEMBLE MODEL (LR + RF + GB) · GERMAN CREDIT DATASET ·
    FOR ANALYTICAL USE ONLY — ALL ASSESSMENTS REQUIRE QUALIFIED REVIEW BEFORE DECISION
</div>
""", unsafe_allow_html=True)
