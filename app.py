"""Streamlit Dashboard for Quantum-Secure Communication in Biomedical Networks.

Track 2: Quantum Cryptography and Communication.
Provides an interactive visualization of the physical BB84 QKD layer, classical
threat classifier (NSL-KDD), adaptive security decision engine, and telemetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.classifier import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    get_attack_probability,
    load_nsl_kdd,
    load_sample_row,
)
from src.decision_engine import calculate_asymptotic_key_rate, evaluate_connection_security
from src.qkd import BB84Simulation, QKDMetrics, run_bb84_simulation

# Page configuration - clean light layout
st.set_page_config(
    page_title="Quantum-Secure Biomedical Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal CSS for clean typography and clean cards
st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .verdict-card {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 6px solid;
    }
    .verdict-accept {
        background-color: #F0FDF4;
        border-color: #22C55E;
        color: #15803D;
    }
    .verdict-monitor {
        background-color: #FEFCE8;
        border-color: #EAB308;
        color: #A16207;
    }
    .verdict-reject {
        background-color: #FEF2F2;
        border-color: #EF4444;
        color: #B91C1C;
    }
    .section-title {
        font-weight: 600;
        color: #1E293B;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_model():
    """Load or train Random Forest pipeline."""
    model_path = Path("models/rf_pipeline.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    # If not yet trained, train on the fly
    train_df, test_df = load_nsl_kdd()
    from src.classifier import train_and_evaluate_models

    results = train_and_evaluate_models(train_df, test_df)
    return results["Random Forest Classifier"]["pipeline"]


@st.cache_data
def load_cached_datasets():
    """Load NSL-KDD dataset splits with caching."""
    return load_nsl_kdd()


@st.cache_data
def execute_qkd_simulation(
    n_qubits: int,
    noise_level: float,
    noise_type: str,
    eve_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    """Run QKD simulation with caching."""
    metrics = run_bb84_simulation(
        n_qubits=n_qubits,
        noise_level=noise_level,
        noise_type=noise_type,  # type: ignore
        eve_fraction=eve_fraction,
        seed=seed,
    )
    return metrics.to_dict()


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("🔐 Control Center")
st.sidebar.markdown("Configure Quantum Channel & Threat Vectors")

st.sidebar.subheader("Quantum Channel Parameters")
noise_level_pct = st.sidebar.slider(
    "Channel Noise Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.5,
    help="Physical quantum optical channel noise (depolarizing/phase-flip).",
)
noise_level = noise_level_pct / 100.0

eve_fraction_pct = st.sidebar.slider(
    "Eve Interception Ratio (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=5.0,
    help="Fraction of qubits intercepted and measured by Eve in intercept-resend attack.",
)
eve_fraction = eve_fraction_pct / 100.0

noise_type = st.sidebar.selectbox(
    "Channel Noise Model",
    options=["depolarizing", "phase_flip"],
    index=0,
    help="Depolarizing affects both X and Z bases; Phase-flip affects X basis predominantly.",
)

col_q1, col_q2 = st.sidebar.columns(2)
with col_q1:
    n_qubits = st.number_input("Qubits Sent", min_value=100, max_value=5000, value=1000, step=100)
with col_q2:
    sim_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Classical Threat Layer")

threat_mode = st.sidebar.radio(
    "Threat Intelligence Source",
    options=["NSL-KDD Network Sample", "Manual Threat Probability Slider"],
    index=0,
)

current_sample_info = None
threat_prob: float = 0.0

if threat_mode == "Manual Threat Probability Slider":
    threat_prob_pct = st.sidebar.slider(
        "P(Attack) Threat Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
    )
    threat_prob = threat_prob_pct / 100.0
else:
    # NSL-KDD Mode
    try:
        train_df, test_df = load_cached_datasets()
        sample_choice = st.sidebar.selectbox(
            "Select Test Flow Type",
            options=["Normal Traffic (Low Risk)", "DoS / Probe / Intrusion (High Risk)", "Custom Index"],
        )

        if sample_choice == "Normal Traffic (Low Risk)":
            sample_df, target_val, raw_label = load_sample_row(split="test", label_type="normal")
            row_idx = int(sample_df.index[0])
        elif sample_choice == "DoS / Probe / Intrusion (High Risk)":
            sample_df, target_val, raw_label = load_sample_row(split="test", label_type="attack")
            row_idx = int(sample_df.index[0])
        else:
            row_idx = st.sidebar.number_input("Row Index", min_value=0, max_value=len(test_df) - 1, value=12)
            sample_df, target_val, raw_label = load_sample_row(index=int(row_idx), split="test")

        threat_prob = float(get_attack_probability(sample_df))
        current_sample_info = {
            "row_idx": row_idx,
            "raw_label": raw_label,
            "target": target_val,
            "sample_df": sample_df,
        }
        st.sidebar.info(
            f"**Sample #{row_idx}** | Label: `{raw_label}`\n\n"
            f"Predicted P(Attack): **{threat_prob:.1%}**"
        )
    except Exception as e:
        st.sidebar.error(f"Error loading NSL-KDD: {e}")
        threat_prob = 0.20

run_button = st.sidebar.button("🚀 Run Full Simulation", type="primary", use_container_width=True)


# -------------------------------------------------------------
# MAIN DASHBOARD CONTENT
# -------------------------------------------------------------
st.title("🛡️ Quantum-Secure Biomedical Network Layer")
st.caption(
    "Hybrid Cryptographic Assurance for Critical Hospital-to-Cloud Telemetry | Track 2: Quantum Cryptography"
)

with st.spinner("Executing BB84 quantum circuit simulation and adaptive security evaluation..."):
    # Execute QKD
    sim_data = execute_qkd_simulation(
        n_qubits=int(n_qubits),
        noise_level=noise_level,
        noise_type=noise_type,
        eve_fraction=eve_fraction,
        seed=int(sim_seed),
    )

    qber = sim_data["qber"]
    qber_ucb = sim_data["qber_ucb"]
    qber_z = sim_data["qber_z"]
    qber_x = sim_data["qber_x"]
    n_sifted = sim_data["n_sifted"]

    # Decision Engine Evaluation
    decision = evaluate_connection_security(
        qber_ucb=qber_ucb,
        n_sifted=n_sifted,
        attack_prob=threat_prob,
        base_qber_threshold=0.11,
        qber=qber,
    )

    verdict = decision["verdict"]
    max_allowed_qber = decision["max_allowed_qber"]
    key_rate = decision["key_rate"]
    reason = decision["reason"]

# Top Row Metrics Cards
st.markdown("### Real-Time Security Telemetry")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Observed QBER",
        value=f"{qber:.2%}",
        delta=f"Threshold: {max_allowed_qber:.1%}",
        delta_color="inverse" if qber > max_allowed_qber else "normal",
    )
with col2:
    st.metric(
        label="95% QBER UCB",
        value=f"{qber_ucb:.2%}",
        delta=f"+{(qber_ucb - qber):.2%} uncertainty",
        delta_color="off",
    )
with col3:
    st.metric(
        label="Secret Key Rate (R)",
        value=f"{key_rate:.4f}",
        delta=f"{int(key_rate * n_sifted)} secret bits" if n_sifted > 0 else "0 bits",
        delta_color="normal" if key_rate > 0 else "inverse",
    )
with col4:
    st.metric(
        label="Classical Threat Score",
        value=f"{threat_prob:.1%}",
        delta="ML Threat Level",
        delta_color="inverse" if threat_prob >= 0.5 else "normal",
    )
with col5:
    verdict_colors = {"ACCEPT": "🟢 ACCEPT", "MONITOR": "🟡 MONITOR", "REJECT": "🔴 REJECT"}
    st.metric(
        label="Channel Decision",
        value=verdict_colors.get(verdict, verdict),
        delta="Adaptive Policy",
        delta_color="off",
    )

# Verdict Alert Banner
card_class = f"verdict-{verdict.lower()}"
st.markdown(
    f"""
    <div class="verdict-card {card_class}">
        <h4 style="margin:0 0 6px 0;">Decision Verdict: <strong>{verdict}</strong></h4>
        <p style="margin:0; font-size:14.5px;">{reason}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Live Results & Telemetry",
        "🗺️ Adaptive Decision Map",
        "🔬 BB84 Protocol Details",
        "🤖 Classical Threat Layer",
    ]
)

# -------------------------------------------------------------
# TAB 1: LIVE RESULTS
# -------------------------------------------------------------
with tab1:
    col_l1, col_l2 = st.columns([1, 1])

    with col_l1:
        st.subheader("QBER vs. Adaptive Threshold")
        fig_g, ax_g = plt.subplots(figsize=(6, 3.8))
        categories = ["Observed QBER", "95% UCB QBER", "Max Allowed Limit"]
        values = [qber * 100, qber_ucb * 100, max_allowed_qber * 100]
        colors = ["#2563EB", "#0D9488", "#EF4444"]

        bars = ax_g.bar(categories, values, color=colors, width=0.55)
        ax_g.axhline(
            max_allowed_qber * 100,
            color="#EF4444",
            linestyle="--",
            label=f"Adaptive Ceiling ({max_allowed_qber*100:.1f}%)",
        )
        ax_g.set_ylabel("Error Rate (%)")
        ax_g.set_ylim(0, max(25.0, max(values) * 1.25))
        for bar in bars:
            height = bar.get_height()
            ax_g.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        ax_g.legend(loc="upper right")
        sns.despine()
        st.pyplot(fig_g)
        plt.close(fig_g)

    with col_l2:
        st.subheader("Basis-Resolved Error Rates")
        fig_b, ax_b = plt.subplots(figsize=(6, 3.8))
        b_labels = ["Z-Basis Error", "X-Basis Error", "Overall QBER"]
        b_vals = [qber_z * 100, qber_x * 100, qber * 100]
        b_colors = ["#6366F1", "#EC4899", "#14B8A6"]

        b_bars = ax_b.bar(b_labels, b_vals, color=b_colors, width=0.55)
        ax_b.set_ylabel("Error Rate (%)")
        ax_b.set_ylim(0, max(25.0, max(b_vals) * 1.25))
        for bar in b_bars:
            height = bar.get_height()
            ax_b.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        sns.despine()
        st.pyplot(fig_b)
        plt.close(fig_b)

    st.markdown("---")
    st.subheader("Key Generation Sifting Funnel")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.metric("1. Qubits Sent by Alice", f"{n_qubits:,}")
    with f_col2:
        sift_pct = (n_sifted / n_qubits) * 100 if n_qubits > 0 else 0
        st.metric("2. Sifted Key Bits", f"{n_sifted:,}", f"{sift_pct:.1f}% basis match")
    with f_col3:
        st.metric("3. Bit Errors in Sifted Key", f"{int(round(qber * n_sifted))}", f"QBER: {qber:.2%}")
    with f_col4:
        secure_bits = int(key_rate * n_sifted) if verdict == "ACCEPT" else 0
        st.metric(
            "4. Usable Secret Bits",
            f"{secure_bits:,}",
            f"{'Approved' if verdict=='ACCEPT' else 'Blocked ('+verdict+')'}",
            delta_color="normal" if secure_bits > 0 else "inverse",
        )

# -------------------------------------------------------------
# TAB 2: DECISION MAP
# -------------------------------------------------------------
with tab2:
    st.subheader("2D Security Decision Boundary Map")
    st.markdown(
        f"Evaluating operating parameter space at current threat level: **P(Attack) = {threat_prob:.1%}** "
        f"(Adaptive QBER Limit = **{max_allowed_qber:.2%}**)"
    )

    n_pts = 40
    noise_grid = np.linspace(0.0, 0.20, n_pts)
    eve_grid = np.linspace(0.0, 1.0, n_pts)
    grid_decisions = np.zeros((n_pts, n_pts))

    # Analytical expectation: QBER ~ 2/3 * noise + 0.25 * eve - (2/3 * noise * 0.25 * eve)
    for j, e_val in enumerate(eve_grid):
        for i, n_val in enumerate(noise_grid):
            # Theoretical expected QBER for depolarizing channel + intercept-resend
            expected_qber = (2.0 / 3.0) * n_val * (1.0 - 0.5 * e_val) + 0.25 * e_val
            # Approximate finite-key UCB
            finite_ucb = expected_qber + 1.645 * np.sqrt(
                max(0.0, expected_qber * (1.0 - expected_qber) / (n_qubits * 0.5))
            )
            d = evaluate_connection_security(
                qber_ucb=finite_ucb,
                n_sifted=int(n_qubits * 0.5),
                attack_prob=threat_prob,
                base_qber_threshold=0.11,
                qber=expected_qber,
            )
            val_map = {"ACCEPT": 0, "MONITOR": 1, "REJECT": 2}
            grid_decisions[j, i] = val_map[d["verdict"]]

    fig_map, ax_map = plt.subplots(figsize=(8, 5))
    from matplotlib.colors import ListedColormap

    cmap_dec = ListedColormap(["#2ECC71", "#F1C40F", "#E74C3C"])
    ax_map.contourf(noise_grid * 100, eve_grid * 100, grid_decisions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_dec)

    # Plot current operating point
    ax_map.scatter(
        [noise_level * 100],
        [eve_fraction * 100],
        color="black",
        s=140,
        marker="X",
        edgecolors="white",
        linewidth=2,
        zorder=6,
        label=f"Current Operating Point ({noise_level*100:.1f}%, {eve_fraction*100:.0f}%)",
    )

    ax_map.set_xlabel("Quantum Channel Noise (%)")
    ax_map.set_ylabel("Eve Interception Ratio (%)")
    ax_map.set_title(f"Operating Envelope (Threat P = {threat_prob:.1%})")
    ax_map.legend(loc="upper right", framealpha=0.9)

    st.pyplot(fig_map)
    plt.close(fig_map)

# -------------------------------------------------------------
# TAB 3: PROTOCOL DETAILS
# -------------------------------------------------------------
with tab3:
    st.subheader("First 20 Qubits Micro-Trace")
    st.markdown("Detailed step-by-step physical state trace from Alice through Eve to Bob:")

    alice_bits = np.array(sim_data["alice_bits"])
    alice_bases = np.array(sim_data["alice_bases"])
    eve_intercepted = np.array(sim_data["eve_intercepted"])
    eve_bases = np.array(sim_data["eve_bases"])
    eve_bits = np.array(sim_data["eve_bits"])
    bob_bases = np.array(sim_data["bob_bases"])
    bob_bits = np.array(sim_data["bob_bits"])

    basis_names = {0: "Z (|0>, |1>)", 1: "X (|+>, |->)"}
    preview_rows = []
    n_display = min(20, len(alice_bits))

    for idx in range(n_display):
        a_b = int(alice_bits[idx])
        a_bas = "Z" if alice_bases[idx] == 0 else "X"
        is_eve = bool(eve_intercepted[idx])
        e_bas = ("Z" if eve_bases[idx] == 0 else "X") if is_eve else "-"
        e_bit = str(eve_bits[idx]) if is_eve else "-"
        b_bas = "Z" if bob_bases[idx] == 0 else "X"
        b_bit = int(bob_bits[idx])

        sifted = a_bas == b_bas
        error_flag = (a_b != b_bit) if sifted else "-"

        preview_rows.append(
            {
                "Qubit #": idx + 1,
                "Alice Bit": a_b,
                "Alice Basis": a_bas,
                "Eve Intercept?": "⚠️ Yes" if is_eve else "No",
                "Eve Basis": e_bas,
                "Eve Measured": e_bit,
                "Bob Basis": b_bas,
                "Bob Measured": b_bit,
                "Sifted?": "✅ Kept" if sifted else "❌ Discarded",
                "Bit Error": "🔴 Error" if error_flag is True else ("🟢 Match" if error_flag is False else "-"),
            }
        )

    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 4: THREAT LAYER
# -------------------------------------------------------------
with tab4:
    st.subheader("Classical ML Threat Classification (NSL-KDD)")
    try:
        model = load_cached_model()
        st.markdown(
            "The classical threat layer models hospital network traffic using Random Forest and Logistic Regression "
            "pipelines trained on NSL-KDD benchmark telemetry."
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Model Performance Metrics (NSL-KDD Test Set)")
            metrics_summary = pd.DataFrame(
                [
                    {
                        "Model": "Random Forest (Production)",
                        "Accuracy": "77.59%",
                        "Precision": "96.79%",
                        "Recall": "62.71%",
                        "F1-Score": "0.7611",
                    },
                    {
                        "Model": "Logistic Regression (Baseline)",
                        "Accuracy": "75.39%",
                        "Precision": "91.74%",
                        "Recall": "62.39%",
                        "F1-Score": "0.7427",
                    },
                ]
            )
            st.table(metrics_summary)

        with col_m2:
            st.markdown("#### Confusion Matrix (Random Forest)")
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
            cm_data = np.array([[9444, 267], [4786, 8047]])
            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                cbar=False,
                ax=ax_cm,
            )
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Label")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        if current_sample_info is not None:
            st.markdown("---")
            st.markdown(f"#### Inspected Network Flow Features (Row #{current_sample_info['row_idx']})")
            sample_features_df = current_sample_info["sample_df"][CATEGORICAL_FEATURES + NUMERICAL_FEATURES[:12]].astype(str)
            st.dataframe(sample_features_df.T.rename(columns={sample_features_df.index[0]: "Value"}))

    except Exception as e:
        st.error(f"Error displaying Threat Layer: {e}")

