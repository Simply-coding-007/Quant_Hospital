"""Presentation Figures Generator for Quantum-Secure Biomedical Networks.

Generates high-resolution (300 DPI) publication-ready plots:
1. results/qber_vs_noise.png: QBER vs Quantum Noise (0-20%) for Eve=0% vs Eve=50% with 11% threshold.
2. results/key_rate_curve.png: Asymptotic Key Rate R vs QBER with ~11% zero-crossing annotation.
3. results/detection_heatmap.png: 8x8 Decision Heatmap (Noise x Eve) for fixed threat p=0.3 with disk caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns

from src.decision_engine import calculate_asymptotic_key_rate, evaluate_connection_security
from src.qkd import run_bb84_simulation


def setup_style() -> None:
    """Configure clean, professional seaborn whitegrid style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def generate_qber_vs_noise_plot(results_dir: Path = Path("results")) -> Path:
    """Generate Figure 1: QBER vs Noise level for Eve=0% and Eve=50%."""
    print("Generating Figure 1: qber_vs_noise.png...")
    noise_levels = np.linspace(0.0, 0.20, 11)
    n_reps = 5
    n_qubits = 1000

    qber_no_eve = np.zeros((len(noise_levels), n_reps))
    qber_eve_50 = np.zeros((len(noise_levels), n_reps))

    for i, noise in enumerate(noise_levels):
        for r in range(n_reps):
            seed = 100 + i * 10 + r
            # Eve = 0%
            res_0 = run_bb84_simulation(
                n_qubits=n_qubits, noise_level=noise, eve_fraction=0.0, seed=seed
            )
            qber_no_eve[i, r] = res_0.qber

            # Eve = 50%
            res_50 = run_bb84_simulation(
                n_qubits=n_qubits, noise_level=noise, eve_fraction=0.5, seed=seed
            )
            qber_eve_50[i, r] = res_50.qber

    mean_no_eve = np.mean(qber_no_eve, axis=1) * 100
    std_no_eve = np.std(qber_no_eve, axis=1) * 100

    mean_eve_50 = np.mean(qber_eve_50, axis=1) * 100
    std_eve_50 = np.std(qber_eve_50, axis=1) * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    # Colorblind friendly colors
    c_blue = "#0072B2"
    c_orange = "#D55E00"
    c_red = "#CC79A7"

    ax.plot(
        noise_levels * 100,
        mean_no_eve,
        "o-",
        color=c_blue,
        linewidth=2.2,
        markersize=6,
        label="No Eavesdropper (Eve = 0%)",
    )
    ax.fill_between(
        noise_levels * 100,
        mean_no_eve - std_no_eve,
        mean_no_eve + std_no_eve,
        color=c_blue,
        alpha=0.18,
    )

    ax.plot(
        noise_levels * 100,
        mean_eve_50,
        "s-",
        color=c_orange,
        linewidth=2.2,
        markersize=6,
        label="Active Interception (Eve = 50%)",
    )
    ax.fill_between(
        noise_levels * 100,
        mean_eve_50 - std_eve_50,
        mean_eve_50 + std_eve_50,
        color=c_orange,
        alpha=0.18,
    )

    # Base QBER threshold line
    ax.axhline(
        11.0,
        color="#D9534F",
        linestyle="--",
        linewidth=1.8,
        label="BB84 Theoretical Limit (11.0% QBER)",
    )

    ax.set_xlabel("Channel Depolarizing Noise Level (%)")
    ax.set_ylabel("Measured QBER (%)")
    ax.set_title("Quantum Bit Error Rate (QBER) vs. Channel Noise")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 30.0)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    out_path = results_dir / "qber_vs_noise.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_key_rate_curve_plot(results_dir: Path = Path("results")) -> Path:
    """Generate Figure 2: Asymptotic Key Generation Rate R vs QBER."""
    print("Generating Figure 2: key_rate_curve.png...")
    qber_vals = np.linspace(0.0, 0.16, 200)
    key_rates = [calculate_asymptotic_key_rate(q) for q in qber_vals]

    fig, ax = plt.subplots(figsize=(8, 5))

    c_green = "#009E73"
    ax.plot(qber_vals * 100, key_rates, color=c_green, linewidth=2.8, label="Key Rate $R = 1 - 2h(QBER)$")

    # Fill secure zone
    ax.fill_between(
        qber_vals[qber_vals <= 0.11] * 100,
        [calculate_asymptotic_key_rate(q) for q in qber_vals if q <= 0.11],
        color=c_green,
        alpha=0.15,
        label="Secure Region ($R > 0$)",
    )

    # Fill abort zone
    ax.axvspan(11.0, 16.0, color="#D9534F", alpha=0.12, label="Insecure Region ($R = 0$, Abort)")

    # Zero crossing annotation
    ax.axvline(11.0, color="#D9534F", linestyle="--", linewidth=1.6)
    ax.scatter([11.0], [0.0], color="#D9534F", s=70, zorder=5)
    ax.annotate(
        "Shor-Preskill / Devetak-Winter Bound\n$QBER \\approx 11.00\\% \\rightarrow R = 0$",
        xy=(11.0, 0.02),
        xytext=(6.5, 0.35),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9),
        fontsize=10.5,
        fontweight="medium",
    )

    ax.set_xlabel("Quantum Bit Error Rate - QBER (%)")
    ax.set_ylabel("Asymptotic Secret Key Rate $R$ (bits / sifted bit)")
    ax.set_title("Theoretical Secret Key Generation Rate vs. Error Rate")
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    out_path = results_dir / "key_rate_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_detection_heatmap_plot(
    results_dir: Path = Path("results"), force_recompute: bool = False
) -> Path:
    """Generate Figure 3: 8x8 Decision Heatmap (Noise x Eve) for fixed threat p=0.3 with caching."""
    print("Generating Figure 3: detection_heatmap.png...")
    cache_path = results_dir / "detection_heatmap_cache.json"

    n_bins = 8
    noise_vals = np.linspace(0.0, 0.15, n_bins)
    eve_vals = np.linspace(0.0, 1.0, n_bins)
    threat_prob = 0.30  # Fixed threat p = 0.3
    n_qubits = 800
    n_reps = 3

    # Verdict encoding: 0 = ACCEPT (Green), 1 = MONITOR (Yellow), 2 = REJECT (Red)
    verdict_map = {"ACCEPT": 0, "MONITOR": 1, "REJECT": 2}

    grid_matrix = np.zeros((n_bins, n_bins), dtype=int)
    qber_matrix = np.zeros((n_bins, n_bins), dtype=float)

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached heatmap data from {cache_path}...")
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                grid_matrix = np.array(data["verdicts"])
                qber_matrix = np.array(data["qbers"])
        except Exception:
            print("Cache load failed, recomputing...")
            force_recompute = True

    if force_recompute or not cache_path.exists():
        print("Computing 8x8 decision grid simulations...")
        for j, eve in enumerate(eve_vals):
            for i, noise in enumerate(noise_vals):
                rep_verdicts = []
                rep_qbers = []
                for r in range(n_reps):
                    sim = run_bb84_simulation(
                        n_qubits=n_qubits,
                        noise_level=float(noise),
                        eve_fraction=float(eve),
                        seed=500 + i * 20 + j * 5 + r,
                    )
                    dec = evaluate_connection_security(
                        qber_ucb=sim.qber_ucb,
                        n_sifted=sim.n_sifted,
                        attack_prob=threat_prob,
                        base_qber_threshold=0.11,
                        qber=sim.qber,
                    )
                    rep_verdicts.append(verdict_map[dec["verdict"]])
                    rep_qbers.append(sim.qber)

                # Majority verdict
                majority_v = int(np.round(np.mean(rep_verdicts)))
                grid_matrix[j, i] = majority_v
                qber_matrix[j, i] = float(np.mean(rep_qbers))

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "verdicts": grid_matrix.tolist(),
                    "qbers": qber_matrix.tolist(),
                    "noise_vals": noise_vals.tolist(),
                    "eve_vals": eve_vals.tolist(),
                    "threat_prob": threat_prob,
                },
                f,
                indent=2,
            )
        print(f"Saved heatmap cache to {cache_path}")

    # Plotting heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Discrete colormap: Green (Accept), Yellow/Amber (Monitor), Red (Reject)
    cmap = ListedColormap(["#2ECC71", "#F1C40F", "#E74C3C"])

    # Y-axis is Eve fraction (descending top-to-bottom for intuitive view)
    # We display matrix with imshow
    im = ax.imshow(grid_matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")

    # Set tick labels
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{x*100:.1f}%" for x in noise_vals])
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels([f"{y*100:.0f}%" for y in eve_vals])

    # Annotate QBER percentages in each cell
    for j in range(n_bins):
        for i in range(n_bins):
            val_str = f"{qber_matrix[j, i]*100:.1f}%"
            text_color = "black" if grid_matrix[j, i] == 1 else "white"
            ax.text(
                i,
                j,
                val_str,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=9.5,
            )

    ax.set_xlabel("Channel Depolarizing Noise Level (%)")
    ax.set_ylabel("Eve Interception Ratio (%)")
    ax.set_title(f"Security Decision Matrix ($P(Threat) = {threat_prob*100:.0f}\\%$, Base Threshold = $11.0\\%$)")

    # Custom discrete legend
    patch_accept = mpatches.Patch(color="#2ECC71", label="ACCEPT (Key Gen Permitted)")
    patch_monitor = mpatches.Patch(color="#F1C40F", label="MONITOR (Uncertain / Elevated Risk)")
    patch_reject = mpatches.Patch(color="#E74C3C", label="REJECT (Channel Compromised / Abort)")

    ax.legend(
        handles=[patch_accept, patch_monitor, patch_reject],
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
    )

    out_path = results_dir / "detection_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    """Generate all presentation figures into results/ directory."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate Publication Figures for Quantum-Secure Biomedical Networks")
    parser.add_argument("--all", action="store_true", help="Generate all extended research plots (calibration, policies, fingerprint, changepoint)")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("=========================================================")
    print("PHASE 5: GENERATING CORE PRESENTATION FIGURES (300 DPI)")
    print("=========================================================")

    p1 = generate_qber_vs_noise_plot(results_dir)
    p2 = generate_key_rate_curve_plot(results_dir)
    p3 = generate_detection_heatmap_plot(results_dir)

    print("\n[SUCCESS] Core presentation figures verified:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")

    if args.all:
        print("\n=========================================================")
        print("EXTENDED RESEARCH FIGURE SUITE (Tasks 1 - 4)")
        print("=========================================================")
        from src.calibration import run_calibration_and_explainability_suite
        from src.policy_eval import run_policy_benchmark_suite
        from src.fingerprint import run_fingerprint_suite
        from src.changepoint import run_changepoint_suite

        run_calibration_and_explainability_suite()
        run_policy_benchmark_suite()
        run_fingerprint_suite()
        run_changepoint_suite()


if __name__ == "__main__":
    main()
