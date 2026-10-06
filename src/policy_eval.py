"""Decision Policy Evaluation and Benchmarking Framework.

Compares three quantum security decision policies:
  Policy A: Fixed 11% threshold on Point-Estimate QBER
  Policy B: Fixed 11% threshold on 95% Upper Confidence Bound (QBER_UCB)
  Policy C: Adaptive Rule (Max Allowed QBER = 11% * (1 - 0.5 * P(Attack)) with UCB & MONITOR)

Evaluates:
  1. False-Reject Rate (FRR) under noise-only conditions.
  2. Miss Rate & Detection Rate under Eve intercept-resend attacks across noise levels.
  3. Multi-Threat performance across Low (0.1), Medium (0.5), and High (0.9) threat.
  4. Detection-Rate vs. False-Alarm-Rate tradeoff curves.
  5. Finite-key scaling across qubit counts (N = 200 to 5000).
  6. Dual-convention reporting (MONITOR as Alert vs. MONITOR as Accept).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.decision_engine import evaluate_connection_security
from src.qkd import run_bb84_simulation


def decide_policy_a(qber: float, base_threshold: float = 0.11) -> str:
    """Policy A: Fixed 11% threshold on point-estimate QBER."""
    return "ACCEPT" if qber <= base_threshold else "REJECT"


def decide_policy_b(qber_ucb: float, base_threshold: float = 0.11) -> str:
    """Policy B: Fixed 11% threshold on 95% Upper Confidence Bound."""
    return "ACCEPT" if qber_ucb <= base_threshold else "REJECT"


def decide_policy_c(
    qber_ucb: float,
    n_sifted: int,
    attack_prob: float,
    base_threshold: float = 0.11,
    qber: float | None = None,
) -> str:
    """Policy C: Our Adaptive Rule combining QBER_UCB and Classical Threat Score."""
    dec = evaluate_connection_security(
        qber_ucb=qber_ucb,
        n_sifted=n_sifted,
        attack_prob=attack_prob,
        base_qber_threshold=base_threshold,
        qber=qber,
    )
    return dec["verdict"]


class PolicyEvaluator:
    """Rigorous statistical evaluation framework for QKD security decision policies."""

    def __init__(self, results_dir: Union[str, Path] = "results", seed: int = 42) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def run_noise_sweep_experiment(
        self,
        noise_levels: np.ndarray,
        threat_levels: List[float] = [0.10, 0.50, 0.90],
        n_trials: int = 50,
        n_qubits: int = 1000,
    ) -> pd.DataFrame:
        """Experiment 1 & 3: Noise-only false-reject rate (FRR) across threat levels."""
        print(f"Running Experiment 1: Noise-only FRR sweep ({n_trials} trials/cell)...", flush=True)
        records = []

        for noise in noise_levels:
            # Simulate n_trials for this noise level once
            sim_runs = []
            for trial in range(n_trials):
                trial_seed = self.seed + int(noise * 10000) + trial * 7
                sim = run_bb84_simulation(
                    n_qubits=n_qubits,
                    noise_level=float(noise),
                    eve_fraction=0.0,
                    seed=trial_seed,
                )
                sim_runs.append(sim)

            for p_threat in threat_levels:
                decisions_a = []
                decisions_b = []
                decisions_c = []

                for sim in sim_runs:
                    v_a = decide_policy_a(sim.qber)
                    v_b = decide_policy_b(sim.qber_ucb)
                    v_c = decide_policy_c(sim.qber_ucb, sim.n_sifted, p_threat, qber=sim.qber)

                    decisions_a.append(v_a)
                    decisions_b.append(v_b)
                    decisions_c.append(v_c)

                frr_a = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_a]))
                frr_b = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_b]))
                frr_c_strict = float(np.mean([1 if d in ("REJECT", "MONITOR") else 0 for d in decisions_c]))
                frr_c_permissive = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_c]))

                records.append(
                    {
                        "Threat Level P(Att)": p_threat,
                        "Channel Noise": noise,
                        "FRR Policy A (Point 11%)": frr_a,
                        "FRR Policy B (UCB 11%)": frr_b,
                        "FRR Policy C [Monitor=Alert]": frr_c_strict,
                        "FRR Policy C [Monitor=Accept]": frr_c_permissive,
                    }
                )

        df = pd.DataFrame(records)
        return df

    def run_eve_sweep_experiment(
        self,
        eve_levels: np.ndarray,
        noise_levels: List[float] = [0.0, 0.05, 0.10],
        threat_levels: List[float] = [0.10, 0.50, 0.90],
        n_trials: int = 50,
        n_qubits: int = 1000,
    ) -> pd.DataFrame:
        """Experiment 2 & 3: Eve intercept sweep (Detection Rate & Miss Rate)."""
        print(f"Running Experiment 2: Eve intercept sweep ({n_trials} trials/cell)...", flush=True)
        records = []

        for noise in noise_levels:
            for eve in eve_levels:
                # Simulate n_trials for this (noise, eve) pair once
                sim_runs = []
                for trial in range(n_trials):
                    trial_seed = self.seed + int(eve * 1000) + int(noise * 10000) + trial * 11
                    sim = run_bb84_simulation(
                        n_qubits=n_qubits,
                        noise_level=float(noise),
                        eve_fraction=float(eve),
                        seed=trial_seed,
                    )
                    sim_runs.append(sim)

                for p_threat in threat_levels:
                    decisions_a = []
                    decisions_b = []
                    decisions_c = []

                    for sim in sim_runs:
                        v_a = decide_policy_a(sim.qber)
                        v_b = decide_policy_b(sim.qber_ucb)
                        v_c = decide_policy_c(sim.qber_ucb, sim.n_sifted, p_threat, qber=sim.qber)

                        decisions_a.append(v_a)
                        decisions_b.append(v_b)
                        decisions_c.append(v_c)

                    det_a = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_a]))
                    miss_a = 1.0 - det_a

                    det_b = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_b]))
                    miss_b = 1.0 - det_b

                    det_c_strict = float(np.mean([1 if d in ("REJECT", "MONITOR") else 0 for d in decisions_c]))
                    miss_c_strict = 1.0 - det_c_strict

                    det_c_perm = float(np.mean([1 if d == "REJECT" else 0 for d in decisions_c]))
                    miss_c_perm = 1.0 - det_c_perm

                    records.append(
                        {
                            "Threat Level P(Att)": p_threat,
                            "Channel Noise": noise,
                            "Eve Fraction": eve,
                            "DetRate Policy A": det_a,
                            "MissRate Policy A": miss_a,
                            "DetRate Policy B": det_b,
                            "MissRate Policy B": miss_b,
                            "DetRate Policy C [Monitor=Alert]": det_c_strict,
                            "MissRate Policy C [Monitor=Alert]": miss_c_strict,
                            "DetRate Policy C [Monitor=Accept]": det_c_perm,
                            "MissRate Policy C [Monitor=Accept]": miss_c_perm,
                        }
                    )

        df = pd.DataFrame(records)
        return df

    def run_qubit_scaling_experiment(
        self,
        qubit_counts: List[int] = [200, 500, 1000, 2000, 3000],
        noise: float = 0.08,
        eve: float = 0.15,
        threat: float = 0.30,
        n_trials: int = 50,
    ) -> pd.DataFrame:
        """Experiment 5: Finite-key scaling and variance collapse across qubit counts."""
        print(f"Running Experiment 5: Qubit count scaling ({n_trials} trials/count)...", flush=True)
        records = []

        for n_q in qubit_counts:
            qber_list = []
            ucb_list = []
            spread_list = []
            c_decisions = []

            for t in range(n_trials):
                sim = run_bb84_simulation(
                    n_qubits=n_q,
                    noise_level=noise,
                    eve_fraction=eve,
                    seed=self.seed + n_q + t * 5,
                )
                qber_list.append(sim.qber)
                ucb_list.append(sim.qber_ucb)
                spread_list.append(sim.qber_ucb - sim.qber)
                dec = decide_policy_c(sim.qber_ucb, sim.n_sifted, threat, qber=sim.qber)
                c_decisions.append(dec)

            records.append(
                {
                    "Qubits Sent": n_q,
                    "Mean Sifted Key Length": int(n_q * 0.49),
                    "Mean QBER": np.mean(qber_list),
                    "Mean 95% UCB": np.mean(ucb_list),
                    "UCB Margin Spread": np.mean(spread_list),
                    "Accept Rate": np.mean([1 if d == "ACCEPT" else 0 for d in c_decisions]),
                    "Monitor Rate": np.mean([1 if d == "MONITOR" else 0 for d in c_decisions]),
                    "Reject Rate": np.mean([1 if d == "REJECT" else 0 for d in c_decisions]),
                }
            )

        df = pd.DataFrame(records)
        return df

    def plot_policy_tradeoff_curves(
        self, noise_df: pd.DataFrame, eve_df: pd.DataFrame
    ) -> Path:
        """Generate ROC-style Detection Rate vs False Alarm Rate tradeoff curves."""
        print("Generating Policy Tradeoff Plots...")
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

        # Subplot 1: False Reject Rate vs Noise Level (Low Threat P=0.1 vs High Threat P=0.9)
        ax1 = axes[0]
        subset_low = noise_df[noise_df["Threat Level P(Att)"] == 0.10]
        subset_high = noise_df[noise_df["Threat Level P(Att)"] == 0.90]

        ax1.plot(
            subset_low["Channel Noise"] * 100,
            subset_low["FRR Policy A (Point 11%)"] * 100,
            "o--",
            color="#2563EB",
            label="Policy A (Point-Est 11%)",
            linewidth=2,
        )
        ax1.plot(
            subset_low["Channel Noise"] * 100,
            subset_low["FRR Policy B (UCB 11%)"] * 100,
            "s--",
            color="#0D9488",
            label="Policy B (UCB 11%)",
            linewidth=2,
        )
        ax1.plot(
            subset_low["Channel Noise"] * 100,
            subset_low["FRR Policy C [Monitor=Alert]"] * 100,
            "^-",
            color="#D97706",
            label="Policy C (Low Threat P=0.1)",
            linewidth=2.2,
        )
        ax1.plot(
            subset_high["Channel Noise"] * 100,
            subset_high["FRR Policy C [Monitor=Alert]"] * 100,
            "d-",
            color="#DC2626",
            label="Policy C (High Threat P=0.9) [Honest Alert]",
            linewidth=2.2,
        )

        ax1.set_xlabel("Channel Depolarizing Noise (%)", fontsize=11)
        ax1.set_ylabel("False-Reject Rate (%) [Noise Only]", fontsize=11)
        ax1.set_title("False Reject Rate vs. Channel Noise", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", framealpha=0.9)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Detection Rate vs Eve Interception (Noise = 5%)
        ax2 = axes[1]
        eve_subset_low = eve_df[(np.isclose(eve_df["Channel Noise"], 0.05)) & (np.isclose(eve_df["Threat Level P(Att)"], 0.10))]
        eve_subset_high = eve_df[(np.isclose(eve_df["Channel Noise"], 0.05)) & (np.isclose(eve_df["Threat Level P(Att)"], 0.90))]

        ax2.plot(
            eve_subset_low["Eve Fraction"] * 100,
            eve_subset_low["DetRate Policy A"] * 100,
            "o--",
            color="#2563EB",
            label="Policy A (Point 11%)",
            linewidth=2,
        )
        ax2.plot(
            eve_subset_low["Eve Fraction"] * 100,
            eve_subset_low["DetRate Policy B"] * 100,
            "s--",
            color="#0D9488",
            label="Policy B (UCB 11%)",
            linewidth=2,
        )
        ax2.plot(
            eve_subset_low["Eve Fraction"] * 100,
            eve_subset_low["DetRate Policy C [Monitor=Alert]"] * 100,
            "^-",
            color="#D97706",
            label="Policy C (Adaptive, Threat P=0.1)",
            linewidth=2.2,
        )
        ax2.plot(
            eve_subset_high["Eve Fraction"] * 100,
            eve_subset_high["DetRate Policy C [Monitor=Alert]"] * 100,
            "d-",
            color="#DC2626",
            label="Policy C (Adaptive, Threat P=0.9)",
            linewidth=2.2,
        )

        ax2.set_xlabel("Eve Interception Ratio (%) [at 5% Noise]", fontsize=11)
        ax2.set_ylabel("Intrusion Detection Rate (%)", fontsize=11)
        ax2.set_title("Wiretap Detection Rate vs. Eve Fraction", fontsize=12, fontweight="bold")
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("Quantum Decision Policy Evaluation: Security vs. Availability Tradeoff", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = self.results_dir / "policy_roc_curves.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved Policy Tradeoff Curves to: {out_path}", flush=True)
        return out_path

    def plot_multi_threat_heatmaps(self) -> Path:
        """Generate 3 side-by-side decision heatmaps at Low (0.1), Med (0.5), and High (0.9) Threat."""
        print("Generating Multi-Threat Decision Heatmaps (Low, Medium, High)...", flush=True)
        threat_levels = [0.10, 0.50, 0.90]
        titles = ["Low Threat (P = 10%)", "Medium Threat (P = 50%)", "High Threat (P = 90%)"]

        n_pts = 35
        noise_grid = np.linspace(0.0, 0.20, n_pts)
        eve_grid = np.linspace(0.0, 1.0, n_pts)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        cmap_dec = ListedColormap(["#2ECC71", "#F1C40F", "#E74C3C"])

        val_map = {"ACCEPT": 0, "MONITOR": 1, "REJECT": 2}

        for idx, (p_th, title) in enumerate(zip(threat_levels, titles)):
            grid_dec = np.zeros((n_pts, n_pts))
            for j, e_val in enumerate(eve_grid):
                for i, n_val in enumerate(noise_grid):
                    exp_qber = (2.0 / 3.0) * n_val * (1.0 - 0.5 * e_val) + 0.25 * e_val
                    fin_ucb = exp_qber + 1.645 * np.sqrt(max(0.0, exp_qber * (1.0 - exp_qber) / 500.0))
                    v = decide_policy_c(fin_ucb, 500, p_th, qber=exp_qber)
                    grid_dec[j, i] = val_map[v]

            ax = axes[idx]
            cf = ax.contourf(
                noise_grid * 100,
                eve_grid * 100,
                grid_dec,
                levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=cmap_dec,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Channel Noise (%)", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Eve Interception Ratio (%)", fontsize=11)

            # Draw 11% baseline reference line
            ax.axvline(16.5, color="white", linestyle=":", alpha=0.7, label="11% QBER Equiv")

        import matplotlib.patches as mpatches

        p_acc = mpatches.Patch(color="#2ECC71", label="ACCEPT (Key Gen Allowed)")
        p_mon = mpatches.Patch(color="#F1C40F", label="MONITOR (Sampling / Elevated Risk)")
        p_rej = mpatches.Patch(color="#E74C3C", label="REJECT (Compromised / Abort)")

        fig.legend(
            handles=[p_acc, p_mon, p_rej],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.06),
            frameon=True,
            facecolor="white",
            fontsize=11,
        )

        plt.suptitle("Adaptive Security Decision Envelopes Across Classical Threat Levels", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = self.results_dir / "multi_threat_heatmaps.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Multi-Threat Heatmaps to: {out_path}", flush=True)
        return out_path


def run_policy_benchmark_suite() -> None:
    """Execute complete policy evaluation benchmarking experiment suite."""
    print("==========================================================================", flush=True)
    print("TASK 2: DECISION POLICY BENCHMARKING (A: Fixed Point, B: Fixed UCB, C: Adaptive)", flush=True)
    print("==========================================================================", flush=True)
    evaluator = PolicyEvaluator(seed=42)

    noise_levels = np.linspace(0.0, 0.20, 7)
    eve_levels = np.linspace(0.0, 1.0, 7)

    noise_df = evaluator.run_noise_sweep_experiment(noise_levels, n_trials=50, n_qubits=1000)
    eve_df = evaluator.run_eve_sweep_experiment(eve_levels, n_trials=50, n_qubits=1000)
    scaling_df = evaluator.run_qubit_scaling_experiment(n_trials=50)

    evaluator.plot_policy_tradeoff_curves(noise_df, eve_df)
    evaluator.plot_multi_threat_heatmaps()

    # Save summary tables
    out_noise = evaluator.results_dir / "policy_noise_frr.csv"
    out_eve = evaluator.results_dir / "policy_eve_detection.csv"
    out_scale = evaluator.results_dir / "policy_qubit_scaling.csv"

    noise_df.to_csv(out_noise, index=False)
    eve_df.to_csv(out_eve, index=False)
    scaling_df.to_csv(out_scale, index=False)

    print("\n--- POLICY BENCHMARK SUMMARY: NOISE-ONLY FRR AT 10% NOISE ---")
    sub = noise_df[noise_df["Channel Noise"] == 0.10]
    print(tabulate(sub, headers="keys", tablefmt="github", showindex=False))

    print("\n--- POLICY BENCHMARK SUMMARY: FINITE-KEY SCALING ---")
    print(tabulate(scaling_df, headers="keys", tablefmt="github", showindex=False))

    print("\n[SUCCESS] Task 2 completed successfully!")


if __name__ == "__main__":
    run_policy_benchmark_suite()
