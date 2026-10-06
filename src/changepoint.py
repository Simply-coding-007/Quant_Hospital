"""Time-Varying Attack Detection & CUSUM Changepoint Monitoring for BB84 QKD.

Detects transient, intermittent, and scheduled burst eavesdropping attacks:
  1. Partitions sifted key into sequential blocks (e.g., B = 50 sifted bits).
  2. Computes per-block error rates q_k = e_k / B.
  3. Implements Cumulative Sum (CUSUM) sequential change-point detector:
       S_0 = 0,  S_k = max(0, S_{k-1} + (q_k - mu_0 - k_ref))
     with calibrated decision limit h_limit achieving target False Alarm Rate (FAR <= 1%).
  4. Compares against standard whole-session QBER decision.
  5. Identifies attack windows, detection delays, and blind spots of whole-session monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.decision_engine import evaluate_connection_security
from src.qkd import BB84Simulation, run_bb84_simulation


class CUSUMChangePointDetector:
    """Sequential CUSUM detector for real-time quantum channel changepoint and burst monitoring."""

    def __init__(
        self,
        block_size: int = 50,
        baseline_noise: float = 0.02,
        expected_attack_jump: float = 0.25,
        h_limit: float = 0.35,
    ) -> None:
        self.block_size = block_size
        self.baseline_noise = baseline_noise
        # Single-qubit depolarizing baseline error rate: mu_0 ~ 2/3 * noise
        self.mu_0 = (2.0 / 3.0) * baseline_noise
        # Expected error rate under active intercept-resend attack: mu_1 ~ mu_0 + 0.25
        self.mu_1 = self.mu_0 + expected_attack_jump
        # CUSUM reference slack parameter
        self.k_ref = (self.mu_1 - self.mu_0) / 2.0
        self.h_limit = h_limit

    def calibrate_threshold(
        self,
        n_clean_sessions: int = 100,
        n_qubits: int = 2000,
        target_far: float = 0.01,
        seed: int = 42,
    ) -> float:
        """Calibrate decision limit h_limit via Monte Carlo simulation on clean/noisy baseline sessions."""
        max_s_scores = []
        for s_idx in range(n_clean_sessions):
            sim = run_bb84_simulation(
                n_qubits=n_qubits,
                noise_level=self.baseline_noise,
                eve_fraction=0.0,
                seed=seed + s_idx * 17,
            )
            res = self.process_sifted_stream(sim.sifted_errors)
            max_s = max(res["cusum_statistic"]) if len(res["cusum_statistic"]) > 0 else 0.0
            max_s_scores.append(max_s)

        # Set threshold to (1 - target_far) percentile + safety margin
        calibrated_h = float(np.percentile(max_s_scores, (1.0 - target_far) * 100)) + 0.05
        self.h_limit = max(0.20, calibrated_h)
        return self.h_limit

    def process_sifted_stream(self, sifted_errors: np.ndarray) -> Dict[str, Any]:
        """Process sequential sifted error stream into blocks and compute CUSUM statistic."""
        n_bits = len(sifted_errors)
        n_blocks = n_bits // self.block_size

        if n_blocks == 0:
            return {
                "n_blocks": 0,
                "block_qber": [],
                "cusum_statistic": [],
                "alarm_triggered": False,
                "alarm_block": None,
                "changepoint_block": None,
            }

        block_qber = []
        for b in range(n_blocks):
            chunk = sifted_errors[b * self.block_size : (b + 1) * self.block_size]
            block_qber.append(float(np.mean(chunk)))

        cusum_scores = []
        current_s = 0.0
        alarm_block = None
        changepoint_block = None

        for b_idx, q_k in enumerate(block_qber):
            # CUSUM update
            current_s = max(0.0, current_s + (q_k - self.mu_0 - self.k_ref))
            cusum_scores.append(current_s)

            if current_s > self.h_limit and alarm_block is None:
                alarm_block = b_idx
                # Backtrack to find when CUSUM began climbing from zero
                zero_indices = [j for j in range(b_idx + 1) if cusum_scores[j] <= 1e-6]
                changepoint_block = zero_indices[-1] if zero_indices else 0

        return {
            "n_blocks": n_blocks,
            "block_qber": block_qber,
            "cusum_statistic": cusum_scores,
            "alarm_triggered": alarm_block is not None,
            "alarm_block": alarm_block,
            "changepoint_block": changepoint_block,
            "h_limit": self.h_limit,
            "mu_0": self.mu_0,
            "k_ref": self.k_ref,
        }


class ChangepointBenchmark:
    """Evaluates CUSUM detector performance against Whole-Session QBER across burst attack scenarios."""

    def __init__(self, results_dir: Union[str, Path] = "results", seed: int = 42) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def run_benchmark_experiments(self) -> pd.DataFrame:
        """Run systematic benchmark across attack durations (50 to 400 qubits) and noise levels."""
        print("Running CUSUM vs Whole-Session Burst Attack Benchmark...")
        detector = CUSUMChangePointDetector(block_size=50, baseline_noise=0.02)
        detector.calibrate_threshold(n_clean_sessions=50, n_qubits=2000, seed=self.seed)
        print(f"Calibrated CUSUM Decision Limit h_limit: {detector.h_limit:.4f} (FAR <= 1%)")

        burst_durations = [30, 80, 150, 250, 400]
        noise_levels = [0.01, 0.03, 0.06]
        n_trials = 30
        n_total_qubits = 2000

        records = []

        for noise in noise_levels:
            detector.baseline_noise = noise
            detector.mu_0 = (2.0 / 3.0) * noise
            detector.k_ref = (0.25) / 2.0

            for burst_len in burst_durations:
                cusum_detections = 0
                whole_session_detections = 0
                delays = []

                # Attack window centered in the session
                start_q = (n_total_qubits - burst_len) // 2
                end_q = start_q + burst_len
                schedule = [(start_q, end_q)]

                for t in range(n_trials):
                    trial_seed = self.seed + burst_len * 10 + int(noise * 1000) + t
                    sim = run_bb84_simulation(
                        n_qubits=n_total_qubits,
                        noise_level=noise,
                        eve_fraction=1.0,
                        attack_schedule=schedule,
                        seed=trial_seed,
                    )

                    # 1. Whole-session decision
                    dec_whole = evaluate_connection_security(
                        qber_ucb=sim.qber_ucb,
                        n_sifted=sim.n_sifted,
                        attack_prob=0.20,  # Low threat baseline
                        base_qber_threshold=0.11,
                        qber=sim.qber,
                    )
                    if dec_whole["verdict"] in ("REJECT", "MONITOR"):
                        whole_session_detections += 1

                    # 2. CUSUM detector
                    res_cusum = detector.process_sifted_stream(sim.sifted_errors)
                    if res_cusum["alarm_triggered"]:
                        cusum_detections += 1
                        # Estimate ground truth start block in sifted stream
                        # ~49% of qubits become sifted bits
                        gt_start_block = int((start_q * 0.49) / detector.block_size)
                        alarm_b = res_cusum["alarm_block"]
                        delay_blocks = max(0, alarm_b - gt_start_block)
                        delays.append(delay_blocks * detector.block_size)

                cusum_det_rate = cusum_detections / n_trials
                whole_det_rate = whole_session_detections / n_trials
                avg_delay_qubits = float(np.mean(delays)) if delays else float("nan")

                records.append(
                    {
                        "Channel Noise": f"{noise*100:.1f}%",
                        "Burst Duration (Qubits)": burst_len,
                        "Burst % of Session": f"{(burst_len/n_total_qubits)*100:.1f}%",
                        "CUSUM Detection Rate": f"{cusum_det_rate*100:.1f}%",
                        "Whole-Session Det Rate": f"{whole_det_rate*100:.1f}%",
                        "Detection Advantage": f"{(cusum_det_rate - whole_det_rate)*100:+.1f}%",
                        "Mean Detection Delay (Bits)": f"{avg_delay_qubits:.0f}" if not np.isnan(avg_delay_qubits) else "-",
                    }
                )

        df = pd.DataFrame(records)
        out_csv = self.results_dir / "changepoint_benchmark.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved Changepoint Benchmark to: {out_csv}")

        print("\n" + "=" * 95)
        print("CUSUM TIME-VARYING BURST ATTACK DETECTION vs WHOLE-SESSION QBER")
        print("=" * 95)
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        print("=" * 95)
        return df

    def plot_changepoint_example(self) -> Path:
        """Generate 3-panel visualization showing per-block QBER, CUSUM statistic, and detection window."""
        print("Generating CUSUM Changepoint Detection 3-Panel Plot...")
        n_total = 2400
        start_q = 800
        end_q = 1300
        schedule = [(start_q, end_q)]
        noise = 0.02

        sim = run_bb84_simulation(
            n_qubits=n_total,
            noise_level=noise,
            eve_fraction=1.0,
            attack_schedule=schedule,
            seed=101,
        )

        detector = CUSUMChangePointDetector(block_size=40, baseline_noise=noise, h_limit=0.30)
        res = detector.process_sifted_stream(sim.sifted_errors)

        block_qber = np.array(res["block_qber"]) * 100
        cusum_scores = np.array(res["cusum_statistic"])
        n_blocks = len(block_qber)
        blocks = np.arange(1, n_blocks + 1)

        gt_start_b = int((start_q * 0.49) / 40)
        gt_end_b = int((end_q * 0.49) / 40)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

        # Panel 1: Per-Block QBER
        ax1 = axes[0]
        ax1.plot(blocks, block_qber, "o-", color="#2563EB", linewidth=2, markersize=5, label="Measured Block QBER")
        ax1.axhline(11.0, color="#DC2626", linestyle="--", linewidth=1.5, label="11% Whole-Session Threshold")
        ax1.axhline((2.0 / 3.0) * noise * 100, color="#059669", linestyle=":", label="Baseline Channel Noise (~1.3%)")
        ax1.axvspan(gt_start_b, gt_end_b, color="#EF4444", alpha=0.15, label="Ground-Truth Attack Window (Eve Active)")

        ax1.set_ylabel("Block QBER (%)", fontsize=11)
        ax1.set_title(f"1. Sequential Per-Block Error Rate (Session QBER = {sim.qber*100:.2f}%)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", framealpha=0.9)

        # Panel 2: CUSUM Test Statistic
        ax2 = axes[1]
        ax2.plot(blocks, cusum_scores, "s-", color="#D97706", linewidth=2.2, markersize=5, label="CUSUM Statistic $S_k$")
        ax2.axhline(detector.h_limit, color="#DC2626", linestyle="--", linewidth=2, label=f"Decision Limit $h = {detector.h_limit:.2f}$")
        ax2.axvspan(gt_start_b, gt_end_b, color="#EF4444", alpha=0.15)

        if res["alarm_triggered"]:
            alarm_b = res["alarm_block"] + 1
            ax2.scatter([alarm_b], [cusum_scores[res["alarm_block"]]], color="#DC2626", s=130, marker="X", zorder=6, label=f"ALARM TRIGGERED (Block {alarm_b})")

        ax2.set_ylabel("CUSUM Score $S_k$", fontsize=11)
        ax2.set_title("2. Sequential CUSUM Cumulative Anomaly Accumulator", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", framealpha=0.9)

        # Panel 3: Security Verdict State & Detection Localization
        ax3 = axes[2]
        status_line = np.zeros(n_blocks)
        if res["alarm_triggered"]:
            status_line[res["alarm_block"] :] = 1.0

        ax3.step(blocks, status_line, where="mid", color="#DC2626", linewidth=2.5, label="CUSUM Security State (1 = Alert)")
        ax3.axvspan(gt_start_b, gt_end_b, color="#EF4444", alpha=0.15, label="Ground-Truth Attack Window")
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["Normal (Pass)", "Alert (MITM Caught)"])
        ax3.set_xlabel("Sequential Sifted Block Index (Block Size = 40 bits)", fontsize=11)
        ax3.set_ylabel("Security State", fontsize=11)
        ax3.set_title("3. Real-Time MITM Localization & Intrusion Interception", fontsize=12, fontweight="bold")
        ax3.legend(loc="upper left", framealpha=0.9)

        plt.suptitle("Time-Varying Burst Attack Detection: CUSUM Changepoint vs. Whole-Session Monitoring", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = self.results_dir / "changepoint_detection.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved Changepoint Detection Plot to: {out_path}")
        return out_path


def run_changepoint_suite() -> None:
    """Execute complete changepoint monitoring benchmark."""
    print("==========================================================================")
    print("TASK 4: TIME-VARYING ATTACKS & CUSUM CHANGEPOINT DETECTION BENCHMARK")
    print("==========================================================================")
    bench = ChangepointBenchmark(seed=42)
    bench.run_benchmark_experiments()
    bench.plot_changepoint_example()
    print("\n[SUCCESS] Task 4 completed successfully!")


if __name__ == "__main__":
    run_changepoint_suite()
