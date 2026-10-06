"""Quantum Channel Anomaly Fingerprinting & Physical Degradation Attribution.

Analyzes basis-resolved QBER (QBER_Z vs. QBER_X) across 5 physical scenarios:
  1. Standard Intercept-Resend Eve (Random Basis Z/X) -> QBER_Z ~ QBER_X ~ 0.25 * f
  2. Z-Biased Eve (Always measures in Z basis)        -> QBER_Z ~ 0, QBER_X ~ 0.50 * f
  3. Depolarizing Channel Noise                       -> QBER_Z ~ QBER_X ~ 0.67 * p
  4. Dephasing / Phase-Flip Channel Noise             -> QBER_Z ~ 0, QBER_X ~ p
  5. Bit-Flip Channel Noise                           -> QBER_Z ~ p, QBER_X ~ 0

Features:
  - Theoretical state evolution models and simulation curves.
  - Multi-scenario dataset generation with randomized strengths.
  - Scatter plot with overlaid theoretical trajectory lines (results/fingerprint_scatter.png).
  - Anomaly Attribution Classifier with 5-fold cross-validation and Confusion Matrix (results/fingerprint_confusion_matrix.png).
  - Explicit documentation of physical degeneracies (e.g., Dephasing vs. Z-Biased Eve).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tabulate import tabulate


def simulate_fingerprint_run(
    n_qubits: int = 1000,
    scenario: Literal[
        "standard_eve", "z_biased_eve", "depolarizing_noise", "dephasing_noise", "bit_flip_noise"
    ] = "standard_eve",
    strength: float = 0.10,
    seed: int = 42,
) -> Dict[str, float]:
    """Simulate a single BB84 session tailored to one of 5 physical degradation scenarios.

    Theoretical Formulas:
    - standard_eve:       QBER_Z = QBER_X = 0.25 * f (Eve measures in random Z/X basis)
    - z_biased_eve:       QBER_Z = 0.0, QBER_X = 0.50 * f (Eve measures only in Z basis)
    - depolarizing_noise: QBER_Z = QBER_X = (2/3) * p (Isotropic Pauli X, Y, Z noise)
    - dephasing_noise:    QBER_Z = 0.0, QBER_X = p (Pauli Z phase noise; commutes with Z)
    - bit_flip_noise:     QBER_Z = p, QBER_X = 0.0 (Pauli X bit-flip noise; commutes with X)
    """
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, size=n_qubits, dtype=np.int32)
    alice_bases = rng.integers(0, 2, size=n_qubits, dtype=np.int32)
    bob_bases = rng.integers(0, 2, size=n_qubits, dtype=np.int32)
    bob_bits = np.zeros(n_qubits, dtype=np.int32)

    noise_model = NoiseModel()
    has_noise = False

    # Configure noise model if scenario is channel noise
    if scenario == "depolarizing_noise" and strength > 1e-6:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(strength, 1), ["id"])
        has_noise = True
    elif scenario == "dephasing_noise" and strength > 1e-6:
        # Phase flip (Pauli Z)
        noise_model.add_all_qubit_quantum_error(
            pauli_error([("Z", strength), ("I", max(0.0, 1.0 - strength))]), ["id"]
        )
        has_noise = True
    elif scenario == "bit_flip_noise" and strength > 1e-6:
        # Bit flip (Pauli X)
        noise_model.add_all_qubit_quantum_error(
            pauli_error([("X", strength), ("I", max(0.0, 1.0 - strength))]), ["id"]
        )
        has_noise = True

    sim = AerSimulator(
        noise_model=noise_model if has_noise else None, seed_simulator=seed
    )

    eve_intercepted = np.zeros(n_qubits, dtype=bool)
    eve_bases = np.full(n_qubits, -1, dtype=np.int32)

    if scenario in ("standard_eve", "z_biased_eve"):
        eve_fraction = strength
        eve_intercepted = rng.random(n_qubits) < eve_fraction
        if np.any(eve_intercepted):
            eve_indices = np.where(eve_intercepted)[0]
            if scenario == "standard_eve":
                eve_bases[eve_indices] = rng.integers(0, 2, size=len(eve_indices), dtype=np.int32)
            else:  # z_biased_eve: Eve ALWAYS measures in Z basis (basis = 0)
                eve_bases[eve_indices] = 0

    # Execute circuits
    # 1. Non-intercepted
    if np.any(~eve_intercepted):
        active_direct = []
        for k in range(8):
            a_bit = (k >> 2) & 1
            a_bas = (k >> 1) & 1
            b_bas = k & 1
            idx = np.where(
                (~eve_intercepted)
                & (alice_bits == a_bit)
                & (alice_bases == a_bas)
                & (bob_bases == b_bas)
            )[0]
            if len(idx) == 0:
                continue
            qc = QuantumCircuit(1, 1)
            if a_bit: qc.x(0)
            if a_bas: qc.h(0)
            qc.id(0)
            if b_bas: qc.h(0)
            qc.measure(0, 0)
            active_direct.append((qc, idx))

        if active_direct:
            circs = [item[0] for item in active_direct]
            tcircs = transpile(circs, sim, optimization_level=0)
            if not isinstance(tcircs, list):
                tcircs = [tcircs]
            for c_i, (_, idx) in enumerate(active_direct):
                res = sim.run(tcircs[c_i], shots=len(idx), memory=True).result()
                for m_i, bit_str in enumerate(res.get_memory()):
                    bob_bits[idx[m_i]] = int(bit_str.strip())

    # 2. Intercepted
    if np.any(eve_intercepted):
        active_eve = []
        for k in range(16):
            a_bit = (k >> 3) & 1
            a_bas = (k >> 2) & 1
            e_bas = (k >> 1) & 1
            b_bas = k & 1
            idx = np.where(
                (eve_intercepted)
                & (alice_bits == a_bit)
                & (alice_bases == a_bas)
                & (eve_bases == e_bas)
                & (bob_bases == b_bas)
            )[0]
            if len(idx) == 0:
                continue
            qc = QuantumCircuit(1, 2)
            if a_bit: qc.x(0)
            if a_bas: qc.h(0)
            if e_bas: qc.h(0)
            qc.measure(0, 0)
            qc.reset(0)
            with qc.if_test((qc.clbits[0], 1)):
                qc.x(0)
            if e_bas: qc.h(0)
            qc.id(0)
            if b_bas: qc.h(0)
            qc.measure(0, 1)
            active_eve.append((qc, idx))

        if active_eve:
            circs = [item[0] for item in active_eve]
            tcircs = transpile(circs, sim, optimization_level=0)
            if not isinstance(tcircs, list):
                tcircs = [tcircs]
            for c_i, (_, idx) in enumerate(active_eve):
                res = sim.run(tcircs[c_i], shots=len(idx), memory=True).result()
                for m_i, bit_str in enumerate(res.get_memory()):
                    clean = bit_str.replace(" ", "")
                    bob_bits[idx[m_i]] = int(clean[0]) if len(clean) > 1 else int(clean[-1])

    # Sifting and basis-resolved QBER
    sifted_mask = alice_bases == bob_bases
    sifted_idx = np.where(sifted_mask)[0]
    n_sifted = len(sifted_idx)

    if n_sifted > 0:
        errors = alice_bits[sifted_idx] != bob_bits[sifted_idx]
        qber = float(np.mean(errors))

        z_mask = alice_bases[sifted_idx] == 0
        x_mask = alice_bases[sifted_idx] == 1

        qber_z = float(np.mean(errors[z_mask])) if np.any(z_mask) else 0.0
        qber_x = float(np.mean(errors[x_mask])) if np.any(x_mask) else 0.0
    else:
        qber = qber_z = qber_x = 0.0

    return {
        "scenario": scenario,
        "strength": strength,
        "n_sifted": n_sifted,
        "qber": qber,
        "qber_z": qber_z,
        "qber_x": qber_x,
        "ratio_zx": (qber_z + 1e-4) / (qber_x + 1e-4),
        "diff_zx": qber_z - qber_x,
    }


class QuantumChannelFingerprinter:
    """Generates benchmark datasets, scatter visual plots, and attribution classifiers."""

    def __init__(self, results_dir: Union[str, Path] = "results", seed: int = 42) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def generate_fingerprint_dataset(self, n_samples_per_class: int = 60) -> pd.DataFrame:
        """Generate diverse synthetic dataset across 5 physical degradation modes."""
        print(f"Generating fingerprint dataset ({n_samples_per_class} samples/class)...")
        scenarios = [
            "standard_eve",
            "z_biased_eve",
            "depolarizing_noise",
            "dephasing_noise",
            "bit_flip_noise",
        ]
        records = []

        rng = np.random.default_rng(self.seed)

        for sc in scenarios:
            for s_i in range(n_samples_per_class):
                # Sample random strength parameter between 2% and 35%
                strength = float(rng.uniform(0.02, 0.35))
                seed = self.seed + s_i * 13 + len(sc) * 100
                res = simulate_fingerprint_run(
                    n_qubits=1000, scenario=sc, strength=strength, seed=seed  # type: ignore
                )
                records.append(res)

        df = pd.DataFrame(records)
        out_csv = self.results_dir / "fingerprint_dataset.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved fingerprint dataset to: {out_csv}")
        return df

    def plot_fingerprint_scatter(self, df: pd.DataFrame) -> Path:
        """Generate scatter plot of QBER_Z vs QBER_X with overlaid theoretical trajectories."""
        print("Generating QBER_Z vs QBER_X Fingerprint Scatter Plot...")
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 7))

        label_map = {
            "standard_eve": "Standard Eve (Random Basis) [QBER_Z ≈ QBER_X]",
            "depolarizing_noise": "Depolarizing Noise [QBER_Z ≈ QBER_X]",
            "z_biased_eve": "Z-Biased Eve (Eve in Z) [QBER_Z ≈ 0, QBER_X > 0]",
            "dephasing_noise": "Dephasing (Phase-Flip) [QBER_Z ≈ 0, QBER_X > 0]",
            "bit_flip_noise": "Bit-Flip Noise [QBER_X ≈ 0, QBER_Z > 0]",
        }

        palette = {
            "standard_eve": "#2563EB",
            "depolarizing_noise": "#0D9488",
            "z_biased_eve": "#EA580C",
            "dephasing_noise": "#DC2626",
            "bit_flip_noise": "#9333EA",
        }

        # Scatter points
        for sc, group in df.groupby("scenario"):
            ax.scatter(
                group["qber_x"] * 100,
                group["qber_z"] * 100,
                color=palette[sc],
                label=label_map[sc],
                alpha=0.8,
                s=45,
                edgecolors="white",
                linewidth=0.5,
            )

        # Theoretical Trajectory Lines
        # 1. Symmetric line (y = x): Standard Eve & Depolarizing Noise
        line_sym = np.linspace(0, 30, 100)
        ax.plot(line_sym, line_sym, "k--", alpha=0.6, linewidth=1.5, label="Theory: Symmetric $QBER_Z = QBER_X$")

        # 2. X-error only (y = 0): Dephasing & Z-biased Eve
        ax.axhline(0, color="#DC2626", linestyle=":", alpha=0.7, linewidth=1.5, label="Theory: Dephasing / Z-Eve ($QBER_Z = 0$)")

        # 3. Z-error only (x = 0): Bit-Flip Noise
        ax.axvline(0, color="#9333EA", linestyle=":", alpha=0.7, linewidth=1.5, label="Theory: Bit-Flip ($QBER_X = 0$)")

        # Highlight Physical Degeneracy / Ambiguity Box
        ax.annotate(
            "CRITICAL PHYSICAL DEGENERACY:\nDephasing Noise & Z-Biased Eve\nOverlap completely at QBER_Z ≈ 0",
            xy=(15, 0.5),
            xytext=(12, 12),
            arrowprops=dict(facecolor="red", shrink=0.08, width=1.2, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.5", fc="#FEF2F2", ec="#EF4444", alpha=0.95),
            fontsize=9.5,
            fontweight="bold",
        )

        ax.set_xlabel("X-Basis Error Rate QBER_X (%)", fontsize=11)
        ax.set_ylabel("Z-Basis Error Rate QBER_Z (%)", fontsize=11)
        ax.set_title("Quantum Channel Fingerprint: Basis-Resolved Anomaly Attribution", fontsize=13, fontweight="bold")
        ax.set_xlim(-1, 35)
        ax.set_ylim(-1, 35)
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9.5, framealpha=0.95)

        out_path = self.results_dir / "fingerprint_scatter.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Fingerprint Scatter Plot to: {out_path}")
        return out_path

    def train_attribution_classifier(self, df: pd.DataFrame) -> Tuple[RandomForestClassifier, pd.DataFrame]:
        """Train Random Forest anomaly classifier with Stratified 5-Fold Cross-Validation."""
        print("Training Anomaly Attribution Classifier with 5-Fold Cross-Validation...")
        features = ["qber_z", "qber_x", "ratio_zx", "qber", "diff_zx"]
        X = df[features]
        y = df["scenario"]

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.seed)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        classes = [
            "bit_flip_noise",
            "dephasing_noise",
            "depolarizing_noise",
            "standard_eve",
            "z_biased_eve",
        ]
        cm = confusion_matrix(y, y_pred, labels=classes)

        print("\n--- ANOMALY ATTRIBUTION CLASSIFICATION REPORT ---")
        print(classification_report(y, y_pred, labels=classes, zero_division=0))

        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=[c.replace("_", " ") for c in classes],
            yticklabels=[c.replace("_", " ") for c in classes],
            ax=ax,
        )
        ax.set_title("Physical Anomaly Attribution Confusion Matrix (5-Fold CV)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Anomaly Scenario", fontsize=11)
        ax.set_ylabel("True Physical Scenario", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        out_cm = self.results_dir / "fingerprint_confusion_matrix.png"
        plt.savefig(out_cm, dpi=300)
        plt.close(fig)
        print(f"Saved Anomaly Confusion Matrix to: {out_cm}")

        # Train final classifier on full dataset
        clf.fit(X, y)
        return clf, pd.DataFrame(cm, index=classes, columns=classes)


def run_fingerprint_suite() -> None:
    """Execute complete quantum channel fingerprinting and attribution benchmark."""
    print("==========================================================================")
    print("TASK 3: QUANTUM CHANNEL ANOMALY FINGERPRINTING & DEGRADATION ATTRIBUTION")
    print("==========================================================================")
    fp = QuantumChannelFingerprinter(seed=42)
    df = fp.generate_fingerprint_dataset(n_samples_per_class=60)
    fp.plot_fingerprint_scatter(df)
    clf, cm_df = fp.train_attribution_classifier(df)
    print("\n[SUCCESS] Task 3 completed successfully!")


if __name__ == "__main__":
    run_fingerprint_suite()
