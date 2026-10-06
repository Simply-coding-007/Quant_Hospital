"""BB84 Quantum Key Distribution (QKD) Simulation.

Implements the physical quantum layer simulation using Qiskit >= 1.0 and Qiskit-Aer.
Simulates Alice state preparation, Eve intercept-resend attack, quantum channel
noise (depolarizing and phase-flip), Bob measurement, key sifting, and QBER estimation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error


@dataclass
class QKDMetrics:
    """Dataclass holding QKD protocol simulation results and diagnostic metrics."""

    n_qubits: int
    n_sifted: int
    qber: float
    qber_z: float
    qber_x: float
    qber_ucb: float
    noise_level: float
    noise_type: str
    eve_fraction: float
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    eve_intercepted: np.ndarray
    eve_bases: np.ndarray
    eve_bits: np.ndarray
    bob_bases: np.ndarray
    bob_bits: np.ndarray
    sifted_indices: np.ndarray
    sifted_errors: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


class BB84Simulation:
    """BB84 QKD Protocol Simulator using Qiskit Aer backend.

    Attributes:
        n_qubits: Number of photon pulses/qubits transmitted by Alice.
        noise_level: Error rate on the quantum channel [0.0, 1.0].
        noise_type: 'depolarizing' or 'phase_flip'.
        eve_fraction: Fraction of transmitted qubits intercepted by Eve [0.0, 1.0].
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 1000,
        noise_level: float = 0.0,
        noise_type: Literal["depolarizing", "phase_flip"] = "depolarizing",
        eve_fraction: float = 0.0,
        seed: int = 42,
    ) -> None:
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError(f"noise_level must be in [0.0, 1.0], got {noise_level}")
        if not (0.0 <= eve_fraction <= 1.0):
            raise ValueError(f"eve_fraction must be in [0.0, 1.0], got {eve_fraction}")
        if noise_type not in ("depolarizing", "phase_flip"):
            raise ValueError(f"Unknown noise_type '{noise_type}', expected 'depolarizing' or 'phase_flip'")

        self.n_qubits = n_qubits
        self.noise_level = float(noise_level)
        self.noise_type = noise_type
        self.eve_fraction = float(eve_fraction)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _build_noise_model(self) -> Optional[NoiseModel]:
        """Build Qiskit NoiseModel attached only to the channel identity gate."""
        if self.noise_level <= 1e-9:
            return None

        noise_model = NoiseModel()
        if self.noise_type == "depolarizing":
            # Single-qubit depolarizing error attached only to 'id' gate
            error = depolarizing_error(self.noise_level, 1)
            noise_model.add_all_qubit_quantum_error(error, ["id"])
        elif self.noise_type == "phase_flip":
            # Phase flip (Pauli Z) error attached only to 'id' gate
            p_z = self.noise_level
            p_i = max(0.0, 1.0 - p_z)
            error = pauli_error([("Z", p_z), ("I", p_i)])
            noise_model.add_all_qubit_quantum_error(error, ["id"])

        return noise_model

    def run(self) -> QKDMetrics:
        """Run the BB84 protocol simulation and return detailed metrics."""
        # 1. Alice generates random bits and random bases (0 = Z basis {|0>, |1>}, 1 = X basis {|+>, |->})
        alice_bits = self.rng.integers(0, 2, size=self.n_qubits, dtype=np.int32)
        alice_bases = self.rng.integers(0, 2, size=self.n_qubits, dtype=np.int32)

        # 2. Eve intercept-resend configuration
        eve_intercepted = self.rng.random(self.n_qubits) < self.eve_fraction
        eve_bases = np.full(self.n_qubits, -1, dtype=np.int32)
        eve_bits = np.full(self.n_qubits, -1, dtype=np.int32)

        if np.any(eve_intercepted):
            eve_indices = np.where(eve_intercepted)[0]
            eve_bases[eve_indices] = self.rng.integers(0, 2, size=len(eve_indices), dtype=np.int32)

        # 3. Bob chooses random measurement bases (0 = Z, 1 = X)
        bob_bases = self.rng.integers(0, 2, size=self.n_qubits, dtype=np.int32)

        # Build quantum simulator with channel noise
        noise_model = self._build_noise_model()
        simulator = AerSimulator(noise_model=noise_model, seed_simulator=self.seed)

        bob_bits = np.zeros(self.n_qubits, dtype=np.int32)

        # 1. Non-intercepted circuits: 8 possible (a_bit, a_basis, b_basis) configurations
        if np.any(~eve_intercepted):
            # Collect circuits and batch execute
            active_direct = []
            active_indices = []
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

                qc = QuantumCircuit(1, 1, name=f"d_{k}")
                if a_bit == 1:
                    qc.x(0)
                if a_bas == 1:
                    qc.h(0)
                qc.id(0)  # Channel noise
                if b_bas == 1:
                    qc.h(0)
                qc.measure(0, 0)

                active_direct.append((qc, idx))

            if active_direct:
                circuits_to_run = [item[0] for item in active_direct]
                transpiled_circs = transpile(circuits_to_run, simulator, optimization_level=0)
                if not isinstance(transpiled_circs, list):
                    transpiled_circs = [transpiled_circs]

                for circ_i, (_, idx) in enumerate(active_direct):
                    res = simulator.run(transpiled_circs[circ_i], shots=len(idx), memory=True).result()
                    mem = res.get_memory()
                    for m_i, bit_str in enumerate(mem):
                        bob_bits[idx[m_i]] = int(bit_str.strip())

        # 2. Intercepted circuits: 16 possible (a_bit, a_basis, e_basis, b_basis) configurations
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

                qc = QuantumCircuit(1, 2, name=f"e_{k}")
                if a_bit == 1:
                    qc.x(0)
                if a_bas == 1:
                    qc.h(0)
                if e_bas == 1:
                    qc.h(0)
                qc.measure(0, 0)

                qc.reset(0)
                with qc.if_test((qc.clbits[0], 1)):
                    qc.x(0)
                if e_bas == 1:
                    qc.h(0)
                qc.id(0)  # Channel noise from Eve to Bob
                if b_bas == 1:
                    qc.h(0)
                qc.measure(0, 1)

                active_eve.append((qc, idx))

            if active_eve:
                circuits_to_run = [item[0] for item in active_eve]
                transpiled_circs = transpile(circuits_to_run, simulator, optimization_level=0)
                if not isinstance(transpiled_circs, list):
                    transpiled_circs = [transpiled_circs]

                for circ_i, (_, idx) in enumerate(active_eve):
                    res = simulator.run(transpiled_circs[circ_i], shots=len(idx), memory=True).result()
                    mem = res.get_memory()
                    for m_i, bit_str in enumerate(mem):
                        clean_bits = bit_str.replace(" ", "")
                        eve_bits[idx[m_i]] = int(clean_bits[-1])
                        bob_bits[idx[m_i]] = int(clean_bits[0]) if len(clean_bits) > 1 else int(clean_bits[-1])

        # 4. Sifting step: Keep indices where Alice and Bob used the same basis
        sifted_mask = alice_bases == bob_bases
        sifted_indices = np.where(sifted_mask)[0]
        n_sifted = int(len(sifted_indices))

        if n_sifted > 0:
            alice_sifted = alice_bits[sifted_indices]
            bob_sifted = bob_bits[sifted_indices]
            errors = alice_sifted != bob_sifted
            sifted_errors = errors.astype(np.int32)
            qber = float(np.mean(errors))

            # Basis-resolved QBER
            z_mask = (alice_bases[sifted_indices] == 0)
            x_mask = (alice_bases[sifted_indices] == 1)

            qber_z = float(np.mean(errors[z_mask])) if np.any(z_mask) else 0.0
            qber_x = float(np.mean(errors[x_mask])) if np.any(x_mask) else 0.0

            # 95% one-sided upper confidence bound: QBER + 1.645 * sqrt(QBER * (1 - QBER) / n_sifted)
            variance_term = qber * (1.0 - qber) / n_sifted
            qber_ucb = float(min(1.0, qber + 1.645 * math.sqrt(max(0.0, variance_term))))
        else:
            sifted_errors = np.array([], dtype=np.int32)
            qber = 0.0
            qber_z = 0.0
            qber_x = 0.0
            qber_ucb = 1.0

        return QKDMetrics(
            n_qubits=self.n_qubits,
            n_sifted=n_sifted,
            qber=qber,
            qber_z=qber_z,
            qber_x=qber_x,
            qber_ucb=qber_ucb,
            noise_level=self.noise_level,
            noise_type=self.noise_type,
            eve_fraction=self.eve_fraction,
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            eve_intercepted=eve_intercepted,
            eve_bases=eve_bases,
            eve_bits=eve_bits,
            bob_bases=bob_bases,
            bob_bits=bob_bits,
            sifted_indices=sifted_indices,
            sifted_errors=sifted_errors,
        )


def run_bb84_simulation(
    n_qubits: int = 1000,
    noise_level: float = 0.0,
    noise_type: str = "depolarizing",
    eve_fraction: float = 0.0,
    seed: int = 42,
) -> QKDMetrics:
    """Helper functional interface to run BB84 simulation."""
    sim = BB84Simulation(
        n_qubits=n_qubits,
        noise_level=noise_level,
        noise_type=noise_type,  # type: ignore
        eve_fraction=eve_fraction,
        seed=seed,
    )
    return sim.run()


def main() -> None:
    """Demonstrate BB84 simulation with 1000 qubits, 5% noise, 20% Eve."""
    print("=========================================================")
    print("PHASE 3: BB84 QKD SIMULATION DEMO")
    print("=========================================================")
    print("Running BB84 with 1000 qubits, 5% noise, 20% Eve...")

    metrics = run_bb84_simulation(
        n_qubits=1000,
        noise_level=0.05,
        noise_type="depolarizing",
        eve_fraction=0.20,
        seed=42,
    )

    print(f"\n--- BB84 Simulation Results (Qiskit Aer) ---")
    print(f"Total Qubits Sent:      {metrics.n_qubits}")
    print(f"Sifted Key Bits:        {metrics.n_sifted} ({metrics.n_sifted / metrics.n_qubits * 100:.1f}%)")
    print(f"Channel Noise Level:    {metrics.noise_level * 100:.1f}% ({metrics.noise_type})")
    print(f"Eve Intercept Fraction: {metrics.eve_fraction * 100:.1f}%")
    print(f"Observed QBER:          {metrics.qber:.4f} ({metrics.qber * 100:.2f}%)")
    print(f"  Z-Basis QBER:         {metrics.qber_z:.4f} ({metrics.qber_z * 100:.2f}%)")
    print(f"  X-Basis QBER:         {metrics.qber_x:.4f} ({metrics.qber_x * 100:.2f}%)")
    print(f"QBER 95% UCB:           {metrics.qber_ucb:.4f} ({metrics.qber_ucb * 100:.2f}%)")

    # Theoretical sanity check: at 100% Eve and 0 noise, QBER should be ~ 25% (0.25 +/- 0.04)
    print("\nRunning theoretical sanity check: 100% Eve, 0% noise...")
    sanity_metrics = run_bb84_simulation(
        n_qubits=2000,
        noise_level=0.0,
        noise_type="depolarizing",
        eve_fraction=1.0,
        seed=100,
    )
    print(f"Sanity Check QBER (100% Eve, 0% Noise): {sanity_metrics.qber:.4f} (Expected ~0.2500)")
    assert 0.20 <= sanity_metrics.qber <= 0.30, f"QBER {sanity_metrics.qber} deviates significantly from 25%"
    print("[SUCCESS] Sanity check passed! QBER matches theoretical intercept-resend expectation (~25%).")


if __name__ == "__main__":
    main()
