"""Decision Engine for Hybrid Quantum-Classical Security.

Evaluates physical QKD parameters (QBER, UCB, sifted key length) together with
classical network threat classification (P(Attack)) to produce an adaptive
security decision (ACCEPT, MONITOR, REJECT).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional

VerdictType = Literal["ACCEPT", "MONITOR", "REJECT"]


def binary_entropy(p: float) -> float:
    """Compute the Shannon binary entropy function h(p) with boundary protection.

    h(p) = -p * log2(p) - (1 - p) * log2(1 - p)
    Defined on [0, 1] with h(0) = h(1) = 0.

    Args:
        p: Error probability in [0, 1].

    Returns:
        Entropy value in [0, 1].
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if not (0.0 < p < 1.0):
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def calculate_asymptotic_key_rate(qber: float) -> float:
    """Calculate asymptotic secret key generation rate R for BB84.

    Under one-way classical post-processing in the asymptotic limit (Shor-Preskill / Devetak-Winter):
    R = max(0, 1 - 2 * h(QBER))

    Args:
        qber: Observed or estimated Quantum Bit Error Rate.

    Returns:
        Fractional key rate R in [0, 1].
    """
    if qber < 0.0:
        return 0.0
    if qber >= 0.5:
        return 0.0
    h_q = binary_entropy(qber)
    return float(max(0.0, 1.0 - 2.0 * h_q))


def evaluate_connection_security(
    qber_ucb: float,
    n_sifted: int,
    attack_prob: float,
    base_qber_threshold: float = 0.11,
    qber: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate overall connection security and determine channel verdict.

    Combines physical-layer QBER statistics with classical threat probability
    using an adaptive threshold function:
        max_allowed_qber = base_qber_threshold * (1 - 0.5 * attack_prob)

    Verdict Rules:
    - REJECT:  qber_ucb > max_allowed_qber OR attack_prob >= 0.85
    - ACCEPT:  qber_ucb <= max_allowed_qber AND attack_prob < 0.5
    - MONITOR: point-estimate qber <= max_allowed_qber but qber_ucb > max_allowed_qber
               (statistical uncertainty)
    - FALLBACK / ELEVATED: qber_ucb <= max_allowed_qber but 0.5 <= attack_prob < 0.85
               (elevated network threat)

    Args:
        qber_ucb: 95% upper confidence bound on QBER.
        n_sifted: Number of sifted key bits available.
        attack_prob: Classical threat probability P(Attack) in [0, 1].
        base_qber_threshold: Base theoretical QBER threshold (default: 0.11 ~ 11%).
        qber: Observed point-estimate QBER (if None, defaults to qber_ucb).

    Returns:
        Dictionary containing verdict, max_allowed_qber, key_rate, and reason.
    """
    if qber is None:
        qber = qber_ucb

    # Clamping and bounds check
    attack_prob = max(0.0, min(1.0, float(attack_prob)))
    qber = max(0.0, min(1.0, float(qber)))
    qber_ucb = max(0.0, min(1.0, float(qber_ucb)))

    # Adaptive QBER threshold scales down when classical threat is higher
    max_allowed_qber = float(base_qber_threshold * (1.0 - 0.5 * attack_prob))
    key_rate = calculate_asymptotic_key_rate(qber)

    verdict: VerdictType
    reason: str

    if n_sifted < 10:
        verdict = "REJECT"
        reason = (
            f"Insufficient sifted key bits (n_sifted={n_sifted} < 10) to guarantee "
            f"information-theoretic security or statistical confidence."
        )
    elif attack_prob >= 0.85:
        verdict = "REJECT"
        reason = (
            f"Critical classical network threat detected (P(Attack) = {attack_prob:.1%} >= 85.0%). "
            f"Network segment untrusted; key distribution aborted."
        )
    elif qber_ucb > max_allowed_qber:
        if qber <= max_allowed_qber:
            # Point estimate passes, but UCB fails due to finite sample statistical variance
            verdict = "MONITOR"
            reason = (
                f"Statistical uncertainty: Observed QBER ({qber:.1%}) <= threshold ({max_allowed_qber:.1%}), "
                f"but 95% UCB ({qber_ucb:.1%}) exceeds threshold. Key generation paused for further sampling."
            )
        else:
            verdict = "REJECT"
            reason = (
                f"Quantum Bit Error Rate ({qber_ucb:.1%} UCB) exceeds adaptive threshold "
                f"({max_allowed_qber:.1%}). Channel eavesdropping or intolerable noise detected."
            )
    elif attack_prob < 0.5:
        verdict = "ACCEPT"
        reason = (
            f"Channel verified secure: QBER UCB ({qber_ucb:.1%}) <= threshold ({max_allowed_qber:.1%}) "
            f"and network threat level normal ({attack_prob:.1%}). Key generation approved."
        )
    else:  # 0.5 <= attack_prob < 0.85 and qber_ucb <= max_allowed_qber
        verdict = "MONITOR"
        reason = (
            f"Elevated classical threat detected (P(Attack) = {attack_prob:.1%}). "
            f"QBER is currently within bounds ({qber_ucb:.1%} <= {max_allowed_qber:.1%}), "
            f"but increased surveillance is advised."
        )

    return {
        "verdict": verdict,
        "max_allowed_qber": max_allowed_qber,
        "key_rate": key_rate,
        "reason": reason,
        "qber": qber,
        "qber_ucb": qber_ucb,
        "attack_prob": attack_prob,
        "n_sifted": n_sifted,
    }
