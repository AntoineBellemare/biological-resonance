"""n:m coupling indices beyond the PLV family — canonically homed in biotuner now.

These all-moment n:m indices (Tass 1998 entropy index; conditional-probability index; Palus 1997 phase
mutual information) live in ``biotuner.resonance.coupling`` and are re-exported here so the
resonance_paper studies keep a stable import path. The validated cross-signal detector that wraps them
with bandpass+Hilbert, the correct Tass multipliers, an IAAFT surrogate-z, and the within-signal
scope guard is ``biotuner.resonance.detect_nm_coupling``.

Call with the CORRECT n:m multipliers (for f_b/f_a = p/q use n=p, m=q); each returns a scalar in ~[0,1].
"""
from biotuner.resonance.coupling import (
    nm_rho_entropy as rho_entropy,
    nm_conditional_prob as conditional_prob,
    nm_phase_mi as phase_mi,
)

TECHNIQUES = {"rho_entropy": rho_entropy, "conditional_prob": conditional_prob, "phase_mi": phase_mi}

__all__ = ["rho_entropy", "conditional_prob", "phase_mi", "TECHNIQUES"]
