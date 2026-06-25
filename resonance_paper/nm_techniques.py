"""n:m coupling techniques BEYOND the PLV family — purpose-built for phase-phase synchronization.

The PLV-family metrics (biotuner's nm_plv/pli/wpli/rrci) capture only the FIRST circular moment of the
n:m relative-phase distribution psi = n*phi_a - m*phi_b: they are maximal only for a UNIMODAL psi and
underestimate coupling when psi is multimodal (multistable locks, non-sinusoidal coupling, harmonic
distortion). These measures, introduced for n:m synchronization (Tass et al. 1998 PRL 81:3291;
Rosenblum/Pikovsky), detect ANY departure from uniformity / statistical dependence:

  rho_entropy        Tass entropy index: (Hmax - H)/Hmax from the Shannon entropy of the psi histogram.
  conditional_prob   Tass conditional-probability index: mean concentration of m*phi_b given n*phi_a.
  phase_mi           normalized mutual information between (n*phi_a) and (m*phi_b) — model-free dependence.

All take instantaneous phase arrays (phi_a, phi_b) and multipliers (n, m); call with the CORRECT n:m
multipliers (for f_b/f_a = b/a use n=b, m=a). Each returns a scalar in ~[0, 1]; 0 = no coupling.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi


def _wrap(x):
    return np.angle(np.exp(1j * x))


def rho_entropy(phi_a, phi_b, n, m, nbins=18):
    """Tass n:m entropy synchronization index in [0,1]. 1 = delta-concentrated relative phase."""
    psi = _wrap(n * np.asarray(phi_a) - m * np.asarray(phi_b))
    h, _ = np.histogram(psi, bins=nbins, range=(-np.pi, np.pi))
    p = h / h.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    Hmax = np.log(nbins)
    return float((Hmax - H) / Hmax)


def conditional_prob(phi_a, phi_b, n, m, nbins=18, min_count=5):
    """Tass conditional-probability index in [0,1]: mean resultant of m*phi_b within bins of n*phi_a."""
    x = _wrap(n * np.asarray(phi_a))
    y = m * np.asarray(phi_b)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    rs = []
    for k in range(nbins):
        sel = idx == k
        if sel.sum() > min_count:
            rs.append(np.abs(np.mean(np.exp(1j * y[sel]))))
    return float(np.mean(rs)) if rs else 0.0


def phase_mi(phi_a, phi_b, n, m, nbins=16):
    """Normalized mutual information in [0,1] between (n*phi_a) and (m*phi_b)."""
    x = _wrap(n * np.asarray(phi_a))
    y = _wrap(m * np.asarray(phi_b))
    c, _, _ = np.histogram2d(x, y, bins=nbins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    pxy = c / c.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    outer = px * py
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz])))
    Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float(mi / (min(Hx, Hy) + 1e-12))


TECHNIQUES = {"rho_entropy": rho_entropy, "conditional_prob": conditional_prob, "phase_mi": phase_mi}
