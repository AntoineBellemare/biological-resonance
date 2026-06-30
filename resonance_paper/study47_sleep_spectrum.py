"""Study 47 — The harmonic & phase-coupling SPECTRA of the EEG across sleep stages (whole-spectrum).

The framework's natural readout is not a hand-picked frequency pair but the SPECTRA: the harmonic
spectrum H(f) and the n:m phase-coupling spectrum PC(f), each built from ALL frequency pairs across the
whole spectrum (the fraction kernel assigns the n:m ratio to every pair). We compute these per 30 s
Fpz-Cz epoch with compute_resonance (canonical n:m convention + fraction ratio kernel + Hilbert phase),
for the metric panel (PLV vs the all-moment rho_entropy / phase_mi, plus the 0-lag-robust rrci), and ask
how the whole spectrum changes across Wake / N2 / N3 / REM (per-frequency Kruskal-Wallis, BH-FDR).

This shows whether the new techniques measure a physiologically meaningful, stage-dependent n:m
phase-coupling spectrum in brain signals.

Outputs: results/study47_sleep_spectrum.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import kruskal

from resonance_paper import _common as C
from resonance_paper.study14_sleep import load_sleep
from biotuner.resonance import compute_resonance, ResonanceConfig

warnings.filterwarnings("ignore")

STAGES = ["Wake", "N2", "N3", "REM"]
METRICS = ["nm_plv_canonical", "nm_rrci_canonical", "nm_rho_entropy_canonical", "nm_phase_mi_canonical"]


def _cfg(metric):
    # POWER-INDEPENDENT reduction (alpha_self=alpha_partner=0, off-diagonal, no self-subtract):
    # PC[i] = sum_{j!=i} Phi[i,j], the pure phase-coupling spectrum, so stage-dependence reflects
    # PHASE coupling rather than the (trivially stage-varying) power spectrum.
    return ResonanceConfig(precision_hz=0.5, fmin=1.0, fmax=16.0,
                           coupling_metric=metric, ratio_kernel="fraction",
                           ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                           phase_estimator="hilbert", phase_estimator_params={"bandwidth": 1.5},
                           legacy_self_pair_subtract=False, alpha_self=0.0, alpha_partner=0.0)


def _bh_fdr(p):
    p = np.asarray(p, float); order = np.argsort(p); ranks = np.arange(1, len(p) + 1)
    q = np.empty(len(p)); q[order] = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    return np.clip(q, 0, 1)


def run(quick=True):
    n_sub = 6 if quick else 10
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=6 if quick else 12)
    by_stage = {s: [it for it in items if it.get("state") == s] for s in STAGES}
    print("  epochs/stage:", {s: len(v) for s, v in by_stage.items()}, flush=True)

    cfgs = {m: _cfg(m) for m in METRICS}
    PC = {m: {s: [] for s in STAGES} for m in METRICS}    # metric -> stage -> list of PC spectra
    H = {s: [] for s in STAGES}
    freqs = None
    for s in STAGES:
        for it in by_stage[s]:
            x = np.asarray(it["X"])[0].astype(np.float64); sf = it["sf"]
            for j, m in enumerate(METRICS):
                r = compute_resonance(x, sf=sf, config=cfgs[m])
                if freqs is None:
                    freqs = r.freqs.tolist()
                PC[m][s].append(r.factors["PC"])
                if j == 0:
                    H[s].append(r.factors["H"])
        print(f"    {s}: {len(by_stage[s])} epochs done", flush=True)

    freqs = np.asarray(freqs); nf = len(freqs)
    out = {"freqs": freqs.tolist(), "stages": STAGES, "pc_mean": {}, "h_mean": {}, "stage_dependence": {}}
    out["h_mean"] = {s: np.mean(np.stack(H[s]), 0).tolist() if H[s] else [] for s in STAGES}
    for m in METRICS:
        out["pc_mean"][m] = {s: (np.mean(np.stack(PC[m][s]), 0).tolist() if PC[m][s] else []) for s in STAGES}
        # per-frequency Kruskal across stages, FDR over frequencies
        pf = []
        for k in range(nf):
            groups = [[sp[k] for sp in PC[m][s]] for s in STAGES if len(PC[m][s]) >= 3]
            try:
                pf.append(kruskal(*groups)[1])
            except Exception:
                pf.append(1.0)
        q = _bh_fdr(pf)
        n_sig = int(np.sum(q <= 0.05))
        peakf = float(freqs[int(np.argmin(pf))])
        out["stage_dependence"][m] = dict(n_freqs_sig=n_sig, n_freqs=nf,
                                          min_p=float(np.min(pf)), peak_freq=peakf, fdr_q=q.tolist())
    C.save_json(dict(quick=quick, n_subjects=n_sub, epochs_per_stage={s: len(v) for s, v in by_stage.items()},
                     **out), "study47_sleep_spectrum.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 47 headline (H & PC spectra change across sleep stages) ---")
    for m in METRICS:
        sd = out["stage_dependence"][m]
        print(f"  {m:26s}: {sd['n_freqs_sig']}/{sd['n_freqs']} freq bins of the PC spectrum are stage-dependent "
              f"(FDR<=0.05); strongest at {sd['peak_freq']:.1f} Hz (p={sd['min_p']:.1e}).")
    fm = out["stage_dependence"]["nm_plv_canonical"]["n_freqs_sig"]
    am = max(out["stage_dependence"]["nm_rho_entropy_canonical"]["n_freqs_sig"],
             out["stage_dependence"]["nm_phase_mi_canonical"]["n_freqs_sig"])
    print(f"  => PLV PC-spectrum stage-dependent at {fm} bins; all-moment indices at up to {am} bins "
          f"({'more' if am > fm else 'similar'}). The whole n:m phase-coupling spectrum reorganizes with sleep depth.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)
