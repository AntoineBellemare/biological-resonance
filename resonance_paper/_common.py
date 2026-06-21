"""Shared infrastructure for the resonance validation suite (resonance_paper/).

Everything the four studies need in common:

  * path setup + results/figures IO (numpy-aware JSON)
  * publication plotting style
  * ResonanceConfig presets and the strategy-sweep grids
  * feature extraction: turn a ResonanceResult into a flat scalar feature dict
  * detection helpers for ground-truth recovery (band z-score, ROC/AUC)

The whole suite uses the SINGLE-SIGNAL resonance API
(``biotuner.resonance.compute_resonance`` / ``with_surrogate_null``) plus the
per-spectrum complexity metrics, and sweeps the strategy registry — matching
the scope chosen for the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
PAPER_DIR = HERE / "paper"
for _d in (RESULTS_DIR, FIG_DIR, PAPER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Resonance API
# --------------------------------------------------------------------------
from biotuner.resonance import (  # noqa: E402
    compute_resonance,
    ResonanceConfig,
    with_surrogate_null,
)

# Primary surrogate null for single-signal inference.
# NOTE: with_surrogate_null's documented default 'IAAFT' is NOT implemented for
# single signals (biotuner.surrogates.generate_surrogate only knows
# AAFT/TFT/phase/shuffle/white/pink/brown/blue), so calling it with defaults
# raises ValueError. 'AAFT' (amplitude-adjusted Fourier transform) is the right
# null here: it preserves the power spectrum AND the amplitude distribution while
# randomizing phase, so a surviving resonance peak reflects genuine phase
# structure rather than spectral shape.
PRIMARY_SURROGATE = "AAFT"

# --------------------------------------------------------------------------
# Config presets
# --------------------------------------------------------------------------
def default_config(fmin=2.0, fmax=45.0, precision_hz=0.5, remove_aperiodic=True,
                   **overrides) -> ResonanceConfig:
    """Recommended config for NEW analyses.

    Uses the canonical n:m phase-coupling convention (nm_plv_canonical) paired
    with the fraction ratio kernel so true n:m mode-locks are tested exactly,
    not just the small binary preset table.

    ``remove_aperiodic=True`` by default (Tier-2 refinement): harmonicity is
    confounded by the aperiodic (1/f) component — on pure colored noise both
    H_avg and H_max rise with the 1/f slope — so the aperiodic component is
    removed before harmonicity is computed unless a comparison explicitly needs
    it left in.
    """
    params = dict(
        precision_hz=precision_hz,
        fmin=fmin,
        fmax=fmax,
        remove_aperiodic=remove_aperiodic,
        harmonic_kernel="harmsim",
        ratio_kernel="fraction",
        ratio_kernel_params={"max_denom": 16, "beta": 1.0},
        coupling_metric="nm_plv_canonical",
        combine="product",
    )
    params.update(overrides)
    return ResonanceConfig(**params)


def legacy_config(fmin=2.0, fmax=45.0, precision_hz=0.5, **overrides) -> ResonanceConfig:
    """Bit-exact reproduction preset for the pre-refactor compute_global_harmonicity."""
    params = dict(
        precision_hz=precision_hz,
        fmin=fmin,
        fmax=fmax,
        psd_normalization="minmax_prob",
        harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 20, "min_notes": 2},
        ratio_kernel="binary",
        ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
        coupling_metric="nm_plv",
        gaussian_smooth_sigma=1.0,
        legacy_self_pair_subtract=True,
        normalize=True,
        bandwidth_correction=False,
        combine="product",
    )
    params.update(overrides)
    return ResonanceConfig(**params)


# --------------------------------------------------------------------------
# Strategy-sweep grids (the "method comparison" axis of the paper)
# --------------------------------------------------------------------------
HARMONIC_KERNELS = ["harmsim", "subharm_tension"]
RATIO_KERNELS = ["binary", "fraction"]
COUPLING_METRICS = ["nm_plv", "nm_plv_canonical", "nm_pli", "nm_wpli", "nm_rrci"]
COMBINE_RULES = ["product", "geomean", "harmmean", "min", "weighted_log"]
SURROGATE_TYPES = ["AAFT", "phase", "shuffle"]


def ratio_params_for(ratio_kernel):
    if ratio_kernel == "binary":
        return {"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True}
    return {"max_denom": 16, "beta": 1.0}


def strategy_grid(
    harmonic_kernels=None,
    ratio_kernels=None,
    coupling_metrics=None,
    combine="product",
    base_fmin=2.0,
    base_fmax=45.0,
    precision_hz=0.5,
):
    """Yield (label, ResonanceConfig) for the cartesian product of strategies."""
    hk = harmonic_kernels or HARMONIC_KERNELS
    rk = ratio_kernels or RATIO_KERNELS
    cm = coupling_metrics or COUPLING_METRICS
    for h in hk:
        for r in rk:
            for c in cm:
                label = f"{h}|{r}|{c}"
                cfg = ResonanceConfig(
                    precision_hz=precision_hz,
                    fmin=base_fmin,
                    fmax=base_fmax,
                    harmonic_kernel=h,
                    ratio_kernel=r,
                    ratio_kernel_params=ratio_params_for(r),
                    coupling_metric=c,
                    combine=combine,
                )
                yield label, cfg


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
SPECTRA = ("H", "PC", "R")
SCALAR_METRICS = (
    "avg", "max", "flatness", "entropy", "spread", "higuchi",
    "peaks_avg", "peak_harmsim_avg", "peak_harmsim_max",
)


def resonance_features(result, prefix=""):
    """Flatten a ResonanceResult into a dict of scalar features.

    One ``{spectrum}_{metric}`` entry per (H/PC/R) × scalar metric, plus the
    dominant peak frequency of each spectrum. ``prefix`` is prepended to every
    key (e.g. ``"R_"`` or a strategy label).
    """
    feats = {}
    for sp in SPECTRA:
        summary = result.summaries.get(sp, {})
        for m in SCALAR_METRICS:
            val = summary.get(m, np.nan)
            feats[f"{prefix}{sp}_{m}"] = float(val) if np.isscalar(val) else np.nan
        # dominant peak frequency
        peaks = result.peaks.get(sp) if result.peaks else None
        if peaks is not None and len(peaks):
            feats[f"{prefix}{sp}_peak_top"] = float(peaks[0])
        else:
            feats[f"{prefix}{sp}_peak_top"] = np.nan
    return feats


# --------------------------------------------------------------------------
# Detection helpers (ground-truth recovery)
# --------------------------------------------------------------------------
def band_value(freqs, values, f_center, half_width=1.0):
    """Max of ``values`` within ``f_center ± half_width`` Hz."""
    mask = np.abs(np.asarray(freqs) - f_center) <= half_width
    if not mask.any():
        return np.nan
    return float(np.nanmax(np.asarray(values)[mask]))


def zscored_resonance(signal, sf, config, surr_type=PRIMARY_SURROGATE, n=100, rng_seed=0):
    """Run with_surrogate_null and return (freqs, R_observed, R_z)."""
    res = with_surrogate_null(
        signal, sf=sf, config=config, surr_type=surr_type,
        n=n, correction="both", parallel=False, rng_seed=rng_seed,
    )
    return res.freqs, res.resonance_spectrum, res.resonance_spectrum_z


def _surrogate_factor_arrays(signal, sf, config, surr_type, n):
    """Run the pipeline on n surrogates, returning stacked (H, PC, R) arrays.

    Parallelized with joblib when available (the surrogate loop is the suite's
    compute bottleneck), serial fallback otherwise.
    """
    from biotuner.surrogates import generate_surrogate

    def one(_i):
        surr = np.asarray(generate_surrogate(signal, surr_type=surr_type, sf=sf), dtype=np.float64)
        r = compute_resonance(surr, sf=sf, config=config)
        return r.factors["H"], r.factors["PC"], r.resonance_spectrum

    try:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=-1, prefer="processes")(delayed(one)(i) for i in range(n))
    except Exception:
        res = [one(i) for i in range(n)]
    H = np.array([r[0] for r in res])
    PC = np.array([r[1] for r in res])
    R = np.array([r[2] for r in res])
    return H, PC, R


def factor_surrogate_z(signal, sf, config, surr_type=PRIMARY_SURROGATE, n=100, seed=0):
    """Per-frequency surrogate z-score for EACH factor (H, PC, R).

    This is the suite's core inference tool. ``with_surrogate_null`` only
    z-scores the resonance spectrum R, but the validation shows R is
    H-dominated (hence PSD-driven) and therefore blind to phase coupling under a
    PSD-preserving null. The phase-coupling factor PC is the correct detector for
    n:m coupling, so we z-score all three factors against the same surrogate
    ensemble.

    Returns
    -------
    freqs : ndarray
    z : dict ``{'H': z_H, 'PC': z_PC, 'R': z_R}`` each (n_freqs,)
    observed : ResonanceResult on the real signal
    """
    observed = compute_resonance(signal, sf=sf, config=config)
    freqs = observed.freqs
    H, PC, R = _surrogate_factor_arrays(signal, sf, config, surr_type, n)
    obs_vals = {"H": observed.factors["H"], "PC": observed.factors["PC"],
                "R": observed.resonance_spectrum}
    stacks = {"H": H, "PC": PC, "R": R}
    z = {}
    for k, arr in stacks.items():
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0) + 1e-12
        z[k] = (obs_vals[k] - mu) / sd
    return freqs, z, observed


def roc_auc(scores_pos, scores_neg):
    """AUC of the score distribution separating positives from negatives.

    Implemented as the Mann-Whitney U statistic / (n_pos * n_neg) so we avoid a
    hard sklearn dependency in the hot path. Returns 0.5 for no separation,
    1.0 for perfect.
    """
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    # rank-based U
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    # simple tie handling: recompute ranks as average
    sort_idx = np.argsort(allv, kind="mergesort")
    sorted_v = allv[sort_idx]
    avg_ranks = np.empty(allv.size, dtype=float)
    i = 0
    while i < allv.size:
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_ranks[sort_idx[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = avg_ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


# --------------------------------------------------------------------------
# Tier-1 statistics: CIs, permutation tests, per-subject stats, FDR
# --------------------------------------------------------------------------
def bootstrap_auc_ci(scores_pos, scores_neg, n_boot=2000, alpha=0.05, seed=0):
    """Point AUC + bootstrap (1-alpha) CI by resampling pos/neg with replacement."""
    pos = np.asarray(scores_pos, float); pos = pos[np.isfinite(pos)]
    neg = np.asarray(scores_neg, float); neg = neg[np.isfinite(neg)]
    point = roc_auc(pos, neg)
    if pos.size == 0 or neg.size == 0:
        return dict(auc=point, lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(pos, size=pos.size, replace=True)
        bn = rng.choice(neg, size=neg.size, replace=True)
        boots[b] = roc_auc(bp, bn)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(auc=float(point), lo=float(lo), hi=float(hi))


def permutation_test_value(observed, null_fn, n_perm=1000, seed=0, tail="greater"):
    """Generic permutation p: fraction of null draws >= observed (+1 smoothing)."""
    rng = np.random.default_rng(seed)
    null = np.array([null_fn(rng) for _ in range(n_perm)], dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return dict(observed=float(observed), p=np.nan, null_mean=np.nan)
    if tail == "greater":
        p = (np.sum(null >= observed) + 1) / (null.size + 1)
    else:
        p = (np.sum(null <= observed) + 1) / (null.size + 1)
    return dict(observed=float(observed), p=float(p), null_mean=float(null.mean()))


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns (rejected_bool_array, qvalues)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out_q = np.empty(n); out_q[order] = np.clip(q, 0, 1)
    return out_q <= alpha, out_q


def paired_by_group(rows, feature, group_key, cond_key, cond_a, cond_b):
    """Per-group (e.g. per-subject) paired comparison of ``feature`` between two
    conditions. Returns Wilcoxon signed-rank p, mean within-pair difference,
    rank-biserial effect size, and the per-group means.
    """
    from scipy.stats import wilcoxon
    groups = sorted(set(r[group_key] for r in rows))
    a_means, b_means = [], []
    for g in groups:
        a = [r[feature] for r in rows if r[group_key] == g and r[cond_key] == cond_a]
        b = [r[feature] for r in rows if r[group_key] == g and r[cond_key] == cond_b]
        a = [v for v in a if np.isfinite(v)]; b = [v for v in b if np.isfinite(v)]
        if a and b:
            a_means.append(np.mean(a)); b_means.append(np.mean(b))
    a_means = np.array(a_means); b_means = np.array(b_means)
    if a_means.size < 2:
        return dict(n_groups=int(a_means.size), p=np.nan, mean_diff=np.nan, rank_biserial=np.nan)
    diff = b_means - a_means
    try:
        stat, p = wilcoxon(b_means, a_means)
    except ValueError:
        p = np.nan
    n_pos = np.sum(diff > 0); n_neg = np.sum(diff < 0)
    rb = (n_pos - n_neg) / max(1, (n_pos + n_neg))
    return dict(n_groups=int(a_means.size), p=float(p) if np.isfinite(p) else np.nan,
                mean_diff=float(np.mean(diff)), rank_biserial=float(rb),
                cond_a_mean=float(a_means.mean()), cond_b_mean=float(b_means.mean()))


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj, name):
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, cls=_NumpyEncoder))
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def load_json(name):
    return json.loads((RESULTS_DIR / name).read_text())


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False, "lines.linewidth": 1.4,
    })
    return plt


def save_fig(fig, name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.relative_to(REPO_ROOT)} (+pdf)")
    import matplotlib.pyplot as plt
    plt.close(fig)
