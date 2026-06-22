"""Study 24 — Operating characteristics: SNR robustness, null calibration, scaling.

A methods paper should state WHERE a method works, that its inference CONTROLS false
positives, and how it SCALES. Three parts:

  (A) SNR ROBUSTNESS — detection AUC vs SNR. Harmonicity detection (harmonic stack vs
      pink noise, statistic H_max) and n:m coupling detection (locked vs unlocked
      single signal, statistic max PC_z against an AAFT null) across a sweep of SNR.
      Gives the operating range, not a single ceiling number.

  (B) NULL CALIBRATION — on signals with NO phase coupling (PSD-matched nulls), the
      per-frequency surrogate z should exceed the alpha=0.05 one-sided threshold
      (z>1.645) about 5% of the time. Confirms PC_z controls type-I error (and flags
      that max-over-frequency detection needs a multiple-comparison correction).

  (C) SCALING — wall-clock of compute_resonance vs signal length and vs frequency
      resolution (n_freqs), so users can budget runtime.

Outputs: results/study24_operating_characteristics.json, figures/study24_*.{png,pdf}
"""
from __future__ import annotations

import time
import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from resonance_paper.study5_cross_signal import gen_pair, cross_target_z, _config_for, SF as SF5
from biotuner.resonance import compute_resonance

SF = 500.0           # >= 320 so the AAFT surrogate's 150 Hz post-filter stays below Nyquist
DUR = 12.0
F0 = 10.0
LOCK_PAIRS = [(6.0, 9.0)]    # study5 lock_2to3 regime: A@6, B@9 (2:3), targeted PC entry


def harmonic_sig(snr_db, seed=0):
    rng = np.random.default_rng(seed); n = int(SF * DUR); t = np.arange(n) / SF
    x = sum(np.sin(2 * np.pi * F0 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 5)) / 4
    nz = 10 ** (-snr_db / 20.0)
    return _norm(_norm(x) + nz * pink_noise(n, SF, seed=seed + 1)).astype(np.float64)


def noise_sig(seed=0):
    n = int(SF * DUR)
    return _norm(pink_noise(n, SF, seed=seed)).astype(np.float64)


def _Hmax(sig, cfg):
    return float(compute_resonance(sig, sf=SF, config=cfg).summaries["H"]["max"])


def part_a_snr(quick):
    """Detection AUC vs SNR for harmonicity (H, harmonic vs noise) and n:m coupling
    (targeted cross PC z-score, study5 lock_2to3 regime, locked vs unlocked)."""
    cfg = C.default_config(fmin=2, fmax=45, precision_hz=0.5)
    cfg5 = _config_for("lock_2to3")
    snrs = [-36, -30, -24, -18, -12, -6, 0] if quick else [-42, -36, -30, -24, -18, -12, -6, 0, 6]
    cpl_snrs = [-36, -30, -24, -18, -12, -6] if quick else [-42, -36, -30, -24, -18, -12, -6, 0]
    seeds = range(8) if quick else range(16)
    cpl_seeds = range(6) if quick else range(12)
    n_surr = 20 if quick else 40
    out = {"snrs": snrs, "H_auc": [], "cpl_snrs": cpl_snrs, "PC_auc": []}
    for snr in snrs:                                   # H detection (cheap)
        h_pos = [_Hmax(harmonic_sig(snr, s), cfg) for s in seeds]
        h_neg = [_Hmax(noise_sig(s + 500), cfg) for s in seeds]
        out["H_auc"].append(C.bootstrap_auc_ci(h_pos, h_neg)["auc"])
        print(f"  [A-H]  SNR={snr:+d}dB  H_AUC={out['H_auc'][-1]:.2f}", flush=True)
    for snr in cpl_snrs:                               # coupling detection (cross PC_z; expensive)
        pos, neg = [], []
        for s in cpl_seeds:
            A, B = gen_pair("lock_2to3", True, snr_db=snr, seed=s)
            pos.append(cross_target_z(A, B, SF5, cfg5, LOCK_PAIRS, "PC", n=n_surr, seed=s))
            A, B = gen_pair("lock_2to3", False, snr_db=snr, seed=s + 700)
            neg.append(cross_target_z(A, B, SF5, cfg5, LOCK_PAIRS, "PC", n=n_surr, seed=s + 700))
        out["PC_auc"].append(C.bootstrap_auc_ci(pos, neg)["auc"])
        print(f"  [A-PC] SNR={snr:+d}dB  coupling_AUC={out['PC_auc'][-1]:.2f}", flush=True)
    return out


def part_b_null(quick):
    """Null calibration of the cross PC z-score: on UNCOUPLED pairs (the genuine null)
    the z should be ~N(0,1), so ~5% exceed the one-sided alpha=0.05 threshold."""
    cfg5 = _config_for("lock_2to3")
    n_inst = 40 if quick else 120
    n_surr = 30 if quick else 60
    zs = []
    for s in range(n_inst):
        A, B = gen_pair("lock_2to3", False, snr_db=6, seed=s + 9000)
        zs.append(cross_target_z(A, B, SF5, cfg5, LOCK_PAIRS, "PC", n=n_surr, seed=s + 9000))
        if s % 10 == 0:
            print(f"  [B] null instance {s}/{n_inst}", flush=True)
    Z = np.array(zs); Z = Z[np.isfinite(Z)]
    return dict(n_instances=int(Z.size), n_surr=n_surr,
                fpr_005=float(np.mean(Z > 1.645)), fpr_001=float(np.mean(Z > 2.326)),
                z_mean=float(Z.mean()), z_std=float(Z.std()))


def part_c_scaling(quick):
    lengths = [1000, 2000, 4000, 8000] if quick else [1000, 2000, 4000, 8000, 16000, 32000]
    cfg = C.default_config(fmin=2, fmax=45, precision_hz=0.5)
    rng = np.random.default_rng(0)
    rows = []
    for L in lengths:
        x = _norm(pink_noise(L, SF, seed=1)).astype(np.float64)
        t0 = time.time()
        reps = 3
        for _ in range(reps):
            compute_resonance(x, sf=SF, config=cfg)
        rows.append(dict(length=L, sec_per_call=(time.time() - t0) / reps))
        print(f"  [C] length={L}  {rows[-1]['sec_per_call']*1000:.0f} ms/call", flush=True)
    # vs frequency resolution (n_freqs ~ (fmax-fmin)/precision)
    res_rows = []
    x = _norm(pink_noise(4000, SF, seed=2)).astype(np.float64)
    for prec in ([1.0, 0.5, 0.25] if quick else [2.0, 1.0, 0.5, 0.25, 0.1]):
        cfg2 = C.default_config(fmin=2, fmax=45, precision_hz=prec)
        t0 = time.time()
        for _ in range(3):
            compute_resonance(x, sf=SF, config=cfg2)
        res_rows.append(dict(precision_hz=prec, n_freqs=int((45 - 2) / prec),
                             sec_per_call=(time.time() - t0) / 3))
        print(f"  [C] precision={prec}Hz  {res_rows[-1]['sec_per_call']*1000:.0f} ms/call", flush=True)
    return dict(by_length=rows, by_resolution=res_rows)


def run(quick=True):
    A = part_a_snr(quick); B = part_b_null(quick); Cc = part_c_scaling(quick)
    result = dict(quick=quick, snr=A, null=B, scaling=Cc)
    C.save_json(result, "study24_operating_characteristics.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    A = result["snr"]; B = result["null"]
    print("\n  --- Study 24 headline (operating characteristics) ---")
    print("  (A) harmonicity detection AUC vs SNR: "
          + "  ".join(f"{s:+d}dB:{a:.2f}" for s, a in zip(A["snrs"], A["H_auc"])))
    print("      coupling detection AUC vs SNR:    "
          + "  ".join(f"{s:+d}dB:{a:.2f}" for s, a in zip(A["cpl_snrs"], A["PC_auc"])))
    print(f"  (B) cross PC z-score null calibration (uncoupled pairs): false-positive rate "
          f"at alpha=0.05 = {B['fpr_005']:.3f} (target 0.05); at 0.01 = {B['fpr_001']:.3f} (target 0.01); "
          f"null z = {B['z_mean']:.2f}+/-{B['z_std']:.2f}")
    sc = result["scaling"]["by_length"]; rr = result["scaling"]["by_resolution"]
    print(f"  (C) scaling: runtime ~flat in length ({sc[0]['length']}->{sc[0]['sec_per_call']*1000:.0f}ms, "
          f"{sc[-1]['length']}->{sc[-1]['sec_per_call']*1000:.0f}ms) — dominated by n_freqs: "
          f"{rr[0]['n_freqs']}->{rr[0]['sec_per_call']*1000:.0f}ms, {rr[-1]['n_freqs']}->{rr[-1]['sec_per_call']*1000:.0f}ms")


def _figures(result):
    plt = C.setup_mpl()
    A = result["snr"]; B = result["null"]; sc = result["scaling"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(A["snrs"], A["H_auc"], "o-", color="#0072B2", label="harmonicity (H)")
    axes[0].plot(A["cpl_snrs"], A["PC_auc"], "s-", color="#CC79A7", label="coupling (cross PC_z)")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("detection AUC"); axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title("A. Operating range\n(detection AUC vs SNR)", fontsize=10); axes[0].legend(fontsize=7)

    axes[1].bar([0, 1], [B["fpr_005"], B["fpr_001"]], color="#009E73", width=0.6, alpha=0.85)
    axes[1].axhline(0.05, color="k", ls="--", lw=0.8); axes[1].axhline(0.01, color="grey", ls=":", lw=0.8)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["α=0.05", "α=0.01"])
    axes[1].set_ylabel("per-frequency false-positive rate")
    axes[1].set_title("B. Null calibration of PC_z\n(targets = dashed)", fontsize=10)

    nf = [r["n_freqs"] for r in sc["by_resolution"]]; ms = [r["sec_per_call"] * 1000 for r in sc["by_resolution"]]
    axes[2].plot(nf, ms, "o-", color="#D55E00")
    axes[2].set_xlabel("n frequency bins"); axes[2].set_ylabel("runtime (ms/call)")
    axes[2].set_title("C. Scaling: cost is set by n_freqs\n(~flat in signal length)", fontsize=10)

    fig.suptitle("Study 24 — Operating characteristics: SNR robustness, null calibration, scaling",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study24_operating_characteristics")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)
