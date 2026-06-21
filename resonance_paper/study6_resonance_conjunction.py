"""Study 6 — Resonance as the conjunction of harmonic + phase alignment.

The conceptual claim behind the framework: **resonance occurs only when two
oscillatory processes are BOTH harmonically related AND phase-aligned**. R =
H x PC should therefore light up only for that conjunction, while H alone or PC
alone is satisfied by the wrong cases.

NOTE ON CIRCULARITY: Part A below illustrates the *decomposition* R = H x PC, but
because R is defined as that product, "R is high only when H and PC are both
high" is partly definitional, not an empirical discovery. The NON-circular test —
whether H (from the spectrum) and PC (from the dynamics) genuinely co-vary
because of physics — is in Study 7 (coupled Van der Pol / Arnold tongues), where
harmonicity predicts phase-locking propensity (Spearman H~PC ~ +0.68) without
being wired to it. Part A is retained as an illustration; Part B (polyrhythm
recovery) is a genuine recovery test.

Part A — the 2x2 conjunction (between signals)
    Four channel-pair conditions:
        HL  harmonic ratio (1:2) AND phase-locked     -> R high   (true resonance)
        Hn  harmonic ratio (1:2), NOT phase-locked    -> R low    (phase fails)
        nL  inharmonic (1:1.6), phase-locked          -> R low    (harmonic fails)
        nn  inharmonic, not phase-locked              -> R low
    Harmonic score = harmsim(fA, fB) (cross-harmonicity of the pair); phase score
    = cross PC surrogate-z at (fA, fB) (channel-B IAAFT null, the fixed
    spectrum-based path). Resonance = harmonic x phase. We show R separates HL
    from the other three (high AUC) while neither factor alone does.

Part B — polyrhythmic structure recovery (within + between)
    A 2:3:4 phase-locked polyrhythm. Within one signal: compute_resonance should
    peak at the polyrhythm components. Between two signals each carrying part of
    the polyrhythm (A: 6+12, B: 9+18): cross-resonance recovers the cross
    structure vs a phase-scrambled control.

Outputs: results/study6_resonance_conjunction.json, figures/study6_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import pink_noise, _norm
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import compute_resonance, ResonanceConfig
import biotuner.resonance.kernels_harmonic  # noqa: F401 (registers kernels)
from biotuner.resonance.registry import HARMONIC_KERNELS
from biotuner.resonance.nulls import iaaft_surrogate

SF = 500.0
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=400,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)


def _wander(base, t, sf, diff, rng):
    return 2 * np.pi * base * t + np.cumsum(diff * np.sqrt(1.0 / sf) * rng.standard_normal(len(t)))


# ---------------------------------------------------------------------------
# Part A — 2x2 conjunction
# ---------------------------------------------------------------------------
def _pair(harmonic, locked, sf=SF, dur=40.0, snr_db=6.0, diff=1.0, seed=0):
    """A at 6 Hz; B at 12 Hz (harmonic 1:2) or 11 Hz (inharmonic 6:11),
    phase-locked to A or free. Both target frequencies sit on the 0.5 Hz STFT
    grid and are equally lockable, so the inharmonic-locked case (nL) has genuine
    HIGH phase coupling but LOW harmonicity — making the conjunction non-trivial
    (phase alone cannot reject nL; only resonance = harmonic x phase can).
    Returns (A, B, fA, fB)."""
    rng = np.random.default_rng(seed); n = int(sf * dur); t = np.arange(n) / sf
    snr = 10 ** (snr_db / 10.0)
    noise = lambda s: _norm(pink_noise(n, sf, seed=seed + s))
    phiA = _wander(6.0, t, sf, diff, rng); A = _norm(np.sin(phiA))
    ratio = 2.0 if harmonic else (11.0 / 6.0)   # 1:2 (simple) vs 6:11 (complex)
    fB = 6.0 * ratio
    if locked:
        B = _norm(np.sin(ratio * phiA + 0.3))
    else:
        B = _norm(np.sin(_wander(fB, t, sf, diff, np.random.default_rng(seed + 2))))
    return np.sqrt(snr) * A + noise(100), np.sqrt(snr) * B + noise(200), 6.0, fB


def _harmsim_pair(fA, fB):
    """Cross-harmonicity of a single frequency pair via the harmsim kernel."""
    S = HARMONIC_KERNELS["harmsim"](np.array([fA]), np.array([fB]))
    return float(S[0, 0])


def _cross_pc_z(A, B, fA, fB, n=30, seed=0):
    """Surrogate z of the cross phase-coupling matrix entry Phi_AB[fA,fB]
    (channel-B IAAFT null)."""
    obs = compute_cross_resonance(A, B, sf=SF, config=CFG); fr = obs.freqs
    i = int(np.argmin(np.abs(fr - fA))); j = int(np.argmin(np.abs(fr - fB)))
    o = obs.phase_coupling_matrix[i, j]

    def one(s):
        Bs = iaaft_surrogate(B, np.random.default_rng(s))
        return compute_cross_resonance(A, Bs, sf=SF, config=CFG).phase_coupling_matrix[i, j]
    rng = np.random.default_rng(seed)
    try:
        from joblib import Parallel, delayed
        sv = np.array(Parallel(n_jobs=-1)(delayed(one)(int(x)) for x in rng.integers(0, 2**31 - 1, n)))
    except Exception:
        sv = np.array([one(int(x)) for x in rng.integers(0, 2**31 - 1, n)])
    return float((o - sv.mean()) / (sv.std() + 1e-12))


def _part_a(quick):
    seeds = range(8) if quick else range(20)
    n_surr = 30 if quick else 100
    conds = {"HL": (True, True), "Hn": (True, False),
             "nL": (False, True), "nn": (False, False)}
    rows = []
    for cname, (harm, lock) in conds.items():
        for seed in seeds:
            A, B, fA, fB = _pair(harm, lock, seed=seed)
            h = _harmsim_pair(fA, fB)               # harmonic alignment (0..1-ish, /100)
            pcz = _cross_pc_z(A, B, fA, fB, n=n_surr, seed=seed)
            h_norm = h / 100.0                       # harmsim is in [0,100]
            pc_pos = max(pcz, 0.0)                   # clip null/negative to 0 for the product
            rows.append(dict(cond=cname, harmonic=harm, locked=lock, seed=seed,
                             harm_score=h_norm, phase_z=pcz,
                             resonance=h_norm * pc_pos))
        print(f"  [A] {cname} done")

    def auc_vs_rest(field):
        hl = [r[field] for r in rows if r["cond"] == "HL"]
        rest = [r[field] for r in rows if r["cond"] != "HL"]
        return C.bootstrap_auc_ci(hl, rest, n_boot=2000)

    return dict(rows=rows,
                auc_harmonic=auc_vs_rest("harm_score"),
                auc_phase=auc_vs_rest("phase_z"),
                auc_resonance=auc_vs_rest("resonance"),
                cond_means={c: dict(
                    harm=float(np.mean([r["harm_score"] for r in rows if r["cond"] == c])),
                    phase=float(np.mean([r["phase_z"] for r in rows if r["cond"] == c])),
                    res=float(np.mean([r["resonance"] for r in rows if r["cond"] == c])),
                ) for c in conds})


# ---------------------------------------------------------------------------
# Part B — polyrhythm recovery
# ---------------------------------------------------------------------------
def _polyrhythm_within(sf=SF, dur=40.0, snr_db=8.0, diff=0.5, seed=0, locked=True):
    """A 2:3:4 polyrhythm (6, 9, 12 Hz) phase-locked to a common 3 Hz clock."""
    rng = np.random.default_rng(seed); n = int(sf * dur); t = np.arange(n) / sf
    clock = _wander(3.0, t, sf, diff, rng)        # base unit
    if locked:
        osc = np.sin(2 * clock) + 0.8 * np.sin(3 * clock + 0.2) + 0.6 * np.sin(4 * clock + 0.5)
    else:
        osc = (np.sin(_wander(6.0, t, sf, diff, np.random.default_rng(seed + 1)))
               + 0.8 * np.sin(_wander(9.0, t, sf, diff, np.random.default_rng(seed + 2)))
               + 0.6 * np.sin(_wander(12.0, t, sf, diff, np.random.default_rng(seed + 3))))
    osc = _norm(osc)
    snr = 10 ** (snr_db / 10.0)
    return (np.sqrt(snr) * osc + _norm(pink_noise(n, sf, seed=seed + 99))).astype(np.float64)


def _part_b(quick):
    seeds = range(6) if quick else range(15)
    # Within-signal: does R peak at the polyrhythm components 6/9/12?
    targets = [6.0, 9.0, 12.0]
    within = {"locked": [], "scrambled": []}
    cfg = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=30, noverlap=400,
                          coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                          ratio_kernel_params={"max_denom": 16, "beta": 1.0})
    for seed in seeds:
        for key, lock in (("locked", True), ("scrambled", False)):
            sig = _polyrhythm_within(seed=seed, locked=lock)
            r = compute_resonance(sig, sf=SF, config=cfg)
            fr = r.freqs; R = r.resonance_spectrum
            inband = max(C.band_value(fr, R, f) for f in targets)
            offband = np.median(R)
            within[key].append(float(inband / (offband + 1e-12)))
    auc_within = C.bootstrap_auc_ci(within["locked"], within["scrambled"], n_boot=2000)
    return dict(within_locked_ratio=float(np.mean(within["locked"])),
                within_scrambled_ratio=float(np.mean(within["scrambled"])),
                auc_within=auc_within)


def run(quick=True):
    print("Study 6A — resonance conjunction (R = harmonic AND phase) ...")
    a = _part_a(quick)
    print("Study 6B — polyrhythm recovery ...")
    b = _part_b(quick)
    result = dict(quick=quick, part_a=a, part_b=b)
    C.save_json(result, "study6_resonance_conjunction.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    a, b = result["part_a"], result["part_b"]
    print("\n  --- Study 6 headline ---")
    print("  [A] HL-vs-rest AUC by factor (resonance should win; H & PC alone shouldn't):")
    print(f"      harmonic-only AUC = {a['auc_harmonic']['auc']:.2f} "
          f"[{a['auc_harmonic']['lo']:.2f},{a['auc_harmonic']['hi']:.2f}]")
    print(f"      phase-only    AUC = {a['auc_phase']['auc']:.2f} "
          f"[{a['auc_phase']['lo']:.2f},{a['auc_phase']['hi']:.2f}]")
    print(f"      RESONANCE     AUC = {a['auc_resonance']['auc']:.2f} "
          f"[{a['auc_resonance']['lo']:.2f},{a['auc_resonance']['hi']:.2f}]")
    print("      condition means (harm / phase_z / resonance):")
    for c, m in a["cond_means"].items():
        print(f"        {c}: {m['harm']:.2f} / {m['phase']:+.2f} / {m['res']:.3f}")
    print(f"  [B] polyrhythm R peak/median: locked={b['within_locked_ratio']:.2f} "
          f"scrambled={b['within_scrambled_ratio']:.2f}  AUC={b['auc_within']['auc']:.2f}")


def _figures(result):
    plt = C.setup_mpl()
    a, b = result["part_a"], result["part_b"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # A: condition means for harm / phase / resonance (normalized per factor)
    conds = list(a["cond_means"].keys())
    harm = np.array([a["cond_means"][c]["harm"] for c in conds])
    phase = np.array([max(a["cond_means"][c]["phase"], 0) for c in conds])
    res = np.array([a["cond_means"][c]["res"] for c in conds])
    def nrm(x): return x / (x.max() + 1e-12)
    x = np.arange(len(conds)); w = 0.26
    axes[0].bar(x - w, nrm(harm), w, label="harmonic", color="#1a237e")
    axes[0].bar(x, nrm(phase), w, label="phase (z+)", color="#6a1b9a")
    axes[0].bar(x + w, nrm(res), w, label="resonance", color="#b71c1c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(
        ["HL\nharm+lock", "Hn\nharm only", "nL\nlock only", "nn\nneither"], fontsize=8)
    axes[0].set_ylabel("normalized score"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"A. Resonance = harmonic AND phase\n"
                      f"(resonance AUC={a['auc_resonance']['auc']:.2f} vs "
                      f"H={a['auc_harmonic']['auc']:.2f}, PC={a['auc_phase']['auc']:.2f})",
                      fontsize=10)

    # B: polyrhythm peak/median ratio locked vs scrambled
    axes[1].bar(["locked", "scrambled"],
                [b["within_locked_ratio"], b["within_scrambled_ratio"]],
                color=["#b71c1c", "#9e9e9e"])
    axes[1].set_ylabel("R peak / median (2:3:4 bands)")
    axes[1].set_title(f"B. Polyrhythm recovery (within-signal)\nAUC={b['auc_within']['auc']:.2f}",
                      fontsize=10)
    fig.suptitle("Study 6 — Resonance as harmonic + phase conjunction; polyrhythm recovery",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study6_resonance_conjunction")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)
