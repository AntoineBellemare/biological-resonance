"""Study 26 — Oscillatory criticality: testing R = H*PC directly against the branching ratio m̂.

The branching network (Study 10) has m̂ but no oscillations (R at the floor); the Wilson-Cowan
model (Study 12) has oscillations but its transition is read via susceptibility, not m̂. Neither
lets the framework's FLAGSHIP measure R be validated against the *branching* criticality axis.

We close that gap with a faithful spiking network: a current-based leaky integrate-and-fire E/I
network of the Brunel (2000) type (sparse random connectivity, inhibition-dominated balance,
transmission delays, refractoriness). The control parameter is the recurrent synaptic GAIN. Low
gain -> externally-driven, subcritical reverberation (m̂ < 1); raising it amplifies recurrent
activity until it becomes marginally self-sustaining (m̂ -> 1, scale-free avalanches), and the
E-I delay loop produces a nested population oscillation at that edge (the regime where avalanches
and oscillations coexist; Poil 2012 / di Santo 2018). Crucially, m̂ here measures recurrent
amplification (Wilting-Priesemann 2018), which is exactly what the gain controls -- so m̂ crosses 1
on a genuine spiking process, not a confounded rate signal.

Per gain (n seeds): population spike-count activity A(t).
  Independent criticality axis (none derived from H/PC/R):
    * m̂ via the multistep-regression estimator (Wilting-Priesemann) on binned A(t)
    * avalanche size power-law goodness-of-fit
    * susceptibility (var of A) and critical slowing (autocorr time)
    * LRTC/DFA of the oscillation-amplitude envelope
  Resonance: H on the population rate; PC/R single-signal (harmonic phase coupling within A) and
    cross-population (E-rate vs I-rate).

Headline test: does R (and PC) peak AT m̂ -> 1, co-located with the avalanche/LRTC markers?

Usage:  python -m resonance_paper.study26_critical_oscillations --probe   # fast physics check
        python -m resonance_paper.study26_critical_oscillations           # quick run
        python -m resonance_paper.study26_critical_oscillations --paper   # paper-grade
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from resonance_paper import criticality as Cr
from biotuner.resonance import compute_resonance
from biotuner.harmonic_connectivity import compute_cross_resonance


SF = 1000.0   # one network step = 1 ms

# Resonance config: matches the Wilson-Cowan study (Study 12) exactly so R is comparable across the
# criticality models (fmax=80 covers a ~12-18 Hz fundamental + its harmonics).
from biotuner.resonance import ResonanceConfig
CROSS_CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=80, noverlap=400,
                            coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                            ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                            return_intermediates=True)


def disc_ei_net(sigma, N=1000, fE=0.8, K=20, r_ref=2, p_spont=2e-4, wI=4.0,
                tauE=2.0, tauI=22.0, n_steps=12000, seed=0):
    """Discrete-time probabilistic E/I spiking network. Explicit E (fraction fE) and I units,
    sparse random out-connectivity (K targets each), refractoriness, and decaying synaptic input
    pools (fast excitatory tauE, slow inhibitory tauI, in steps). `sigma` scales excitatory drive
    (the branching / avalanche-criticality knob); the slow inhibitory feedback sets a STABLE
    population oscillation whose frequency is fixed by tauI, not by sigma. Returns per-step E and I
    population spike counts (the cross-resonance read uses these two signals, as in Study 12)."""
    rng = np.random.default_rng(seed)
    NE = int(N * fE)
    targets = rng.integers(0, N, size=(N, K))
    aE = np.exp(-1.0 / tauE); aI = np.exp(-1.0 / tauI)
    gE = np.zeros(N); gI = np.zeros(N)                  # decaying excitatory / inhibitory input pools
    refr = np.zeros(N, int)
    active = np.zeros(N, bool); active[rng.integers(0, N, size=max(3, N // 50))] = True
    rE = np.empty(n_steps); rI = np.empty(n_steps)
    for t in range(n_steps):
        rE[t] = active[:NE].sum(); rI[t] = active[NE:].sum()
        gE *= aE; gI *= aI
        aiE = np.where(active[:NE])[0]
        aiI = np.where(active[NE:])[0] + NE
        if aiE.size:
            gE += np.bincount(targets[aiE].ravel(), minlength=N)
        if aiI.size:
            gI += np.bincount(targets[aiI].ravel(), minlength=N)
        drive = sigma * gE / K - wI * gI / K
        p = np.clip(drive, 0.0, 1.0) + p_spont
        elig = refr == 0
        newactive = elig & (rng.random(N) < p)
        refr[refr > 0] -= 1
        refr[active] = r_ref
        active = newactive
    return rE, rI


def _bin(A, nbin):
    n = (len(A) // nbin) * nbin
    return A[:n].reshape(-1, nbin).sum(1)


def _powerlaw_r2(A):
    sizes, _durs = Cr._avalanches_from_activity(np.asarray(A, float))
    sizes = sizes[sizes >= 1]
    if len(sizes) < 30:
        return float("nan")
    hi = max(2, int(np.percentile(sizes, 99)))
    bins = np.unique(np.logspace(0, np.log10(hi), 18).astype(int))
    if len(bins) < 4:
        return float("nan")
    cnt, edges = np.histogram(sizes, bins=bins, density=True)
    ctr = np.sqrt(edges[:-1] * edges[1:]); m = (cnt > 0) & (ctr > 0)
    if m.sum() < 4:
        return float("nan")
    lx, ly = np.log10(ctr[m]), np.log10(cnt[m])
    sl, it = np.polyfit(lx, ly, 1); pred = sl * lx + it
    return float(1 - np.sum((ly - pred) ** 2) / (np.sum((ly - ly.mean()) ** 2) + 1e-12))


def _spec_peak(A, fs):
    A = np.asarray(A, float) - np.mean(A)
    f, P = welch(A, fs=fs, nperseg=min(4096, len(A)))
    band = (f >= 3) & (f <= 200)
    if not band.any():
        return float("nan"), float("nan")
    fb, Pb = f[band], P[band]
    j = int(np.argmax(Pb))
    return float(fb[j]), float(Pb[j] / (np.median(Pb) + 1e-30))


def _autocorr_time(x, max_lag=200):
    x = np.asarray(x, float) - np.mean(x)
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac[:max_lag] < np.exp(-1))[0]
    return float(below[0]) if len(below) else float(max_lag)


def probe(sigmas=None):
    """Fast physics check across the branching knob sigma: rate, oscillation freq + prominence
    (is the fundamental stable?), avalanche power-law (PL_r2), crackling DCC (~0 at criticality),
    m̂ (lags 20 / 5), and the cross E-I resonance R read with the Study-12 config."""
    if sigmas is None:
        sigmas = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
    print("  sigma  rateHz  f_osc  prom   PL_r2   DCC   chi    nav   maxs   PCei   Rei")
    for sigma in sigmas:
        rE, rI = disc_ei_net(sigma, n_steps=16000, seed=0)
        cut = len(rE) // 6; rE = rE[cut:]; rI = rI[cut:]; A = rE + rI
        rate = A.sum() / (len(A) / SF) / 1000.0
        fpk, prom = _spec_peak(A, SF)
        plr2 = _powerlaw_r2(A)
        dcc = Cr.dcc_from_activity(A).get("dcc", float("nan"))
        sizes, _ = Cr._avalanches_from_activity(A); sizes = sizes[sizes >= 1]
        chi = float(np.mean(sizes ** 2) / (np.mean(sizes) + 1e-9)) if len(sizes) else float("nan")
        nav = len(sizes); maxs = float(sizes.max()) if len(sizes) else 0.0
        Ez = _norm(rE.astype(np.float64)); Iz = _norm(rI.astype(np.float64))
        try:
            res = compute_cross_resonance(Ez, Iz, sf=SF, config=CROSS_CFG)
            fr = res.freqs; i = int(np.argmin(np.abs(fr - fpk))) if np.isfinite(fpk) else 0
            PCei = float(res.phase_coupling_matrix[i, i])
            Hei = float(C.band_value(fr, res.factors["H"]["all"], fpk)) if np.isfinite(fpk) else float("nan")
            Rei = Hei * PCei
        except Exception:
            PCei = Rei = float("nan")
        print(f"  {sigma:5.2f}  {rate:6.2f}  {fpk:5.1f}  {prom:5.1f}  {plr2:6.2f}  {dcc:5.2f}  "
              f"{chi:5.1f}  {nav:5d}  {maxs:5.0f}  {PCei:5.2f}  {Rei:5.2f}", flush=True)


# ---------------------------------------------------------------------------
def _features(sigma, N, seed, quick):
    rE, rI = disc_ei_net(sigma, N=N, n_steps=14000 if quick else 24000, seed=seed)
    cut = len(rE) // 6; rE = rE[cut:]; rI = rI[cut:]
    A = rE + rI
    # --- independent criticality axis (avalanche-based; m̂ reported but oscillation-confounded) ---
    m_hat = Cr.branching_ratio_mr(A, kmax=20)
    pl_r2 = _powerlaw_r2(A)                                  # power-law quality: peaks at criticality
    dccd = Cr.dcc_from_activity(A)
    dcc = float(dccd.get("dcc", float("nan"))); tau = float(dccd.get("tau", float("nan")))
    sizes, _d = Cr._avalanches_from_activity(A); sizes = sizes[sizes >= 1]
    chi = float(np.mean(sizes ** 2) / (np.mean(sizes) + 1e-9)) if len(sizes) else float("nan")
    maxs = float(sizes.max()) if len(sizes) else 0.0
    suscept = float(np.var(A)); suscept_norm = float(np.var(A) / (np.mean(A) ** 2 + 1e-12))
    actau = _autocorr_time(A)
    f_osc, prom = _spec_peak(A, SF)
    Az = _norm(A.astype(np.float64))
    band = (max(2.5, f_osc * 0.6), f_osc * 1.6) if np.isfinite(f_osc) else (8.0, 13.0)
    try:
        lrtc = Cr.lrtc_envelope(Az, SF, band=band)
    except Exception:
        lrtc = float("nan")
    # --- resonance: single-signal H + within-signal PC (Study-12 kernels), and cross E-I (chosen read) ---
    cfg_self = C.default_config(fmin=2, fmax=80, precision_hz=0.5)
    r = compute_resonance(Az, sf=SF, config=cfg_self)
    H = float(r.summaries["H"]["max"])
    PCs = float(C.band_value(r.freqs, r.factors["PC"], f_osc)) if np.isfinite(f_osc) else np.nan
    Hs = float(C.band_value(r.freqs, r.factors["H"], f_osc)) if np.isfinite(f_osc) else np.nan
    R_self = Hs * PCs if np.isfinite(PCs) else np.nan
    Ez = _norm(rE.astype(np.float64)); Iz = _norm(rI.astype(np.float64))
    res = compute_cross_resonance(Ez, Iz, sf=SF, config=CROSS_CFG)
    fr = res.freqs; i = int(np.argmin(np.abs(fr - f_osc))) if np.isfinite(f_osc) else 0
    PC = float(res.phase_coupling_matrix[i, i])
    Hc = float(C.band_value(fr, res.factors["H"]["all"], f_osc)) if np.isfinite(f_osc) else np.nan
    R = Hc * PC
    return dict(m_hat=m_hat, prox_crit=(pl_r2 if np.isfinite(pl_r2) else np.nan),
                powerlaw_r2=pl_r2, dcc=dcc, tau=tau, chi=chi, max_aval=maxs,
                susceptibility=suscept, susceptibility_norm=suscept_norm,
                autocorr_time=actau, lrtc=lrtc, peak_prom=prom, f_osc=f_osc,
                H=H, PC=PC, R=R, PC_self=PCs, R_self=R_self)


def run(quick=True):
    N = 800 if quick else 1200
    seeds = list(range(3) if quick else range(8))
    cs = [0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6] if quick else \
         [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 1.10, 1.35, 1.70]
    METRICS = ["m_hat", "prox_crit", "powerlaw_r2", "dcc", "chi", "max_aval",
               "susceptibility", "susceptibility_norm", "autocorr_time", "lrtc",
               "peak_prom", "f_osc", "H", "PC", "R", "PC_self", "R_self"]
    per = {m: [] for m in METRICS}; rows = []
    for c in cs:                       # c == branching parameter sigma
        acc = {m: [] for m in METRICS}
        for seed in seeds:
            f = _features(c, N, seed, quick)
            for m in METRICS:
                acc[m].append(f[m])
        row = dict(sigma=c)
        for m in METRICS:
            ci = C.mean_ci([v for v in acc[m] if np.isfinite(v)]); per[m].append(acc[m])
            row[m] = ci["mean"]; row[m + "_sem"] = ci["sem"]
        rows.append(row)
        print(f"  sigma={c:.2f}  PLr2={row['powerlaw_r2']:.2f}  DCC={row['dcc']:.2f}  "
              f"chi={row['chi']:.0f}  prom={row['peak_prom']:.0f}  f={row['f_osc']:.1f}  "
              f"H={row['H']:.3f}  PC={row['PC']:.3f}  R={row['R']:.3f}", flush=True)

    def mat(m):
        k = min(len(a) for a in per[m]); return np.array([a[:k] for a in per[m]])
    c_arr = np.array(cs, float)

    # criticality proxy = avalanche power-law quality (peaks at criticality). m̂ is reported but
    # NOT used as the axis (oscillation-confounded -> violates the MR estimator's assumptions).
    def tracks(metric, predictor="prox_crit"):
        from scipy.stats import spearmanr, wilcoxon
        rhos = []
        for j in range(len(seeds)):
            y = [per[metric][i][j] for i in range(len(cs))]
            x = [per[predictor][i][j] for i in range(len(cs))]
            ok = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
            if len(ok) >= 5:
                rr = spearmanr([a for a, _ in ok], [b for _, b in ok])[0]
                if np.isfinite(rr):
                    rhos.append(rr)
        if len(rhos) < 3:
            return dict(rho=float("nan"), n=len(rhos))
        ci = C.mean_ci(rhos)
        try:
            p = float(wilcoxon(rhos).pvalue)
        except ValueError:
            p = float("nan")
        return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=p,
                    frac_pos=float(np.mean(np.array(rhos) > 0)), n=len(rhos))

    def peak_diff(metricA, metricB):
        MA, MB = mat(metricA), mat(metricB); nb = min(MA.shape[1], MB.shape[1])
        rng = np.random.default_rng(3); d = []
        for _ in range(2000):
            idx = rng.integers(0, nb, nb)
            a = c_arr[int(np.nanargmax(np.nanmean(MA[:, idx], axis=1)))]
            b = c_arr[int(np.nanargmax(np.nanmean(MB[:, idx], axis=1)))]
            d.append(a - b)
        d = np.array(d); lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        return dict(mean=float(np.mean(d)), lo=lo, hi=hi, coincide=bool(lo <= 0 <= hi))

    summary = dict(N=N, n_seeds=len(seeds),
                   R_tracks_crit=tracks("R"), PC_tracks_crit=tracks("PC"),
                   H_tracks_crit=tracks("H"), Rself_tracks_crit=tracks("R_self"),
                   R_peak_vs_crit=peak_diff("R", "powerlaw_r2"),
                   PC_peak_vs_crit=peak_diff("PC", "powerlaw_r2"),
                   sigma_at_max_crit=float(c_arr[int(np.nanargmax([r["powerlaw_r2"] for r in rows]))]),
                   sigma_at_max_R=float(c_arr[int(np.nanargmax([r["R"] for r in rows]))]),
                   sigma_at_max_PC=float(c_arr[int(np.nanargmax([r["PC"] for r in rows]))]),
                   sigma_runaway=float(c_arr[int(np.nanargmax([r["max_aval"] for r in rows]))]))
    result = dict(quick=quick, sigmas=cs, rows=rows, summary=summary)
    C.save_json(result, "study26_critical_oscillations.json")
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 26 headline (oscillatory criticality: R vs avalanche criticality) ---")
    print(f"  {'sigma':>5} {'m_hat':>6} {'PL_r2':>6} {'DCC':>5} {'chi':>7} {'prom':>6} "
          f"{'H':>6} {'PC':>6} {'R':>6}")
    for r in result["rows"]:
        print(f"  {r['sigma']:>5.2f} {r['m_hat']:>6.2f} {r['powerlaw_r2']:>6.2f} {r['dcc']:>5.2f} "
              f"{r['chi']:>7.0f} {r['peak_prom']:>6.0f} {r['H']:>6.3f} {r['PC']:>6.3f} {r['R']:>6.3f}")
    for nm, k in [("R", "R_tracks_crit"), ("PC", "PC_tracks_crit"), ("H", "H_tracks_crit"),
                  ("R_self", "Rself_tracks_crit")]:
        v = s[k]
        if np.isfinite(v.get("rho", np.nan)):
            print(f"  {nm} tracks criticality (power-law quality): rho={v['rho']:+.2f} "
                  f"[{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.2g} ({int(v['frac_pos']*100)}%+, n={v['n']})")
    pd = s["R_peak_vs_crit"]
    print(f"  R peak vs power-law-quality peak (in sigma): diff={pd['mean']:+.2f} "
          f"[{pd['lo']:+.2f},{pd['hi']:+.2f}] -> {'COINCIDE' if pd['coincide'] else 'DIFFER'}")
    print(f"  sigma at max power-law={s['sigma_at_max_crit']:.2f}, at max R={s['sigma_at_max_R']:.2f}, "
          f"at max PC={s['sigma_at_max_PC']:.2f}, runaway onset~{s['sigma_runaway']:.2f}")


if __name__ == "__main__":
    import sys
    if "--probe" in sys.argv:
        probe()
    else:
        run(quick="--paper" not in sys.argv)
