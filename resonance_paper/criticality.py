"""Neural-criticality estimators for the real-data criticality studies.

Proper criticality markers (not stand-ins like raw LZc / 1-f slope):

  * dfa(x)               -- detrended-fluctuation-analysis exponent. alpha~0.5
                            uncorrelated, ->1.0 long-range correlated (near
                            criticality), >1 non-stationary. Validated below:
                            white~0.5, 1/f~1.0, brown~1.5.
  * lrtc_envelope(sig)   -- DFA of a band's amplitude envelope = the canonical
                            human-EEG long-range-temporal-correlation marker
                            (Linkenkaer-Hansen 2001).
  * branching_ratio_mr(A)-- multistep-regression estimator of the branching ratio
                            m_hat (Wilting & Priesemann 2018). m~1 = critical.
                            This is the REAL-BRAIN analog of Study 10's swept
                            branching ratio sigma; validated below against the
                            branching process at known sigma.
  * eeg_branching_ratio(data, sf) -- m_hat from multichannel EEG via thresholded
                            avalanche event-rate + the MR estimator.

A criticality-proximity score |alpha-1| or |m_hat-1| (smaller = closer to
critical) lets us test the in-silico prediction (Study 10): harmonicity H is
MAXIMIZED near the critical point.
"""
from __future__ import annotations

import numpy as np


def dfa(x, scales=None):
    """Detrended fluctuation analysis exponent of 1-D signal x."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    n = len(y)
    if n < 64:
        return float("nan")
    if scales is None:
        scales = np.unique(np.logspace(np.log10(16), np.log10(max(32, n // 4)), 18).astype(int))
    F, used = [], []
    for s in scales:
        s = int(s)
        if s < 8 or s > n // 2:
            continue
        nseg = n // s
        seg = y[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        coef = np.polyfit(t, seg.T, 1)               # (2, nseg): slope, intercept
        fit = (np.outer(t, coef[0]) + coef[1]).T     # (nseg, s)
        rms = np.sqrt(np.mean((seg - fit) ** 2, axis=1))
        F.append(np.sqrt(np.mean(rms ** 2))); used.append(s)
    F = np.asarray(F); s_arr = np.asarray(used, float)
    ok = np.isfinite(F) & (F > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(s_arr[ok]), np.log(F[ok]), 1)[0])


def lrtc_envelope(sig, sf, band=(8, 13)):
    """DFA of a band-limited amplitude envelope (Linkenkaer-Hansen LRTC)."""
    from scipy.signal import butter, filtfilt, hilbert
    ny = sf / 2.0
    b, a = butter(4, [band[0] / ny, band[1] / ny], btype="band")
    env = np.abs(hilbert(filtfilt(b, a, np.asarray(sig, float))))
    return dfa(env)


def branching_ratio_mr(activity, kmax=20):
    """Multistep-regression branching ratio m_hat (Wilting-Priesemann 2018).

    r_k = regression slope of A_{t+k} on A_t decays as ~ m^k; m_hat from the
    slope of log r_k vs k. Robust to subsampling. m~1 critical, <1 subcritical.
    """
    A = np.asarray(activity, float); A = A - A.mean()
    den = np.sum(A[:-1] ** 2)
    if den <= 0:
        return float("nan")
    rk, ks = [], []
    for k in range(1, kmax + 1):
        d = np.sum(A[:-k] ** 2)
        if d <= 0:
            break
        rk.append(np.sum(A[:-k] * A[k:]) / d); ks.append(k)
    rk = np.asarray(rk); ks = np.asarray(ks, float)
    ok = np.isfinite(rk) & (rk > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.exp(np.polyfit(ks[ok], np.log(rk[ok]), 1)[0]))


def eeg_event_rate(data, sf, thresh=3.0, bin_ms=4.0):
    """Population event-rate time series from multichannel EEG: count of
    |z-score|>thresh excursions across channels per time-bin."""
    data = np.atleast_2d(np.asarray(data, float))
    z = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-12)
    rate = (np.abs(z) > thresh).sum(0).astype(float)        # (n_times,)
    bs = max(1, int(round(bin_ms * sf / 1000.0)))
    n = (len(rate) // bs) * bs
    if n < bs * 8:
        return rate
    return rate[:n].reshape(-1, bs).sum(1)


def eeg_branching_ratio(data, sf, thresh=3.0, bin_ms=4.0, kmax=20):
    """m_hat from multichannel EEG avalanche event-rate."""
    return branching_ratio_mr(eeg_event_rate(data, sf, thresh, bin_ms), kmax=kmax)


# ---------------------------------------------------------------------------
# Population activity + neuronal-avalanche analysis (DCC)
# ---------------------------------------------------------------------------
def population_activity(data, sf, mode="gfp", thresh=2.5):
    """A scalar population-activity TIME SERIES from multichannel EEG -- the
    in-vivo analog of Study 10's branching-activity A(t), on which to compute
    *avalanche-resonance* (Investigation #1).
      mode='gfp'        -> global field power (std across channels per sample)
      mode='event_rate' -> count of |z|>thresh channels per sample
    """
    data = np.atleast_2d(np.asarray(data, float))
    if mode == "gfp":
        return data.std(axis=0)
    z = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-12)
    return (np.abs(z) > thresh).sum(0).astype(float)


def _avalanches_from_activity(A):
    """Avalanche (size, duration) pairs from a 1-D activity series: an avalanche
    is a run of A>0 bounded by A==0 bins."""
    A = np.asarray(A, float)
    thr = np.median(A[A > 0]) * 0.0 if np.any(A > 0) else 0.0   # active = A>0
    active = A > thr
    sizes, durs = [], []
    cs = cd = 0.0
    for a, v in zip(active, A):
        if a:
            cs += v; cd += 1
        elif cd > 0:
            sizes.append(cs); durs.append(cd); cs = cd = 0.0
    if cd > 0:
        sizes.append(cs); durs.append(cd)
    return np.asarray(sizes, float), np.asarray(durs, float)


def powerlaw_mle(x, xmin=1.0):
    """Clauset discrete-power-law MLE exponent."""
    x = np.asarray(x, float); x = x[x >= xmin]
    if x.size < 20:
        return float("nan")
    return float(1.0 + x.size / np.sum(np.log(x / (xmin - 0.5))))


def dcc_from_activity(A):
    """DCC + avalanche exponents from a 1-D population-activity series."""
    sizes, durs = _avalanches_from_activity(A)
    out = dict(tau=float("nan"), alpha=float("nan"), gamma_fit=float("nan"),
               gamma_pred=float("nan"), dcc=float("nan"), n_avalanches=int(len(sizes)))
    if len(sizes) < 50:
        return out
    tau = powerlaw_mle(sizes, 1.0); alpha = powerlaw_mle(durs, 1.0)
    out["tau"] = tau; out["alpha"] = alpha
    uniq = np.unique(durs)
    mS = np.array([sizes[durs == d].mean() for d in uniq])
    ok = (uniq > 0) & (mS > 0)
    if ok.sum() >= 4:
        gamma_fit = float(np.polyfit(np.log(uniq[ok]), np.log(mS[ok]), 1)[0])
        out["gamma_fit"] = gamma_fit
        if np.isfinite(tau) and np.isfinite(alpha) and abs(tau - 1) > 1e-6:
            gp = (alpha - 1.0) / (tau - 1.0)
            out["gamma_pred"] = float(gp); out["dcc"] = float(abs(gamma_fit - gp))
    return out


def avalanche_dcc(data, sf, thresh=2.5, bin_ms=4.0):
    """Deviation-from-Criticality Coefficient (Ma et al. 2019) from multichannel
    EEG: avalanches in the population event-rate -> size exponent tau, duration
    exponent alpha; crackling scaling predicts gamma_pred=(alpha-1)/(tau-1);
    DCC=|gamma_fit-gamma_pred|, ~0 at criticality."""
    return dcc_from_activity(eeg_event_rate(data, sf, thresh=thresh, bin_ms=bin_ms))


# ---------------------------------------------------------------------------
# self-validation against known ground truth
# ---------------------------------------------------------------------------
def _validate():
    rng = np.random.default_rng(0)
    n = 20000
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    # pink (1/f) via spectral shaping
    f = np.fft.rfftfreq(n, 1.0); f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f)
    pink = np.fft.irfft(spec, n)
    print("DFA validation (expect white~0.5, pink~1.0, brown~1.5):")
    print(f"  white={dfa(white):.2f}  pink={dfa(pink):.2f}  brown={dfa(brown):.2f}")

    # branching ratio vs known sigma (reuse Study 10's branching process)
    try:
        from resonance_paper.study10_criticality import branching_activity
        print("branching_ratio_mr validation (m_hat peaks at sigma=1):")
        for sigma in (0.80, 0.90, 1.00, 1.10, 1.20):
            ms = [branching_ratio_mr(branching_activity(sigma, N=500, T=12000, seed=s)[0]) for s in range(6)]
            print(f"  sigma={sigma:.2f} -> m_hat={np.nanmean(ms):.3f}")
    except Exception as exc:  # pragma: no cover
        print(f"  [branching validation skipped: {exc}]")

    # DCC validation on a Poisson-branching avalanche process (real silent bins)
    def _sim_aval(sigma, n_av, rng):
        s = []
        for _ in range(n_av):
            active = 1
            while active > 0 and len(s) < 400000:
                s.append(active)
                active = min(int(rng.poisson(min(sigma * active, 50.0))), 200)  # cap to avoid runaway
            s.append(0)
        return np.asarray(s, float)
    print("avalanche-exponent validation (Poisson branching; size tau->1.5 at sigma=1):")
    for sigma in (0.85, 0.95, 1.00, 1.05, 1.15):
        rng = np.random.default_rng(int(sigma * 100))
        d = dcc_from_activity(_sim_aval(sigma, 4000, rng))
        print(f"  sigma={sigma:.2f} -> tau={d['tau']:.2f} |tau-1.5|={abs(d['tau']-1.5):.2f} "
              f"DCC={d['dcc']:.3f} (n_av={d['n_avalanches']})")


if __name__ == "__main__":
    _validate()
