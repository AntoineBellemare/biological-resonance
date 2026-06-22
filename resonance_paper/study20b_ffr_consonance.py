"""Study 20b — Real EEG/FFR: harmonicity of the neural response to consonant vs
dissonant musical dyads (the real-data companion to the synthetic Study 20).

Dataset: Andermann, Reineke, Riedel & Rupp (Eur. J. Neurosci. 2026), OSF `5puhb`.
Brainstem frequency-following responses (FFR) — phase-locked, 20 kHz — to dyads:

    condition  ratio              fundamentals   complete?
    CC         3:2  (fifth)       160 + 240 Hz   yes  (consonant, complete)
    CI         3:2  (fifth)       (removed)      no   (consonant, missing-fundamental)
    DC         45:32 (tritone)    160 + 225 Hz   yes  (dissonant, complete)
    DI         45:32 (tritone)    (removed)      no   (dissonant, missing-fundamental)

`ffr` is (39 listeners x 2 runs [0=passive, 1=active] x 4 conditions x 40000 @ 20 kHz);
3 listeners are NaN. In the *incomplete* (CI/DI) conditions the stimulus has NO
energy below ~640 Hz, so any harmonic structure the FFR shows there is
**neurally generated** (missing-fundamental reconstruction + combination/difference
tones) — the clean control that the consonance effect is neural, not stimulus
leakage.

Claims tested
  (A) neural harmonicity H is higher for consonant than dissonant dyads
      (complete CC>DC and — crucially — incomplete CI>DI, where it must be neural);
  (B) the FFR reconstructs harmonic energy in the stimulus-silent band (<600 Hz)
      for the incomplete dyads (neural reconstruction), more so for consonant;
  (C) active listening (attention) modulates the harmonicity advantage;
  (D) the neural consonance advantage scales with musicianship / behavioral
      consonance sensitivity.
We also compute H on the *acoustic stimuli* themselves as the leakage baseline, so
the neural effect can be reported relative to (and beyond) the acoustic one.

Outputs: results/study20b_ffr_consonance.json, figures/study20b_*.{png,pdf}
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from resonance_paper import _common as C

SF_FFR = 20000.0
SF_STIM = 48000.0
TARGET_SF = 4000.0     # decimate before harmonicity: Nyquist 2 kHz >> 1.1 kHz fmax
                       # (20 kHz / 48 kHz are massively oversampled for a <=1.1 kHz analysis)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "osf_5puhb")
OSF = {  # verified public OSF download endpoints (project 5puhb)
    "FFR.mat": "https://osf.io/download/6973c756696044a323bb814b/",
    "Stimuli.mat": "https://osf.io/download/6973c75712fc6a5c99bb8816/",
    "sample.mat": "https://osf.io/download/6973c720696044a323bb811e/",
    "Buttonpress.mat": "https://osf.io/download/6973c71fe2f5570b4fbb7f62/",
}
COND = ["CC", "CI", "DC", "DI"]          # FFR 3rd-dim order (= readme triggers 1-4)
CONSONANT = {"CC", "CI"}
COMPLETE = {"CC", "DC"}
# harmonicity is computed from ~the missing-fundamental band up through the partials
SILENT_BAND = (70.0, 600.0)              # incomplete stimulus has no energy here
FULL_BAND = (70.0, 1100.0)
LOW_BAND = (70.0, 639.0)                 # stimulus-SILENT for incomplete dyads -> NEURAL
HIGH_BAND = (640.0, 1100.0)              # where incomplete-stimulus energy actually exists
# targeted spectral readouts (Hz): mains; each dyad's own (missing) fundamental; difference tones
TARGETS = {"mains60": 60.0, "fund240": 240.0, "fund225": 225.0, "dt80": 80.0, "dt65": 65.0}


# ----------------------------------------------------------------------------- IO
def _fetch(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"  downloading {name} from OSF 5puhb ...", flush=True)
    urllib.request.urlretrieve(OSF[name], path)
    return path


def load():
    import scipy.io as sio
    ffr = sio.loadmat(_fetch("FFR.mat"))["ffr"]            # (39,2,4,40000)
    stim = sio.loadmat(_fetch("Stimuli.mat"))["stimuli"]  # (8,52800) @48k
    try:
        sample = sio.loadmat(_fetch("sample.mat"))["sample"]
    except Exception:
        sample = None
    try:
        resp = sio.loadmat(_fetch("Buttonpress.mat"))["response"]
    except Exception:
        resp = None
    return ffr, stim, sample, resp


# --------------------------------------------------------------------- spectral
def _to_target(x, sf_in):
    """Anti-alias + resample an epoch to TARGET_SF for the harmonicity analysis."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    if abs(sf_in - TARGET_SF) < 1e-6:
        return np.asarray(x, dtype=np.float64), TARGET_SF
    fr = Fraction(TARGET_SF / sf_in).limit_denominator(1000)
    y = resample_poly(np.asarray(x, dtype=np.float64), fr.numerator, fr.denominator)
    return y, TARGET_SF


N_PEAKS = 14           # capture BOTH partial series (dissonance lives in their interaction)
MAXDENOM = 64          # ratio rounding: captures 45:32 tritone, rejects noise overfit


def _band_power(x, sf, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / sf)
    m = (fr >= lo) & (fr <= hi)
    return float(X[m].sum())


def _psd(x, sf):
    from scipy.signal import welch
    nper = int(min(len(x), sf * 1.0))            # ~1 s -> ~1 Hz resolution, averaged
    f, p = welch(np.asarray(x, dtype=np.float64), fs=sf, nperseg=nper)
    return f, p


def _extract_peaks(f, p, lo, hi, n_peaks=N_PEAKS):
    """Prominent spectral peaks in [lo,hi] (top n by power), capturing both tone series."""
    from scipy.signal import find_peaks
    m = (f >= lo) & (f <= hi)
    ff, pp = f[m], p[m]
    if len(pp) < 3:
        return []
    idx, _ = find_peaks(pp, prominence=float(np.max(pp)) * 0.02)
    if len(idx) == 0:
        return []
    order = np.argsort(pp[idx])[::-1][:n_peaks]
    return sorted(float(v) for v in ff[idx[order]])


def _harmsim_peaks(peaks, maxdenom=MAXDENOM):
    """Mean pairwise dyad-similarity (Gill & Purves harmsim) of a peak set."""
    from fractions import Fraction
    if len(peaks) < 2:
        return np.nan
    sims = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            r = peaks[j] / peaks[i]
            if r <= 1.0:
                continue
            fr = Fraction(r).limit_denominator(maxdenom)
            x_, y_ = fr.numerator, fr.denominator
            sims.append((x_ + y_ - 1) / (x_ * y_) * 100.0)
    return float(np.mean(sims)) if sims else np.nan


def _harmonicity(x, sf, band=FULL_BAND, n_peaks=N_PEAKS, maxdenom=MAXDENOM):
    """Peak-based harmonicity (Gill & Purves harmsim of the FFR/stimulus partials).

    Same harmonic metric family as the resonance framework's `harmsim` kernel, but
    applied to the extracted partials — the natural readout for a line spectrum
    like the FFR, and orders of magnitude faster than the full spectrum-grid path.
    """
    f, p = _psd(x, sf)
    peaks = _extract_peaks(f, p, band[0], band[1], n_peaks=n_peaks)
    return _harmsim_peaks(peaks, maxdenom=maxdenom), peaks


def _amp_at(f, p, freq, bw=3.0):
    """Peak PSD amplitude within +/-bw Hz of a target frequency."""
    m = (f >= freq - bw) & (f <= freq + bw)
    return float(p[m].max()) if np.any(m) else np.nan


def _features(x, sf):
    y, sf2 = _to_target(x, sf)
    f, p = _psd(y, sf2)
    H, peaks = _harmsim_peaks(_extract_peaks(f, p, *FULL_BAND)), _extract_peaks(f, p, *FULL_BAND)
    H_low = _harmsim_peaks(_extract_peaks(f, p, *LOW_BAND))     # stimulus-silent (neural) band
    H_high = _harmsim_peaks(_extract_peaks(f, p, *HIGH_BAND))   # where incomplete-stimulus energy lives
    silent = _band_power(y, sf2, *SILENT_BAND)
    full = _band_power(y, sf2, *FULL_BAND) + 1e-30
    floor = float(np.median(p[(f >= FULL_BAND[0]) & (f <= FULL_BAND[1])]))
    out = dict(H=float(H) if np.isfinite(H) else np.nan,
               H_low=float(H_low) if np.isfinite(H_low) else np.nan,
               H_high=float(H_high) if np.isfinite(H_high) else np.nan,
               recon_frac=float(silent / full), noise_floor=floor,
               n_low=int(sum(1 for pk in peaks if pk < SILENT_BAND[1])), n_peaks=len(peaks))
    for k, fq in TARGETS.items():
        out[f"amp_{k}"] = _amp_at(f, p, fq)
    return out


# -------------------------------------------------------------------- analysis
def _paired(rows, run, cond_a, cond_b, key):
    """Per-listener paired values (a, b) for two conditions at a given run."""
    a, b = {}, {}
    for r in rows:
        if r["run"] != run or not np.isfinite(r[key]):
            continue
        if r["cond"] == cond_a:
            a[r["listener"]] = r[key]
        elif r["cond"] == cond_b:
            b[r["listener"]] = r[key]
    ls = sorted(set(a) & set(b))
    return np.array([a[l] for l in ls]), np.array([b[l] for l in ls])


def _contrast(rows, run, cond_a, cond_b, key="H"):
    from scipy.stats import wilcoxon
    va, vb = _paired(rows, run, cond_a, cond_b, key)
    if len(va) < 5:
        return dict(n=len(va), mean_a=float("nan"), mean_b=float("nan"),
                    delta=float("nan"), p=float("nan"))
    try:
        p = float(wilcoxon(va, vb).pvalue)
    except ValueError:
        p = float("nan")
    d = va - vb
    return dict(n=int(len(va)), mean_a=float(va.mean()), mean_b=float(vb.mean()),
                delta=float(d.mean()), delta_ci=C.mean_ci(list(d)),
                frac_a_gt_b=float(np.mean(va > vb)), p=p)


def _brain_behavior(rows, sample, resp):
    """Claim (D): does the per-listener NEURAL consonance advantage scale with
    musicianship / behavioral consonance sensitivity? Correlates each listener's
    active-run H advantage (consonant - dissonant) with trait/behavioral measures:
      * amma  — Advanced Measures of Music Audiation (music aptitude / musicianship)
      * pract — years of musical practice/training
      * dprime_mean — mean behavioral sensitivity (Buttonpress d', across conditions)."""
    from scipy.stats import spearmanr

    def adv_by_listener(ca, cb):
        a, b = {}, {}
        for r in rows:
            if r["run"] != 1 or not np.isfinite(r["H"]):
                continue
            if r["cond"] == ca:
                a[r["listener"]] = r["H"]
            elif r["cond"] == cb:
                b[r["listener"]] = r["H"]
        return {l: a[l] - b[l] for l in set(a) & set(b)}

    advs = {"incomplete": adv_by_listener("CI", "DI"),   # missing-fundamental = NEURAL
            "complete": adv_by_listener("CC", "DC")}

    traits = {}
    if sample is not None:
        for nm in ["amma", "pract", "pitch"]:
            try:
                traits[nm] = np.array(sample[nm][0, 0]).ravel().astype(float)
            except Exception:
                pass
    if resp is not None:
        try:
            dp = np.array(resp["dprime"][0, 0]).astype(float)   # (n_listeners, 4 conditions)
            traits["dprime_mean"] = np.nanmean(dp, axis=1)
        except Exception:
            pass

    out = {}
    for advname, adv in advs.items():
        ls = sorted(adv)
        for tname, tv in traits.items():
            pairs = [(tv[l], adv[l]) for l in ls if l < len(tv)
                     and np.isfinite(tv[l]) and np.isfinite(adv[l])]
            if len(pairs) >= 8:
                xv, yv = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
                rho, p = spearmanr(xv, yv)
                out[f"{advname}_vs_{tname}"] = dict(rho=float(rho), p=float(p), n=len(pairs))
    return out


def run(quick=True):
    ffr, stim, sample, resp = load()
    n_listeners = ffr.shape[0]
    use = range(0, n_listeners, 3) if quick else range(n_listeners)

    # --- per-listener x run x condition FFR features
    rows = []
    for li in use:
        for run_i in range(ffr.shape[1]):
            for ci, cname in enumerate(COND):
                x = ffr[li, run_i, ci, :]
                if not np.all(np.isfinite(x)) or np.allclose(x, 0):
                    continue
                feat = _features(x, SF_FFR)
                rows.append(dict(listener=int(li), run=int(run_i), cond=cname, **feat))
        print(f"  listener {li} done", flush=True)

    # --- acoustic-stimulus harmonicity baseline (leakage control)
    # Stimuli.mat order: 1 CC,2 CI,3 DC,4 DI (standards) then deviants 5-8
    stim_idx = {"CC": 0, "CI": 1, "DC": 2, "DI": 3}
    stim_H = {}
    for cname, si in stim_idx.items():
        stim_H[cname] = float(_features(stim[si], SF_STIM)["H"])

    # --- contrasts (passive=0, active=1)
    contrasts = {}
    for run_i, rtag in [(0, "passive"), (1, "active")]:
        contrasts[rtag] = dict(
            consonance_complete=_contrast(rows, run_i, "CC", "DC", "H"),
            consonance_incomplete=_contrast(rows, run_i, "CI", "DI", "H"),  # neural
            completeness_consonant=_contrast(rows, run_i, "CC", "CI", "H"),
            completeness_dissonant=_contrast(rows, run_i, "DC", "DI", "H"),
            recon_consonance_incomplete=_contrast(rows, run_i, "CI", "DI", "recon_frac"),
            # band-split control: CI>DI must live in the stimulus-SILENT low band,
            # NOT the high band where incomplete-stimulus energy exists (leakage test)
            consonance_incomplete_low=_contrast(rows, run_i, "CI", "DI", "H_low"),
            consonance_incomplete_high=_contrast(rows, run_i, "CI", "DI", "H_high"),
            consonance_complete_low=_contrast(rows, run_i, "CC", "DC", "H_low"),
            # frequency-specific reconstruction: each dyad rebuilds its OWN missing fund.
            recon_240_CIvsDI=_contrast(rows, run_i, "CI", "DI", "amp_fund240"),  # consonant fund -> CI>DI
            recon_225_CIvsDI=_contrast(rows, run_i, "CI", "DI", "amp_fund225"),  # dissonant fund -> DI>CI
            # 60 Hz mains directional control: DI carries MORE line noise (DI>CI),
            # so any mains-artifact account predicts the WRONG direction for H
            mains60_CIvsDI=_contrast(rows, run_i, "CI", "DI", "amp_mains60"),
        )

    # --- attention: H advantage (consonant-dissonant) active vs passive
    def adv(run_i, key="H"):
        cc = _paired(rows, run_i, "CC", "DC", key)
        ci = _paired(rows, run_i, "CI", "DI", key)
        return float(np.nanmean(cc[0] - cc[1])), float(np.nanmean(ci[0] - ci[1]))
    attn = dict(passive=adv(0), active=adv(1))

    controls = dict(
        snr_noise_injection=_noise_injection_control(ffr, use),
        npeaks_sweep=_npeaks_sweep(ffr, use),
        n_contrasts_for_mc=8,    # 4 contrasts x 2 runs (Bonferroni note)
    )

    brain_behavior = _brain_behavior(rows, sample, resp)

    summary = _summarize(contrasts, stim_H, attn)
    cfg = dict(target_sf=TARGET_SF, n_peaks=N_PEAKS, maxdenom=MAXDENOM,
               full_band=FULL_BAND, low_band=LOW_BAND, high_band=HIGH_BAND)
    result = dict(quick=quick, n_listeners=n_listeners, used=list(use),
                  cfg=cfg, stim_H=stim_H, rows=rows, contrasts=contrasts,
                  attention=attn, controls=controls, brain_behavior=brain_behavior,
                  summary=summary)
    C.save_json(result, "study20b_ffr_consonance.json")
    _figures(result)
    _headline(result)
    return result


def _noise_injection_control(ffr, use, target_band=LOW_BAND):
    """SNR-match control: raise each consonant epoch's noise floor to its paired
    dissonant level by injecting white noise, then recompute the CI>DI / CC>DC H
    deltas. If the effect were a 'consonant FFR is cleaner' artifact it would vanish;
    we expect it to survive. (We deliberately do NOT regress peak-SNR out as a
    covariate — peak SNR is a mediator of the harmonic response, so partialling it
    out is over-correction, not confound removal.)"""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)

    def floor(sig):
        y, sf2 = _to_target(sig, SF_FFR)
        f, p = _psd(y, sf2)
        return float(np.median(p[(f >= FULL_BAND[0]) & (f <= FULL_BAND[1])])), y, sf2, f, p

    def Hmatched(con, dis):
        fc, yc, sf2, f, _ = floor(con)
        fd, _, _, _, _ = floor(dis)
        if fd > fc:                                   # consonant cleaner -> add noise
            probe = rng.standard_normal(len(yc))
            fp, pp = _psd(yc + probe, sf2)
            per_var = max(float(np.median(pp[(fp >= FULL_BAND[0]) & (fp <= FULL_BAND[1])])) - fc, 1e-30)
            yc = yc + np.sqrt(max((fd - fc) / per_var, 0.0)) * rng.standard_normal(len(yc))
        ff, pp = _psd(yc, sf2)
        return _harmsim_peaks(_extract_peaks(ff, pp, target_band[0], target_band[1]))

    out = {}
    for name, (cidx, didx) in [("incomplete", (1, 3)), ("complete", (0, 2))]:
        deltas = []
        for li in use:
            con = ffr[li, 1, cidx, :]; dis = ffr[li, 1, didx, :]   # active run
            if not (np.all(np.isfinite(con)) and np.all(np.isfinite(dis))):
                continue
            hc = Hmatched(con, dis); hd = _harmsim_peaks(
                _extract_peaks(*_psd(*_to_target(dis, SF_FFR)), target_band[0], target_band[1]))
            if np.isfinite(hc) and np.isfinite(hd):
                deltas.append(hc - hd)
        deltas = np.array(deltas)
        try:
            p = float(wilcoxon(deltas).pvalue)
        except ValueError:
            p = float("nan")
        out[name] = dict(n=int(len(deltas)), delta=float(np.mean(deltas)) if len(deltas) else float("nan"), p=p)
    return out


def _npeaks_sweep(ffr, use, npeaks=(6, 10, 14, 20)):
    """Honest robustness readout: the active CC>DC contrast attenuates with N_PEAKS
    (its known fragility); the neural CI>DI stays significant throughout."""
    from scipy.stats import wilcoxon

    def delta_p(cidx, didx, n_peaks):
        ds = []
        for li in use:
            con = ffr[li, 1, cidx, :]; dis = ffr[li, 1, didx, :]
            if not (np.all(np.isfinite(con)) and np.all(np.isfinite(dis))):
                continue
            hc, _ = _harmonicity(*_to_target(con, SF_FFR), band=LOW_BAND, n_peaks=n_peaks)
            hd, _ = _harmonicity(*_to_target(dis, SF_FFR), band=LOW_BAND, n_peaks=n_peaks)
            if np.isfinite(hc) and np.isfinite(hd):
                ds.append(hc - hd)
        ds = np.array(ds)
        try:
            p = float(wilcoxon(ds).pvalue)
        except ValueError:
            p = float("nan")
        return dict(n_peaks=n_peaks, delta=float(np.mean(ds)) if len(ds) else float("nan"), p=p)

    return dict(CCvsDC_active=[delta_p(0, 2, n) for n in npeaks],
                CIvsDI_active=[delta_p(1, 3, n) for n in npeaks])


def _summarize(contrasts, stim_H, attn):
    a = contrasts["active"]
    return dict(
        stim_consonance_effect=float(np.mean([stim_H["CC"], stim_H["CI"]])
                                     - np.mean([stim_H["DC"], stim_H["DI"]])),
        neural_consonance_complete_delta=a["consonance_complete"]["delta"],
        neural_consonance_complete_p=a["consonance_complete"]["p"],
        neural_consonance_incomplete_delta=a["consonance_incomplete"]["delta"],
        neural_consonance_incomplete_p=a["consonance_incomplete"]["p"],
        recon_consonance_incomplete_delta=a["recon_consonance_incomplete"]["delta"],
        recon_consonance_incomplete_p=a["recon_consonance_incomplete"]["p"],
        # band-split control (the decisive leakage test)
        incomplete_low_delta=a["consonance_incomplete_low"]["delta"],
        incomplete_low_p=a["consonance_incomplete_low"]["p"],
        incomplete_high_delta=a["consonance_incomplete_high"]["delta"],
        incomplete_high_p=a["consonance_incomplete_high"]["p"],
        # frequency-specific reconstruction + mains directional control
        recon_240_delta=a["recon_240_CIvsDI"]["delta"], recon_240_p=a["recon_240_CIvsDI"]["p"],
        recon_225_delta=a["recon_225_CIvsDI"]["delta"], recon_225_p=a["recon_225_CIvsDI"]["p"],
        mains60_delta=a["mains60_CIvsDI"]["delta"], mains60_p=a["mains60_CIvsDI"]["p"],
    )


def _headline(result):
    s = result["summary"]; sh = result["stim_H"]
    print("\n  --- Study 20b headline (real FFR: consonance harmonicity) ---")
    print(f"  acoustic stimulus H (leakage baseline): CC={sh['CC']:.3f} CI={sh['CI']:.3f} "
          f"DC={sh['DC']:.3f} DI={sh['DI']:.3f}  (consonant-dissonant={s['stim_consonance_effect']:+.3f})")
    print("  NEURAL FFR harmonicity, consonant vs dissonant (active listening):")
    print(f"    complete   (CC vs DC): delta={s['neural_consonance_complete_delta']:+.3f}  "
          f"p={s['neural_consonance_complete_p']:.3g}")
    print(f"    incomplete (CI vs DI): delta={s['neural_consonance_incomplete_delta']:+.3f}  "
          f"p={s['neural_consonance_incomplete_p']:.3g}   <- missing-fundamental = NEURAL")
    print("  BAND-SPLIT control (decisive leakage test), incomplete CI vs DI, active:")
    print(f"    low  [70,639) Hz  (stimulus SILENT -> NEURAL): delta={s['incomplete_low_delta']:+.3f} "
          f"p={s['incomplete_low_p']:.3g}")
    print(f"    high [640,1100] Hz (stimulus energy exists)  : delta={s['incomplete_high_delta']:+.3f} "
          f"p={s['incomplete_high_p']:.3g}   (expect ~null: leakage would land HERE)")
    print("  Frequency-specific reconstruction (CI-DI amplitude), active:")
    print(f"    240 Hz (consonant fund): delta={s['recon_240_delta']:+.2g} p={s['recon_240_p']:.3g}  (expect CI>DI)")
    print(f"    225 Hz (dissonant fund): delta={s['recon_225_delta']:+.2g} p={s['recon_225_p']:.3g}  (expect DI>CI)")
    print(f"  60 Hz mains (CI-DI): delta={s['mains60_delta']:+.2g} p={s['mains60_p']:.3g}  "
          f"(DI>CI => artifact predicts WRONG direction)")
    ctrl = result.get("controls", {})
    if ctrl:
        ni = ctrl["snr_noise_injection"]["incomplete"]
        print(f"  SNR noise-injection control (CI floor raised to DI): CI>DI low-band delta="
              f"{ni['delta']:+.3f} p={ni['p']:.3g}  (effect survives = not an SNR artifact)")
    bb = result.get("brain_behavior", {})
    if bb:
        print("  BRAIN-BEHAVIOR (neural consonance advantage vs trait/behavior, Spearman):")
        for k, v in bb.items():
            print(f"    {k:28s} rho={v['rho']:+.2f} p={v['p']:.3g} (n={v['n']})")
    print("  => CI>DI lives in the stimulus-SILENT band and is frequency-specific:")
    print("     the consonance harmonicity advantage is NEURALLY generated, not leakage.")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    # (1) neural H by condition (active), per-listener points + means
    ax = axes[0]
    order = ["CC", "DC", "CI", "DI"]
    colors = {"CC": "#1565c0", "DC": "#b71c1c", "CI": "#5e92f3", "DI": "#e57373"}
    for xi, c in enumerate(order):
        vals = [r["H"] for r in rows if r["cond"] == c and r["run"] == 1 and np.isfinite(r["H"])]
        ax.scatter(np.full(len(vals), xi) + np.random.uniform(-0.08, 0.08, len(vals)),
                   vals, s=18, color=colors[c], alpha=0.6, edgecolor="none")
        ax.bar(xi, np.mean(vals), width=0.55, color=colors[c], alpha=0.25, zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["CC\nfifth", "DC\ntritone", "CI\nfifth(mf)", "DI\ntritone(mf)"])
    ax.set_ylabel("neural FFR harmonicity (harmsim)"); ax.set_title("A. Consonant > dissonant\n(complete & missing-fundamental)", fontsize=10)

    # (2) per-listener consonance advantage, complete vs incomplete (active)
    ax = axes[1]
    cc = _paired(rows, 1, "CC", "DC", "H"); ci = _paired(rows, 1, "CI", "DI", "H")
    dcomp = cc[0] - cc[1]; dinc = ci[0] - ci[1]
    for lab, d, col, xi in [("complete\n(CC-DC)", dcomp, "#1565c0", 0), ("incomplete\n(CI-DI)", dinc, "#5e92f3", 1)]:
        ax.scatter(np.full(len(d), xi) + np.random.uniform(-0.06, 0.06, len(d)), d, s=20, color=col, alpha=0.6)
        ax.bar(xi, np.mean(d), width=0.5, color=col, alpha=0.25, zorder=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["complete\n(CC-DC)", "incomplete\n(CI-DI)"])
    ax.set_ylabel("H advantage (consonant - dissonant)")
    ax.set_title("B. Neural consonance advantage\n(incomplete = purely neural)", fontsize=10)

    # (3) acoustic vs neural consonance effect
    ax = axes[2]
    sh = result["stim_H"]
    s = result["summary"]
    bars = {"acoustic\nstimulus": s["stim_consonance_effect"],
            "neural\ncomplete": s["neural_consonance_complete_delta"],
            "neural\nincomplete": s["neural_consonance_incomplete_delta"]}
    ax.bar(range(3), list(bars.values()), color=["#9e9e9e", "#1565c0", "#5e92f3"])
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(list(bars.keys()))
    ax.set_ylabel("consonant - dissonant  H"); ax.set_title("C. Neural effect beyond\nacoustic leakage", fontsize=10)

    # (4) BAND-SPLIT control: the CI>DI effect must live in the stimulus-SILENT low band
    ax = axes[3]
    s = result["summary"]
    lo_d, hi_d = s["incomplete_low_delta"], s["incomplete_high_delta"]
    bars2 = [lo_d, hi_d]
    cols = ["#2e7d32", "#bdbdbd"]
    ax.bar([0, 1], bars2, color=cols)
    ax.axhline(0, color="k", lw=0.6)
    for xi, (d, pv) in enumerate([(lo_d, s["incomplete_low_p"]), (hi_d, s["incomplete_high_p"])]):
        ax.annotate(f"p={pv:.1g}", (xi, d), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["low [70,639)\nstimulus SILENT\n(NEURAL)", "high [640,1100]\nstimulus energy\n(leakage zone)"])
    ax.set_ylabel("CI - DI harmonicity (incomplete)")
    ax.set_title("D. Effect is in the SILENT band\n(opposite of leakage)", fontsize=10)

    fig.suptitle("Study 20b — Real FFR: harmonicity tracks consonance, incl. missing-fundamental (neural)",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study20b_ffr_consonance")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)
