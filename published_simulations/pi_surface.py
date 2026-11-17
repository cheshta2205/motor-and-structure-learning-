"""PI as a surface over lesion strength and post-change window, and a fit to it.

MEASURE (--measure):  PI at every (sigma, tau) for each seed, saved to npz.
  The windows are nested slices of the same rollouts, so ten of them cost what
  four did -- the expense is the sigma sweep, not the window sweep.

FIT (--fit):  per seed, per task, per site, fit

    f(s,t) = 1 / (1 + (s / s50(t))**n)          f = (PI - .5)/(PI(0) - .5)
    s50(t) = s_inf + (s_0 - s_inf)*exp(-(t-1)/tau_c)

  All 6x10 = 60 points enter one fit for 4 parameters; there is no train/test
  split, because the parameters are a summary of the data rather than a
  predictor.  The replication is the ten independently trained networks.

  f is normalised by the intact index at the SAME window, because OLM and ORM
  start from very different places (0.91 vs 0.999) and raw PI is not comparable
  between them.  PI cannot fall below 0.5 -- that is chance, both objects
  equally surprising -- so a plain logistic bottoming out at zero would be
  fitting a region the measure cannot enter.
"""
import argparse
import os

import numpy as np
import torch

import olm_orm_final as O

TAUS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)

# SIGMA MUST REACH HALF-COST IN EVERY CONDITION, or s50 is a bound rather than a
# measurement.  olm_orm_final's SIGMAS stop at 0.8, where OLM's cerebellar cost
# has passed half (s50 ~ 0.35) but the other three have not: ORM/cb loses only
# 6.2% there.  Extending to 5.0 lets all four cross, so the bar chart can show
# four measured values instead of one number and three arrows.  Note 3.0 means
# noise at three times the signal's own scale -- far beyond anything
# physiological.  The RATIO between conditions is the result; the absolute value
# is not a claim about biology.
SIGMAS_EXT = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 5.0)
OUT = os.path.join(O.FIG_DIR, "pooled_ac_seeds", "pi_surface.npz")


def measure(seeds, n=384, r_id=O.R_ID):
    torch.manual_seed(0)                       # same lesion draw as the figures
    rows = {}
    for s in seeds:
        p = O.ckpt_path(s)
        if not os.path.exists(p):
            print(f"  no checkpoint for seed {s}"); continue
        mtl, cb = O.load(p)
        aw = O.dose_response_windows(mtl, cb, s, n=n, r_id=r_id,
                                     ks=TAUS, sigmas=SIGMAS_EXT)
        for (task, site, sig, tau), v in aw.items():
            rows.setdefault((task, site, sig, tau), []).append(v)
        print(f"  seed {s} done", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, taus=np.array(TAUS), sigmas=np.array(SIGMAS_EXT),
             **{"|".join(map(str, k)): np.array(v) for k, v in rows.items()})
    print(f"saved -> {OUT}")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _load():
    z = np.load(OUT)
    taus, sigmas = z["taus"], z["sigmas"]
    d = {}
    for k in z.files:
        if k in ("taus", "sigmas"):
            continue
        task, site, sig, tau = k.split("|")
        d[(task, site, float(sig), int(tau))] = z[k]
    n_seed = len(next(iter(d.values())))
    return taus, sigmas, d, n_seed


def _s50_at_tau(sigmas, f):
    """Two-parameter Hill fit at ONE window.  Returns (s50, n) or (nan, nan).

    Done before the joint fit as a check: plotting s50 against tau shows whether
    it decays exponentially, rather than assuming it does.  Returns nan when the
    curve never reaches half -- which is the ORM case and is a bound, not a
    failure.
    """
    from scipy.optimize import curve_fit
    m = sigmas > 0
    if f[m].min() > 0.5:                 # never reaches half-cost in range
        return np.nan, np.nan
    try:
        p, _ = curve_fit(lambda s, s50, n: 1.0 / (1.0 + (s / s50) ** n),
                         sigmas[m], f[m], p0=[0.3, 2.0],
                         bounds=([1e-3, 0.2], [20.0, 20.0]), maxfev=20000)
        return float(p[0]), float(p[1])
    except Exception:
        return np.nan, np.nan


def _joint(sigmas, taus, F):
    """Fit all 60 points at once.  F is (n_tau, n_sigma) of retained fraction."""
    from scipy.optimize import curve_fit
    S, T = np.meshgrid(sigmas, taus)
    m = S > 0
    def model(X, s_inf, s_0, tau_c, n):
        s, t = X
        s50 = s_inf + (s_0 - s_inf) * np.exp(-(t - 1.0) / tau_c)
        return 1.0 / (1.0 + (s / s50) ** n)
    try:
        p, _ = curve_fit(model, (S[m], T[m]), F[m],
                         p0=[0.3, 2.0, 3.0, 2.0],
                         bounds=([1e-3, 1e-3, 0.1, 0.2], [10, 50, 100, 20]),
                         maxfev=60000)
        pred = model((S[m], T[m]), *p)
        ss = 1.0 - np.sum((F[m] - pred) ** 2) / np.sum((F[m] - F[m].mean()) ** 2)
        return dict(s_inf=p[0], s_0=p[1], tau_c=p[2], n=p[3], r2=ss)
    except Exception as e:
        return dict(s_inf=np.nan, s_0=np.nan, tau_c=np.nan, n=np.nan, r2=np.nan)


def fit():
    taus, sigmas, d, n_seed = _load()
    print(f"{n_seed} seeds, {len(sigmas)} sigmas x {len(taus)} taus "
          f"= {len(sigmas)*len(taus)} points per fit\n")
    out = {}
    for task in ("OLM", "ORM"):
        for site in ("cb", "e", "h"):
            per_seed = []
            for i in range(n_seed):
                F = np.zeros((len(taus), len(sigmas)))
                for a, t in enumerate(taus):
                    base = d[(task, site, 0.0, t)][i] - 0.5
                    for b, s in enumerate(sigmas):
                        F[a, b] = (d[(task, site, s, t)][i] - 0.5) / max(base, 1e-12)
                per_seed.append(_joint(sigmas, taus, F))
            out[(task, site)] = per_seed
    # report
    print(f"{'task':>4} {'site':>5} {'tau_c':>16} {'s_inf':>16} {'n':>14} {'R2':>7}")
    for k, v in out.items():
        g = lambda f: np.array([x[f] for x in v], float)
        def ms(f):
            a = g(f); a = a[np.isfinite(a)]
            return f"{a.mean():7.2f}+-{a.std(ddof=1):5.2f}" if len(a) > 1 else "     n/a"
        print(f"{k[0]:>4} {k[1]:>5} {ms('tau_c')} {ms('s_inf')} {ms('n'):>14} "
              f"{np.nanmean(g('r2')):7.3f}")
    # the per-tau check
    print(f"\ns50 at each window (mean over seeds; nan = never reaches half-cost)")
    print(f"{'task':>4} {'site':>5} " + "".join(f"{t:>7}" for t in taus))
    for task in ("OLM", "ORM"):
        for site in ("cb", "h"):
            row = []
            for t in taus:
                vals = []
                for i in range(n_seed):
                    base = d[(task, site, 0.0, t)][i] - 0.5
                    f = np.array([(d[(task, site, s, t)][i] - 0.5) / max(base, 1e-12)
                                  for s in sigmas])
                    vals.append(_s50_at_tau(sigmas, f)[0])
                vals = np.array(vals, float)
                row.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)
            print(f"{task:>4} {site:>5} " +
                  "".join(f"{v:7.2f}" if np.isfinite(v) else "    >.8" for v in row))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _model(S, T, s_inf, s_0, tau_c, n):
    s50 = s_inf + (s_0 - s_inf) * np.exp(-(T - 1.0) / tau_c)
    return 1.0 / (1.0 + (S / s50) ** n)


def figure(path=None):
    """Fit quality, then the two parameters that answer 'how much' and 'how fast'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus, sigmas, d, n_seed = _load()
    out = fit_quiet(taus, sigmas, d, n_seed)

    ARMS = (("OLM", "cb", "#2f7ab8"), ("ORM", "cb", "#c0392b"),
            ("OLM", "h", "#7f8c8d"), ("ORM", "h", "#b08b4f"))
    SHOW_T = (1, 4, 16)

    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 1, 1, 0.05], wspace=0.34)

    # (a) fit quality -- data points with fitted curves, OLM and ORM, cb site
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(sigmas.min(), sigmas.max(), 200)
    for task, col in (("OLM", "#2f7ab8"), ("ORM", "#c0392b")):
        p = {k: np.nanmean([x[k] for x in out[(task, "cb")]])
             for k in ("s_inf", "s_0", "tau_c", "n")}
        for i, t in enumerate(SHOW_T):
            f = [np.mean([(d[(task, "cb", s, t)][j] - .5) /
                          max(d[(task, "cb", 0.0, t)][j] - .5, 1e-12)
                          for j in range(n_seed)]) for s in sigmas]
            ax.plot(sigmas, f, "o", ms=4, color=col, alpha=[.35, .65, 1.0][i],
                    label=f"{task}, $\\tau$={t}" if True else None)
            ax.plot(xs, _model(xs, t, **p), "-", lw=1.4, color=col,
                    alpha=[.35, .65, 1.0][i])
    ax.axhline(0.5, ls=":", color="#888", lw=1)
    # name the SITE on the axis: panel (a) is the cerebellar lesion only, and
    # nothing else in the panel says so -- the MTL site appears only in the bars.
    ax.set_xlabel(r"cerebellar lesion strength  $\sigma_{\mathrm{CB}}$")
    ax.set_ylabel("preference retained\n(1 = intact, 0 = chance)")
    ax.set_title("fit to the measured surface", fontsize=10.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    _style_ax(ax)

    # (b) sigma_inf and (c) tau_c
    # Title names the parameter, y-axis names what it is a quantity OF.  Read
    # together the two panels say: after tau_c steps, a lesion of sigma_inf
    # costs half the above-chance preference.
    for k, (field, lab, ylab, gsi) in enumerate(
            ("s_inf", r"$\sigma_\infty$",
             "lesion strength that halves\nthe preference", 1),
            ) if False else enumerate(
            (("s_inf", r"$\sigma_\infty$",
              "lesion strength that halves\nthe preference", 1),
             ("tau_c", r"$\tau_c$",
              "steps before that potency\nis reached", 2))):
        ax = fig.add_subplot(gs[0, gsi])
        for i, (task, site, col) in enumerate(ARMS):
            v = np.array([x[field] for x in out[(task, site)]], float)
            v = v[np.isfinite(v)]
            ax.bar(i, v.mean(), 0.66, color=col, edgecolor="none", zorder=2)
            ax.plot(np.full(len(v), i), v, ls="none", marker="o", ms=2.8,
                    mfc="white", mec="#1a1a1a", mew=0.6, zorder=3, alpha=.9)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([f"{t}\n{s}" for t, s, _ in ARMS], fontsize=8.5)
        ax.set_title(lab, fontsize=11.5)
        ax.set_ylabel(ylab, fontsize=9)
        _style_ax(ax)

    fig.suptitle("Lesion potency and its time course, fitted jointly over "
                 f"{len(sigmas)} lesion strengths and {len(taus)} windows "
                 f"({n_seed} seeds; dots are individual networks)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = path or os.path.join(os.path.dirname(OUT), "pi_surface_fit.png")
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"saved -> {path}")


def _style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fit_quiet(taus, sigmas, d, n_seed):
    out = {}
    for task in ("OLM", "ORM"):
        for site in ("cb", "e", "h"):
            per = []
            for i in range(n_seed):
                F = np.zeros((len(taus), len(sigmas)))
                for a, t in enumerate(taus):
                    base = d[(task, site, 0.0, t)][i] - 0.5
                    for b, s in enumerate(sigmas):
                        F[a, b] = (d[(task, site, s, t)][i] - 0.5) / max(base, 1e-12)
                per.append(_joint(sigmas, taus, F))
            out[(task, site)] = per
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n", type=int, default=384)
    a = ap.parse_args()
    if a.measure:
        measure(a.seeds, n=a.n)
    if a.fit:
        fit()
    if a.figure:
        figure()
