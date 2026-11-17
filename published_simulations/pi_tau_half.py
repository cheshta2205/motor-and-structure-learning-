"""The window-length analogue of the sigma_inf panel.

sigma_inf answers "how much noise halves the preference".  This answers the
same question along the other axis: "how long a window must PI be averaged over
before the lesion has halved it".

    tau_half(sigma) = the window tau at which
                      f = (PI(sigma,tau) - .5) / (PI(0,tau) - .5)  falls to .5

MEASURED, NOT FITTED.  This replaces tau_c from pi_surface.fit, which is a
parameter of an exponential whose starting point s_0 lands at 11.6 (OLM/cb) and
16.0 (ORM/cb) -- 2.3x and 3.2x beyond the largest sigma ever swept (5.0).  At
the short windows tau_c is about, neither cerebellar condition has a measurable
half-point at all, so tau_c describes a decay from a value that was never
observed.  tau_half uses only points that exist.

tau_half IS ONLY DEFINED ABOVE sigma_inf.  Below that dose the lesion never
removes half the preference at any window, so the curve diverges there rather
than going missing -- the divergence is the same fact sigma_inf reports, seen
from the other axis.  Plotted as a gap running to the top of the panel.

RESOLUTION FLOOR.  The window grid starts at 1, so a condition that has already
halved by the shortest window reads ~1 and cannot read lower.  OLM/cb sits at
1.4 over most of its range for that reason.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pi_surface as P

ARMS = (("OLM", "cb", "#2f7ab8", "OLM, cerebellar"),
        ("ORM", "cb", "#c0392b", "ORM, cerebellar"),
        ("OLM", "h",  "#7f8c8d", "OLM, MTL"),
        ("ORM", "h",  "#b08b4f", "ORM, MTL"))
BAR_SIGMA = 3.0          # the lowest dose at which all four are defined
SHOW_S = (0.5, 2.0, 5.0)  # slices for panel A
OUT = os.path.join(P.O.FIG_DIR, "pooled_ac_seeds")


def _f(d, taus, task, site, s, i):
    """Preference retained at each window, for one seed."""
    return np.array([(d[(task, site, s, t)][i] - .5)
                     / max(d[(task, site, 0.0, t)][i] - .5, 1e-12) for t in taus])


def _tau_half(fv, taus):
    """First crossing of .5, interpolated in log tau.  nan = never crosses."""
    lt = np.log(np.asarray(taus, float))
    if fv[0] < .5:
        return float(taus[0])                      # already halved at tau=1
    for a in range(len(fv) - 1):
        if fv[a] >= .5 > fv[a + 1]:
            w = (fv[a] - .5) / (fv[a] - fv[a + 1])
            return float(np.exp(lt[a] + w * (lt[a + 1] - lt[a])))
    return np.nan


def table(d, taus, sigmas, n):
    """{(task, site, sigma): per-seed tau_half}"""
    return {(t, st, s): np.array([_tau_half(_f(d, taus, t, st, s, i), taus)
                                  for i in range(n)], float)
            for t, st, _, _ in ARMS for s in sigmas}


def figure(path=None):
    taus, sigmas, d, n = P._load()
    TH = table(d, taus, sigmas, n)
    tmax = float(max(taus))

    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1.25], wspace=.34)

    # (a) the raw curves: preference retained against window length
    ax = fig.add_subplot(gs[0, 0])
    for task, col in (("OLM", "#2f7ab8"), ("ORM", "#c0392b")):
        for i, s in enumerate(SHOW_S):
            y = np.mean([_f(d, taus, task, "cb", s, j) for j in range(n)], 0)
            ax.plot(taus, y, "-o", ms=4, color=col, lw=1.8,
                    alpha=[.35, .65, 1.0][i],
                    label=f"{task}, $\\sigma$={s:g}")
    ax.axhline(.5, ls=":", color="#888", lw=1)
    ax.set_xscale("log"); ax.set_xticks(taus)
    ax.set_xticklabels([str(t) for t in taus], fontsize=8)
    ax.set_xlabel("window  $\\tau$  (post-change steps PI is averaged over)")
    ax.set_ylabel("preference retained\n(1 = intact, 0 = chance)")
    ax.set_title("cerebellar noise: preference against window length", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # (b) tau_half at one matched dose
    ax2 = fig.add_subplot(gs[0, 1])
    for j, (task, site, col, _) in enumerate(ARMS):
        v = TH[(task, site, BAR_SIGMA)]
        ok = v[np.isfinite(v)]
        ax2.bar([j], [ok.mean()], color=col, width=.65,
                yerr=[ok.std(ddof=1) / np.sqrt(len(ok))], capsize=4)
        ax2.plot(np.full(len(ok), j) + np.random.default_rng(0)
                 .normal(0, .055, len(ok)), ok, "o", ms=3.5, mfc="w",
                 mec="#333", mew=.7, zorder=3)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"{t}\n{s}" for t, s, _, _ in ARMS], fontsize=9)
    ax2.set_ylabel("steps of window needed\nto halve the preference")
    ax2.set_title(f"$\\tau_{{1/2}}$ at $\\sigma$ = {BAR_SIGMA:g}", fontsize=10)

    # (c) the whole curve -- it diverges at each condition's sigma_inf
    ax3 = fig.add_subplot(gs[0, 2])
    for task, site, col, lab in ARMS:
        y = [np.nanmean(TH[(task, site, s)]) if np.isfinite(TH[(task, site, s)]).any()
             else np.nan for s in sigmas]
        y = np.array(y, float)
        m = np.isfinite(y)
        ax3.plot(np.array(sigmas)[m], y[m], "-o", ms=4, color=col, lw=1.8, label=lab)
        # the dose below which it never halves, at any window
        first = np.array(sigmas)[m][0] if m.any() else np.nan
        if np.isfinite(first):
            ax3.plot([first, first], [y[m][0], tmax * 1.6], color=col, lw=1.2,
                     ls=(0, (2, 2)), alpha=.7)
    ax3.axhline(tmax, ls=":", color="#888", lw=1)
    ax3.text(sigmas[-1], tmax * 1.06, "longest window measured", ha="right",
             fontsize=7.5, color="#888")
    ax3.set_yscale("log"); ax3.set_ylim(.8, tmax * 1.9)
    ax3.set_xlabel("lesion strength  $\\sigma$")
    ax3.set_ylabel("$\\tau_{1/2}$  (steps)")
    ax3.set_title("dashed = below this dose the preference\nnever halves, at any window",
                  fontsize=9)
    ax3.legend(fontsize=8, frameon=False)

    for a in (ax, ax2, ax3):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("How long a window must PI be averaged over before the lesion has "
                 f"halved it  ({n} seeds; measured, not fitted)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .92))
    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, "pi_tau_half.png")
    fig.savefig(path, dpi=170); fig.savefig(path.replace(".png", ".svg"))
    plt.close(fig)

    print(f"{'task':>4} {'site':>4}  tau_half at sigma=" + "".join(f"{s:>7}" for s in sigmas))
    for task, site, _, _ in ARMS:
        row = [np.nanmean(TH[(task, site, s)]) if np.isfinite(TH[(task, site, s)]).any()
               else np.nan for s in sigmas]
        print(f"{task:>4} {site:>4} " + 19 * " " +
              "".join(f"{v:7.1f}" if np.isfinite(v) else "    nan" for v in row))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    figure()
