"""gap_ALL_X split into two rows: intact on top, cerebellum perturbed below.

Same numbers as olm_orm_final.fig_gap -- read straight out of pooled.npz -- but
one row per condition instead of both overlaid, so the shrinking of the gap is
read vertically rather than through two superimposed pairs of curves.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import olm_orm_final as O

POOL = os.path.join(O.FIG_DIR, "pooled_ac_seeds")
ROWS = [c for c in O.COND if c[0] in ("intact", "cerebellar output")]
RLAB = {"intact": "intact", "cerebellar output": "cerebellum perturbed"}


def figure(path=None):
    z = np.load(os.path.join(POOL, "pooled.npz"), allow_pickle=True)
    cur = {eval(k.split("|", 1)[1]): z[k] for k in z.files if k.startswith("cur|")}

    fig, AX = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
    for i, (lab, _, col) in enumerate(ROWS):
        for j, task in enumerate(("OLM", "ORM")):
            ax = AX[i, j]
            ch = np.nanmean(cur[(task, lab, True)], 0)
            un = np.nanmean(cur[(task, lab, False)], 0)
            x = np.arange(1, len(ch) + 1)
            ax.plot(x, ch, color=col, lw=2.0, label="changed")
            ax.plot(x, un, color=col, lw=2.0, ls=":", label="unchanged")
            ax.fill_between(x, un, ch, color=col, alpha=0.10, lw=0)
            ax.set_yscale("log")
            O._style(ax)
            if i == 0:
                ax.set_title(f"{task}   (object "
                             f"{'location' if task == 'OLM' else 'identity'}"
                             " changed)", fontsize=11)
            if i == 1:
                ax.set_xlabel("steps since the change")
            if j == 0:
                ax.set_ylabel(f"{RLAB[lab]}\nsquared error / channel variance")
            if j == 0:
                ax.legend(fontsize=8, frameon=False)

    sub = (f"10 seeds, reach {O.R_ID}, n=384 arenas per seed; "
           f"error bars are across independently trained networks")
    fig.suptitle("The gap between changed and unchanged is the preference "
                 "signal\n" + sub, fontsize=10.5, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = path or os.path.join(POOL, "gap_rows_ALL_X.png")
    O._save(fig, path)
    plt.close(fig)
    print(f"saved -> {path}  (+ .svg)")


if __name__ == "__main__":
    figure()
