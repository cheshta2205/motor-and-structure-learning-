"""
sac_figures.py -- per-seed diagnostic plots for one sac.py training run.

Mirrors figures.py's per-seed/across-seed split: this is the per-run view
(one seed, one config). Cross-config ablation analysis lives separately in
make_ablations_plot.py, which reads the aggregated_results.csv sac.main
writes per config rather than the per-seed curve.npz this file plots.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from figures import _despine
from plotting_utils import use_paper_style, save

use_paper_style()


def smooth(x, w=25):
    x = np.asarray(x)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def _block_key(p):
    """Row label for a block: its context and which visit it was."""
    if p.get("context_id") is None:
        return "warmup (null)"
    return f"{p['label']} (x{int(p['occurrence']) + 1})"


def plot_cross_context(seed, probes, out_dir, config_name=None):
    """Transfer matrix: every context evaluated at the end of every block.

    Rows are blocks in temporal order, columns are the context evaluated. Read
    it two ways:

      DOWN a column -- how performance in one context evolves as training moves
        through the others. A cell well above the warmup row for a context that
        is NOT currently being trained is a memory trace: something learned
        earlier survived.
      ACROSS a row -- what the policy at that moment can do in each context.
        The gap between the diagonal cell (the context just trained, boxed) and
        the rest is the interference cost of specializing.

    Two metrics, because they fail differently: success rate says whether the
    agent arrives, path ratio (path length / straight line, successful reaches
    only) says how directly -- so it still separates configs once success has
    saturated. Failures are excluded from path ratio because a truncated
    trajectory is short and would masquerade as efficient.

    The row labels carry each block's DEFAULT AIM as signed mean / |mean| of
    first_angle. It is deliberately not a matrix cell: the first action is
    taken before the agent has felt the field (zero velocity, zero prediction
    error, freshly reset EC state), so it is identical in every context and
    describes the policy's standing bias rather than per-context compensation.

    BOTH statistics are shown because the sign is the informative half and the
    two together separate cases a single number conflates:

      signed ~= +-|mean|  coherent, direction-independent compensation. Under
        rot +45 the agent aims about -45 for every target, under rot -45 about
        +45 -- opposite signs that an absolute mean would collapse onto the
        same value.
      signed ~= 0, |mean| large  the offset flips sign across targets. That is
        reflection's signature, since task.py's reflection maps angle alpha to
        2*phi - alpha, so the required offset 2*(phi - alpha) DEPENDS on target
        direction. Not the same thing as no compensation at all, which is the
        untrained warmup row -- also signed ~= 0 with a large |mean|, but
        scattered rather than systematic. The trajectory plot tells those apart.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not probes or "eval_label" not in probes[0]:
        return      # probe file predates cross-context evaluation

    row_keys = list(dict.fromkeys(_block_key(p) for p in probes))
    cols = list(dict.fromkeys(p["eval_label"] for p in probes))

    success = np.full((len(row_keys), len(cols)), np.nan)
    path = np.full((len(row_keys), len(cols)), np.nan)
    current = np.zeros((len(row_keys), len(cols)), dtype=bool)
    default_aim = {}

    for p in probes:
        r, c = row_keys.index(_block_key(p)), cols.index(p["eval_label"])
        success[r, c] = float(np.mean(p["success"]))
        ratios = p.get("path_ratio")
        if ratios is not None and np.any(~np.isnan(ratios)):
            path[r, c] = float(np.nanmean(ratios))
        current[r, c] = bool(p.get("is_current", False))
        fa = np.asarray(p["first_angle"], dtype=float)
        default_aim.setdefault(_block_key(p), (float(np.mean(fa)),
                                               float(np.mean(np.abs(fa)))))

    rows = [f"{k}   [aim {default_aim[k][0]:+.0f}° / |{default_aim[k][1]:.0f}°|]"
            for k in row_keys]
    pmax = float(np.nanmax(path)) if np.any(~np.isnan(path)) else 2.0
    panels = [
        ("Success rate", success, 0.0, 1.0, "plasma", "{:.2f}"),
        ("Path length / straight line\n(successful reaches)", path, 1.0, max(pmax, 1.05),
         "plasma_r", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(1.6 * len(cols) + 5.5, 0.55 * len(rows) + 2.4),
                              constrained_layout=True)

    for ax, (title, mat, vmin, vmax, cmap, fmt) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("evaluated in", fontsize=8)
        for r in range(len(rows)):
            for c in range(len(cols)):
                if np.isnan(mat[r, c]):
                    continue
                # label in whichever of black/white reads against the cell
                rel = (mat[r, c] - vmin) / (vmax - vmin + 1e-12)
                ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center",
                        fontsize=7, color="white" if rel > 0.55 else "black")
                if current[r, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                ec="red", lw=1.8, zorder=5))
        fig.colorbar(im, ax=ax, fraction=0.035)

    axes[0].set_ylabel("after training block", fontsize=8)
    title = f"Cross-context transfer (seed {seed}) — red = context just trained"
    if config_name:
        title = f"{config_name} — {title}"
    fig.suptitle(title, fontsize=10)
    save(plt.gcf(), os.path.join(out_dir, "cross_context"), close=False, formats=("png",))
    plt.close(fig)


def plot_center_out(seed, probes, out_dir, config_name=None):
    """Center-out reach trajectories, one panel per phase.

    The diagnostic the learning curves cannot give: an agent that compensates
    for the force field feedforward reaches STRAIGHT, while one that only
    corrects online reaches on a curve that straightens late. Both can end up
    with the same success rate. Panels share a start and target set across
    every phase, seed and config, so shapes are directly comparable; the
    columns are contexts and the rows are repeat visits (occurrence), so
    savings show up as later rows being straighter than earlier ones.
    """
    os.makedirs(out_dir, exist_ok=True)
    # probes now cover every context per block; this plot is about the context
    # actually being trained, so keep only those (see plot_cross_context for
    # the rest). Older probe files predate the flag, hence the default.
    probes = [p for p in probes if p.get("is_current", True)]
    if not probes:
        return

    labels = list(dict.fromkeys(p["label"] for p in probes))
    occurrences = sorted({int(p["occurrence"]) for p in probes})
    n_col, n_row = len(labels), len(occurrences)

    fig, axes = plt.subplots(n_row, n_col, figsize=(3.1 * n_col, 3.1 * n_row),
                              squeeze=False, constrained_layout=True)

    for probe in probes:
        r = occurrences.index(int(probe["occurrence"]))
        c = labels.index(probe["label"])
        ax = axes[r][c]

        goals, traj = probe["goals"], probe["trajectories"]
        # per-reach starts when present; older probes only carry a shared
        # `center`, so broadcast it to keep them plottable
        starts = probe.get("starts")
        if starts is None:
            starts = np.repeat(np.asarray(probe["center"])[None, :], len(goals), 0)
        colors = plt.get_cmap("twilight")(np.linspace(0, 1, len(goals), endpoint=False))

        for k in range(len(goals)):
            xy = traj[k]
            xy = xy[~np.isnan(xy[:, 0])]
            ax.plot(xy[:, 0], xy[:, 1], lw=1.2, color=colors[k], zorder=2)
            # straight line start -> target, for reference against curvature
            ax.plot([starts[k, 0], goals[k, 0]], [starts[k, 1], goals[k, 1]],
                    lw=0.5, ls=":", color="0.7", zorder=1)
            ax.add_patch(plt.Circle(goals[k], probe["goal_radius"], fill=False,
                                     lw=0.6, ec=colors[k], zorder=3))

        ax.plot(starts[:, 0], starts[:, 1], marker="+", ms=7, ls="none",
                color="black", zorder=4)
        # LIMITS FROM THE DATA, not from `radius`. With starts drawn across the
        # grid the reaches no longer share an origin, so a fixed pad around one
        # centre clips most of them; `radius` is only meaningful in the old
        # centre-out mode and is kept for those probe files.
        xy = np.concatenate([starts, goals])
        pad = probe["goal_radius"] + 0.15
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{probe['label']}  (x{int(probe['occurrence']) + 1})\n"
                     f"{int(probe['success'].sum())}/{len(goals)} reached, "
                     f"first angle {np.mean(probe['first_angle']):+.0f}°",
                     fontsize=8)

    title = f"Center-out reaches (seed {seed})"
    if config_name:
        title = f"{config_name} — {title}"
    fig.suptitle(title, fontsize=11)
    save(plt.gcf(), os.path.join(out_dir, "center_out"), close=False, formats=("png",))
    plt.close(fig)


def plot_seed(seed, curve, out_dir, key_params=None):
    os.makedirs(out_dir, exist_ok=True)
    n = len(curve["success"])
    ep = np.arange(1, n + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for ax, key, color, label in [
        (axes[0], "q_loss", "tomato", "Q loss"),
        (axes[1], "actor_loss", "steelblue", "Actor loss"),
        (axes[2], "alpha", "darkorange", "Alpha"),
    ]:
        y = smooth(curve[key])
        ax.plot(ep[-len(y):], y, color=color, lw=1.5)
        ax.axhline(0, color="black", ls=":", lw=0.7)
        ax.set_ylabel(label, fontsize=8)
        _despine(ax)
    axes[2].set_xlabel("Episode (global)")
    fig.suptitle(f"SAC diagnostic curves (seed {seed})", fontsize=11)
    save(plt.gcf(), os.path.join(out_dir, "diagnostic_losses"), close=False, formats=("png",))
    plt.close(fig)

    for metric, color, ylim in [("success", "steelblue", (0, 1)), ("return", "seagreen", None)]:
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        y = smooth(curve[metric])
        ax.plot(ep[-len(y):], y, lw=1.7, color=color)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(ep[0], ep[-1] + 1)
        ax.set_ylabel(metric.title())
        ax.set_xlabel("Episode (global)")
        fig.suptitle(f"Episode {metric} curve (seed {seed})", fontsize=11)
        if key_params is not None:
            param_str = ", ".join(f"{k}={v}" for k, v in key_params.items())
            ax.set_title(param_str, fontsize=7)
        _despine(ax)
        save(plt.gcf(), os.path.join(out_dir, f"{metric}_curve"), close=False, formats=("png",))
        plt.close(fig)


def _circular_mean_deg(a, axis=0):
    """Mean of angles in degrees, done on the circle.

    An arithmetic mean is wrong here: aims near +180 and -180 are adjacent,
    and averaging them linearly gives 0 -- the opposite direction.
    """
    r = np.deg2rad(np.asarray(a, dtype=float))
    with np.errstate(invalid="ignore"):
        s = np.nanmean(np.sin(r), axis=axis)
        c = np.nanmean(np.cos(r), axis=axis)
    return np.degrees(np.arctan2(s, c))


def plot_aim_by_step(seed, probes, out_dir, config_name=None, max_blocks=16):
    """Aim angle against step, one line per EVALUATED context, one panel per
    training block.

    WHY THIS EXISTS. The probe summary reports first_angle, and for any config
    whose context arrives through the circuit that number is identical in every
    context BY CONSTRUCTION: at t=0 the probe has vel=0, e_t=0, cb_h2=0 and a
    freshly reset EC state, so z is a fixed vector and the whole observation is
    context-independent. Reading "@-19deg in all four columns" as evidence that
    the policy ignores context is therefore reading an artifact of the probe.

    The oracle has no such delay -- its code is present at t=0 -- so comparing
    the two on first_angle compares a config that cannot yet know the context
    against one that always does.

    Context reaches EC a few steps in (6-way decodability 0.74
    over t in [0,5), 0.99 by [5,15)), so this is where a circuit config's
    context dependence can appear at all. Lines that stay superimposed past
    ~5 steps are genuine context-blindness; lines that separate are not.
    """
    os.makedirs(out_dir, exist_ok=True)
    probes = [p for p in probes if "angle_by_step" in p]
    if not probes:
        return          # probe file predates per-step angles

    row_keys = list(dict.fromkeys(_block_key(p) for p in probes))[:max_blocks]
    cols = list(dict.fromkeys(p["eval_label"] for p in probes))
    colors = plt.get_cmap("tab10")(np.arange(len(cols)) % 10)

    n_col = min(4, len(row_keys))
    n_row = int(np.ceil(len(row_keys) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.4 * n_col, 2.6 * n_row),
                              squeeze=False, sharex=True, sharey=True,
                              constrained_layout=True)
    for ax in axes.ravel():
        ax.set_visible(False)

    for i, key in enumerate(row_keys):
        ax = axes[i // n_col][i % n_col]
        ax.set_visible(True)
        for p in (q for q in probes if _block_key(q) == key):
            c = cols.index(p["eval_label"])
            mean = _circular_mean_deg(p["angle_by_step"], axis=0)
            steps = np.arange(len(mean))
            ok = ~np.isnan(mean)
            ax.plot(steps[ok], mean[ok], lw=2.0 if p.get("is_current") else 1.1,
                    color=colors[c], alpha=1.0 if p.get("is_current") else 0.75,
                    label=p["eval_label"] if i == 0 else None,
                    zorder=3 if p.get("is_current") else 2)
        ax.axhline(0, color="black", ls=":", lw=0.7)
        ax.axvline(5, color="0.6", ls="--", lw=0.7)
        ax.set_title(key, fontsize=7)
        _despine(ax)

    axes[-1][0].set_xlabel("step")
    axes[0][0].set_ylabel("aim angle (deg)")
    if len(row_keys):
        axes[0][0].legend(fontsize=6, loc="best", frameon=False)
    title = (f"Aim by step (seed {seed}) — thick = context trained on; "
             f"dashed line = when z resolves")
    if config_name:
        title = f"{config_name} — {title}"
    fig.suptitle(title + "\nStep 0 is context-blind by construction for circuit "
                        "configs; only later steps can show context dependence",
                 fontsize=9)
    save(plt.gcf(), os.path.join(out_dir, "aim_by_step"), close=False, formats=("png",))
    plt.close(fig)


def plot_transfer_trajectories(seed, probes, out_dir, config_name=None,
                               occurrence=None):
    """Center-out reaches for every EVALUATED context, one row per block.

    plot_center_out shows only the context being trained (is_current); this is
    the same picture for the off-diagonal, so "the behavior looks the same in
    every context" can be checked by eye rather than inferred from a summary
    statistic. Defaults to the LAST occurrence of each context, which is the
    most-trained state; pass occurrence="all" for every block.
    """
    os.makedirs(out_dir, exist_ok=True)
    probes = [p for p in probes if p.get("context_id") is not None]
    if not probes:
        return
    if occurrence != "all":
        last = max(int(p["occurrence"]) for p in probes)
        occurrence = last if occurrence is None else occurrence
        probes = [p for p in probes if int(p["occurrence"]) == occurrence]

    row_keys = list(dict.fromkeys(_block_key(p) for p in probes))
    cols = list(dict.fromkeys(p["eval_label"] for p in probes))
    fig, axes = plt.subplots(len(row_keys), len(cols),
                              figsize=(2.3 * len(cols), 2.3 * len(row_keys)),
                              squeeze=False, constrained_layout=True)

    for p in probes:
        r, c = row_keys.index(_block_key(p)), cols.index(p["eval_label"])
        ax = axes[r][c]
        goals, traj = p["goals"], p["trajectories"]
        starts = p.get("starts")
        if starts is None:
            starts = np.repeat(np.asarray(p["center"])[None, :], len(goals), 0)
        tc = plt.get_cmap("twilight")(np.linspace(0, 1, len(goals), endpoint=False))
        for k in range(len(goals)):
            xy = traj[k][~np.isnan(traj[k][:, 0])]
            ax.plot(xy[:, 0], xy[:, 1], lw=1.0, color=tc[k], zorder=2)
            ax.plot([starts[k, 0], goals[k, 0]], [starts[k, 1], goals[k, 1]],
                    lw=0.4, ls=":", color="0.75", zorder=1)
        ax.plot(starts[:, 0], starts[:, 1], marker="+", ms=5, ls="none",
                color="black", zorder=4)
        # from the data, for the same reason as plot_center_out: the reaches no
        # longer share an origin, so `radius` about one point clips most of them
        xy = np.concatenate([starts, goals])
        pad = p["goal_radius"] + 0.15
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if p.get("is_current"):
            for sp in ax.spines.values():
                sp.set_color("red"); sp.set_linewidth(2.0)
        ax.set_title(f"{int(p['success'].sum())}/{len(goals)}", fontsize=7)
        if r == 0:
            ax.text(0.5, 1.28, p["eval_label"], transform=ax.transAxes,
                    ha="center", fontsize=7)
        if c == 0:
            ax.set_ylabel(_block_key(p), fontsize=6)

    title = f"Transfer trajectories (seed {seed}) — red = context trained on"
    if config_name:
        title = f"{config_name} — {title}"
    fig.suptitle(title, fontsize=10)
    save(plt.gcf(), os.path.join(out_dir, "transfer_trajectories"), close=False, formats=("png",))
    plt.close(fig)
