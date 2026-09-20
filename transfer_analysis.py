"""
transfer_analysis.py -- seed-averaged cross-context transfer, across configs.

The third analysis layer. sac_figures.py plots ONE seed of one config;
make_ablations_plot.py compares configs from the aggregated learning-curve
CSVs; this file compares configs from the PROBES (sac_runs/*/seed_*/probes.npz),
which is where the cross-context evaluation lives.

Two scalar summaries of the transfer matrix, one per axis:

  INTERFERENCE, across a row -- the participation ratio of the row. Normalize
    the row to p_i = s_i / sum(s), then PR = 1 / sum(p_i^2). PR near 1 means
    the policy is competent in exactly one context (fully specialized, maximal
    interference); PR near N means it is equally competent in all of them.
    PR describes the SHAPE of the profile, not its level: a policy scoring 0.2
    everywhere has PR = N while being bad everywhere, so PR is always reported
    beside the row mean.

  MEMORY, down a column -- performance in a context as a function of LAG, the
    number of blocks since that context was last trained. Lag 0 is the block
    that just trained it, lag k is k blocks later. A curve that decays slowly
    with lag is a memory trace; one that collapses to the naive level (the
    warmup row, drawn as a floor) is not. Indexing by lag rather than
    smoothing down the raw column makes configs comparable and gives both
    anchors -- just-trained and never-trained -- a fixed position.
"""
import glob
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sac
from sac_figures import _block_key
from plotting_utils import set_plt_params, make_axis_nice

def load_probes(config_name, seeds=None, run_dir=None):
    """Load every seed's probe list for one config -> {seed: probes}."""
    root = run_dir or sac.RUN_DIR
    found = {}
    for path in sorted(glob.glob(os.path.join(root, config_name, "seed_*", "probes.npz"))):
        seed = int(os.path.basename(os.path.dirname(path)).split("_")[1])
        if seeds is not None and seed not in set(seeds):
            continue
        probes = np.load(path, allow_pickle=True)["probes"].tolist()
        if probes and "eval_label" in probes[0]:
            found[seed] = probes
    return found


def block_matrix(probes, metric="success"):
    """One seed's transfer matrix -> (row_keys, cols, values, is_current).

    Rows are blocks in temporal order, columns are the context evaluated.
    """
    row_keys = list(dict.fromkeys(_block_key(p) for p in probes))
    cols = list(dict.fromkeys(p["eval_label"] for p in probes))
    values = np.full((len(row_keys), len(cols)), np.nan)
    current = np.zeros((len(row_keys), len(cols)), dtype=bool)

    for p in probes:
        r, c = row_keys.index(_block_key(p)), cols.index(p["eval_label"])
        v = p.get(metric)
        if v is not None:
            with np.errstate(invalid="ignore"):
                values[r, c] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else np.nan
        current[r, c] = bool(p.get("is_current", False))
    return row_keys, cols, values, current


def stack_seeds(config_name, seeds=None, metric="success", run_dir=None):
    """Stack every seed's matrix -> (row_keys, cols, arr[seed, row, col], current).

    Raises if seeds disagree on the block or context ordering, which would
    otherwise average unrelated cells together.
    """
    per_seed = load_probes(config_name, seeds=seeds, run_dir=run_dir)
    if not per_seed:
        return None

    mats, ref = [], None
    for seed in sorted(per_seed):
        row_keys, cols, values, current = block_matrix(per_seed[seed], metric=metric)
        if ref is None:
            ref = (row_keys, cols, current)
        elif (row_keys, cols) != (ref[0], ref[1]):
            raise ValueError(f"{config_name} seed {seed} has a different block/context "
                              f"layout; cannot average across seeds")
        mats.append(values)

    return ref[0], ref[1], np.stack(mats), ref[2]


def participation_ratio(row):
    """PR = 1 / sum(p^2) for the row normalized to sum 1.

    NaN when the row is all zeros or entirely missing -- an untrained policy
    that fails everywhere has no profile to describe, and returning N there
    would read as "perfectly generalist".
    """
    r = np.asarray(row, dtype=float)
    r = r[~np.isnan(r)]
    r = np.clip(r, 0.0, None)
    total = r.sum()
    if r.size == 0 or total <= 0:
        return np.nan
    p = r / total
    return float(1.0 / np.sum(p ** 2))


def interference_table(config_names, seeds=None, metric="success", run_dir=None):
    """Per (config, seed, block): participation ratio and mean level.

    Skips the warmup row, whose profile is undefined -- it is the naive floor,
    not a trained policy.
    """
    rows = []
    for name in config_names:
        stacked = stack_seeds(name, seeds=seeds, metric=metric, run_dir=run_dir)
        if stacked is None:
            continue
        row_keys, cols, arr, _ = stacked
        for si, seed in enumerate(sorted(load_probes(name, seeds=seeds, run_dir=run_dir))):
            for ri, key in enumerate(row_keys):
                if key.startswith("warmup"):
                    continue
                vals = arr[si, ri]
                rows.append({
                    "config": name, "seed": seed, "block": ri, "block_label": key,
                    "participation_ratio": participation_ratio(vals),
                    "mean_level": float(np.nanmean(vals)),
                    "n_contexts": int(np.sum(~np.isnan(vals))),
                })
    return pd.DataFrame(rows)


def retention_table(config_names, seeds=None, metric="success", run_dir=None):
    """Per (config, seed, context, lag): performance since that context was
    last trained.

    lag 0 is the block that trained it; lag k is k blocks later. Rows before a
    context has ever been trained are recorded with lag = -1, the naive
    reference the decay is measured against.
    """
    rows = []
    for name in config_names:
        stacked = stack_seeds(name, seeds=seeds, metric=metric, run_dir=run_dir)
        if stacked is None:
            continue
        row_keys, cols, arr, current = stacked
        seed_list = sorted(load_probes(name, seeds=seeds, run_dir=run_dir))

        for ci, col in enumerate(cols):
            trained_at = [ri for ri in range(len(row_keys)) if current[ri, ci]]
            for ri in range(len(row_keys)):
                earlier = [t for t in trained_at if t <= ri]
                lag = (ri - max(earlier)) if earlier else -1
                for si, seed in enumerate(seed_list):
                    v = arr[si, ri, ci]
                    if np.isnan(v):
                        continue
                    rows.append({"config": name, "seed": seed, "context": col,
                                  "block": ri, "lag": lag, "value": float(v)})
    return pd.DataFrame(rows)


def _mean_sem(frame, x, y):
    g = frame.groupby(x)[y]
    out = g.agg(["mean", "std", "count"])
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_transfer_summary(config_names, seeds=None, out_dir=None, metric="success",
                           run_dir=None, colors=None):
    """Seed-averaged matrices per config, plus the two scalar summaries."""
    out_dir = out_dir or os.path.join(sac.FIG_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    stacks = {}
    for name in config_names:
        s = stack_seeds(name, seeds=seeds, metric=metric, run_dir=run_dir)
        if s is not None:
            stacks[name] = s
    if not stacks:
        print("no probe files found; run a sweep first")
        return None

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = {n: cmap(i % 10) for i, n in enumerate(stacks)}

    # ---- seed-averaged matrices, one panel per config -------------------
    n = len(stacks)
    row_keys, cols = next(iter(stacks.values()))[0], next(iter(stacks.values()))[1]
    fig, axes = plt.subplots(1, n, figsize=(1.35 * len(cols) * n + 2.5,
                                             0.42 * len(row_keys) + 2.6),
                              squeeze=False, constrained_layout=True)
    for ax, (name, (rk, cl, arr, current)) in zip(axes[0], stacks.items()):
        with np.errstate(invalid="ignore"):
            # cells that no seed ever reached are all-NaN; nanmean warns rather
            # than erroring on those, and NaN is the answer we want
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(arr, axis=0)
        im = ax.imshow(mean, cmap="plasma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels(cl, rotation=30, ha="right", fontsize=6)
        ax.set_yticks(range(len(rk)))
        ax.set_yticklabels(rk if ax is axes[0][0] else [""] * len(rk), fontsize=6)
        ax.set_title(f"{name}\n(n={arr.shape[0]} seeds)", fontsize=8)
        for r in range(len(rk)):
            for c in range(len(cl)):
                if np.isnan(mean[r, c]):
                    continue
                ax.text(c, r, f"{mean[r, c]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if mean[r, c] > 0.55 else "black")
                if current[r, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                ec="red", lw=1.2, zorder=5))
        ax.set_xlabel("evaluated in", fontsize=7)
    axes[0][0].set_ylabel("after training block", fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.02)
    make_axis_nice(fig)
    fig.suptitle(f"Seed-averaged cross-context transfer ({metric})", fontsize=10)
    plt.savefig(os.path.join(out_dir, f"transfer_matrices_{metric}"))
    plt.close(fig)

    # ---- the two scalar summaries ---------------------------------------
    inter = interference_table(list(stacks), seeds=seeds, metric=metric, run_dir=run_dir)
    reten = retention_table(list(stacks), seeds=seeds, metric=metric, run_dir=run_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)

    ax = axes[0]
    for name in stacks:
        d = _mean_sem(inter[inter.config == name], "block", "participation_ratio")
        ax.errorbar(d.index, d["mean"], yerr=d["sem"], marker="o", ms=3, lw=1.2,
                    capsize=2, label=name, color=colors[name])
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax.axhline(len(cols) - 1, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("training block")
    ax.set_ylabel("participation ratio")
    ax.set_title("Interference: contexts competent in at once\n"
                 "(1 = specialized, dashed = all contexts)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1]
    for name in stacks:
        d = _mean_sem(inter[inter.config == name], "block", "mean_level")
        ax.errorbar(d.index, d["mean"], yerr=d["sem"], marker="o", ms=3, lw=1.2,
                    capsize=2, label=name, color=colors[name])
    ax.set_xlabel("Training block")
    ax.set_ylabel(f"Mean {metric} across contexts")

    ax = axes[2]
    for name in stacks:
        d = _mean_sem(reten[(reten.config == name) & (reten.lag >= 0)], "lag", "value")
        ax.errorbar(d.index, d["mean"], yerr=d["sem"], marker="o", ms=3, lw=1.2,
                    capsize=2, label=name, color=colors[name])
        naive = reten[(reten.config == name) & (reten.lag < 0)]["value"]
        if len(naive):
            ax.axhline(naive.mean(), color=colors[name], ls=":", lw=0.9)
    ax.set_xlabel("Blocks since trained (lag)")
    ax.set_ylabel(metric)
    ax.set_title("Memory: retention vs lag\n", fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    make_axis_nice(fig, offset=10, line_width=0.5)

    fig.suptitle(f"Transfer summaries ({metric})", fontsize=10)
    plt.savefig(os.path.join(out_dir, f"transfer_summaries_{metric}"))
    plt.close(fig)

    return {"interference": inter, "retention": reten}


def main(config_names=None, seeds=None, metric="success"):
    set_plt_params(style='fast', file_format="png")
    if config_names is None:
        import sweep_ablations
        config_names = [n for n in sweep_ablations.CONFIG_BY_NAME
                        if load_probes(n, seeds=seeds)]
    tables = plot_transfer_summary(config_names, seeds=seeds, metric=metric)
    if tables is None:
        return

    out_dir = os.path.join(sac.FIG_DIR, "plots")
    for name, frame in tables.items():
        frame.to_csv(os.path.join(out_dir, f"transfer_{name}_{metric}.csv"), index=False)

    inter, reten = tables["interference"], tables["retention"]
    print(f"\nInterference (participation ratio, mean over blocks and seeds), "
          f"1 = specialized:")
    for name, d in inter.groupby("config"):
        print(f"  {name:20s} PR {d.participation_ratio.mean():.2f} "
              f"| mean {metric} {d.mean_level.mean():.2f}")
    print(f"\nMemory (retention by lag):")
    for name, d in reten[reten.lag >= 0].groupby("config"):
        by_lag = d.groupby("lag")["value"].mean()
        naive = reten[(reten.config == name) & (reten.lag < 0)]["value"].mean()
        shown = "  ".join(f"lag{int(k)}={v:.2f}" for k, v in by_lag.items())
        print(f"  {name:20s} {shown}   (naive {naive:.2f})")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
