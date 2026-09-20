"""
make_ablations_plot.py -- cross-config ablation analysis.

Ported from the reference repo's script of the same name. This is the
"plotting scripts from the previous iteration": Hill-curve learning-curve
fits, episodes-to-criterion analysis with Wilcoxon+Holm annotation, and
learning-curve/parameter-distribution plots across the six sac.py configs.
sac_figures.py's plot_seed is the per-run diagnostic view; this file is the
cross-config comparison, reading the aggregated_results.csv sac.main writes
per config.

Two things were NOT ported: the two bare pdb.set_trace() calls that were in
add_phase_occurrence (debug leftovers, not intended behavior), and the
reference's own BASE_FIG_DIR/PLOT_OUTPUT_DIR, repointed here at this repo's
sac_runs/ and figures/sac/plots/ layout.

The criterion-based figures (criterion_crossing_*, traces_*, the consistency
report) are restricted to the "cycle" schedule -- see cycle_only. The COIN
paradigms have their own figures, schedule_timecourse_*.

One thing deliberately DIVERGES from the reference: the criterion crossing is
measured on a 25-episode moving average rather than the reference's 5-episode
rolling median, so it reports the same episode the trace figures show. See
estimate_criterion_crossing_episode for why the median was wrong here.
"""
import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import offset_copy
import seaborn as sns

from scipy.optimize import curve_fit
from statannotations.Annotator import Annotator

from plotting_utils import make_axis_nice, set_plt_params
import sac
import task as T

set_plt_params(style='fast', file_format="png")
BASE_FIG_DIR = sac.RUN_DIR
PLOT_OUTPUT_DIR = os.path.join(sac.FIG_DIR, "plots")
METRICS = ["success", "return"]

os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# sns.set_style("whitegrid")

# A CONFIG MISSING FROM CONFIG_ORDERS IS SILENTLY DROPPED FROM EVERY PLOT --
# get_config_order filters against it -- so a new preset in
# sweep_ablations.CONFIG_BY_NAME has to be added here too or its runs produce
# data that no figure shows. Buffer twins (_epbuf/_clearbuf) and circuit
# variants (_iid, _noisy<beta>_sigma<sigma>, ...) of a listed preset are
# DERIVED and need no entry -- see VARIANT_SUFFIXES and CIRCUIT_INFIX_RE.
#
# The three oracles share a hue ramp on purpose: they are the same perfect
# context label in three GEOMETRIES -- near-orthogonal, the circuit's measured
# one, and the circuit's with its common component removed -- so they should
# read as a family, and their spread is the cost of the code's shape.
CONFIG_COLORS = {
    "baseline": "#0e3b5c",
    "cb_only_end_to_end": "#5e4f6d",
    "oracle": "#f4ac7b",
    "coin": "#2a9d8f",           # the inferred oracle: its own hue, not an oracle shade
    "coin_cb": "#1b6f65",        # the same, scored through the learned CB
    "oracle_geom": "#e0913f",
    # "main"/"control" are the frozen circuits (frozen is the default and so
    # unsuffixed; renamed from main_frozen/control_frozen on 2026-09-16)
    "main": "#d8897b",
    "control": "#c25b6b",
    "end_to_end": "#9f6671",
}

CONFIG_ORDERS = {
    "baseline": 0,
    "cb_only_end_to_end": 1,
    "oracle": 2,
    "oracle_geom": 3,
    "main": 4,
    "control": 7,
    "end_to_end": 8,
    "coin": 9,
    "coin_cb": 10,
}

# VARIANTS ARE DERIVED, NOT LISTED. sac.apply_cli_variants names a run
# <parent>_epbuf, <parent>_clearbuf or <parent>_epbuf_clearbuf -- the same
# config with one or two knobs turned -- and the comparison that matters is
# parent against variant. So a variant takes its parent's colour washed out
# (a shadow of the parent, not a new hue) and sits immediately after it in the
# order. Derived for every base config at import, so a new variant of any
# preset plots without anyone editing these tables, and so the palette dicts
# seaborn receives contain the variant as a real key (seaborn checks key
# membership; a lookup fallback would not do).
VARIANT_SUFFIXES = (
    # suffix, order offset past the parent, fraction of the way to white
    ("_epbuf", 0.25, 0.35),
    ("_clearbuf", 0.5, 0.55),
)


def _lighten(hex_colour: str, frac: float) -> str:
    """Mix a #rrggbb colour `frac` of the way toward white."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * frac))
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


# CIRCUIT VARIANTS are the other kind of derived name. sac.apply_cli_variants
# also infixes WHICH frozen circuit drives MC -- `main_iid`, `main_ar1`,
# `main_iid_noisy0.001`, `main_iid_noisy0.001_sigma0.1` -- between the parent
# and the buffer suffixes. Unlike a buffer twin, a circuit variant is a
# different network answering the same question as its parent, so it gets a
# hue of its own (a shade toward black, never the washed-out buffer look) and
# sits between the parent and the next base. They cannot be enumerated at
# import -- beta and sigma are continuous -- so load_all_configs registers one
# when it finds its runs, via register_circuit_variant, and its buffer twins
# with it. Order offsets (0.05 per circuit variant) stay below the buffer
# offsets so `main`, `main_iid_noisy…`, `main_clearbuf`, `main_iid_noisy…_clearbuf`
# read parent / variant / twin / variant's twin.
_ACTION_MODES = "|".join(sorted(sac.M.ACTION_MODES, key=len, reverse=True))
_NUM = r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
# ... followed, optionally, by the TASK infix (which contexts: _rot / _refl,
# sac.PHASE_SET_SUFFIX). A task variant of a preset is registered the same way
# as a circuit variant: its own shade, right after the parent.
_TASK_INFIX = "|".join(re.escape(s) for s in sac.PHASE_SET_SUFFIX.values() if s)
# ... then the COIN-paradigm knobs (schedule, EC prior, relabelling, count prior)
_TASK_INFIX = (_TASK_INFIX + "|" if _TASK_INFIX else "") + "|".join(
    [re.escape(s) for s in sac.SCHEDULE_SUFFIX.values() if s]
    + ["_prior", "_carry", "_relabel", "_cprior", sac.TRANSBUF_SUFFIX])
_TASK_INFIX = rf"(?:{_TASK_INFIX})*"
_TRAIN_INFIX = "|".join(re.escape(t) for t in sac.M._TRAIN_TAG.values() if t)
CIRCUIT_INFIX_RE = re.compile(
    rf"(?:_(?:{_ACTION_MODES}))?(?:{_TRAIN_INFIX})?(?:_noisy{_NUM}(?:_sigma{_NUM})?)?"
    rf"{_TASK_INFIX}$")
CIRCUIT_VARIANT_ORDER_STEP = 0.05
CIRCUIT_VARIANT_SHADE = 0.35        # fraction of the way toward black


def _shade(hex_colour: str, frac: float) -> str:
    """Mix a #rrggbb colour `frac` of the way toward black."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c * (1 - frac)))
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _strip_buffer_suffixes(name: str) -> str:
    for suffix, _, _ in reversed(VARIANT_SUFFIXES):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def split_run_name(name: str):
    """(base preset, circuit infix, buffer suffix) of a run name, e.g.
    'main_iid_noisy0.001_sigma0.1_clearbuf' -> ('main', '_iid_noisy0.001_sigma0.1',
    '_clearbuf'). The circuit infix is '' for a run of the default circuit."""
    circuit_level = _strip_buffer_suffixes(name)
    buffer_suffix = name[len(circuit_level):]
    m = CIRCUIT_INFIX_RE.search(circuit_level)
    infix = m.group(0) if m else ""
    return circuit_level[: len(circuit_level) - len(infix)], infix, buffer_suffix


def base_config(name: str) -> str:
    """The preset a run name was derived from: buffer suffixes stripped in the
    order apply_cli_variants appends them (epbuf first, clearbuf last), then
    the circuit infix."""
    return split_run_name(name)[0]


def _schedule_of(cfg: str) -> str:
    """The sac.SCHEDULES entry a run name carries (_sr / _er / _savw as a
    whole '_'-token), "cycle" when none."""
    tokens = set(cfg.split("_"))
    for schedule, suffix in sac.SCHEDULE_SUFFIX.items():
        if suffix and suffix.lstrip("_") in tokens:
            return schedule
    return "cycle"


def _register_buffer_variants(base: str):
    """<base>_epbuf, <base>_clearbuf, <base>_epbuf_clearbuf: the parent's
    colour washed out, just after it in the order."""
    combos = [(s,) for s in VARIANT_SUFFIXES] + [tuple(VARIANT_SUFFIXES)]
    for combo in combos:
        name = base + "".join(s for s, _, _ in combo)
        CONFIG_ORDERS.setdefault(name, CONFIG_ORDERS[base] + sum(o for _, o, _ in combo))
        colour = CONFIG_COLORS[base]
        for _, _, frac in combo:
            colour = _lighten(colour, frac)
        CONFIG_COLORS.setdefault(name, colour)


def _register_variants():
    for base in list(CONFIG_ORDERS):
        _register_buffer_variants(base)


def register_circuit_variant(name: str) -> bool:
    """Admit a run whose name is a circuit variant of a registered preset
    (plus, optionally, buffer suffixes): give the circuit-level name its own
    order slot and shade, derive its buffer twins, and return True. A name
    already registered returns True untouched; a name whose base preset is
    unknown, or that carries no circuit infix, returns False -- unknown
    directories stay dropped, as before."""
    if name in CONFIG_ORDERS:
        return True
    base, infix, _ = split_run_name(name)
    if not infix or base not in CONFIG_ORDERS:
        return False
    circuit_level = base + infix
    if circuit_level not in CONFIG_ORDERS:
        # successive variants of one base step further from it, in the order
        # they are discovered (sorted directory listing -> stable across runs)
        n_prior = 0
        for other in CONFIG_ORDERS:
            b, i, s = split_run_name(other)
            n_prior += b == base and bool(i) and not s
        CONFIG_ORDERS[circuit_level] = (CONFIG_ORDERS[base]
                                        + CIRCUIT_VARIANT_ORDER_STEP * (n_prior + 1))
        CONFIG_COLORS[circuit_level] = _shade(
            CONFIG_COLORS[base], CIRCUIT_VARIANT_SHADE * (1 + 0.5 * n_prior))
    _register_buffer_variants(circuit_level)
    return name in CONFIG_ORDERS


_register_variants()


def config_order() -> list:
    """Every registered config, in plotting order. A FUNCTION, because
    register_circuit_variant adds names after import."""
    return [config for config, _ in sorted(CONFIG_ORDERS.items(), key=lambda item: item[1])]


CONFIG_ORDER = config_order()       # the import-time snapshot, kept for callers

# The window sac_figures.smooth uses for the per-seed <metric>_curve figures.
# Reused here so a trace plotted by this script is the same curve the user is
# already reading off those, rather than a differently-filtered version of it.
TRACE_SMOOTH_WINDOW = 25

PHASE_COLORS = {
    0: "#d2e3e1",
    1: "#a6c8c3",
    2: "#79aca6",
    3: "#4d9188",
    4: "#20756a",
}


def get_config_order(data: Optional[pd.DataFrame] = None) -> list:
    """Return the canonical config order, optionally filtered to configs present in data."""
    order = config_order()
    if data is None or data.empty or "config" not in data.columns:
        return order

    present = [config for config in order if config in set(data["config"].dropna().astype(str))]
    extras = sorted(set(data["config"].dropna().astype(str)) - set(present))
    return present + extras


def cycle_only(data: pd.DataFrame) -> pd.DataFrame:
    """The rows of the "cycle" schedule alone -- the _sr / _er / _savw runs
    dropped.

    Every criterion-based figure goes through this. The criterion is a
    fraction of the best value SEEN (global_criterion_targets) and "episodes
    to criterion" counts from a block onset, so both only mean something
    across runs that visit the same contexts in the same block structure. A
    COIN-paradigm run has its own blocks (a null field, a 1-reach evoke
    block, an error-clamp channel it cannot learn in) and drags the global
    target around, which would silently move every cycle run's crossing.
    The paradigms have their own figures -- plot_schedule_timecourse and
    plot_savings_washout."""
    if data is None or data.empty or "config" not in data.columns:
        return data
    keep = data["config"].astype(str).map(_schedule_of) == "cycle"
    return data[keep].copy()


def logistic(x, k, x0):
    """Generalized logistic curve: 1 / (1 + exp(-k(x - x0)))"""
    return 1 / (1 + np.exp(-k * (x - x0)))


def hill(x, a, L, n, K):
    """Hill curve: a + (L - a) * x^n / (K^n + x^n)"""
    return a + (L - a) * (x ** n) / (K ** n + x ** n)


def fit_learning_curve(episodes, values, metric_name):
    """Fit a Hill function to learning data, normalized to [0, 1]."""
    valid_idx = ~np.isnan(values)
    x = episodes[valid_idx].astype(float)
    y = values[valid_idx].astype(float)

    if len(x) < 4:
        return None

    y_min, y_max = np.nanmin(y), np.nanmax(y)
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    try:
        a_guess = np.clip(y_norm[0], 0, 1)
        L_guess = np.clip(np.max(y_norm), 0.5, 1)
        K_guess = len(x) / 2
        n_guess = 1.0

        popt, _ = curve_fit(
            hill, x, y_norm,
            p0=[a_guess, L_guess, n_guess, K_guess],
            maxfev=5000, ftol=1e-6,
            bounds=([0, 0, 0.1, 0], [1, 1, 10, len(x) * 2]),
        )

        a, L, n, K = popt
        return {"model": "hill", "a": a, "L": L, "n": n, "K": K}
    except Exception as e:
        print(f"Fitting failed for {metric_name}: {e}")
        return None


def load_all_configs(base_dir: str) -> Optional[pd.DataFrame]:
    """Load and concatenate all aggregated CSVs into a single DataFrame."""
    dfs = []

    csv_paths = glob.glob(os.path.join(base_dir, "*/aggregated_results.csv"))

    for csv_path in sorted(csv_paths):
        config_name = os.path.basename(os.path.dirname(csv_path))
        # a circuit variant of a known preset registers itself here (see
        # register_circuit_variant); anything else is dropped, as before
        if config_name in CONFIG_ORDERS or register_circuit_variant(config_name):
            try:
                df = pd.read_csv(csv_path)
                df['config'] = config_name
                dfs.append(df)
                print(f"Loaded {config_name}: {len(df)} rows, {df['seed'].nunique()} seeds")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def add_phase_occurrence(data: pd.DataFrame, label_col: str = "label", phase_col: str = "phase") -> pd.DataFrame:
    """Derive an integer per-label occurrence counter by walking episodes in
    CHRONOLOGICAL order.

    The ordering column matters and is easy to get wrong: sac.py's `ep` is the
    episode index WITHIN a phase block (1..N_PHASE_EP), so sorting by it
    interleaves every block and this counter silently collapses to all-zeros.
    `global_ep` is the monotonic one, so prefer it whenever present.

    Note main() does not use this -- it keys phase visits off the `occurrence`
    column sac.py already emits (see build_phase_occurrence_label). This stays
    for data that lacks that column.
    """
    if label_col not in data.columns or phase_col not in data.columns:
        raise KeyError(f"Expected '{label_col}' and '{phase_col}' in the CSV")

    ep_col = "global_ep" if "global_ep" in data.columns else "ep"
    data = data.sort_values(["config", "seed", ep_col]).copy()
    data["phase_occurrence"] = 0
    for (config, seed), group in data.groupby(["config", "seed"], sort=False):
        label_counts = {}
        prev_phase = None

        # iterate episode-by-episode to avoid counting multiple rows within same episode
        for ep, ep_group in group.groupby(ep_col, sort=False):
            lbl = ep_group[label_col].iloc[0]
            ph = ep_group[phase_col].iloc[0]

            if lbl not in label_counts:
                label_counts[lbl] = 0
            else:
                # increment only when the phase actually changed since previous episode
                if ph != prev_phase:
                    label_counts[lbl] += 1

            data.loc[ep_group.index, "phase_occurrence"] = label_counts[lbl]
            prev_phase = ph

    data["phase_label"] = data[label_col] + " (x" + (data["phase_occurrence"] + 1).astype(str) + ")"
    data["occurrence_label"] = "x" + (data["phase_occurrence"] + 1).astype(str)
    return data


def build_phase_occurrence_label(data: pd.DataFrame) -> pd.DataFrame:
    """Key each PHASE VISIT as "<label> (xN)" -- e.g. "C1: rot +45 (x2)".

    This is the grouping key the downstream analyses want: a phase visit is one
    context's Nth block, so it must split on label AND occurrence together.
    Keying on the occurrence index alone would merge every context within a
    cycle -- fitting one learning curve across rot+45 and curl+.5 episodes
    mixed together, which is meaningless.

    sac.py emits `occurrence` (which repetition of the phase cycle) directly,
    so this reads it rather than re-deriving it from the episode sequence.
    """
    occ_col = None
    for c in ("occurrence", "phase_occurrence", "occurrence_label", "occ"):
        if c in data.columns:
            occ_col = c
            break

    if occ_col is None:
        raise KeyError("Expected an 'occurrence' column (e.g. 'occurrence' or "
                        "'phase_occurrence') in the CSVs")

    data = data.copy()
    if data[occ_col].dtype == object:
        occ_numeric = pd.to_numeric(
            data[occ_col].astype(str).str.replace(r"[^0-9]", "", regex=True),
            errors="coerce").fillna(0).astype(int)
    else:
        occ_numeric = data[occ_col].fillna(0).astype(int)

    data["occurrence"] = occ_numeric
    data["phase_occurrence"] = data["label"].astype(str) + " (x" + (occ_numeric + 1).astype(str) + ")"
    return data


def fit_curves_per_phase(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Fit learning curves for each (config, seed, phase, occurrence) and extract parameters."""
    results = []

    group_keys = ['config', 'seed', 'phase', 'occurrence', 'phase_occurrence']
    for keys, group in data.groupby(group_keys):
        config, seed, phase, occurrence, phase_occurrence = keys
        group = group.sort_values('ep')
        n = len(group)
        if n < 3:
            continue

        episodes = np.arange(n).astype(float)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].values
            fit_params = fit_learning_curve(episodes, values, metric)
            row = {
                'config': config, 'seed': seed, 'phase': int(phase),
                'occurrence': int(occurrence), 'phase_occurrence': phase_occurrence,
            }
            if fit_params:
                row[f"{metric}_a"] = fit_params["a"]
                row[f"{metric}_L"] = fit_params["L"]
                row[f"{metric}_n"] = fit_params["n"]
                row[f"{metric}_K"] = fit_params["K"]
            else:
                row[f"{metric}_a"] = np.nan
                row[f"{metric}_L"] = np.nan
                row[f"{metric}_n"] = np.nan
                row[f"{metric}_K"] = np.nan
            results.append(row)

    if not results:
        return pd.DataFrame(columns=['config', 'seed', 'phase', 'phase_occurrence']
                             + [f"{m}_{p}" for m in metrics for p in ['a', 'L', 'n', 'K']])
    return pd.DataFrame(results)


def fit_curves_per_occurrence(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    results = []

    group_keys = ["config", "seed", "occurrence"]
    for keys, group in data.groupby(group_keys):
        config, seed, occurrence = keys
        group = group.sort_values("ep")
        n = len(group)
        if n < 3:
            continue

        episodes = np.arange(n).astype(float)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].values
            fit_params = fit_learning_curve(episodes, values, metric)
            row = {"config": config, "seed": seed, "occurrence": int(occurrence)}
            if fit_params:
                row[f"{metric}_a"] = fit_params["a"]
                row[f"{metric}_L"] = fit_params["L"]
                row[f"{metric}_n"] = fit_params["n"]
                row[f"{metric}_K"] = fit_params["K"]
            else:
                row[f"{metric}_a"] = np.nan
                row[f"{metric}_L"] = np.nan
                row[f"{metric}_n"] = np.nan
                row[f"{metric}_K"] = np.nan
            results.append(row)

    if not results:
        return pd.DataFrame(
            columns=["config", "seed", "occurrence"]
            + [f"{m}_{p}" for m in metrics for p in ["a", "L", "n", "K"]])
    return pd.DataFrame(results)


def smooth_trace(values: np.ndarray, window: int = TRACE_SMOOTH_WINDOW) -> tuple:
    """Moving average of a trace, returned as (offset, smoothed).

    Deliberately identical to sac_figures.smooth so a trace drawn here lines up
    episode-for-episode with the per-seed <metric>_curve figures: a 'valid'
    convolution, whose first output summarizes episodes 0..window-1 and so is
    plotted at index window-1 (hence the offset).
    """
    y = np.asarray(values, dtype=float)
    if window is None or window <= 1 or len(y) < window:
        return 0, y
    return window - 1, np.convolve(y, np.ones(window) / window, mode="valid")


def estimate_criterion_crossing_episode(
        episodes: np.ndarray, values: np.ndarray, target: float,
        smooth_window: int = TRACE_SMOOTH_WINDOW) -> float:
    """Estimate the first episode where the trajectory crosses a fixed target.

    Smoothing is a `smooth_window`-episode MOVING AVERAGE, aligned exactly the
    way smooth_trace (and sac_figures.smooth) align it, so the episode reported
    here is the episode at which the PLOTTED trace crosses the PLOTTED
    threshold -- see plot_metric_traces_by_phase_visit.

    This diverges from the reference repo, which used a 5-episode rolling
    MEDIAN. That disagreed with its own trace figures by up to ~550 episodes on
    this task's data: per-episode `return` is near-bimodal because the success
    bonus dominates it (~+6.8 on success, ~-2 on failure) and the criterion
    falls BETWEEN the two modes, so three lucky successes out of five moved the
    median past threshold while the 25-episode mean was still far below it. At
    one such reported crossing the local hit rate was 0.13 -- a coin flip, not
    a learning milestone. build_criterion_trace_consistency_table exists to
    keep the two definitions pinned together from here on.

    Consequence worth knowing: a 25-episode mean does not exist until episode
    25, so no crossing is reported earlier than that. A block already at
    criterion when it starts reports the first episode where the mean can be
    computed, not episode 0.
    """
    valid_idx = ~np.isnan(values)
    x = np.asarray(episodes[valid_idx], dtype=float)
    y = np.asarray(values[valid_idx], dtype=float)

    if len(x) < 4:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.isnan(target):
        return np.nan

    offset, y = smooth_trace(y, smooth_window)
    x = x[offset:]

    if len(y) == 0:
        return np.nan

    increasing = y[-1] >= y[0]
    crossings = np.where(y >= target)[0] if increasing else np.where(y <= target)[0]

    if len(crossings) == 0:
        return x.max()

    idx = int(crossings[0])
    if idx == 0:
        return float(x[0])

    x0, x1 = float(x[idx - 1]), float(x[idx])
    y0, y1 = float(y[idx - 1]), float(y[idx])

    if np.isclose(y0, y1):
        return x1

    frac = (target - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return x0 + frac * (x1 - x0)


def global_criterion_targets(data: pd.DataFrame, metrics: list,
                             criterion_fraction: float = 0.8) -> dict:
    """The per-metric criterion threshold: a fraction of the best value seen
    ANYWHERE in `data`.

    One shared threshold per metric is the point -- a per-seed or per-phase
    target would make "episodes to criterion" mean something different in every
    panel. Kept as one function so the trace plots draw exactly the threshold
    the crossing tables measured against.
    """
    targets = {}
    for metric in metrics:
        if metric in data.columns:
            global_max = np.nanmax(data[metric].to_numpy(dtype=float))
            targets[metric] = criterion_fraction * global_max if np.isfinite(global_max) else np.nan
    return targets


def build_criterion_crossing_table(data: pd.DataFrame, metrics: list, criterion_fraction: float = 0.8) -> pd.DataFrame:
    """Per-seed episode-to-criterion table; criterion is global per metric."""
    rows = []

    global_targets = global_criterion_targets(data, metrics, criterion_fraction)

    for (config, seed, occurrence), group in data.groupby(["config", "seed", "occurrence"]):
        group = group.sort_values("ep")
        episodes = np.arange(len(group), dtype=float)

        for metric in metrics:
            if metric not in group.columns:
                continue
            target = global_targets.get(metric, np.nan)
            t50 = estimate_criterion_crossing_episode(episodes, group[metric].values, target=target)
            rows.append({
                "config": config, "seed": seed, "occurrence": int(occurrence),
                "metric": metric, "criterion_target": target, "criterion_episode": t50,
            })

    if not rows:
        return pd.DataFrame(columns=["config", "seed", "occurrence", "metric", "criterion_target", "criterion_episode"])
    return pd.DataFrame(rows)


def build_criterion_crossing_phase_table(data: pd.DataFrame, metrics: list, criterion_fraction: float = 0.8) -> pd.DataFrame:
    """Per-seed episode-to-criterion table split by phase occurrence."""
    rows = []

    global_targets = global_criterion_targets(data, metrics, criterion_fraction)

    if "phase_occurrence" not in data.columns:
        raise KeyError("Expected a 'phase_occurrence' column for the phase-aware criterion plot")

    for (config, seed, phase_occurrence), group in data.groupby(["config", "seed", "phase_occurrence"]):
        group = group.sort_values("ep")
        episodes = np.arange(len(group), dtype=float)

        for metric in metrics:
            if metric not in group.columns:
                continue
            target = global_targets.get(metric, np.nan)
            criterion_episode = estimate_criterion_crossing_episode(episodes, group[metric].values, target=target)

            labels = group["label"].unique()
            label = labels[0] if len(labels) > 0 else "Unknown"
            occurrences = group["occurrence"].unique()
            occurrence = occurrences[0] if len(occurrences) > 0 else -1
            rows.append({
                "config": config, "seed": seed, "phase_occurrence": phase_occurrence,
                "metric": metric, "criterion_target": target, "criterion_episode": criterion_episode,
                "label": label, "occurrence": occurrence,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "config", "seed", "phase_occurrence", "metric", "criterion_target", "criterion_episode"])
    return pd.DataFrame(rows)


def build_criterion_trace_consistency_table(
        data: pd.DataFrame, metrics: list, criterion_fraction: float = 0.8,
        smooth_window: int = TRACE_SMOOTH_WINDOW) -> pd.DataFrame:
    """Per phase visit, the reported criterion episode next to the episode at
    which the PLOTTED trace clears the same threshold.

    Now that estimate_criterion_crossing_episode smooths the same way the trace
    figures do, `gap` should sit in [0, 1] everywhere: the two find the same
    crossing, and the estimator only interpolates back to the sub-episode point
    where the smoothed curve met the threshold. This table is therefore a
    standing regression check rather than a diagnostic -- a `gap` of hundreds of
    episodes is how the old 5-episode-median estimator showed up, so if one
    reappears the two definitions have drifted apart again.

    `frac_above` stays useful on its own: it is the local per-episode hit rate
    at the reported crossing, so it says whether "at criterion" means the
    policy is reliable there (near 1) or merely averaging past the line (near
    0.5 -- expected when, as for `return`, the threshold falls between two
    modes of a bimodal metric).
    """
    targets = global_criterion_targets(data, metrics, criterion_fraction)
    reported = build_criterion_crossing_phase_table(data, metrics, criterion_fraction=criterion_fraction)
    reported = reported.set_index(["config", "seed", "phase_occurrence", "metric"])

    rows = []
    for (config, seed, visit), group in data.groupby(["config", "seed", "phase_occurrence"]):
        group = group.sort_values("ep")
        for metric in metrics:
            if metric not in group.columns:
                continue
            target = targets.get(metric, np.nan)
            values = group[metric].to_numpy(dtype=float)

            offset, smoothed = smooth_trace(values, smooth_window)
            above = np.where(smoothed >= target)[0]
            trace_episode = float(above[0] + offset) if len(above) else np.nan

            try:
                criterion_episode = float(reported.loc[(config, seed, visit, metric), "criterion_episode"])
            except KeyError:
                criterion_episode = np.nan

            frac_above = np.nan
            if np.isfinite(criterion_episode):
                lo = max(0, int(criterion_episode) - smooth_window // 2)
                frac_above = float(np.mean(values[lo:lo + smooth_window] >= target))

            rows.append({
                "config": config, "seed": seed, "phase_occurrence": visit, "metric": metric,
                "criterion_target": target,
                "criterion_episode": criterion_episode,
                "trace_episode": trace_episode,
                "gap": trace_episode - criterion_episode,
                "frac_above": frac_above,
            })

    return pd.DataFrame(rows)


def report_criterion_trace_consistency(data: pd.DataFrame, metrics: list, output_dir: str,
                                       criterion_fraction: float = 0.8,
                                       smooth_window: int = TRACE_SMOOTH_WINDOW) -> pd.DataFrame:
    """Print and save the criterion-vs-trace comparison, over the
    cycle-schedule runs the criterion figures use (cycle_only)."""
    table = build_criterion_trace_consistency_table(
        cycle_only(data), metrics, criterion_fraction=criterion_fraction,
        smooth_window=smooth_window)
    if table.empty:
        return table

    summary = (table.groupby("metric")
                    .agg(median_criterion=("criterion_episode", "median"),
                         median_trace=("trace_episode", "median"),
                         median_gap=("gap", "median"),
                         max_gap=("gap", "max"),
                         median_frac_above=("frac_above", "median"))
                    .round(2))
    print(f"\nCriterion episode vs. plotted-trace crossing (both {smooth_window}-episode mean; "
          "gap should stay in [0, 1] -- see build_criterion_trace_consistency_table):")
    print(summary.to_string())

    output_path = os.path.join(output_dir, "criterion_trace_consistency.csv")
    table.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return table


def testable_pairs(facet_data: pd.DataFrame, pairs: list, min_pairs: int = 2) -> list:
    """Drop comparisons a paired Wilcoxon can't actually take.

    scipy discards zero-difference pairs before testing, so a comparison needs
    at least `min_pairs` seeds whose criterion episodes actually DIFFER. Two
    things trigger that in practice: too few seeds per config (n=1 leaves
    nothing to pair), and two configs that both saturate at the same criterion
    episode because neither reached the threshold in that phase. Either way
    scipy raises, which would otherwise cost the whole figure -- so skip the
    untestable comparison and still draw the plot.
    """
    keep = []
    for (occ_a, cfg_a), (occ_b, cfg_b) in pairs:
        a = facet_data[(facet_data["occurrence"] == occ_a) & (facet_data["config"] == cfg_a)]
        b = facet_data[(facet_data["occurrence"] == occ_b) & (facet_data["config"] == cfg_b)]

        if "seed" in facet_data.columns:
            merged = a.merge(b, on="seed", suffixes=("_a", "_b"))
            diffs = (merged["criterion_episode_a"] - merged["criterion_episode_b"]).to_numpy()
        else:
            n = min(len(a), len(b))
            diffs = (a["criterion_episode"].to_numpy()[:n]
                     - b["criterion_episode"].to_numpy()[:n])

        if np.count_nonzero(diffs) >= min_pairs:
            keep.append(((occ_a, cfg_a), (occ_b, cfg_b)))
    return keep


def get_shared_seeds(data: pd.DataFrame, configs: list) -> set:
    seed_sets = []
    for config in configs:
        seeds = set(data.loc[data["config"] == config, "seed"].dropna().astype(int).unique())
        seed_sets.append(seeds)

    if not seed_sets:
        return set()
    return set.intersection(*seed_sets)


def plot_criterion_crossing(data: pd.DataFrame, metrics: list, output_dir: str, criterion_fraction: float = 0.8):
    """Plot episode-to-criterion estimates with a consistent global threshold.

    Cycle-schedule runs only (cycle_only)."""
    data = cycle_only(data)
    if data.empty:
        print("No cycle-schedule runs; skipping criterion_crossing_*")
        return
    config_order = get_config_order(data)
    shared_seeds = get_shared_seeds(data, config_order)
    restricted_data = data[data["seed"].isin(shared_seeds) & data["config"].isin(config_order)].copy()

    crossing_table = build_criterion_crossing_phase_table(restricted_data, metrics, criterion_fraction=criterion_fraction)

    for metric in metrics:
        plot_data = crossing_table[crossing_table["metric"] == metric].dropna(subset=["criterion_episode"])

        facet_data = plot_data.copy()
        facet_data["occurrence"] = pd.to_numeric(facet_data["occurrence"], errors="coerce")
        facet_data = facet_data.dropna(subset=["occurrence"])
        facet_data["occurrence"] = facet_data["occurrence"].astype(int)

        if plot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(3, 3))
        sns.violinplot(
            data=plot_data, x="occurrence", y="criterion_episode", hue="config",
            palette=CONFIG_COLORS, hue_order=get_config_order(plot_data), ax=ax,
            inner="box", inner_kws={"box_width": 1, "whis_width": 0.5, "color": "black"}, width=1,
        )

        pairs = []
        present = set(zip(facet_data["occurrence"], facet_data["config"]))
        # Both orders must come from facet_data, the frame handed to Annotator --
        # taking them from the full `data` lets an occurrence that got dropped
        # here (NaN criterion, filtered seed) stay in `order`, and
        # statannotations rejects an order value it can't find in the data.
        occurrence_order = sorted(facet_data["occurrence"].dropna().unique())
        config_order = get_config_order(facet_data)

        for occ in occurrence_order:
            baseline_pair = (occ, "baseline")
            if baseline_pair not in present:
                continue
            for cfg in config_order:
                if cfg == "baseline":
                    continue
                if (occ, cfg) in present:
                    pairs.append((baseline_pair, (occ, cfg)))

        pairs = testable_pairs(facet_data, pairs)
        if pairs:
            annot = Annotator(
                ax, pairs, data=facet_data, x="occurrence", y="criterion_episode",
                hue="config", order=occurrence_order, hue_order=config_order,
            )
            annot.configure(
                test="Wilcoxon", text_format="star", comparisons_correction="holm",
                correction_format='replace', verbose=False,
            )
            annot.apply_and_annotate()
        else:
            print(f"  {metric}: no testable comparisons (need >=2 seeds whose "
                  f"criterion episodes differ) -- plotting without annotation")

        ax.set_xlabel("Previous Experiences")
        ax.set_ylabel("Episodes to Criterion")
        ax.set_title(f"Criterion Crossing Episode: {metric.replace('_', ' ').title()}")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=15)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        output_path = os.path.join(output_dir, f"criterion_crossing_{metric}")
        plt.tight_layout()
        make_axis_nice(fig)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_criterion_crossing_by_phase_occurrence(data: pd.DataFrame, metrics: list, output_dir: str, criterion_fraction: float = 0.8):
    """Plot episode-to-criterion estimates separated by phase occurrence.

    Cycle-schedule runs only (cycle_only)."""
    data = cycle_only(data)
    if data.empty:
        print("No cycle-schedule runs; skipping criterion_crossing_by_phase_occurrence_*")
        return
    config_order = get_config_order(data)
    shared_seeds = get_shared_seeds(data, config_order)
    restricted_data = data[data["seed"].isin(shared_seeds) & data["config"].isin(config_order)].copy()

    crossing_table = build_criterion_crossing_phase_table(restricted_data, metrics, criterion_fraction=criterion_fraction)

    for metric in metrics:
        plot_data = crossing_table[crossing_table["metric"] == metric].dropna(subset=["criterion_episode"])
        if plot_data.empty:
            continue

        g = sns.FacetGrid(plot_data, col="label", col_wrap=4, height=3, aspect=1.0)

        def plot_func(data, **kwargs):
            ax = plt.gca()
            facet_data = data.copy()
            facet_data["occurrence"] = pd.to_numeric(facet_data["occurrence"], errors="coerce")
            facet_data = facet_data.dropna(subset=["occurrence"])
            facet_data["occurrence"] = facet_data["occurrence"].astype(int)

            # from facet_data, not this facet's raw `data` -- see the note in
            # plot_criterion_crossing on why order must match what Annotator sees
            occurrence_order = sorted(facet_data["occurrence"].dropna().unique())
            config_order = get_config_order(facet_data)

            sns.violinplot(
                data=facet_data, x="occurrence", y="criterion_episode", hue="config",
                palette=CONFIG_COLORS, hue_order=config_order, order=occurrence_order,
                inner="box", inner_kws={"box_width": 1, "whis_width": 0.5, "color": "black"}, **kwargs,
            )

            pairs = []
            present = set(zip(facet_data["occurrence"], facet_data["config"]))
            for occ in occurrence_order:
                baseline_pair = (occ, "baseline")
                if baseline_pair not in present:
                    continue
                for cfg in config_order:
                    if cfg == "baseline":
                        continue
                    if (occ, cfg) in present:
                        pairs.append((baseline_pair, (occ, cfg)))

            pairs = testable_pairs(facet_data, pairs)
            if pairs:
                annot = Annotator(
                    ax, pairs, data=facet_data, x="occurrence", y="criterion_episode",
                    hue="config", order=occurrence_order, hue_order=config_order,
                )
                annot.configure(
                    test="Wilcoxon", text_format="star", comparisons_correction="holm",
                    correction_format='replace', verbose=False,
                )
                annot.apply_and_annotate()

        g.map_dataframe(plot_func)
        g.figure.set_layout_engine("constrained")
        g.set_axis_labels("Previous Experiences", "Episodes to Criterion")
        g.set(ylim=(0, restricted_data['ep'].max()))
        g.figure.suptitle(f"Criterion Crossing by Phase Occurrence: {metric.replace('_', ' ').title()}", fontsize=14)

        output_path = os.path.join(output_dir, f"criterion_crossing_by_phase_occurrence_{metric}")
        make_axis_nice(g.figure)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves with seaborn."""
    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: {metric} not in data")
            continue

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=data, x="ep", y=metric, hue="config", errorbar=("se", 1), linewidth=1,
            palette=CONFIG_COLORS, hue_order=get_config_order(data),
        )
        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Learning Curve: {metric.replace('_', ' ').title()}")
        plt.legend(title="Config", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        make_axis_nice(plt.gca())
        output_path = os.path.join(output_dir, f"learning_curve_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_episodic_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str, phase_length: int = 500):
    """Plot learning curves against global episode index, with phase-transition markers."""
    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: {metric} not in data")
            continue

        plt.figure(figsize=(12, 2))
        x_max = data['global_ep'].max()

        for phase_start in range(0, int(x_max) + phase_length, phase_length):
            plt.axvline(x=phase_start, color='black', linestyle='--', alpha=0.5)

        sns.lineplot(
            data=data, x="global_ep", y=metric, hue="config", errorbar=("se", 1),
            linewidth=1, palette=CONFIG_COLORS,
        )

        ax = plt.gca()
        plt.xlim(0, data['global_ep'].max())
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))

        if metric == 'return':
            plt.ylim(-35, 5)
        else:
            plt.ylim(0, 1)

        plt.xlabel("Episode")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Learning Curve: {metric.replace('_', ' ').title()}")
        plt.legend(title="Config", bbox_to_anchor=(1.05, 1), loc="upper left")
        make_axis_nice(plt.gca())
        output_path = os.path.join(output_dir, f"stepwise_learning_curve_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def phase_blocks(group: pd.DataFrame) -> pd.DataFrame:
    """One row per phase block of a single run, in chronological order.

    `phase` (not `label`) is the block index sac.py emits, so it separates
    repeat visits to the same context without needing the occurrence counter.
    """
    blocks = (group.groupby("phase", sort=True)
                   .agg(start=("global_ep", "min"), end=("global_ep", "max"),
                        label=("label", "first"), occurrence=("occurrence", "first"))
                   .reset_index())
    blocks["phase_occurrence"] = (blocks["label"].astype(str)
                                  + " (x" + (blocks["occurrence"].astype(int) + 1).astype(str) + ")")
    return blocks


# The criterion marker sits just ABOVE the threshold line, pointing down at
# the episode where the median trace crosses it. Offsetting it keeps the
# threshold line readable underneath, and a thin dark edge keeps the marker
# visible where it lands on a band of its own config colour.
CROSSING_MARKER = dict(marker="v", ms=4, mec="black", mew=0.3, clip_on=False, zorder=5)
CROSSING_MARKER_OFFSET_PT = 5


def crossing_transform(ax, fig):
    """Data coordinates shifted up by a fixed number of POINTS.

    Points, not data units: the offset must look the same in every panel, and
    these axes share a y range that differs between metrics (return spans ~20,
    success spans 1), so any data-unit offset would be invisible in one figure
    and enormous in the other.
    """
    return offset_copy(ax.transData, fig=fig, y=CROSSING_MARKER_OFFSET_PT, units="points")


def median_seed_band(group: pd.DataFrame, metric: str, x_col: Optional[str] = None,
                     smooth_window: int = TRACE_SMOOTH_WINDOW) -> pd.DataFrame:
    """Across-seed median and standard deviation of one config's traces,
    indexed by episode.

    Each seed is smoothed FIRST and the seeds aggregated SECOND. The other
    order folds within-seed episode noise into the band, which would make its
    width a measure of task stochasticity (large, and roughly the same for
    every config) instead of seed-to-seed spread.

    Median rather than mean because a single seed that fails to adapt in a
    block drags a mean far below where any individual run actually sits --
    and those stragglers are common enough here to matter. `x_col` is the
    episode column to index on (global_ep for whole-run traces); None indexes
    on position within the block, which is what the phase-visit panels need
    since every block restarts at zero.
    """
    curves = {}
    for seed, run in group.groupby("seed"):
        run = run.sort_values(x_col if x_col else "ep")
        offset, y = smooth_trace(run[metric].to_numpy(dtype=float), smooth_window)
        x = run[x_col].to_numpy()[offset:] if x_col else np.arange(len(run))[offset:]
        curves[seed] = pd.Series(y, index=x)

    if not curves:
        return pd.DataFrame(columns=["median", "std"], dtype=float)

    # Seeds are aligned on the episode index, so a short run contributes NaN
    # past its end rather than silently shifting the others.
    band = pd.DataFrame(curves)
    return pd.DataFrame({
        "median": band.median(axis=1),
        # ddof=1 is NaN for a single seed; a zero-width band is the honest
        # rendering of "one run, no spread to show".
        "std": band.std(axis=1).fillna(0.0),
    }).sort_index()


def plot_metric_traces(data: pd.DataFrame, metrics: list, output_dir: str,
                       criterion_fraction: float = 0.8,
                       smooth_window: int = TRACE_SMOOTH_WINDOW):
    """Across-seed median +- s.d. traces against global episode, one row per
    config.

    The cross-config counterpart of sac_figures.plot_seed's <metric>_curve.
    The criterion line and the across-seed MEDIAN crossing of each block are
    overlaid, so this figure and criterion_crossing_* can be read against each
    other instead of by eye. One marker per block, matched to the trace colour:
    the violins are where the per-seed spread of those crossings belongs.

    Cycle-schedule runs only (cycle_only), so this figure and the violins are
    computed from exactly the same rows.
    """
    if "global_ep" not in data.columns:
        print("Warning: no 'global_ep' column; skipping trace plots")
        return

    data = cycle_only(data)
    if data.empty:
        print("No cycle-schedule runs; skipping traces_*")
        return

    configs = get_config_order(data)
    targets = global_criterion_targets(data, metrics, criterion_fraction)
    crossings = build_criterion_crossing_phase_table(data, metrics, criterion_fraction=criterion_fraction)

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: {metric} not in data")
            continue

        target = targets.get(metric, np.nan)
        fig, axes = plt.subplots(len(configs), 1, figsize=(12, 1.9 * len(configs)),
                                 sharex=True, sharey=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        for ax, config in zip(axes, configs):
            config_data = data[data["config"] == config]
            color = CONFIG_COLORS.get(config, "gray")

            band = median_seed_band(config_data, metric, "global_ep", smooth_window)
            ax.fill_between(band.index, band["median"] - band["std"], band["median"] + band["std"],
                            color=color, alpha=0.25, lw=0)
            ax.plot(band.index, band["median"], color=color, lw=1.0)

            # Block boundaries and the reported crossings sit on top of the
            # traces: a marker far from where the median clears the dotted
            # threshold is the inconsistency, made visible.
            blocks = phase_blocks(config_data)
            for _, block in blocks.iterrows():
                ax.axvline(block["start"], color="black", ls="--", lw=0.5, alpha=0.4)

            if np.isfinite(target):
                ax.axhline(target, color="black", ls=":", lw=0.7)

            marks = crossings[(crossings["config"] == config) & (crossings["metric"] == metric)]
            starts = blocks.set_index("phase_occurrence")["start"]
            for visit, visit_marks in marks.groupby("phase_occurrence"):
                start = starts.get(visit)
                crossing = visit_marks["criterion_episode"].median()
                if start is None or not np.isfinite(crossing):
                    continue
                ax.plot(start + crossing, target, **CROSSING_MARKER,
                        color=color, transform=crossing_transform(ax, fig))

            ax.set_ylabel(f"{config}\n{metric.title()}", fontsize=5)
            ax.set_xlim(data["global_ep"].min(), data["global_ep"].max())

        for _, block in phase_blocks(data[data["config"] == configs[0]]).iterrows():
            axes[0].annotate(block["phase_occurrence"], (block["start"], 1.02),
                             xycoords=("data", "axes fraction"), fontsize=4, rotation=45,
                             ha="left", va="bottom")

        axes[-1].set_xlabel("Episode (global)")
        fig.suptitle(f"{metric.title()} traces, across-seed median +- s.d. "
                     f"(dotted: criterion {criterion_fraction:g} x max = {target:.3g}; "
                     f"marker: median reported crossing; {smooth_window}-episode mean)", fontsize=6)
        make_axis_nice(fig)
        output_path = os.path.join(output_dir, f"traces_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_metric_traces_by_phase_visit(data: pd.DataFrame, metrics: list, output_dir: str,
                                      criterion_fraction: float = 0.8,
                                      smooth_window: int = TRACE_SMOOTH_WINDOW):
    """Across-seed median +- s.d. traces re-zeroed at each block onset: configs
    down, phase visits across.

    This is the panel-for-panel counterpart of
    criterion_crossing_by_phase_occurrence -- same grouping, same threshold,
    same x-axis units (episodes since the context switch) -- so a crossing
    estimate can be checked against the trace it was computed from.

    Cycle-schedule runs only (cycle_only).
    """
    data = cycle_only(data)
    if data.empty:
        print("No cycle-schedule runs; skipping traces_by_phase_visit_*")
        return

    configs = get_config_order(data)
    targets = global_criterion_targets(data, metrics, criterion_fraction)
    crossings = build_criterion_crossing_phase_table(data, metrics, criterion_fraction=criterion_fraction)

    visit_order = (data.groupby("phase_occurrence")["phase"].min().sort_values().index.tolist()
                   if "phase" in data.columns else sorted(data["phase_occurrence"].unique()))

    for metric in metrics:
        if metric not in data.columns:
            continue

        target = targets.get(metric, np.nan)
        fig, axes = plt.subplots(len(configs), len(visit_order),
                                 figsize=(2.1 * len(visit_order), 2.2 * len(configs)),
                                 sharex=True, sharey=True, squeeze=False,
                                 constrained_layout=True)

        for row, config in enumerate(configs):
            color = CONFIG_COLORS.get(config, "gray")
            for col, visit in enumerate(visit_order):
                ax = axes[row][col]
                block = data[(data["config"] == config) & (data["phase_occurrence"] == visit)]

                band = median_seed_band(block, metric, None, smooth_window)
                ax.fill_between(band.index, band["median"] - band["std"],
                                band["median"] + band["std"], color=color, alpha=0.25, lw=0)
                ax.plot(band.index, band["median"], color=color, lw=1.0)

                if np.isfinite(target):
                    ax.axhline(target, color="black", ls=":", lw=0.7)

                marks = crossings[(crossings["config"] == config)
                                  & (crossings["metric"] == metric)
                                  & (crossings["phase_occurrence"] == visit)]
                crossing = marks["criterion_episode"].median()
                if np.isfinite(crossing):
                    ax.plot(crossing, target, **CROSSING_MARKER,
                            color=color, transform=crossing_transform(ax, fig))

                if row == 0:
                    ax.set_title(visit, fontsize=5)
                if col == 0:
                    ax.set_ylabel(f"{config}\n{metric.title()}", fontsize=5)
                if row == len(configs) - 1:
                    ax.set_xlabel("Episodes since switch", fontsize=5)

        fig.suptitle(f"{metric.title()} traces by phase visit, across-seed median +- s.d. "
                     f"(dotted: criterion = {target:.3g}; marker: median reported crossing; "
                     f"{smooth_window}-episode mean)", fontsize=6)
        make_axis_nice(fig)
        output_path = os.path.join(output_dir, f"traces_by_phase_visit_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_learning_curves_by_phase_occurrence(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves colored by phase occurrence, one subplot per config."""
    for metric in metrics:
        if metric not in data.columns:
            continue

        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]

        for ax_idx, config in enumerate(configs):
            config_data = data[data['config'] == config]
            sns.lineplot(
                data=config_data, x="ep", y=metric, hue="phase_occurrence", palette="tab10",
                errorbar=("se", 1), linewidth=1, ax=axes[ax_idx],
            )
            axes[ax_idx].set_title(f"{config} - {metric.replace('_', ' ').title()}")
            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel(metric.replace("_", " ").title())
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(title="Phase", fontsize=9, loc="best")

        make_axis_nice(plt.gcf())
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"by_phase_occurrence_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_learning_curves_by_occurrence(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves colored by occurrence, one subplot per config."""
    for metric in metrics:
        if metric not in data.columns:
            continue

        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]

        for ax_idx, config in enumerate(configs):
            config_data = data[data['config'] == config]
            sns.lineplot(
                data=config_data, x="ep", y=metric, hue="occurrence", palette=PHASE_COLORS,
                errorbar=("se", 1), linewidth=1, ax=axes[ax_idx],
            )
            axes[ax_idx].set_title(f"{config} - {metric.replace('_', ' ').title()}")
            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel(metric.replace("_", " ").title())
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(title="Phase", fontsize=9, loc="best")

        plt.tight_layout()
        make_axis_nice(fig)
        output_path = os.path.join(output_dir, f"by_occurrence_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_phase_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves by phase x occurrence using a FacetGrid."""
    for metric in metrics:
        if metric not in data.columns:
            continue

        g = sns.FacetGrid(data, col="label", row="occurrence", height=5, aspect=1.2)

        def plot_func(data, **kwargs):
            sns.lineplot(
                data=data, x="ep", y=metric, hue="config", errorbar=("se", 1),
                palette=CONFIG_COLORS, linewidth=1, hue_order=get_config_order(data), **kwargs,
            )

        g.map_dataframe(plot_func)

        config_order = get_config_order(data)
        handles = [
            Line2D([0], [0], color=CONFIG_COLORS.get(cfg, "#333333"), marker="o",
                   linewidth=1, markersize=3, label=cfg)
            for cfg in config_order
        ]
        g.figure.legend(handles=handles, title="Config", loc="upper right")
        g.set_axis_labels("Episode", metric.replace("_", " ").title())

        output_path = os.path.join(output_dir, f"phase_learning_curve_{metric}")
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        make_axis_nice(plt.gcf())
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_occurrence_learning_curves(data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot learning curves by occurrence using a FacetGrid."""
    for metric in metrics:
        if metric not in data.columns:
            continue

        g = sns.FacetGrid(data, col="occurrence", height=5, aspect=1.2)

        def plot_func(data, **kwargs):
            sns.lineplot(data=data, x="ep", y=metric, hue="config", errorbar=("se", 1), linewidth=1, **kwargs)

        g.map_dataframe(plot_func)
        g.set_axis_labels("Episode", metric.replace("_", " ").title())

        output_path = os.path.join(output_dir, f"occurrence_learning_curve_{metric}")
        plt.tight_layout()
        make_axis_nice(plt.gcf())
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


def plot_curve_parameters(fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Plot distributions of fitted Hill-curve parameters, split by phase_occurrence."""
    param_names = {"n": "Hill Coefficient", "K": "Half-Maximum", "L": "Asymptote", "a": "Baseline"}

    for metric in metrics:
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            if param_col not in fit_data.columns:
                continue

            plot_data = fit_data[[param_col, 'config', 'phase_occurrence']].dropna()
            if plot_data.empty:
                continue
            plot_data.columns = ['value', 'config', 'phase_occurrence']

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                data=plot_data, x="config", y="value", hue="phase_occurrence",
                order=get_config_order(plot_data), ax=ax,
            )
            ax.set_xlabel("Config")
            ax.set_ylabel(param_label)
            ax.set_title(f"{param_label} Distribution: {metric.replace('_', ' ').title()}")
            ax.tick_params(axis="x", rotation=15)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

            output_path = os.path.join(output_dir, f"param_{metric}_{param}")
            plt.tight_layout()
            make_axis_nice(fig)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_path}")


def plot_curve_parameters_per_occurrence(fit_data: pd.DataFrame, metrics: list, output_dir: str,
                                          comparison_table: Optional[pd.DataFrame] = None):
    """Plot distributions of fitted Hill-curve parameters, split by occurrence."""
    param_names = {"n": "Hill Coefficient", "K": "Half-Maximum", "L": "Asymptote", "a": "Baseline"}

    for metric in metrics:
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            if param_col not in fit_data.columns:
                continue

            plot_data = fit_data[[param_col, 'config', 'occurrence']].dropna()
            if plot_data.empty:
                continue
            plot_data.columns = ['value', 'config', 'occurrence']

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                data=plot_data, x="config", y="value", hue="occurrence",
                palette=PHASE_COLORS, order=get_config_order(plot_data), ax=ax,
            )
            ax.set_xlabel("Config")
            ax.set_ylabel(param_label)
            ax.set_title(f"{param_label} Distribution: {metric.replace('_', ' ').title()}")
            ax.tick_params(axis="x", rotation=15)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

            output_path = os.path.join(output_dir, f"param_{metric}_{param}_occurrence")
            plt.tight_layout()
            make_axis_nice(fig)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {output_path}")


def plot_fitted_curves(data: pd.DataFrame, fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Overlay raw data (mean +/- SEM across seeds) with fitted Hill curves, per config x phase_occurrence."""
    fit_subdir = os.path.join(output_dir, "fit_quality")
    os.makedirs(fit_subdir, exist_ok=True)

    for metric in metrics:
        if metric not in data.columns:
            continue

        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]

        for ax_idx, config in enumerate(configs):
            ax = axes[ax_idx]
            config_data = data[data['config'] == config]
            config_fit = fit_data[fit_data['config'] == config]

            phase_occs = sorted(config_data['phase_occurrence'].unique())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(phase_occs)))
            handles = []

            for phase_idx, phase_occ in enumerate(phase_occs):
                phase_data = config_data[config_data['phase_occurrence'] == phase_occ]
                color = colors[phase_idx]

                ep_grouped = phase_data.groupby('ep')[metric].agg(['mean', 'std', 'count'])
                ep_grouped['sem'] = ep_grouped['std'] / np.sqrt(ep_grouped['count'])

                x_vals = ep_grouped.index.values.astype(float)
                y_vals = ep_grouped['mean'].values
                y_err = ep_grouped['sem'].values

                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                y_norm = (y_vals - y_min) / (y_max - y_min + 1e-8)

                sample_idx = np.random.RandomState(42).choice(
                    len(x_vals), size=max(1, int(len(x_vals) * .1)), replace=False)
                sample_idx = np.sort(sample_idx)

                ax.errorbar(x_vals[sample_idx], y_norm[sample_idx],
                            yerr=y_err[sample_idx] / (y_max - y_min + 1e-8),
                            fmt='o', alpha=0.6, color=color, markersize=5)

                phase_occ_fits = config_fit[config_fit['phase_occurrence'] == phase_occ]
                r_squared = np.nan
                if not phase_occ_fits.empty:
                    a_mean = phase_occ_fits[f'{metric}_a'].mean()
                    L_mean = phase_occ_fits[f'{metric}_L'].mean()
                    n_mean = phase_occ_fits[f'{metric}_n'].mean()
                    K_mean = phase_occ_fits[f'{metric}_K'].mean()

                    if all(~np.isnan([a_mean, L_mean, n_mean, K_mean])):
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_smooth = hill(x_smooth, a_mean, L_mean, n_mean, K_mean)
                        y_pred = hill(x_vals, a_mean, L_mean, n_mean, K_mean)
                        ss_res = np.sum((y_norm - y_pred) ** 2)
                        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                        ax.plot(x_smooth, y_smooth, '--', linewidth=2, alpha=0.8, color=color)

                label_text = f'{phase_occ}' if np.isnan(r_squared) else f'{phase_occ} (R²={r_squared:.3f})'
                handles.append(Line2D([0], [0], color=color, marker='o', linestyle='--',
                                       linewidth=2, markersize=5, label=label_text))

            ax.set_title(f'{config}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(f'{metric.title()} (normalized)')
            ax.grid(True, alpha=0.3)
            ax.legend(handles=handles, fontsize=8, loc='best')

        plt.suptitle(f'Fit Quality: {metric.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(fit_subdir, f'fit_quality_{metric}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {output_path}')


def plot_fitted_curves_by_occurrence(data: pd.DataFrame, fit_data_occur: pd.DataFrame, metrics: list, output_dir: str):
    """Overlay occurrence-level raw data with fitted Hill curves, per config."""
    fit_subdir = os.path.join(output_dir, "fit_quality")
    os.makedirs(fit_subdir, exist_ok=True)

    for metric in metrics:
        if metric not in data.columns:
            continue

        configs = get_config_order(data)
        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
        if len(configs) == 1:
            axes = [axes]

        for ax_idx, config in enumerate(configs):
            ax = axes[ax_idx]
            config_data = data[data['config'] == config].copy()
            config_fit = fit_data_occur[fit_data_occur['config'] == config]

            occs = sorted(config_data['occurrence'].dropna().unique())
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(occs)))
            handles = []

            for occ_idx, occurrence in enumerate(occs):
                occ_data = config_data[config_data['occurrence'] == occurrence].copy()
                if occ_data.empty:
                    continue

                color = PHASE_COLORS.get(occurrence, colors[occ_idx])
                occ_data = occ_data.sort_values(['seed', 'ep'])
                occ_data['local_ep'] = occ_data.groupby('seed').cumcount().astype(float)

                grouped = occ_data.groupby('local_ep')[metric].agg(['mean', 'std', 'count'])
                grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

                x_vals = grouped.index.values.astype(float)
                y_vals = grouped['mean'].values
                y_err = grouped['sem'].fillna(0).values
                if len(x_vals) == 0:
                    continue

                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                denom = y_max - y_min + 1e-8
                y_norm = (y_vals - y_min) / denom

                sample_idx = np.random.RandomState(42).choice(
                    len(x_vals), size=max(1, int(len(x_vals) * 0.1)), replace=False)
                sample_idx = np.sort(sample_idx)

                ax.errorbar(x_vals[sample_idx], y_norm[sample_idx], yerr=y_err[sample_idx] / denom,
                            fmt='o', alpha=0.6, color=color, markersize=5)

                occ_fit = config_fit[config_fit['occurrence'] == occurrence]
                r_squared = np.nan
                if not occ_fit.empty:
                    a_mean = occ_fit[f'{metric}_a'].mean()
                    L_mean = occ_fit[f'{metric}_L'].mean()
                    n_mean = occ_fit[f'{metric}_n'].mean()
                    K_mean = occ_fit[f'{metric}_K'].mean()

                    if all(~np.isnan([a_mean, L_mean, n_mean, K_mean])):
                        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_smooth = hill(x_smooth, a_mean, L_mean, n_mean, K_mean)
                        y_pred = hill(x_vals, a_mean, L_mean, n_mean, K_mean)
                        ss_res = np.sum((y_norm - y_pred) ** 2)
                        ss_tot = np.sum((y_norm - np.mean(y_norm)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                        ax.plot(x_smooth, y_smooth, '--', linewidth=2, alpha=0.8, color=color)

                label_text = f'occ {int(occurrence)}' if np.isnan(r_squared) else f'occ {int(occurrence)} (R²={r_squared:.3f})'
                handles.append(Line2D([0], [0], color=color, marker='o', linestyle='--',
                                       linewidth=2, markersize=5, label=label_text))

            ax.set_title(f'{config}')
            ax.set_xlabel('Local Episode (within occurrence)')
            ax.set_ylabel(f'{metric.title()} (normalized)')
            ax.grid(True, alpha=0.3)
            if handles:
                ax.legend(handles=handles, fontsize=8, loc='best')

        plt.suptitle(f'Fit Quality by Occurrence: {metric.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(fit_subdir, f'fit_quality_occurrence_{metric}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {output_path}')


def plot_parameter_summary(fit_data: pd.DataFrame, metrics: list, output_dir: str):
    """Heatmap of mean fitted Hill-curve parameters per config."""
    param_names = {"a": "Baseline", "L": "Asymptote", "n": "Hill Coeff", "K": "Half-Max"}

    for metric in metrics:
        summary_data = {}
        for param, param_label in param_names.items():
            param_col = f"{metric}_{param}"
            if param_col in fit_data.columns:
                summary_data[param_label] = fit_data.groupby("config")[param_col].mean()

        if not summary_data:
            continue

        summary_df = pd.DataFrame(summary_data)
        plt.figure(figsize=(8, 6))
        sns.heatmap(summary_df.T, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "Parameter Value"})
        plt.title(f"Curve Parameters Summary: {metric.replace('_', ' ').title()}")
        plt.xlabel("Config")
        output_path = os.path.join(output_dir, f"param_summary_{metric}")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {output_path}")


# ── Retention vs savings ─────────────────────────────────────────────────
#
# Two things "relearns faster on a revisit" can mean, and the block-level
# metrics above add them together:
#
#   RETENTION  the block STARTS higher than a first visit did. The policy
#              still performs -- washout by the intervening contexts did not
#              erase it, or it switched back. A level.
#   SAVINGS    the block CLIMBS faster from wherever it starts. The classic
#              motor-learning result, measured after washout. A rate.
#
# They dissociate empirically here: with the replay buffer cleared at every
# switch, the frozen circuit keeps its savings and loses all its retention,
# while the one-hot oracle keeps ~60% of its retention and never had rate
# savings to lose. A single "savings@k" number cannot show that, so each block
# is split into an ONSET window and the remainder:
#
#   level = the block's value AT EPISODE 0: OLS intercept of the metric over
#           the first ONSET_FRACTION of the block
#   onset = mean(metric over that same window)
#   gain  = mean(metric over the rest of the block) - onset
#
#   retention = level(revisit blocks) - level(first-visit blocks)
#   savings   = gain(revisit blocks)  - gain(first-visit blocks)
#
# both averaged over blocks WITHIN a seed first, so the unit of independence
# in every error bar is the seed.
#
# `level` is an intercept and not the window mean because the mean leaks rate
# into the level: two blocks starting at the same value, one climbing twice as
# fast, differ in their 20-episode mean by slope * 9.5/100 -- a faster learner
# would be scored as retaining more. The intercept is exact for a locally
# linear start; on binary data it is ~1.9x noisier than the mean, which the
# within-seed block average absorbs. `gain` is relative to the block's own
# onset on purpose: a block that starts higher must not be credited with
# learning faster, which is the conflation in the other direction.
#
# KNOWN LIMIT: gain is bounded above by (ceiling - onset), so a config that
# retains a lot has little room to gain and its savings reads low or even
# NEGATIVE (the one-hot oracle: retention ~0.55, savings ~-0.2). That is the
# metric being honest about a ceiling, not evidence of anti-savings; read the
# savings axis for configs that start low, and keep ONSET_FRACTION small so a
# fast climb is scored as gain rather than absorbed into the onset.
ONSET_FRACTION = 0.2


def _block_level_and_onset(y: np.ndarray, k: int) -> tuple:
    """(intercept at episode 0, window mean) over the first k episodes."""
    w = np.asarray(y[:k], dtype=float)
    ok = np.isfinite(w)
    if ok.sum() < 2:
        m = float(np.nanmean(w)) if ok.any() else np.nan
        return m, m
    t = np.arange(k, dtype=float)[ok]
    coef = np.polyfit(t, w[ok], 1)          # [slope, intercept]
    return float(coef[1]), float(w[ok].mean())


def retention_savings_table(data: pd.DataFrame, metric: str = "success",
                            onset_fraction: float = ONSET_FRACTION,
                            drop_first_block: bool = False) -> pd.DataFrame:
    """One row per (config, seed): level, onset and gain for first visits and
    revisits, and the differences `retention` (of level) and `savings` (of
    gain).

    A block is a `phase`; `occurrence` 0 is a first visit, anything higher a
    revisit. `drop_first_block` excludes phase 0, for schedules WITHOUT a
    warmup block where phase 0 also contains initial task acquisition and so
    is not a clean first visit (the reference repo's runs). With sac.py's
    null-field warmup it is a first visit like any other and stays in.
    """
    rows = []
    for (config, seed), run in data.groupby(["config", "seed"]):
        visits = {0: [], 1: []}
        for phase, blk in run.groupby("phase", sort=True):
            if drop_first_block and phase == 0:
                continue
            y = blk.sort_values("ep")[metric].to_numpy(dtype=float)
            k = int(round(onset_fraction * len(y)))
            if k < 1 or len(y) - k < 1:
                continue
            level, onset = _block_level_and_onset(y, k)
            gain = float(np.nanmean(y[k:])) - onset
            visits[min(int(blk["occurrence"].iloc[0]), 1)].append((level, onset, gain))
        if not visits[0] or not visits[1]:
            continue                        # nothing to contrast in this run
        l0, o0, g0 = np.mean(visits[0], axis=0)
        l1, o1, g1 = np.mean(visits[1], axis=0)
        rows.append(dict(config=config, seed=seed,
                         level_first=l0, level_revisit=l1,
                         onset_first=o0, onset_revisit=o1,
                         gain_first=g0, gain_revisit=g1,
                         retention=l1 - l0, savings=g1 - g0))
    return pd.DataFrame(rows)


def _block_length(group: pd.DataFrame) -> int:
    """Episodes per block for one config: the median block length over
    (seed, phase). Grouping by phase alone counts every seed's copy of the
    block and inflates the length by the number of seeds."""
    return int(group.groupby(["seed", "phase"])["ep"].count().median())


def _visit_band(group: pd.DataFrame, metric: str, revisit: bool,
                smooth_window: int) -> pd.DataFrame:
    """Median +- sd across seeds of the within-block trace, for first visits
    or revisits, indexed by position in the block.

    Blocks are averaged WITHIN a seed at each position before smoothing and
    before the across-seed median -- median_seed_band assumes one block per
    seed and would otherwise concatenate a seed's blocks end to end.
    """
    sel = group[group["occurrence"] > 0] if revisit else group[group["occurrence"] == 0]
    curves = {}
    for seed, run in sel.groupby("seed"):
        by_pos = (run.assign(pos=run.groupby("phase")["ep"].rank(method="first") - 1)
                     .groupby("pos")[metric].mean().sort_index())
        offset, y = smooth_trace(by_pos.to_numpy(dtype=float), smooth_window)
        curves[seed] = pd.Series(y, index=by_pos.index.to_numpy()[offset:])
    if not curves:
        return pd.DataFrame(columns=["median", "std"], dtype=float)
    band = pd.DataFrame(curves)
    return pd.DataFrame({"median": band.median(axis=1),
                         "std": band.std(axis=1).fillna(0.0)}).sort_index()


def adaptation_index(first_angle_deg, schedule="spontaneous_recovery"):
    """Feedforward compensation of the first action, scaled to the schedule's
    baseline and perturbation (sac.schedule_contexts): 0 = full compensation
    for the baseline, +1 = full compensation for P+. Under the P0 schedules
    the baseline is the null field, so this is -first_angle / 30: +1 = P+,
    -1 = P-, 0 = aiming straight. Under savings-washout the baseline is
    rot +30 and P+ rot +60, so 0 = compensating +30, +1 = compensating +60,
    -1 = aiming straight. The sign flip puts adaptation to the long
    perturbation UP, as the COIN and Smith et al. figures draw it."""
    roles = sac.schedule_contexts(schedule)
    base = 0.0 if roles["base"] is None else float(T.C_PARAM[roles["base"]])
    pert = float(T.C_PARAM[roles["pplus"]])
    return (-np.asarray(first_angle_deg, dtype=float) - base) / (pert - base)


def _reference_labels(schedule):
    """What the +1 / 0 / -1 lines of panel A mean under a schedule."""
    roles = sac.schedule_contexts(schedule)
    if roles["base"] is None:
        return {1: "full P+", -1: "full P−"}
    base, pert = float(T.C_PARAM[roles["base"]]), float(T.C_PARAM[roles["pplus"]])
    straight = (0.0 - base) / (pert - base)          # where aiming at the goal lands
    return {1: f"full P+ ({T.C_LABEL[roles['pplus']]})",
            0: f"full base ({T.C_LABEL[roles['base']]})", straight: "straight aim"}


def _paradigm_runs(data, schedule):
    """The runs of ONE paradigm, in config order. Keyed on the run name's
    schedule suffix: a spontaneous-recovery run and a savings-washout run
    both have P0 blocks and must never share a figure."""
    return [c for c in get_config_order(data) if _schedule_of(c) == schedule]


CLEARBUF_SUFFIX_PLOT = sac.CLEARBUF_SUFFIX


def _replay_off(cfg):
    """Whether a run has no cross-block replay: the *_clearbuf twin."""
    return CLEARBUF_SUFFIX_PLOT in cfg


def _model_style(cfg):
    """Colour follows the MODEL (the parent preset), line style the replay
    condition: solid with the buffer, dashed without. The lightened twin
    colour is not used here -- two encodings of one bit is enough, and the
    dashed line stays legible in print."""
    parent = base_config(cfg)
    return CONFIG_COLORS.get(parent, CONFIG_COLORS.get(cfg, "#333")), ("--" if _replay_off(cfg) else "-")


def _model_label(cfg):
    return f"{base_config(cfg)} · {'no replay' if _replay_off(cfg) else 'replay'}"


def _after_channel_probes(cfg, run_dir):
    """Per seed: adaptation index of the FIRST action in the probe of P+ and
    of P-, after the LAST channel block, over probe targets 1..7 -- the probe
    observer persists across a probe's targets, so from the second target on
    the first action is belief-informed; target 0 is the uniform-belief aim
    and is left out. Returns {"P+": [per seed], "P-": [per seed]} or None."""
    pp, pm = sac.coin_pair()
    out = {"P+": [], "P-": []}
    for f in sorted(glob.glob(os.path.join(run_dir, cfg, "seed_*", "probes.npz"))):
        probes = np.load(f, allow_pickle=True)["probes"]
        ch = [pr for pr in probes if pr["label"] == sac.CHANNEL_LABEL]
        if not ch:
            return None
        last_phase = max(int(pr["phase"]) for pr in ch)
        for pr in ch:
            if int(pr["phase"]) != last_phase:
                continue
            fa = np.asarray(pr["first_angle"], dtype=float)
            if int(pr["eval_context_id"]) == pp:
                out["P+"].append(float(adaptation_index(fa[1:]).mean()))
            elif int(pr["eval_context_id"]) == pm:
                out["P-"].append(float(adaptation_index(fa[1:]).mean()))
    return out if out["P+"] and out["P-"] else None


READOUT_COLUMN = "probe_first_angle"


def _readout_column(d):
    """The first-action column a paradigm run is read from: the per-reach
    8-direction probe (sac.READOUT_PROBE_STEPS; mean over directions and, with
    the experience prior, over eight bank draws) where the run logged it, the
    single training reach's first action for runs from before it existed."""
    if READOUT_COLUMN in d.columns and d[READOUT_COLUMN].notna().any():
        return READOUT_COLUMN
    return "first_angle"


def _channel_by_prior_source(d):
    """For a run with the experience prior: the channel first action's
    adaptation index by the source of EC's dealt start state, per seed --
    {"P+": [...], "P-": [...]} -- or None when the run logged no prior.
    From the readout probe's per-source means (probe_fa_pplus / _pminus,
    each already a mean over the targets dealt that source) when the run has
    them, else from the training reaches' own draws (prior_src)."""
    if "prior_src" not in d.columns:
        return None
    ch = d[d.channel == 1]
    if ch.empty:
        return None
    pp, pm = sac.coin_pair()
    out = {}
    probe_cols = {"P+": "probe_fa_pplus", "P-": "probe_fa_pminus"}
    if all(c in d.columns and ch[c].notna().any() for c in probe_cols.values()):
        for key, c in probe_cols.items():
            per_seed = ch.groupby("seed")[c].mean().dropna()
            if per_seed.empty:
                return None
            out[key] = adaptation_index(per_seed.to_numpy()).tolist()
        return out
    ch = ch[ch.prior_src != sac.PRIOR_SRC_NONE]
    if ch.empty:
        return None
    for key, cid in (("P+", pp), ("P-", pm)):
        per_seed = ch[ch.prior_src == cid].groupby("seed").first_angle.mean()
        if per_seed.empty:
            return None
        out[key] = adaptation_index(per_seed.to_numpy()).tolist()
    return out


def _evoke_segments(d):
    """The channel reaches of an evoked-recovery run split at the evoke.

    The schedule is ... P- | CHANNEL (short) | P+ (one reach) | CHANNEL, and
    the two channel blocks are the same label, so they are told apart by the
    block's `occurrence` (sac.build_blocks numbers repeats of a label). Pooling
    them -- one `k` running through both, one opening window spanning the
    evoke -- is what made the evoked figure measure spontaneous recovery
    instead: the quantity of interest is a STEP at a known reach, and a window
    that straddles it averages the step away.

    Returns (pre, post) with a per-seed `k` counted from the evoke: the
    pre-evoke block ends at k = 0 and the post-evoke block starts at k = 1.
    None when the run has fewer than two channel blocks.
    """
    ch = d[d.channel == 1]
    if ch.empty or "occurrence" not in ch.columns:
        return None
    occs = sorted(ch["occurrence"].dropna().unique())
    if len(occs) < 2:
        return None
    pre = ch[ch.occurrence == occs[0]].copy()
    post = ch[ch.occurrence == occs[-1]].copy()
    if pre.empty or post.empty:
        return None
    pre["k"] = -pre.groupby("seed").cumcount(ascending=False)
    post["k"] = post.groupby("seed").cumcount() + 1
    return pre, post


def _seed_mean(frame, column="index"):
    return frame.groupby("seed")[column].mean()


def _evoked_stats(pre, post, evoke_window, close_reaches):
    """Per seed, the two numbers that define evoked recovery, from the
    segments _evoke_segments returns:

      jump  = mean(first `evoke_window` post-evoke reaches) - mean(pre-evoke)
              -- one P+ reach reinstates a memory the channel was not
              expressing, so this is positive;
      decay = mean(last `close_reaches`) - mean(first `evoke_window`), both
              post-evoke -- the reinstated memory fades, so this is negative.

    None when neither can be formed. Per seed throughout: each seed is its own
    control, since runs leave P- at different depths."""
    pre_level = _seed_mean(pre)
    early = _seed_mean(post[post.k <= evoke_window])
    late = _seed_mean(post[post.k > post.k.max() - close_reaches])
    jump, decay = (early - pre_level).dropna(), (late - early).dropna()
    if not len(jump) and not len(decay):
        return None
    return {"jump": jump.to_numpy(), "decay": decay.to_numpy()}


def evoked_recovery_stats(d, evoke_window: int = 10, close_reaches: int = 50):
    """The evoked jump and decay of one run's frame (needs an `index` column
    -- see adaptation_index). None when it is not an evoked-recovery run."""
    seg = _evoke_segments(d)
    return None if seg is None else _evoked_stats(*seg, evoke_window, close_reaches)


def _evoked_panels(axB, axC, runs, frames, channel_window, evoke_window, close_reaches):
    """Panels B and C of the evoked-recovery figure.

    Evoked recovery is not "the channel drifts up", which is what the
    spontaneous panels measure; it is two things, both keyed on the single
    evoking P+ reach (user, 2026-09-20):

      1. the expressed P+ memory is LOWER over the pre-evoke channel reaches
         than over the first few post-evoke ones -- one reach of P+ reinstates
         a memory that the channel alone was not expressing;
      2. it then DECAYS over the post-evoke channel, back toward the level the
         channel was sitting at before the evoke.

    B draws both: the index re-zeroed on each seed's own pre-evoke level,
    against reach relative to the evoke, smoothed WITHIN each segment so the
    step at the evoke survives. C is the same two effects as bars, one dot
    per seed (_evoked_stats).
    """
    axB.axhline(0, **_REF_KW)
    axB.axvline(0.5, color="#444", lw=1.0, ls="--", zorder=1)
    stats = {}
    for cfg, d in frames.items():
        seg = _evoke_segments(d)
        if seg is None:
            continue
        pre, post = seg
        col, ls = _model_style(cfg)
        pre_level = _seed_mean(pre)

        for part in (pre, post):
            part["rebound"] = part["index"] - part["seed"].map(pre_level)
            # smoothed inside the segment: a window that spans the evoke would
            # average the very step this panel exists to show
            part["smc"] = (part.sort_values(["seed", "k"]).groupby("seed")["rebound"]
                           .transform(lambda x: x.rolling(channel_window, min_periods=1,
                                                          center=True).mean()))
            gk = part.groupby("k")["smc"]
            gm, gsd = gk.mean(), gk.std()
            axB.plot(gm.index, gm.values, ls, color=col, lw=1.8)
            axB.fill_between(gm.index, (gm - gsd).values, (gm + gsd).values,
                             color=col, alpha=0.12, lw=0)

        run_stats = _evoked_stats(pre, post, evoke_window, close_reaches)
        if run_stats is not None:
            stats[cfg] = run_stats

    axB.set_xlabel("relative reach")
    axB.set_ylabel("index − pre-evoke channel level")
    axB.set_title(f"Evoked recovery",
                  fontsize=10)
    make_axis_nice(axB)

    _paired_bars(axC, runs, stats, ("jump", "decay"),
                 "adaptation index",
                 f"Evoked Recovery: jump (filled) and decay (open)",
                 refs=(0.0,), ylim=None)


# ── shared panels of the paradigm figures ───────────────────────────────────

_REF_KW = dict(color="#9a9a9a", lw=0.8, ls=":", zorder=0)


def _timecourse_panel(axA, data, runs, smooth_window, schedule="spontaneous_recovery"):
    """Panel A of every paradigm figure: adaptation index across the whole
    schedule, one trace per run (smoothed per seed, mean ± sem over seeds),
    blocks shaded and labelled, references at +1 / 0 / -1, legend below with
    one column per model. Returns ({cfg: frame with ep_in_run / index}, the
    set of readout columns used)."""
    for y in sorted(set(_reference_labels(schedule)) | {0.0}):
        axA.axhline(y, **_REF_KW)
    frames, readouts = {}, set()
    for cfg in runs:
        d = data[data.config == cfg].copy()
        d["ep_in_run"] = d.groupby("seed").cumcount()
        col_name = _readout_column(d); readouts.add(col_name)
        d["index"] = adaptation_index(d[col_name], schedule)
        col, ls = _model_style(cfg)
        # Smoothed WITHIN a block visit, not across the run: a centred window
        # that spans a block boundary reports a switch before it happens, and
        # the evoked schedule's evoke is a single reach that a 15-reach window
        # across the boundary would erase outright.
        within = ["seed", "phase"] if "phase" in d.columns else ["seed"]
        d["sm"] = (d.sort_values(["seed", "ep_in_run"]).groupby(within)["index"]
                    .transform(lambda x: x.rolling(smooth_window, min_periods=1, center=True).mean()))
        g = d.groupby("ep_in_run")["sm"]
        m, se = g.mean(), g.std() / np.sqrt(g.count().clip(lower=1))
        axA.plot(m.index, m.values, ls, color=col, lw=1.6, label=_model_label(cfg))
        axA.fill_between(m.index, (m - se).values, (m + se).values, color=col, alpha=0.12, lw=0)
        frames[cfg] = d
    # blocks, from one run's label sequence
    one = data[data.config == runs[0]]; one = one[one.seed == one.seed.iloc[0]].reset_index(drop=True)
    edges = list(one.index[one.label.ne(one.label.shift())]) + [len(one)]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if hi - lo <= 2:
            # the evoked schedule's evoke is one reach: a shaded span that
            # narrow is invisible, so mark it with a line and label it above
            axA.axvline(lo, color="#444", lw=1.0, ls="--", zorder=1)
            # along the line, low down: the block labels run across the top
            axA.text(lo, 0.03, one.label.iloc[lo], ha="right", va="bottom", fontsize=8,
                     rotation=90, color="#444", transform=axA.get_xaxis_transform())
            continue
        if i % 2:
            axA.axvspan(lo - 0.5, hi - 0.5, color="#000", alpha=0.04, lw=0)
        axA.text((lo + hi) / 2, 0.98, one.label.iloc[lo], ha="center", va="top", fontsize=9,
                 color="#444", transform=axA.get_xaxis_transform())
    axA.set_ylim(-1.5, 1.5); axA.set_xlim(-1, len(one))
    for y, lab in _reference_labels(schedule).items():
        axA.text(len(one) + 1, y, lab, va="center", fontsize=8, color="#777")
    axA.set_ylabel("adaptation index\n(+1 = full P+ compensation)")
    axA.set_xlabel("reach")
    what = ("first action, 8-direction probe per reach" if readouts == {READOUT_COLUMN}
            else "first action" if READOUT_COLUMN not in readouts
            else "first action; probe where logged, else the reach's own")
    axA.set_title(f"Adaptation index ({what}; {smooth_window}-reach "
                  f"smoothing per seed, mean ± sem over seeds)", fontsize=10)
    # legend: ONE COLUMN PER MODEL, replay above no-replay. matplotlib fills a
    # legend column-major, so order the entries model by model and pad a model
    # that has only one variant with an invisible entry.
    handles, labels = axA.get_legend_handles_labels()
    by_model = {}
    for h, lab in zip(handles, labels):
        model, _, variant = lab.partition(" · ")
        by_model.setdefault(model, {})[variant] = (h, lab)
    blank = Line2D([], [], color="none", label=" ")
    ordered_h, ordered_l = [], []
    for model, variants in by_model.items():
        for variant in ("replay", "no replay"):
            h, lab = variants.get(variant, (blank, " "))
            ordered_h.append(h); ordered_l.append(lab)
    axA.legend(ordered_h, ordered_l, frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=max(len(by_model), 1), columnspacing=2.0)
    make_axis_nice(axA)
    return frames, readouts


def _grouped_bars(ax, runs, values, ylabel, title, reference=None):
    """Panel C of the paradigm figures: one bar per run (mean over seeds,
    one dot per seed), grouped replay / no replay. `values`: cfg -> per-seed
    array; runs without a value are skipped. `reference`: cfg -> per-seed
    array of a comparison quantity, drawn as a grey × at its mean."""
    groups = [("replay", [c for c in runs if not _replay_off(c)]),
              ("no replay", [c for c in runs if _replay_off(c)])]
    x = 0; ticks, tlabels = [], []
    for gname, cfgs in groups:
        cfgs = [c for c in cfgs if c in values]
        if not cfgs:
            continue
        x0 = x
        for cfg in cfgs:
            col, _ = _model_style(cfg); v = np.asarray(values[cfg], dtype=float)
            ax.bar(x, v.mean(), width=0.7, color=col, alpha=0.85 if not _replay_off(cfg) else 0.45,
                   edgecolor=col, lw=1.2, ls="--" if _replay_off(cfg) else "-")
            ax.scatter(np.full(len(v), x) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color="#222", zorder=3)
            if reference and cfg in reference and len(reference[cfg]):
                ax.scatter([x], [np.mean(reference[cfg])], marker="x", s=60, color="#777",
                           lw=1.5, zorder=4)
            x += 1
        ticks.append((x0 + x - 1) / 2); tlabels.append(gname); x += 0.8
    ax.axhline(0, color="#9a9a9a", lw=0.8); ax.set_xticks(ticks); ax.set_xticklabels(tlabels)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    make_axis_nice(ax)


def _paired_bars(ax, runs, values, keys, ylabel, title, reference=None,
                 refs=(1, 0, -1), ylim=(-1.6, 1.6)):
    """Panel D of the paradigm figures: per run, up to two bars -- keys[0]
    filled, keys[1] open -- with one dot per seed. `values`: cfg -> {key:
    per-seed list} or None (skipped, leaving its slot). `reference`: cfg ->
    per-seed array of a comparison level, drawn as a grey × at its mean.

    `refs` / `ylim` are the index scale's landmarks; a panel of DIFFERENCES
    (the evoked jump and decay) passes refs=(0,) and ylim=None to autoscale,
    since ±1 means nothing there."""
    x = 0; ticks, tlabels = [], []
    for cfg in runs:
        pr = values.get(cfg)
        col, _ = _model_style(cfg)
        if pr is None:
            x += 1.4; continue
        if reference and cfg in reference and len(reference[cfg]):
            ax.scatter([x + 0.2], [np.mean(reference[cfg])], marker="x", s=60, color="#777",
                       lw=1.5, zorder=4)
        for j, key in enumerate(k for k in keys if k in pr):
            v = np.asarray(pr[key], dtype=float)
            ax.bar(x + 0.4 * j, v.mean(), width=0.38, color=col if j == 0 else "white",
                   edgecolor=col, lw=1.4, ls="--" if _replay_off(cfg) else "-")
            ax.scatter(np.full(len(v), x + 0.4 * j) + np.linspace(-0.08, 0.08, len(v)), v,
                       s=12, color="#222", zorder=3)
        ticks.append(x + 0.2); tlabels.append(_model_label(cfg).replace(" · ", "\n")); x += 1.4
    for y in refs:
        ax.axhline(y, **_REF_KW)
    ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=7)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    make_axis_nice(ax)


def _paradigm_figure():
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.32)
    return fig, (fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]),
                 fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))


def _schedule_fig_path(output_dir, schedule):
    return os.path.join(output_dir, "schedule_timecourse" + sac.SCHEDULE_SUFFIX[schedule])


def plot_schedule_timecourse(data: pd.DataFrame, output_dir: str, smooth_window: int = 15,
                             channel_window: int = 9, run_dir: str = None,
                             open_reaches: int = 25, close_reaches: int = 50,
                             evoke_window: int = 10,
                             schedule: str = "spontaneous_recovery"):
    """The channel-paradigm figure (spontaneous or evoked recovery), for the
    runs of ONE schedule -- file schedule_timecourse_sr / _er.

      A  adaptation index across the whole schedule (_timecourse_panel).
      B  SPONTANEOUS: the channel REBOUND -- each seed's channel index minus
         its own opening level (first `open_reaches` reaches), mean ± sd over
         seeds, so runs that left P- at different depths read on one scale.
         This is the human measure -- a partial drift back toward P+ -- and
         does not require the P+ memory to be intact, only expressed more
         than the P- one as the belief or prior shifts. (The observer's
         belief is logged per reach in curve["belief_pplus" /
         "belief_pminus"] but not drawn; see the memory of 2026-09-18.)
         EVOKED: the step at the evoke and the decay after it (_evoked_panels).
      C  SPONTANEOUS: recovery = mean index over the last `close_reaches`
         channel reaches minus the first `open_reaches`, one dot per seed,
         grouped replay-on / replay-off.
         EVOKED: the jump and the decay as paired bars (_evoked_panels).
      D  memory after the channel: the probe first action under an inferred
         P+ and under an inferred P- (targets 1..7). Intact memories put
         the two bars at opposite signs; an overwritten P+ puts both on the
         P- side.

    Colour = model, solid = replay, dashed = no replay. Only the first action
    is plotted: the reach-mean angle is dominated by braking. Each point is
    the per-reach readout probe's mean over 8 directions where the run logged
    it (_readout_column), otherwise the training reach's own first action.

    `evoke_window` is the post-evoke window the evoked jump is read over;
    its default matches sac.COIN_BLOCK_EP["evoke_after"], the length of the
    pre-evoke channel block it is compared against.
    """
    if "channel" not in data.columns or "label" not in data.columns:
        return
    runs = _paradigm_runs(data, schedule)
    if not runs:
        return
    run_dir = run_dir or BASE_FIG_DIR
    fig, (axA, axB, axC, axD) = _paradigm_figure()
    frames, _ = _timecourse_panel(axA, data, runs, smooth_window, schedule)
    if schedule == "evoked_recovery":
        _evoked_panels(axB, axC, runs, frames, channel_window, evoke_window, close_reaches)
    else:
        for y in (1, 0):
            axB.axhline(y, **_REF_KW)
        recovery = {}          # cfg -> per-seed delta
        for cfg, d in frames.items():
            col, ls = _model_style(cfg)
            # B: the channel, smoothed on its own (shorter window) within the block
            ch = d[d.channel == 1].copy()
            if not len(ch):
                continue
            ch["k"] = ch.groupby("seed").cumcount() + 1
            # B: the REBOUND -- each seed's channel index minus its own opening
            # level (first `open_reaches` reaches), so runs that left P- at
            # different depths are read on one scale: how far the channel drifts
            # back toward P+. The raw channel level stays visible in A.
            opening = ch[ch.k <= open_reaches].groupby("seed")["index"].mean()
            ch["rebound"] = ch["index"] - ch["seed"].map(opening)
            ch["smc"] = (ch.sort_values(["seed", "k"]).groupby("seed")["rebound"]
                         .transform(lambda x: x.rolling(channel_window, min_periods=1, center=True).mean()))
            gk = ch.groupby("k")["smc"]
            gm, gsd = gk.mean(), gk.std()
            axB.plot(gm.index, gm.values, ls, color=col, lw=1.8)
            axB.fill_between(gm.index, (gm - gsd).values, (gm + gsd).values, color=col, alpha=0.12, lw=0)
            closing = ch[ch.k > ch.k.max() - close_reaches].groupby("seed")["index"].mean()
            recovery[cfg] = (closing - opening).dropna().to_numpy()
        axB.set_ylim(-0.75, 1.5); axB.set_xlabel("channel reach")
        axB.set_ylabel(f"rebound: index − opening level\n(first {open_reaches} channel reaches)")
        axB.set_title(f"Channel rebound toward P+ (mean ± sd,\n{channel_window}-reach smoothing)",
                      fontsize=10)
        make_axis_nice(axB)
        _grouped_bars(axC, runs, recovery, "Recovery Index", "Spontaneous Recovery")
    # D: memory after the channel. Two sources, by what the run has:
    #   an inferred oracle -> the probe's first action under the probe
    #     observer's inferred P+ / P- (targets 1..7);
    #   an experience prior -> the CHANNEL first action conditioned on the
    #     context whose end state EC was dealt: the readout probe's per-source
    #     means (probe_fa_pplus / _pminus, eight draws per reach) where logged,
    #     else the training reaches' own draws (curve["prior_src"]).
    memory = {cfg: (_channel_by_prior_source(data[data.config == cfg])
                    or _after_channel_probes(cfg, run_dir)) for cfg in runs}
    _paired_bars(axD, runs, memory, ("P+", "P-"), "first action, adaptation index",
                 " Memory specificity: P+ (filled) / P− (open)")
    path = _schedule_fig_path(output_dir, schedule)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


def _pplus_visits(d):
    """The two P+ exposures of a savings-washout run: (label, [occurrence 0,
    occurrence 1]) -- the block labels start with 'P+ '."""
    labs = [l for l in d.label.unique() if str(l).startswith("P+ ")]
    if not labs:
        return None, []
    occs = sorted(d[d.label == labs[0]].occurrence.unique())
    return labs[0], occs


def _early_block(d, label, occurrence, n):
    """The first n reaches of one block visit, with k = 1..n per seed."""
    b = d[(d.label == label) & (d.occurrence == occurrence)].copy()
    b["k"] = b.groupby("seed").cumcount() + 1
    return b[b.k <= n]


def _late_block(d, label, occurrence, n):
    """The last n reaches of one block visit, per seed."""
    b = d[(d.label == label) & (d.occurrence == occurrence)].copy()
    b["k_from_end"] = b.groupby("seed").cumcount(ascending=False) + 1
    return b[b.k_from_end <= n]


def _base_label(d):
    """The baseline block's label of a savings-washout run: 'base …' (the
    trained baseline, sac.savings_pair) or the null-field label of runs from
    before it."""
    labs = [l for l in d.label.unique() if str(l).startswith("base ")]
    if labs:
        return labs[0]
    return sac.NULL_LABEL if (d.label == sac.NULL_LABEL).any() else None


def _around_switch(d, base_label, pplus_label, occurrence, before, after):
    """The reaches around one switch into P+, with k = -before+1..0 for the
    last `before` reaches of the block before it and k = 1..after for the
    first `after` reaches of P+ -- per seed, columns seed / k / index."""
    pre = _late_block(d, base_label, occurrence, before)
    pre = pre.assign(k=1 - pre.k_from_end)
    post = _early_block(d, pplus_label, occurrence, after)
    return pd.concat([pre[["seed", "k", "index"]], post[["seed", "k", "index"]]], ignore_index=True)


def _reexposure_jump(d, n):
    """Savings as the JUMP at re-exposure: the mean index over the first n
    reaches of P+2 minus the mean over the last n reaches of the washout, per
    seed -- and, as the reference a naive learner sets, the same jump at the
    FIRST exposure (first n of P+1 minus last n of the baseline block).
    Returns (jump2, jump1) as per-seed Series, or (None, None)."""
    label, occs = _pplus_visits(d)
    base = _base_label(d)
    if len(occs) < 2 or base is None:
        return None, None
    jumps = []
    for occ in occs[:2]:
        before = _late_block(d, base, occ, n).groupby("seed")["index"].mean()
        after = _early_block(d, label, occ, n).groupby("seed")["index"].mean()
        jumps.append((after - before).dropna())
    return jumps[1], jumps[0]


def plot_savings_washout(data: pd.DataFrame, output_dir: str, smooth_window: int = 15,
                         early_window: int = 9, early_reaches: int = 10,
                         jump_reaches: int = 5):
    """The savings figure (sac.SCHEDULES "savings_washout": base | P+ | base
    washout | P+, the base a TRAINED context, rot +30, and P+ rot +60 -- see
    sac.savings_pair), for the runs of that schedule alone -- file
    schedule_timecourse_savw. The index is relative to the base: 0 = full
    base compensation, +1 = full P+, -1 = aiming straight.

      A  adaptation index across the whole schedule (_timecourse_panel).
      B  the switch into P+, aligned at reach 0: the last `jump_reaches`
         reaches of the block before it (raw) and the first `early_reaches`
         of P+ (smoothed per seed on `early_window`), for the FIRST exposure
         (thin, faint) and the SECOND (full), mean ± sd over seeds. Savings
         is the second curve jumping where the first one climbs.
      C  savings = the JUMP at re-exposure: mean index over the first
         `jump_reaches` reaches of P+2 minus the mean over the last
         `jump_reaches` reaches of the washout, one dot per seed, grouped
         replay / no replay. The same jump at the first exposure (P+1 minus
         the end of the baseline block) is marked × on each bar: what a
         learner with nothing to retrieve does in its first reaches.
      D  savings proper: the jump at the second exposure minus the jump at
         the first, per seed, grouped replay / no replay -- the delta panel C
         only implies with its × markers.
    """
    if "label" not in data.columns or "occurrence" not in data.columns:
        return
    runs = _paradigm_runs(data, "savings_washout")
    if not runs:
        return
    fig, (axA, axB, axC, axD) = _paradigm_figure()
    frames, _ = _timecourse_panel(axA, data, runs, smooth_window, "savings_washout")
    for y in (1, 0, -1):
        axB.axhline(y, **_REF_KW)
    savings, reference, savings_delta = {}, {}, {}
    for cfg, d in frames.items():
        col, ls = _model_style(cfg)
        label, occs = _pplus_visits(d)
        if len(occs) < 2:
            continue
        base = _base_label(d)
        for j, occ in enumerate(occs[:2]):
            # B: the switch into P+, aligned at reach 0: the last `jump_reaches`
            # reaches of the block before (raw) and the first `early_reaches`
            # of P+ (smoothed on early_window, within the block only, so the
            # jump itself is not smeared)
            b = _around_switch(d, base, label, occ, jump_reaches, early_reaches)
            post = b.k > 0
            b["sm"] = b["index"]
            b.loc[post, "sm"] = (b[post].sort_values(["seed", "k"]).groupby("seed")["index"]
                                 .transform(lambda x: x.rolling(early_window, min_periods=1, center=True).mean()))
            gk = b.groupby("k")["sm"]; gm, gsd = gk.mean(), gk.std()
            axB.plot(gm.index, gm.values, ls, color=col, lw=1.8 if j else 1.0, alpha=1.0 if j else 0.45)
            if j:
                axB.fill_between(gm.index, (gm - gsd).values, (gm + gsd).values, color=col, alpha=0.12, lw=0)
        jump2, jump1 = _reexposure_jump(d, jump_reaches)
        if jump2 is not None:
            savings[cfg] = jump2.to_numpy(); reference[cfg] = jump1.to_numpy()
        # D: SAVINGS PROPER -- the jump at the second exposure minus the jump
        # at the first, per seed (user, 2026-09-18: the delta directly)
        if jump2 is not None:
            delta = (jump2 - jump1).dropna()
            if len(delta):
                savings_delta[cfg] = delta.to_numpy()
    axB.axvline(0.5, color="#9a9a9a", lw=0.8)
    axB.set_ylim(-1.5, 1.5); axB.set_xlim(-jump_reaches + 0.5, early_reaches + 0.5)
    axB.set_xlabel("episodes relative to switch"); axB.set_ylabel("adaptation index")
    axB.set_title(f"Into P+: exposure 1 (faint) vs 2 (full)\n(last {jump_reaches} reaches before, "
                  f"raw; after, {early_window}-reach smoothing; mean ± sd)", fontsize=10)
    make_axis_nice(axB)
    _grouped_bars(axC, runs, savings,
                  f"jump at re-exposure: Δ index, first {jump_reaches} P+2 reaches − last "
                  f"{jump_reaches} washout reaches", "Jump at re-exposure\n(×: first-exposure baseline)",
                  reference=reference)
    _grouped_bars(axD, runs, savings_delta,
                  f"savings: jump at exposure 2 − jump at exposure 1 ({jump_reaches}-reach windows)",
                  "Savings\n(second-exposure jump minus first)")
    path = _schedule_fig_path(output_dir, "savings_washout")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {path}")


def plot_retention_vs_savings(data: pd.DataFrame, metrics: list, output_dir: str,
                              onset_fraction: float = ONSET_FRACTION,
                              drop_first_block: bool = False,
                              smooth_window: int = 10):
    """Left: within-block traces, first visit (dashed) against revisit
    (solid), one panel per config, with the onset window shaded. Right: every
    config as a point in (retention, savings) space, per-seed points faint,
    seed mean with s.e. bars. Retention-only configs sit on the x axis,
    savings-only configs on the y axis; that separation is the figure.

    `smooth_window` is short (10) rather than TRACE_SMOOTH_WINDOW (25) because
    blocks here can be 100 episodes and a 25-episode 'valid' convolution
    would drop the onset window entirely.
    """
    for metric in metrics:
        if metric not in data.columns:
            continue
        table = retention_savings_table(data, metric, onset_fraction, drop_first_block)
        if table.empty:
            print(f"retention_vs_savings[{metric}]: no run has both a first visit and a revisit")
            continue
        configs = [c for c in get_config_order(data) if c in set(table["config"])]
        n = len(configs)

        fig = plt.figure(figsize=(1.7 * n + 3.4, 2.5))
        gs = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [1.9], wspace=0.45)
        trace_axes = []
        for i, config in enumerate(configs):
            ax = fig.add_subplot(gs[0, i], sharey=trace_axes[0] if trace_axes else None)
            trace_axes.append(ax)
            group = data[data["config"] == config]
            colour = CONFIG_COLORS.get(config, "grey")
            ax.axvspan(0, onset_fraction * _block_length(group), color="0.92", lw=0, zorder=0)
            for revisit, ls, lab in ((False, "--", "first visit"), (True, "-", "revisit")):
                band = _visit_band(group, metric, revisit, smooth_window)
                if band.empty:
                    continue
                ax.plot(band.index, band["median"], ls=ls, color=colour, lw=1.2, label=lab)
                ax.fill_between(band.index, band["median"] - band["std"],
                                band["median"] + band["std"], color=colour,
                                alpha=0.12 if revisit else 0.06, lw=0)
            ax.set_title(config.replace("_", " "), fontsize=6)
            ax.set_xlabel("episode in block")
            if i == 0:
                ax.set_ylabel(metric)
                ax.legend(frameon=False, loc="lower right", fontsize=5)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            make_axis_nice(ax)

        ax = fig.add_subplot(gs[0, n])
        ax.axhline(0, color="0.75", lw=0.6, zorder=0)
        ax.axvline(0, color="0.75", lw=0.6, zorder=0)
        for config in configs:
            rows = table[table["config"] == config]
            colour = CONFIG_COLORS.get(config, "grey")
            ax.scatter(rows["retention"], rows["savings"], s=8, color=colour,
                       alpha=0.45, lw=0, zorder=2)
            m = rows[["retention", "savings"]].mean()
            se = rows[["retention", "savings"]].std(ddof=1) / np.sqrt(len(rows)) \
                if len(rows) > 1 else pd.Series({"retention": 0.0, "savings": 0.0})
            ax.errorbar(m["retention"], m["savings"], xerr=se["retention"], yerr=se["savings"],
                        fmt="o", ms=4.5, color=colour, ecolor=colour, elinewidth=0.8,
                        capsize=0, mec="black", mew=0.3, zorder=3,
                        label=f"{config.replace('_', ' ')} (n={len(rows)})")
        ax.set_xlabel(f"retention: level at ep 0, revisit $-$ first  [{metric}]")
        ax.set_ylabel(f"savings: gain(revisit) $-$ gain(first)  [{metric}]")
        ax.legend(frameon=False, fontsize=5, loc="best")
        make_axis_nice(ax)

        fig.suptitle(f"retention (level) vs savings (rate); onset = first "
                     f"{onset_fraction:.0%} of block; unit = seed", fontsize=7, y=1.04)
        output_path = os.path.join(output_dir, f"retention_vs_savings_{metric}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def main():
    sns.set_style("ticks")

    print(f"Loading aggregated results from {BASE_FIG_DIR}...")
    data = load_all_configs(BASE_FIG_DIR)

    if data is None or data.empty:
        print(f"No aggregated_results.csv files found in {BASE_FIG_DIR}")
        return

    data = build_phase_occurrence_label(data)

    print("Fitting learning curves...")
    fit_data = fit_curves_per_phase(data, METRICS)
    fit_data_occur = fit_curves_per_occurrence(data, METRICS)

    print(f"\nCreating plots in {PLOT_OUTPUT_DIR}...")
    # plot_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_learning_curves_by_phase_occurrence(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_learning_curves_by_occurrence(data, METRICS, PLOT_OUTPUT_DIR)
    # plot_episodic_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)

    # plot_phase_learning_curves(data, METRICS, PLOT_OUTPUT_DIR)

    # plot_fitted_curves(data, fit_data, METRICS, PLOT_OUTPUT_DIR)
    # plot_fitted_curves_by_occurrence(data, fit_data_occur, METRICS, PLOT_OUTPUT_DIR)

    # plot_curve_parameters(fit_data, METRICS, PLOT_OUTPUT_DIR)
    plot_metric_traces(data, METRICS, PLOT_OUTPUT_DIR)
    plot_metric_traces_by_phase_visit(data, METRICS, PLOT_OUTPUT_DIR)
    plot_criterion_crossing(data, METRICS, PLOT_OUTPUT_DIR)
    plot_criterion_crossing_by_phase_occurrence(data, METRICS, PLOT_OUTPUT_DIR)
    report_criterion_trace_consistency(data, METRICS, PLOT_OUTPUT_DIR)
    plot_retention_vs_savings(data, METRICS, PLOT_OUTPUT_DIR)
    # the COIN-paradigm figures, one per schedule; each is a no-op without runs
    for schedule in ("spontaneous_recovery", "evoked_recovery"):
        plot_schedule_timecourse(data, PLOT_OUTPUT_DIR, schedule=schedule)
    plot_savings_washout(data, PLOT_OUTPUT_DIR)

    # plot_curve_parameters_per_occurrence(fit_data_occur, METRICS, PLOT_OUTPUT_DIR)
    print(f"\nDone! All plots saved to {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
