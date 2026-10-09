"""
plotting_utils.py -- the paper figure style and the helpers that apply it.

    use_paper_style()            paper.mplstyle + bundled Source Sans 3
    figsize("single"|"double")   final print width, so type is never rescaled
    offset_spines(fig)           offset spines that end on few, round ticks
    label_panels(fig)            bold a, b, c ...
    label_lines(ax)              labels at line ends instead of a legend
    chance_line(ax, y, "chance") recessive reference line, label on the line
    plot_band(ax, x, m, s)       mean line + flat uncertainty band
    save(fig, path)              PDF + PNG together

    swarm_ci(ax, groups)         jittered points + mean and 95% CI, no bar
    colorbar(im, ax, label)      thin, three ticks, no outline
    cmaps "paper_seq"/"paper_div" one sequential and one diverging ramp

STATUS of the 2026-09-21 improvement list. Done as helpers/style: 3 (line
labels), 4 (few round ticks), 6 (reference lines, bands), 7 (swarm + CI, in
make_ablations_plot._grouped_bars/_paired_bars), 8 (paper_seq/paper_div,
colorbar; applied in representation_analysis.fig_confusion), 9 (save PDF+PNG
+ caption file), 10 (palette re-tuned and checked with colorspacious: teal
#238b7e, amber #edb060, grey dropped from the cycle).
Per-figure items 1 (build at figsize), 2 (no fontsize= overrides, prose to
the caption) and 5 (shared y, one label per row) are applied to the four
paper figures so far: figures.fig_ablation, representation_analysis.
fig_paper_pca_z, make_ablations_plot.plot_savings_washout and
plot_recovery_comparison (the paradigm frame _paradigm_figure and
_timecourse_panel came with them, so plot_schedule_timecourse is at print
size too), and both figures in multimodality.py. Apply the same to each
further figure as it is chosen for the paper. Titles stay centred (decided
2026-09-21).

`set_plt_params` and `make_axis_nice` at the bottom are the pre-2026-09-21
helpers, kept only for reference; nothing calls them.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paper style: paper.mplstyle next to this file, plus bundled fonts and cmaps.
# ---------------------------------------------------------------------------
PAPER_STYLE = Path(__file__).with_name("paper.mplstyle")

# Fixed categorical order, matching axes.prop_cycle in paper.mplstyle. Colour
# follows the entity: assign these by name, never by rank in a filtered list.
PALETTE = {
    "navy":   "#0e3b5c",
    "salmon": "#d8897b",
    "teal":   "#238b7e",
    "amber":  "#edb060",
    "violet": "#5e4f6d",
    "rose":   "#c25b6b",
    "grey":   "#8a8f94",
}
COLORS = list(PALETTE.values())
INK = "#1c1c1c"          # text, spines, ticks
MUTED = "#8a8f94"        # chance lines, reference guides, de-emphasised series
SHADE = "#eef0f2"        # phase bands, background regions

# Journal column widths in inches (Nature/Cell/eLife are all within a mm of these).
COLUMN_WIDTH = {"single": 3.5, "onehalf": 5.0, "double": 7.2}


FONT_DIR = Path(__file__).with_name("fonts")   # bundled Source Sans 3 (OFL)


def register_fonts() -> None:
    """Make the bundled fonts visible to matplotlib without installing them."""
    from matplotlib import font_manager
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))


def register_cmaps() -> None:
    """The paper's two colour ramps, registered by name so they work anywhere a
    cmap string does (imshow, scatter, seaborn):

        paper_seq   magnitude: near-white -> navy, one hue, light -> dark
        paper_div   polarity:  rose <- near-white -> navy, neutral at zero
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    near_white = "#f5f6f8"
    seq = LinearSegmentedColormap.from_list("paper_seq", [near_white, "#8fb0c9", PALETTE["navy"]])
    div = LinearSegmentedColormap.from_list("paper_div", [PALETTE["rose"], near_white, PALETTE["navy"]])
    for cm in (seq, div, seq.reversed(), div.reversed()):
        if cm.name not in matplotlib.colormaps:      # use_paper_style() may run twice
            matplotlib.colormaps.register(cm)


def use_paper_style(**overrides) -> None:
    """Reset rcParams, register the bundled fonts and colormaps, apply
    paper.mplstyle, then any rcParams overrides
    (e.g. use_paper_style(**{"font.size": 8}))."""
    plt.rcdefaults()
    register_fonts()
    register_cmaps()
    plt.style.use(str(PAPER_STYLE))
    plt.rcParams.update(overrides)


def figsize(width: Union[str, float] = "single", aspect: float = 0.62,
            height: Optional[float] = None) -> Tuple[float, float]:
    """Figure size in inches at final print width. `width` is a COLUMN_WIDTH key
    or a width in inches; `aspect` is height/width (golden ratio by default)
    unless `height` is given explicitly."""
    w = COLUMN_WIDTH.get(width, width) if isinstance(width, str) else width
    return (w, height if height is not None else w * aspect)


# Tick policy for trim_spines: as few round ticks as still read well.
MIN_TICKS, MAX_TICKS = 3, 7
MAX_OVERSHOOT = 0.2      # extension beyond the data, as a fraction of its range
EXTEND_TOL = 0.02        # data past the outer tick by less than this fraction of
                         # the view is not worth an extra tick step


def _fmt_g(v, _pos=None) -> str:
    """Tick text without trailing zeros: 0, 0.5, 1 rather than 0.0, 0.5, 1.0."""
    return f"{v:g}"


def _nice_steps(step: float):
    """Round tick steps (1, 2, 5 x 10^k) from a decade below to a decade above
    `step`, ascending."""
    import numpy as np
    e = np.floor(np.log10(step))
    return sorted({m * 10.0 ** (e + k) for k in (-1, 0, 1) for m in (1, 2, 5)})


def _retickable(axis) -> bool:
    from matplotlib.category import StrCategoryFormatter, StrCategoryLocator
    from matplotlib.ticker import FixedFormatter, FixedLocator
    if axis.get_scale() != "linear":
        return False
    if isinstance(axis.get_major_locator(), FixedLocator) or \
            isinstance(axis.get_major_formatter(), FixedFormatter):
        return False                     # user-set ticks / category labels
    # A categorical axis matplotlib built through its unit machinery (what
    # seaborn's violin/box/strip plots leave behind): the positions are the
    # category indices and the formatter labels a position by ROUNDING it, so
    # re-ticking at 0, 0.5, 1, ... would read "0 0 0 1 2 2 2". Trim only.
    if isinstance(axis.get_major_locator(), StrCategoryLocator) or \
            isinstance(axis.get_major_formatter(), StrCategoryFormatter):
        return False
    return len(axis.get_majorticklocs()) >= 2


def _choose_ticks(axis, lo: float, hi: float, enclose: bool, data=None):
    """Round ticks for the interval [lo, hi].

    `enclose=True` (autoscaled axis): ticks may extend past the data so the
    spine encloses it; `enclose=False` (explicit limits): ticks stay inside.
    Among steps giving MIN_TICKS..MAX_TICKS ticks and (when enclosing) at most
    MAX_OVERSHOOT extension, take the coarsest -- so 0..1 gets 0, 0.5, 1 and
    0..256 epochs gets 0, 100, 200, 300. If nothing satisfies both, take the
    step with the least extension. None if the axis can't be re-ticked.

    With explicit limits, `data=(dlo, dhi)` (the data on show) makes a step
    whose inside ticks already reach past the data win over a coarser one that
    stops short: a +-1.5 frame holding values up to 1.02 gets ticks every 0.5
    rather than -1, 0, 1 with the spine ending inside the data."""
    import numpy as np
    # an axes with nothing plotted (a twinx partner used for a label, a panel
    # that only holds text) has an inverted, infinite data interval
    if not _retickable(axis) or not (np.isfinite(lo) and np.isfinite(hi)) or not hi > lo:
        return None
    locs = np.asarray(axis.get_majorticklocs())
    step0 = float(np.median(np.diff(locs)))
    if not np.isfinite(step0) or step0 <= 0:
        return None

    def ticks_for(step):
        if enclose:
            first, last = np.floor(lo / step + 1e-9), np.ceil(hi / step - 1e-9)
        else:
            first, last = np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9)
        n = int(last - first) + 1
        return np.round((first + np.arange(n)) * step, 10) if n >= 2 else None

    ok, fallback = [], []
    for step in _nice_steps(step0):
        t = ticks_for(step)
        if t is None or not (MIN_TICKS <= len(t) <= MAX_TICKS):
            continue
        over = ((lo - t[0]) + (t[-1] - hi)) / (hi - lo) if enclose else 0.0
        (ok if over <= MAX_OVERSHOOT else fallback).append((over, len(t), t))
    if ok and not enclose and data is not None:
        dlo, dhi = data
        tol = EXTEND_TOL * (hi - lo)
        holds = [r for r in ok if r[2][0] <= dlo + tol and r[2][-1] >= dhi - tol]
        if holds:
            return min(holds, key=lambda r: r[1])[2]   # coarsest that holds the data
    if ok:
        return min(ok, key=lambda r: r[1])[2]          # coarsest acceptable
    if fallback:
        return min(fallback, key=lambda r: r[0])[2]    # least extension
    return None


def _label_style(axis):
    """Rotation/alignment of an axis' tick labels, so they can be put back:
    Spine.set_position and Axis.set_ticks both rebuild the tick objects and
    silently drop whatever set_xticklabels(rotation=..., ha=...) applied."""
    labs = axis.get_ticklabels()
    if not labs:
        return None
    t = labs[0]
    return dict(rotation=t.get_rotation(), ha=t.get_ha(), va=t.get_va(),
                rotation_mode=t.get_rotation_mode())


def _restore_label_style(axis, style) -> None:
    if style:
        for t in axis.get_ticklabels():
            t.set(**style)


def trim_spines(ax: Union[plt.Axes, plt.Figure, None] = None,
                spines: Optional[List[str]] = None, extend: bool = True) -> None:
    """End each spine on a tick rather than at the axis limit, with few, round
    ticks (see _choose_ticks).

    With `extend` (default), an autoscaled axis is snapped outward to round
    ticks enclosing its data and the view limits grow to include them (the
    axes margins are kept): data reaching x = 11.3 gets a 12 tick and a spine
    that ends there, never a spine cut short of the data. An axis with explicit
    limits keeps them and is re-ticked inside them: ylim(0, 1) reads 0, 0.5, 1.

    Categorical or log axes are only trimmed to their existing outermost ticks
    (seaborn's despine(trim=True)). Call after the limits are final."""
    if ax is None:
        ax = plt.gca()
    axes = ax.axes if isinstance(ax, plt.Figure) else [ax]
    for a in axes:
        for s in (spines or ["left", "bottom"]):
            if not a.spines[s].get_visible():
                continue
            horizontal = s in ("bottom", "top")
            axis = a.xaxis if horizontal else a.yaxis
            style = _label_style(axis)
            try:
                _trim_one(a, axis, s, horizontal, extend)
            finally:
                _restore_label_style(axis, style)


def _trim_one(a, axis, s, horizontal, extend):
    """trim_spines for one spine `s` of axes `a` (see trim_spines)."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    set_lim = a.set_xlim if horizontal else a.set_ylim
    auto = a.get_autoscalex_on() if horizontal else a.get_autoscaley_on()
    margin = a.margins()[0 if horizontal else 1]
    ticks = None

    def data_interval():
        # the UNION over axes that share this axis: a shared axis must enclose
        # every sibling's data, not just this panel's (a row of swarms with
        # sharey otherwise clipped the panels whose values exceed the first's)
        import numpy as np
        group = (a.get_shared_x_axes() if horizontal else a.get_shared_y_axes())
        sibs = list(group.get_siblings(a)) or [a]
        ivs = [sorted((s_.xaxis if horizontal else s_.yaxis).get_data_interval()) for s_ in sibs]
        ivs = [iv for iv in ivs if np.all(np.isfinite(iv)) and iv[1] >= iv[0]]
        if not ivs:
            return sorted(axis.get_data_interval())
        return min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)

    if extend and auto:
        dlo, dhi = data_interval()
        ticks = _choose_ticks(axis, dlo, dhi, enclose=True)
        if ticks is not None:
            pad = margin * (ticks[-1] - ticks[0])
            set_lim(ticks[0] - pad, ticks[-1] + pad)
    if ticks is None:
        lo, hi = sorted(axis.get_view_interval())
        dlo, dhi = data_interval()
        dlo, dhi = max(dlo, lo), min(dhi, hi)          # the data on show
        ticks = _choose_ticks(axis, lo, hi, enclose=False, data=(dlo, dhi))
        if ticks is not None:
            # Explicit limits: ticks sit inside them, but the spine must still
            # enclose the data on show. If no step's inside ticks manage that,
            # add one more step on the failing side and widen the limit to it:
            # reach data to 770 with ticks by 200 gets an 800; data at -4 with
            # ticks 0, 5, 10 gets a -5.
            # ... unless the overshoot is negligible (a shaded span ending half
            # a reach past the last point must not buy a whole extra step)
            ticks = list(ticks)
            step = ticks[1] - ticks[0]
            tol = EXTEND_TOL * (hi - lo)
            if dlo < ticks[0] - tol:
                ticks.insert(0, round(ticks[0] - step, 10))
            if dhi > ticks[-1] + tol:
                ticks.append(round(ticks[-1] + step, 10))
            if ticks[0] < lo or ticks[-1] > hi:
                set_lim(min(lo, ticks[0]), max(hi, ticks[-1]))
    if ticks is None:                                   # categorical / log: trim only
        lo, hi = sorted(axis.get_view_interval())
        ticks = [t for t in axis.get_majorticklocs() if lo <= t <= hi]
        if len(ticks) >= 2:
            a.spines[s].set_bounds(ticks[0], ticks[-1])
        return
    if type(axis.get_major_formatter()) is ScalarFormatter:
        axis.set_major_formatter(FuncFormatter(_fmt_g))
    axis.set_ticks(list(ticks))
    a.spines[s].set_bounds(ticks[0], ticks[-1])


def offset_spines(ax: Union[plt.Axes, plt.Figure, None] = None, offset: float = 4,
                  spines: Optional[List[str]] = None, trim: bool = True) -> None:
    """Move the left/bottom spines outward so they don't touch the data, and
    (by default) end them on ticks that enclose the data (see trim_spines). A
    4-pt offset suits the 7-pt type in paper.mplstyle (make_axis_nice's 10 was
    tuned for the older, larger figures)."""
    if ax is None:
        ax = plt.gca()
    axes = ax.axes if isinstance(ax, plt.Figure) else [ax]
    for a in axes:
        for s in (spines or ["left", "bottom"]):
            axis = a.xaxis if s in ("bottom", "top") else a.yaxis
            style = _label_style(axis)       # set_position rebuilds the tick labels
            a.spines[s].set_position(("outward", offset))
            _restore_label_style(axis, style)
    if trim:
        trim_spines(ax, spines)


def bracket_xticks(ax: Union[plt.Axes, plt.Figure, None] = None, half_width: float = 0.4,
                   offset: float = 4, arm: float = 3, drop: float = 3, gap: float = 2,
                   positions=None, labels=None, color=None, lw=None) -> None:
    """Replace a grouped categorical x axis's ticks with BRACKETS: under each
    category a horizontal bar spanning `half_width` either side of its centre
    (data units -- seaborn's `width` / 2, 0.4 for its default 0.8), with its
    ends turned up toward the data (`arm` pt) and a tick down from its middle
    (`drop` pt) to the label. So a group of hue-split violins or bars reads as
    one bracketed unit rather than a row of shapes above a lone tick.

    The bar sits `offset` pt below the axes, where offset_spines puts the
    spine; the bottom spine is hidden since the brackets replace it, and the
    ticks become fixed so trim_spines leaves them alone. Call after the layout
    and after offset_spines. `positions` / `labels` default to the axis's own
    (a seaborn categorical axis: 0, 1, ... and the category names). On a
    figure, every axes is done; axes whose tick labels are hidden (a
    FacetGrid's inner rows) keep them hidden.

    Everything vertical is in points (the arms and the drop are line MARKERS
    of that size), so the bracket keeps its shape whatever the axes' aspect
    or the figure's size."""
    from matplotlib.lines import Line2D, TICKDOWN, TICKUP
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    if ax is None:
        ax = plt.gca()
    axes = ax.axes if isinstance(ax, plt.Figure) else [ax]
    color = color if color is not None else plt.rcParams["axes.edgecolor"]
    lw = lw if lw is not None else plt.rcParams["axes.linewidth"]
    for a in axes:
        if not a.xaxis.get_visible():             # a twinx partner: its host draws
            continue
        pos = list(a.get_xticks()) if positions is None else list(positions)
        labs = (a.xaxis.get_major_formatter().format_ticks(pos) if labels is None
                else list(labels))
        shown = any(t.get_visible() for t in a.xaxis.get_ticklabels())
        fig = a.figure
        # data x, axes y, then `offset` pt straight down: y = 0 is the axes'
        # bottom edge whatever the y limits do afterwards
        tr = (blended_transform_factory(a.transData, a.transAxes)
              + ScaledTranslation(0, -offset / 72.0, fig.dpi_scale_trans))
        style = dict(color=color, transform=tr, clip_on=False, zorder=2.5)
        for c in pos:
            a.add_line(Line2D([c - half_width, c + half_width], [0, 0], lw=lw,
                              solid_capstyle="butt", **style))
            a.add_line(Line2D([c - half_width, c + half_width], [0, 0], linestyle="none",
                              marker=TICKUP, markersize=arm, markeredgewidth=lw, **style))
            a.add_line(Line2D([c], [0], linestyle="none", marker=TICKDOWN, markersize=drop,
                              markeredgewidth=lw, **style))
        a.spines["bottom"].set_visible(False)
        a.set_xticks(pos, labs)                      # fixed: trim_spines skips it
        a.tick_params(axis="x", which="both", length=0, pad=offset + drop + gap,
                      labelbottom=shown)


def label_panels(fig: plt.Figure, axes=None, x: float = -0.12, y: float = 1.05,
                 size: float = 9, weight: str = "bold", start: str = "a",
                 align: bool = False) -> None:
    """Letter each axes (a, b, c, ...) in the top-left, Nature style.

    `align=True` puts every letter on ONE figure x, the left edge of the
    leftmost axes (after a layout pass) minus the usual offset, so a wide
    panel whose long y label pushes its axes box right does not carry its
    letter with it (the probe figure's score panel under four trace columns).
    The y stays per axes."""
    axes = list(fig.axes if axes is None else axes)
    if align and axes:
        # Letters first at their usual per-axes spot (so the layout reserves
        # room for them), then after a layout pass moved to one common x: the
        # left edge of the leftmost axes minus that axes' usual offset. Each
        # letter stays in its own axes' coordinates.
        texts = [a.text(x, y, chr(ord(start) + i), transform=a.transAxes, size=size,
                        weight=weight, ha="right", va="bottom") for i, a in enumerate(axes)]
        fig.canvas.draw()
        pos = [a.get_position() for a in axes]
        lead = min(range(len(axes)), key=lambda i: pos[i].x0)
        common = pos[lead].x0 + x * pos[lead].width          # figure fraction
        for a, p_, t in zip(axes, pos, texts):
            t.set_x((common - p_.x0) / p_.width)
        return
    for i, a in enumerate(axes):
        a.text(x, y, chr(ord(start) + i), transform=a.transAxes, size=size,
               weight=weight, ha="right", va="bottom")


def set_plt_params(font_size: int = 6, legend_size: int = 6, axes_label_size: int = 6, linewidth: float = 1.5,
                    axes_title_size: int = 6, xtick_label_size: int = 5, ytick_label_size: int = 5,
                    ticksize: float = 2, path_simplify: bool = False,
                    fig_title_size: int = 7, style: str = 'ggplot', font: str = 'avenir',
                    backend: str = 'svg', file_format: str = 'svg', fig_dpi: int = 500, figsize: Tuple[float, float] = (11, 8),
                    auto_method: str = None, x_margin: float = None, y_margin: float = None,
                    render_path: bool = False, marker_size: float = 2):
    """Set global rcParams for a consistent plot style."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams['savefig.format'] = file_format
    plt.rcParams['backend'] = backend
    plt.rcParams['savefig.dpi'] = fig_dpi
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['path.simplify'] = path_simplify
    plt.rcParams['lines.markersize'] = marker_size
    if auto_method is not None:
        plt.rcParams['axes.autolimit_mode'] = auto_method
    if x_margin is not None:
        assert 0 <= x_margin <= 1
        plt.rcParams['axes.xmargin'] = x_margin
    if y_margin is not None:
        assert 0 <= y_margin <= 1
        plt.rcParams['axes.ymargin'] = y_margin

    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=axes_title_size)
    plt.rc('axes', labelsize=axes_label_size)
    plt.rc('xtick', labelsize=xtick_label_size)
    plt.rc('ytick', labelsize=ytick_label_size)
    plt.rc('legend', fontsize=legend_size)
    plt.rc('figure', titlesize=fig_title_size)
    plt.rc('lines', linewidth=linewidth, markersize=marker_size)
    plt.rcParams['font.family'] = font
    plt.rcParams['xtick.major.size'] = ticksize
    plt.rcParams['ytick.major.size'] = ticksize
    plt.rcParams['xtick.minor.size'] = ticksize / 3
    plt.rcParams['ytick.minor.size'] = ticksize / 3

    if not render_path:
        plt.rcParams['svg.fonttype'] = 'none'

    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True


def make_axis_nice(ax: Union[plt.Axes, plt.Figure] = None, offset: int = 10,
                    line_width: float = 0.5, spines: Optional[List] = None, color: str = None):
    """Offset the given spines outward and thin them, so axes don't meet at
    the origin. `ax` may be a Figure, in which case every axes in it is
    modified."""
    if ax is None:
        ax = plt.gca()

    if spines is None:
        spines = ['left', 'bottom']
    ax_list = ax.axes if isinstance(ax, plt.Figure) else [ax]

    for a in ax_list:
        for spine in spines:
            a.spines[spine].set_linewidth(line_width)
            if color is not None:
                a.spines[spine].set_color(color)
            a.spines[spine].set_position(('outward', offset))


# ---------------------------------------------------------------------------
# Marks and annotations: the conventions every paper figure should share.
# ---------------------------------------------------------------------------
def _pt(size) -> float:
    """A matplotlib font size (number or name like 'small') in points."""
    from matplotlib.font_manager import FontProperties
    return FontProperties(size=size).get_size_in_points()


def label_lines(ax: plt.Axes, lines=None, pad_pt: float = 3, min_gap_pt: Optional[float] = None,
                fontsize=None, hide_legend: bool = True):
    """Write each line's label at its right end, in the line's colour, instead
    of a legend. Labels that would overlap are pushed apart vertically (the
    group keeps its centre) while the text's anchor stays at the line end.
    Lines whose label starts with '_' are skipped. Returns the Text objects.
    Call after the limits are final."""
    import numpy as np
    lines = [l for l in (lines if lines is not None else ax.get_lines())
             if l.get_label() and not str(l.get_label()).startswith("_")]
    if not lines:
        return []
    fs = _pt(fontsize if fontsize is not None else plt.rcParams["legend.fontsize"])
    gap_px = (min_gap_pt if min_gap_pt is not None else fs * 1.15) * ax.figure.dpi / 72

    ends = []
    for l in lines:
        x, y = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return []
        ends.append((x[ok][-1], y[ok][-1]))
    ends = np.array(ends)
    disp = ax.transData.transform(ends)
    order = np.argsort(disp[:, 1])
    ys = disp[order, 1].copy()
    for i in range(1, len(ys)):                       # push up to clear the one below
        ys[i] = max(ys[i], ys[i - 1] + gap_px)
    ys += disp[order, 1].mean() - ys.mean()           # re-centre the group
    shifted = np.empty_like(ys)
    shifted[order] = ys

    texts = []
    for l, (x, y), y_new, y_old in zip(lines, ends, shifted, disp[:, 1]):
        dy_pt = (y_new - y_old) * 72 / ax.figure.dpi
        texts.append(ax.annotate(l.get_label(), xy=(x, y), xytext=(pad_pt, dy_pt),
                                 textcoords="offset points", ha="left", va="center",
                                 color=l.get_color(), fontsize=fs, clip_on=False,
                                 annotation_clip=False))
    if hide_legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    return texts


def chance_line(ax: plt.Axes, y: float, text: Optional[str] = None, color: str = MUTED,
                ls: str = ":", lw: float = 0.8, x: float = 1.0, above: bool = True,
                fontsize=None, **kw):
    """A recessive horizontal reference (chance, ceiling, zero) with its label
    sitting on the line at the right edge, in the same grey. `x` is in axes
    fraction; `above` puts the text over the line rather than under it."""
    line = ax.axhline(y, color=color, ls=ls, lw=lw, zorder=0.5, **kw)
    if text:
        fs = _pt(fontsize if fontsize is not None else plt.rcParams["xtick.labelsize"])
        ax.annotate(text, xy=(x, y), xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 1.5 if above else -1.5), textcoords="offset points",
                    ha="right", va="bottom" if above else "top", color=color, fontsize=fs)
    return line


def plot_band(ax: plt.Axes, x, m, s=None, lo=None, hi=None, color=None, label=None,
              alpha: float = 0.2, marker="auto", max_marker_points: int = 15, **kw):
    """A mean line with its uncertainty as a flat band: same hue, alpha 0.2, no
    edge. `s` is a half-width (scalar or array) or give `lo`/`hi` directly.
    Markers are drawn only when there are <= `max_marker_points` points, so
    sparse curves get dots and dense ones don't. Returns the Line2D."""
    import numpy as np
    x, m = np.asarray(x, float), np.asarray(m, float)
    line, = ax.plot(x, m, color=color, label=label, **kw)
    if marker == "auto":
        marker = "o" if len(x) <= max_marker_points else None
    if marker:
        line.set_marker(marker)
    if s is not None:
        lo, hi = m - np.asarray(s, float), m + np.asarray(s, float)
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=alpha, lw=0,
                        zorder=line.get_zorder() - 0.1)
    return line


def save(fig: plt.Figure, path, formats=("pdf", "png"), close: bool = True,
         caption: Optional[str] = None, **kw) -> List[Path]:
    """Write `path` in every format and its caption, creating the directory.

    PDF is the manuscript master (fixed page size, fonts embedded, accepted by
    journals); PNG is for looking at. Add "svg" only for hand-editing in
    Illustrator/Inkscape -- SVG references the font by name, so it renders
    with a substitute wherever Source Sans 3 is not installed. Any extension
    on `path` is replaced.

    EVERY figure gets a caption file, <stem>_caption.txt: the standard record
    of what a figure shows and how it was computed (user, 2026-09-21). The
    text is, in order of preference: `caption`; the figure's suptitle (where
    the older figures keep their prose); the docstring of the function that
    called save() (this repo's plot functions describe their panels there);
    the panel titles. The file's first line names which of these it is, so a
    hand-written caption is distinguishable from a fallback. If none exists a
    warning says so. Returns the paths written."""
    import inspect
    import warnings
    path = Path(path)
    stem = path.with_suffix("") if path.suffix.lower() in {".pdf", ".png", ".svg", ".eps"} else path
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = stem.with_suffix("." + f.lstrip("."))
        fig.savefig(p, **kw)
        out.append(p)

    source = "caption"
    if not (caption and caption.strip()):
        caption, source = fig.get_suptitle(), "suptitle"
    if not (caption and caption.strip()):
        frame = inspect.currentframe().f_back
        fn = frame.f_globals.get(frame.f_code.co_name) if frame else None
        caption, source = (inspect.getdoc(fn) or ""), f"docstring of {frame.f_code.co_name}()"
    if not (caption and caption.strip()):
        titles = [a.get_title() for a in fig.axes if a.get_title().strip()]
        caption, source = " | ".join(dict.fromkeys(titles)), "panel titles"
    if caption and caption.strip():
        p = stem.parent / (stem.name + "_caption.txt")
        p.write_text(f"[{source}]\n{caption.strip()}\n")
        out.append(p)
    else:
        warnings.warn(f"no caption for {stem.name}: pass caption= to save()", stacklevel=2)
    if close:
        plt.close(fig)
    return out


def colorbar(mappable, ax=None, label: Optional[str] = None, ticks=3, fraction: float = 0.035,
             pad: float = 0.03, **kw):
    """A thin colorbar with a label and few ticks (default: min, mid, max),
    no outline, sized to `ax` (an Axes or list of Axes)."""
    import numpy as np
    fig = mappable.axes.figure if hasattr(mappable, "axes") and mappable.axes is not None else plt.gcf()
    cb = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad, **kw)
    if isinstance(ticks, int):
        lo, hi = mappable.get_clim()
        ticks = np.round(np.linspace(lo, hi, ticks), 10)
    if ticks is not None:
        cb.set_ticks(list(ticks))
        cb.set_ticklabels([_fmt_g(t) for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6)
    if label:
        cb.set_label(label)
    return cb


def swarm_ci(ax: plt.Axes, groups, colors=None, labels=None, x=None, ci: float = 1.96,
             point_kw=None, mean_kw=None, ci_kw=None, jitter: float = 0.22, seed: int = 0,
             mean_ms: float = 5, point_color: str = INK, point_alpha: float = 0.45,
             stat: str = "mean", hollow=None, orient: str = "v"):
    """Per-group observations as a jittered swarm in muted ink, with a summary
    drawn over them in the group's colour -- the honest replacement for a bar
    behind a swarm. `stat="mean"`: the mean and its confidence interval (mean
    +- ci * sem). `stat="median"`: the median and the interquartile range, for
    skewed or censored data where the mean says little.

    `groups` is a list of 1-d arrays; `x` the group positions (default 0..n-1,
    in which case the x limits are set too); `labels` become the x tick
    labels. `hollow`, if given, is one boolean mask per group: those points
    are drawn as open circles -- e.g. observations censored at a ceiling,
    which count in the summary but should not read as measured values.
    `mean_kw` / `ci_kw` restyle the summary marker and interval line, e.g.
    mean_kw=dict(mfc="white") for a hollow marker or ci_kw=dict(ls="--").
    `orient="h"` lays the groups out along y instead, values along x -- for a
    swarm that shares its value axis with a trace drawn above it.
    Returns the group positions used."""
    import numpy as np
    if stat not in ("mean", "median"):
        raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
    if orient not in ("v", "h"):
        raise ValueError(f"orient must be 'v' or 'h', got {orient!r}")
    horizontal = orient == "h"

    def pv(pos, val):                        # (position, value) -> (x, y)
        return (val, pos) if horizontal else (pos, val)
    rng = np.random.default_rng(seed)
    n = len(groups)
    own_x = x is None
    x = np.arange(n, dtype=float) if own_x else np.asarray(x, float)
    colors = list(colors) if colors is not None else COLORS[:n]
    hollow = [None] * n if hollow is None else list(hollow)
    pk = dict(s=4, lw=0, alpha=point_alpha, color=point_color, zorder=1)
    pk.update(point_kw or {})
    for xi, g, c, h in zip(x, groups, colors, hollow):
        g = np.asarray(g, float)
        keep = np.isfinite(g)
        h = np.zeros(len(g), bool) if h is None else np.asarray(h, bool)
        g, h = g[keep], h[keep]
        if not len(g):
            continue
        jx = xi + rng.uniform(-jitter, jitter, len(g))
        ax.scatter(*pv(jx[~h], g[~h]), **pk)
        if h.any():
            hk = dict(pk, facecolors="none", edgecolors=pk.get("color", INK),
                      lw=max(0.5, pk.get("lw", 0) or 0.5))
            hk.pop("color", None)
            ax.scatter(*pv(jx[h], g[h]), **hk)
        if stat == "median":
            m = float(np.median(g))
            lo, hi = np.percentile(g, [25, 75]) if len(g) > 1 else (m, m)
        else:
            m = float(g.mean())
            half = ci * g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
            lo, hi = m - half, m + half
        if len(g) > 1:
            ck = dict(color=c, lw=1.5, solid_capstyle="butt", zorder=3)
            ck.update(ci_kw or {})
            ax.plot(*pv(np.array([xi, xi]), np.array([lo, hi])), **ck)
        mk = dict(color=c, ms=mean_ms, mew=1.2, zorder=4)
        mk.update(mean_kw or {})
        ax.plot(*pv(xi, m), "o", **mk)
    set_ticks = ax.set_yticks if horizontal else ax.set_xticks
    set_lim = ax.set_ylim if horizontal else ax.set_xlim
    if labels is not None:
        set_ticks(x, labels)
    if own_x:
        set_lim(x[0] - 0.6, x[-1] + 0.6)
    return x
