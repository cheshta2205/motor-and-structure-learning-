"""
plotting_utils.py -- the two style helpers make_ablations_plot.py needs.

Ported from the reference repo's src/PlottingTools/plotting_functions.py
(only these two functions, not the rest of that toolkit) so
make_ablations_plot.py's many make_axis_nice(...)/set_plt_params(...) call
sites don't need rewriting. figures.py's own `_despine` is untouched and
still used by sac_figures.py -- this module is additive, only imported by
the ablation-analysis pipeline.
"""
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt


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
