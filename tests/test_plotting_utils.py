"""trim_spines / offset_spines: the re-ticking must leave categorical axes
alone.

Seen 2026-09-23 on criterion_crossing_*: the x axis read "0 0 0 1 2 2 2".
Seaborn's categorical plots hand matplotlib a StrCategoryLocator /
StrCategoryFormatter (the unit machinery), not the FixedLocator that
_retickable used to look for, so trim_spines re-ticked the axis at 0, 0.5,
1, ... and the category formatter labelled each position by ROUNDING it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import plotting_utils as PU


@pytest.fixture
def categorical_violin():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"occurrence": np.repeat([0, 1, 2], 40),
                       "config": np.tile(["a", "b"], 60),
                       "y": rng.uniform(0, 150, 120)})
    fig, ax = plt.subplots()
    sns.violinplot(data=df, x="occurrence", y="y", hue="config", ax=ax, width=1)
    yield fig, ax
    plt.close(fig)


def test_categorical_x_axis_keeps_its_three_ticks_and_labels(categorical_violin):
    fig, ax = categorical_violin
    xlim = ax.get_xlim()
    PU.offset_spines(fig)
    fig.canvas.draw()
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]
    assert ax.get_xlim() == xlim                      # not widened either
    # trimmed to the outermost categories, as seaborn's despine(trim=True)
    assert ax.spines["bottom"].get_bounds() == (0, 2)


def test_bracket_xticks_replaces_the_spine_with_one_bracket_per_category(categorical_violin):
    fig, ax = categorical_violin
    n_lines = len(ax.lines)
    PU.offset_spines(fig)
    PU.bracket_xticks(ax, half_width=0.5)
    fig.canvas.draw()
    # bar + arms + drop per category, all outside the axes and unclipped
    added = ax.lines[n_lines:]
    assert len(added) == 9 and not any(l.get_clip_on() for l in added)
    bars = [l for l in added if l.get_linestyle() == "-"]
    assert [tuple(l.get_xdata()) for l in bars] == [(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5)]
    assert not ax.spines["bottom"].get_visible()
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]
    assert all(t.get_visible() for t in ax.get_xticklabels())
    # a second trim (a later offset_spines call) leaves the fixed ticks alone
    PU.trim_spines(ax)
    assert list(ax.get_xticks()) == [0, 1, 2]


def test_bracket_xticks_keeps_a_facet_grids_inner_labels_hidden():
    import seaborn as sns
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": np.repeat([0, 1], 60), "h": np.tile(["a", "b"], 60),
                       "y": rng.normal(size=120), "col": np.tile(np.repeat(["p", "q"], 30), 2)})
    grid = sns.FacetGrid(df, col="col", col_wrap=1, height=2)
    grid.map_dataframe(lambda data, **kw: sns.violinplot(data=data, x="g", y="y", hue="h", **kw))
    top, bottom = grid.axes.flat
    grid.figure.canvas.draw()
    assert not any(t.get_visible() for t in top.get_xticklabels())
    PU.bracket_xticks(grid.figure)
    grid.figure.canvas.draw()
    assert not any(t.get_visible() for t in top.get_xticklabels())
    assert [t.get_text() for t in bottom.get_xticklabels()] == ["0", "1"]
    assert len([l for l in top.lines if l.get_linestyle() == "-"]) >= 2   # brackets drawn there too
    plt.close(grid.figure)


def test_trim_spines_survives_an_axes_with_nothing_plotted():
    """An empty axes (or a twinx partner used only for a label) has an
    infinite data interval; trim_spines used to overflow turning it into a
    tick count (seen in test_figures_smoke on representation_analysis)."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "nothing here", transform=ax.transAxes)
    PU.offset_spines(fig)                          # must not raise
    _, a2 = plt.subplots()
    a2.plot([0, 1], [0, 1]); twin = a2.twinx(); twin.set_ylabel("label only")
    PU.offset_spines(twin.figure)
    plt.close("all")


def test_swarm_ci_median_draws_the_iqr_and_hollow_points():
    fig, ax = plt.subplots()
    g = np.array([10., 20., 30., 40., 150., 150.])
    PU.swarm_ci(ax, [g], colors=["#123456"], x=[2.0], stat="median", hollow=[g >= 150])
    # two scatters: four filled points, two open ones
    filled, open_ = ax.collections
    assert len(filled.get_offsets()) == 4 and len(open_.get_offsets()) == 2
    assert open_.get_facecolors().size == 0            # facecolors="none"
    # the interval is the IQR and the marker the median, over ALL six values
    (iqr, med) = ax.lines
    assert list(iqr.get_ydata()) == pytest.approx(list(np.percentile(g, [25, 75])))
    assert med.get_ydata()[0] == pytest.approx(np.median(g)) and med.get_xdata()[0] == 2.0
    with pytest.raises(ValueError):
        PU.swarm_ci(ax, [g], stat="mode")
    plt.close(fig)


def test_swarm_ci_horizontal_swaps_the_axes():
    fig, ax = plt.subplots()
    g = np.array([10., 20., 30., 40., 150., 150.])
    PU.swarm_ci(ax, [g], colors=["k"], labels=["only"], stat="median",
                hollow=[g >= 150], orient="h")
    filled, open_ = ax.collections
    assert np.allclose(filled.get_offsets()[:, 0], [10, 20, 30, 40])   # values on x
    assert np.allclose(open_.get_offsets()[:, 0], [150, 150])
    (iqr, med) = ax.lines
    assert list(iqr.get_xdata()) == pytest.approx(list(np.percentile(g, [25, 75])))
    assert list(iqr.get_ydata()) == [0, 0] and med.get_xdata()[0] == pytest.approx(np.median(g))
    assert [t.get_text() for t in ax.get_yticklabels()] == ["only"]
    with pytest.raises(ValueError):
        PU.swarm_ci(ax, [g], orient="diagonal")
    plt.close(fig)


def test_swarm_ci_mean_is_unchanged():
    fig, ax = plt.subplots()
    g = np.array([1., 2., 3., 4.])
    PU.swarm_ci(ax, [g], colors=["k"])
    (ci, mean) = ax.lines
    half = 1.96 * g.std(ddof=1) / 2
    assert list(ci.get_ydata()) == pytest.approx([2.5 - half, 2.5 + half])
    assert mean.get_ydata()[0] == 2.5 and len(ax.collections) == 1
    plt.close(fig)


def test_the_numeric_y_axis_is_still_reticked(categorical_violin):
    """The categorical guard must not switch off re-ticking elsewhere."""
    fig, ax = categorical_violin
    PU.offset_spines(fig)
    fig.canvas.draw()
    yt = list(ax.get_yticks())
    assert 3 <= len(yt) <= 7 and yt[0] <= 0 and yt[-1] >= 150
    assert ax.spines["left"].get_bounds() == (yt[0], yt[-1])
