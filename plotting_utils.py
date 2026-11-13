"""
plotting_utils.py -- this repo's name for the shared paper figure style.

The style and its helpers now live in the PlottingTools package
(github.com/eaamematsro/PlottingTools, module `plotting_tools.paper_style`;
merged 2026-10-30 with dual_learning's copy). This module re-exports all of
it, private helpers included, so `import plotting_utils as PU` and
`from plotting_utils import ...` keep working unchanged. Install with

    pip install -e /path/to/PlottingTools        # in every env that draws figures

`set_plt_params` and `make_axis_nice`, the pre-2026-09-21 helpers, come from
plotting_tools.plotting_functions.
"""
try:
    from plotting_tools import paper_style as _ps
    from plotting_tools.plotting_functions import set_plt_params, make_axis_nice  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("plotting_utils needs the PlottingTools package: "
                      "pip install -e /path/to/PlottingTools") from e

globals().update({k: v for k, v in vars(_ps).items() if not k.startswith("__")})
