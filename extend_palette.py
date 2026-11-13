"""extend_palette.py -- moved to PlottingTools (plotting_tools.extend_palette).

Kept so `python extend_palette.py ...` and `import extend_palette` still work; see that module
for the method and the flags."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("plotting_tools.extend_palette", run_name="__main__", alter_sys=True)
else:
    from plotting_tools.extend_palette import *  # noqa: F401,F403
