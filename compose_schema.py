"""compose_schema.py -- moved to PlottingTools (plotting_tools.compose_schema).

Kept so `python compose_schema.py ...` and `import compose_schema` still work; see that module
for the method and the flags."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("plotting_tools.compose_schema", run_name="__main__", alter_sys=True)
else:
    from plotting_tools.compose_schema import *  # noqa: F401,F403
