"""Task schematic for the OLM / ORM figure, in the style of Zeidler et al. Fig 1a,c.

Drawn, not measured, but every number is read off olm_orm_final.py so the sketch
cannot drift from the simulation: arena radius, object separation, session
length, the post-change window, and which channels each task is scored on.

THREE PLACES THE MODEL DIFFERS FROM THE EXPERIMENT, drawn rather than papered
over -- each is something a reader would otherwise have to find in the code:

  arena      circular for BOTH tasks (R_ARENA = 1.0).  Zeidler uses a rectangle
             for OLM and a circle for ORM; nothing in the model depends on it.
  objects    the two objects have DIFFERENT shapes in both tasks, because shape
             is drawn independently per object.  The experiment's ORM sample
             phase uses an identical pair.
  the index  PI compares a changed test session against an UNCHANGED one, both
             scored on object 0.  The animal's DI compares the two objects
             inside one session.  Two sessions is what the code does, so two
             sessions is what the diagram shows.

The delay is the fourth difference and the most interesting one: there is no
24 h.  The MTL state h and the forward model's estimate s_hat are carried
straight from the end of the sample session into the test session, so the
"memory" is whatever those two variables still hold.

Everything is drawn in data units on an equal-aspect axis -- arenas are circles
and must stay circles.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

import olm_orm_final as M

OUT = os.path.join(M.FIG_DIR, "schematic")
INK = "#1a1a1a"
PATHC = "#4a90c4"              # the exploration path, as in the paper
HOT = "#c0392b"                # what changed
AR = 1.0                       # arena radius, data units
XS, YS = 13.2, 7.2             # canvas
T_SESSION = 200                # steps per session (the `T=200` in curves/PI)

# object layout in units of the arena radius.  The separation respects
# OBJ_MIN_SEP / 0.75, so the drawn geometry is one the sampler could produce.
A_OLD, A_NEW, B_POS = (-0.42, 0.40), (0.38, 0.48), (0.44, -0.40)
# a circle is just a polygon with enough sides -- keeps one drawing path.
GLYPH = {"pent": (5, 90), "hex": (6, 0), "sq": (4, 45), "tri": (3, 90),
         "circ": (48, 0)}


def _shape(ax, cx, cy, kind, s=0.19, hot=False):
    """Stand-in glyph for the 3-d shape vector (curvature, elongation,
    angularity).  A different side count = a different shape code."""
    n, rot = GLYPH[kind]
    a = np.radians(rot) + np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.add_patch(Polygon(np.c_[cx + s * np.cos(a), cy + s * np.sin(a)],
                         closed=True, fc=HOT if hot else "#4a4a4a",
                         ec=HOT if hot else INK, lw=1.4 if hot else 1.0, zorder=4))


def _mouse(ax, cx, cy, ang, to):
    """Body, ear and a bowed path running off toward `to`."""
    t = np.linspace(0, 2 * np.pi, 60)
    bx, by = 0.13 * np.cos(t), 0.075 * np.sin(t)
    ca, sa = np.cos(ang), np.sin(ang)
    ax.add_patch(Polygon(np.c_[cx + ca * bx - sa * by, cy + sa * bx + ca * by],
                         closed=True, fc="w", ec=INK, lw=1.1, zorder=6))
    ax.add_patch(Circle((cx - ca * .07 - sa * .05, cy - sa * .07 + ca * .05),
                        .032, fc="w", ec=INK, lw=.9, zorder=7))
    x0, y0 = cx + ca * .135, cy + sa * .135
    x1, y1 = to
    nx, ny = -(y1 - y0), (x1 - x0)
    nn = np.hypot(nx, ny) + 1e-9
    u = np.linspace(0, 1, 60)
    bow = .30 * np.sin(np.pi * u)
    ax.plot(x0 + u * (x1 - x0) + bow * nx / nn, y0 + u * (y1 - y0) + bow * ny / nn,
            color=PATHC, lw=1.6, zorder=5, solid_capstyle="round")


def _arena(ax, cx, cy, objs, title=None, ghost=None):
    """objs = [(x, y, kind, hot), ...] in units of the arena radius."""
    ax.add_patch(Circle((cx, cy), AR, fc="w", ec=INK, lw=1.5, zorder=1))
    if ghost is not None:                 # dashed outline of the old position
        ax.add_patch(Circle((cx + ghost[0] * AR, cy + ghost[1] * AR), .25,
                            fc="none", ec=HOT, lw=1.0, ls=(0, (2, 2)), zorder=3))
    for ox, oy, kind, hot in objs:
        _shape(ax, cx + ox * AR, cy + oy * AR, kind, hot=hot)
    _mouse(ax, cx - .10, cy - .22, np.radians(22), (cx + .30, cy + .62))
    if title:
        ax.text(cx, cy + AR + .17, title, ha="center", fontsize=8.6, color="#333")


def _arrow(ax, x0, x1, y, label, sub):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.3, color=INK, zorder=8))
    ax.text((x0 + x1) / 2, y + .14, label, ha="center", fontsize=8.4)
    ax.text((x0 + x1) / 2, y - .20, sub, ha="center", va="top", fontsize=7.4,
            color="#666")


def row(ax, task, y, scored, titles=True):
    kA, kB = "pent", "hex"
    kA_new = "circ" if task == "ORM" else kA        # ORM swaps the shape code
    posA_new = A_OLD if task == "ORM" else A_NEW    # OLM moves the position

    ax.text(.75, y + .14, task, fontsize=13, weight="bold", ha="center")
    ax.text(.75, y - .22, scored, fontsize=7.2, color="#555", ha="center",
            va="top", linespacing=1.45)

    _arena(ax, 3.15, y, [(*A_OLD, kA, False), (*B_POS, kB, False)],
           title=f"sample · {T_SESSION} steps" if titles else None)
    _arrow(ax, 4.30, 5.70, y, "$h$, $\\hat{s}$ carried over",
           "no delay: the state is\nthe only memory")

    _arena(ax, 6.85, y, [(*posA_new, kA_new, True), (*B_POS, kB, False)],
           title="test · one object changed" if titles else None,
           ghost=A_OLD if task == "OLM" else None)
    _arena(ax, 10.5, y, [(*A_OLD, kA, False), (*B_POS, kB, False)],
           title="test · nothing changed" if titles else None)

    ax.text(6.85, y - AR - .22, "object 0 moved" if task == "OLM"
            else "object 0 given a new shape", ha="center", fontsize=7.8, color=HOT)
    ax.text(10.5, y - AR - .22, "the control session", ha="center",
            fontsize=7.8, color="#666")


def figure(path=None):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_xlim(0, XS); ax.set_ylim(0, YS)
    ax.set_aspect("equal"); ax.axis("off")

    row(ax, "OLM", 5.05, "scored on\nbearing + apparent\nsize of object 0")
    row(ax, "ORM", 1.95, "scored on\nthe shape code\nof object 0",
        titles=False)

    ax.plot([8.70, 8.70], [.75, 6.45], color="#bbb", lw=.9, ls=":")
    ax.text(.08, .22,
            "preference index   PI $= e_{\\mathrm{changed}}/"
            "(e_{\\mathrm{changed}}+e_{\\mathrm{unchanged}})$   over the first "
            f"{M.PI_WINDOW} steps after the change        "
            "0.5 = the change went unnoticed",
            ha="left", fontsize=9)

    os.makedirs(OUT, exist_ok=True)
    path = path or os.path.join(OUT, "olm_orm_task_schematic.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig); print(f"saved -> {path}")


if __name__ == "__main__":
    figure()
