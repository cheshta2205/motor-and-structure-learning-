"""Extend (or shrink) a colour palette to any length, perceptually.

    python extend_palette.py "#a8e6cf" "#dcedc1" "#ffd3b6" "#ffaaa5" "#ff8b94" --n 6
    python extend_palette.py ... --n 8 --keep            # keep every input colour
    python extend_palette.py ... --n 6 --swatch pal.png  # and draw the result

The input colours, in order, define a path. That path is traced in OKLab -- a
colour space in which equal distances look equally different, unlike RGB,
where a straight line between two hues passes through muddy greys and spaces
its stops unevenly -- and resampled:

  default   `n` colours evenly spaced along the path by perceptual distance.
            The two ends are kept; the interior is re-spaced, so the result
            is a smooth ramp with the input's character but not necessarily
            its exact colours.
  --keep    every input colour survives and the extra ones are inserted at
            the midpoints of the largest perceptual gaps, biggest gap first.
            Use this when the originals are already in use elsewhere.

Output is one hex code per line (lower case, like the input), or the list as
a Python literal with --python. No dependency beyond numpy (matplotlib only
for --swatch): the sRGB <-> OKLab transform is written out below.
"""
import argparse
import sys

import numpy as np

# ── sRGB <-> OKLab (Björn Ottosson, 2020) ────────────────────────────────

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.cbrt(c) ** (3 / 2.4) - 0.055)


def rgb_to_oklab(rgb):
    """(..., 3) sRGB in [0, 1] -> (..., 3) OKLab."""
    lms = _srgb_to_linear(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab):
    """(..., 3) OKLab -> (..., 3) sRGB, clipped to [0, 1] (a colour outside
    the gamut lands on its nearest displayable neighbour)."""
    lms = (np.asarray(lab, float) @ np.linalg.inv(_M2).T) ** 3
    return np.clip(_linear_to_srgb(lms @ np.linalg.inv(_M1).T), 0.0, 1.0)


# ── hex in, hex out ──────────────────────────────────────────────────────

def parse_hex(s: str) -> np.ndarray:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {s!r}")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def to_hex(rgb) -> str:
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(rgb, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── the extension itself ─────────────────────────────────────────────────

def _path(lab: np.ndarray) -> np.ndarray:
    """Cumulative perceptual distance along the colour sequence, scaled to
    [0, 1]. Flat (all zeros) if every colour is the same."""
    d = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    return t / t[-1] if t[-1] > 0 else np.linspace(0, 1, len(lab))


def _interp_along(lab: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(s, t, lab[:, k]) for k in range(3)], axis=1)


def extend_palette(colors, n: int, keep: bool = False) -> list:
    """`colors` (hex strings) -> `n` hex strings. See the module docstring
    for the two modes. With n <= len(colors) and keep=False the palette is
    subsampled evenly along its path (the ends kept); with keep=True and
    n < len(colors) that is an error, since every input must survive."""
    if n < 1:
        raise ValueError("n must be at least 1")
    lab = rgb_to_oklab(np.array([parse_hex(c) for c in colors]))
    if len(lab) == 1:
        return [to_hex(oklab_to_rgb(lab[0]))] * n
    t = _path(lab)
    if not keep:
        s = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
        out = _interp_along(lab, t, s)
        return [to_hex(c) for c in oklab_to_rgb(out)]
    if n < len(lab):
        raise ValueError(f"--keep needs n >= {len(lab)} (every input colour is kept)")
    pts = list(lab)
    for _ in range(n - len(lab)):
        gaps = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        i = int(np.argmax(gaps))
        pts.insert(i + 1, (pts[i] + pts[i + 1]) / 2.0)
    return [to_hex(c) for c in oklab_to_rgb(np.array(pts))]


def swatch(colors, path: str) -> None:
    """One labelled block per colour, saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.1 * len(colors), 1.6))
    for i, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c, lw=0))
        # label in whichever of black / white reads against the block
        L = rgb_to_oklab(parse_hex(c))[0]
        ax.text(i + 0.5, 0.5, c, ha="center", va="center", fontsize=8,
                color="black" if L > 0.6 else "white", family="monospace")
    ax.set_xlim(0, len(colors)); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n\n")[1:]))
    ap.add_argument("colors", nargs="+", help="input hex colours, in palette order")
    ap.add_argument("--n", type=int, required=True, help="how many colours to produce")
    ap.add_argument("--keep", action="store_true",
                    help="keep every input colour; insert the extras at the largest gaps")
    ap.add_argument("--python", action="store_true", help="print as a Python list")
    ap.add_argument("--swatch", default=None, help="also save a swatch image here")
    a = ap.parse_args(argv)
    out = extend_palette(a.colors, a.n, keep=a.keep)
    print(repr(out) if a.python else "\n".join(out))
    if a.swatch:
        swatch(out, a.swatch)
        print(f"swatch -> {a.swatch}", file=sys.stderr)


if __name__ == "__main__":
    _main()
