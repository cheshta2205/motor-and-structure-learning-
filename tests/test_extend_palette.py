import numpy as np
import pytest

import extend_palette as EP

FIVE = ["#a8e6cf", "#dcedc1", "#ffd3b6", "#ffaaa5", "#ff8b94"]


def test_oklab_round_trips_and_is_lightness_ordered():
    rgb = np.array([EP.parse_hex(c) for c in FIVE + ["#000000", "#ffffff", "#ff0000"]])
    back = EP.oklab_to_rgb(EP.rgb_to_oklab(rgb))
    assert np.allclose(back, rgb, atol=1e-6)
    L = EP.rgb_to_oklab(rgb)[:, 0]
    assert L[-3] == pytest.approx(0.0, abs=1e-6) and L[-2] == pytest.approx(1.0, abs=1e-6)
    assert EP.to_hex(EP.parse_hex("#A8E6CF")) == "#a8e6cf"
    assert EP.to_hex(EP.parse_hex("abc")) == "#aabbcc"


def test_default_mode_keeps_the_ends_and_respaces_the_interior():
    # the input's stops are NOT evenly spaced perceptually, so even at its own
    # length the default mode moves the interior ones; --keep returns them
    five = EP.extend_palette(FIVE, 5)
    assert five[0] == FIVE[0] and five[-1] == FIVE[-1] and five != FIVE
    assert EP.extend_palette(FIVE, 5, keep=True) == FIVE
    six = EP.extend_palette(FIVE, 6)
    assert len(six) == 6 and six[0] == FIVE[0] and six[-1] == FIVE[-1]
    assert len(set(six)) == 6                            # all distinct
    # evenly spaced in OKLab along the path: consecutive distances agree
    lab = EP.rgb_to_oklab(np.array([EP.parse_hex(c) for c in six]))
    d = np.linalg.norm(np.diff(lab, axis=0), axis=1)
    assert d.max() / d.min() < 1.6                       # clipping at the gamut edge allows some slack
    assert len(EP.extend_palette(FIVE, 3)) == 3          # shrinking works too
    assert EP.extend_palette(FIVE, 3)[0] == FIVE[0]


def test_keep_mode_preserves_every_input_colour():
    eight = EP.extend_palette(FIVE, 8, keep=True)
    assert len(eight) == 8 and all(c in eight for c in FIVE)
    assert [c for c in eight if c in FIVE] == FIVE       # in the original order
    with pytest.raises(ValueError):
        EP.extend_palette(FIVE, 4, keep=True)


def test_cli_prints_one_hex_per_line(capsys, tmp_path):
    EP._main(FIVE + ["--n", "6", "--swatch", str(tmp_path / "s.png")])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 6 and all(len(c) == 7 and c.startswith("#") for c in out)
    assert (tmp_path / "s.png").exists()
