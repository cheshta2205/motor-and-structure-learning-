"""
decode.py -- can the context be read out of z? 

THE DECODER- 
    Multinomial logistic regression (a GLM) on standardised z.
        64 features (z) -> K classes,  K = 4 or 16 [distinguish Family OR individual C]
    sklearn: StandardScaler -> LogisticRegression(max_iter=3000).
    An MLP variant exists for the training-time tracking curve; the figures use
    linear only, because the two sat on top of each other. 

EXAMPLE  (the 16-way readout, one condition, one split)

    1. roll the trained network out on 1200 held-out sequences x 40 steps
       -> 48,000 rows of z, each 64-d, each with a context label 0..15
    2. split by SEQUENCE, 70/30      ->  840 train seqs / 360 test seqs
    3. FIT on all 33,600 training rows at once, every tss value mixed together.
       The decoder never sees tss.
    4. SCORE the 14,400 test rows, grouped by tss:
           tss    0      1      2      3      4      6      8     10     12
           rows 1752   1538   1349   1196   1056    823    650    497    374
       one accuracy per bin -> one curve.  Chance is 1/16 here.

    A per-family panel masks first: rotation keeps 12,007 of the 48,000 rows,
    fits on 8,459, tests on 3,548, and asks which of that family's 4 members it
    is (chance 1/4).

WHY SPLIT BY SEQUENCE, NOT BY TIMESTEP
    Steps inside a sequence share a context and a trajectory, so a timestep
    split puts near-duplicates on both sides and inflates every number.

WHY FIT ONCE AND SCORE PER BIN, NOT FIT PER BIN
    Blocks die out geometrically, so the test bins shrink 4.7x from tss=0 to
    tss=12 (see the table above).  Fitting inside each bin gives the late bins a
    genuinely worse decoder, not just a noisier one, and manufactures a decay
    that is not in the representation:

        tss        1      4      8     10     12
        per-bin  0.860  0.871  0.862  0.492  0.756   <- 0.492 is a failed fit
        pooled   0.836  0.900  0.915  0.928  0.935   <- rises and plateaus

    Fitting once also matches what CB does: it is a fixed read-out too, and does
    not retrain itself depending on time since switch.

WHAT CHANGES BETWEEN PANELS -- only the label and the mask
    by_tss(z, cid, sid, t, ks, mask=(fam == i))   told the family: which member?
    by_tss(z, fam, sid, t, ks, mask=None)         not told: which family?
    by_tss(z, cid, sid, t, ks, mask=None)         not told: which of 16?
    Per-family numbers are OPTIMISTIC -- they hand over the family, which MTL
    itself has to infer.  The 16-way readout is the honest one, and the ablation
    is much larger there (shuffled e: 0.19 vs 0.0625 chance, against 0.36-0.53
    in the per-family panels).
    Note these are separate FITS -- the label space differs -- sharing only the
    recipe above.  144 fits per arm per seed (3 conditions x 6 readouts x 8
    splits).

THE `seed` ARGUMENT IS NOT THE NETWORK'S SEED
    It picks which sequences go to test, and seeds the MLP.  Callers loop it
    0..7 for 8 splits of the SAME rollout; that spread is the band inside a
    per-seed figure.  The band across NETWORK seeds is the error bar to quote.

CAVEAT
    If the code changed SHAPE with tss, one decoder would understate every bin.
    Pooled decoding rising smoothly to a plateau argues it does not, but
    train-high/test-low is the check if it ever matters.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _clf(kind, seed=0):
    return (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            if kind == "linear" else
            make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(128,), max_iter=800,
                                        random_state=seed)))


def seq_split(seq_id, frac=0.3, seed=0):
    """Boolean TEST mask, split by SEQUENCE so no timestep leaks across."""
    gs = np.unique(seq_id)
    rng = np.random.default_rng(seed)
    te = set(rng.choice(gs, size=max(1, int(frac * len(gs))),
                        replace=False).tolist())
    return np.isin(seq_id, list(te))


def acc(X, y, groups, kind, seed=0):
    """Single pooled accuracy, no tss binning.  Used for the training-time
    tracking curve."""
    if len(np.unique(y)) < 2:
        return np.nan
    te = seq_split(groups.ravel() if groups.ndim > 1 else groups, seed=seed)
    if te.all() or (~te).all() or len(np.unique(y[~te])) < 2:
        return np.nan
    return _clf(kind, seed).fit(X[~te], y[~te]).score(X[te], y[te])


def family_by_tss(z, cid, fam, seq_id, tss, ks, fam_of, seed=0, min_test=30):
    """Family accuracy DERIVED from the 16-way decoder, scored per tss bin.

    Why this exists.  Fitting a 4-class decoder directly on the family label
    asks "is family linearly separable as four regions".  It is not: each family
    is four SEPARATE clusters in z (its four members), so one set of linear
    boundaries does badly -- 0.62 at tss=12.  Fitting the 16-way decoder and
    mapping its prediction to a family asks the different, and for our purposes
    better, question "does z DETERMINE the family": 0.90 on the same data.

    Both are reported.  The direct one is the stricter geometric claim; this one
    is the amount of family information actually present.  The ablation ordering
    is the same under either (intact .898 / zero .725 / shuffle .388 here).

    fam_of: array mapping context id -> family id.
    """
    te = seq_split(seq_id, seed=seed)
    if te.all() or (~te).all() or len(np.unique(cid[~te])) < 2:
        return [np.nan] * len(ks)
    model = _clf("linear", seed).fit(z[~te], cid[~te])
    out = []
    for k in ks:
        sel = te & (tss == k)
        if sel.sum() < min_test:
            out.append(np.nan); continue
        out.append(float((fam_of[model.predict(z[sel])] == fam[sel]).mean()))
    return out


def by_tss(z, y, seq_id, tss, ks, kind="linear", mask=None, seed=0,
           min_test=30):
    """Fit one decoder on the pooled training split, score each tss bin.

    z, y, seq_id, tss are FLAT arrays over (sequence, time).  `mask` restricts
    to a subset (e.g. one family) BEFORE fitting, which is what makes a
    within-family decoder within-family.
    Returns a list aligned with `ks`; np.nan where a bin has too few test points.
    """
    # Split on the FULL sequence list, BEFORE masking, so every panel holds out
    # the same sequences.  Splitting after the mask draws from a different
    # sequence list per family (rotation appears in only ~900 of 1200
    # sequences), so the panels ended up ~27% overlapping -- still valid, but
    # unpaired, which put independent split noise on top of every between-family
    # comparison.
    te = seq_split(seq_id, seed=seed)
    m = np.ones(len(tss), bool) if mask is None else mask
    z, y, seq_id, tss, te = z[m], y[m], seq_id[m], tss[m], te[m]
    if te.all() or (~te).all() or len(np.unique(y[~te])) < 2:
        return [np.nan] * len(ks)
    model = _clf(kind, seed).fit(z[~te], y[~te])
    out = []
    for k in ks:
        sel = te & (tss == k)
        out.append(model.score(z[sel], y[sel]) if sel.sum() >= min_test else np.nan)
    return out
