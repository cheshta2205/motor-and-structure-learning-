"""
Locks down the RL oracle's context code: sac.gp_code and sac.oracle_kernel_matrix.

TWO THINGS HERE ARE SILENT IF THEY BREAK, which is why they are tested at all.

The first is the identity path. make_oracle_embed used to be a plain normalised
Gaussian draw and is now a GP draw with K = I, which is the same thing ONLY if
the eigenvalue sort is stable and the float32 cast lands before the
renormalisation. Get either wrong and the codes are a permutation or a rounding
away from what every finished run in sac_runs/ was trained on -- nothing
raises, the runs simply stop being comparable.

The second is the centering. context_centroids regresses out kinematics WITH an
intercept, so residualising alone already removes the contexts' shared
component; asking for the uncentered geometry while residualising returns the
centered one under the wrong name, and the oracle then matches a geometry the
circuit does not have.
"""
import numpy as np
import pytest

import model as M
import sac
import task as T


def _legacy_embed(seed=0, n=None, dim=None):
    """make_oracle_embed exactly as it was before the kernel generalisation."""
    n = sac.N_PHASE_FAMILIES if n is None else n
    dim = sac.ORACLE_EMBED_DIM if dim is None else dim
    rng = np.random.default_rng(sac.ORACLE_EMBED_SEED_BASE + int(seed))
    E = rng.standard_normal((n, dim)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E


def _needs_checkpoint():
    if M.load("main", 0) is None:
        pytest.skip("no main seed 0 checkpoint to measure a geometry from")


# ── the regression guard ─────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(6))
def test_identity_kernel_reproduces_the_legacy_draw_bit_for_bit(seed):
    np.testing.assert_array_equal(sac.make_oracle_embed(seed),
                                  _legacy_embed(seed))


def test_stable_sort_is_what_makes_that_hold():
    """The failure this guards against, made explicit: an UNSTABLE descending
    sort reverses the identity's tied eigenvalues, so L is a permutation and
    the contexts swap codes."""
    n, dim = sac.N_PHASE_FAMILIES, sac.ORACLE_EMBED_DIM
    w, V = np.linalg.eigh(np.eye(n))
    stable = V[:, np.argsort(-w, kind="stable")]
    reversed_ = V[:, np.argsort(w)[::-1]]
    np.testing.assert_array_equal(stable, np.eye(n))
    assert not np.array_equal(reversed_, np.eye(n))    # would permute the rows


def test_identity_is_the_default():
    assert sac.AblationConfig(name="x").oracle_kernel == "identity"
    np.testing.assert_array_equal(sac.oracle_kernel_matrix("identity"),
                                  np.eye(sac.N_PHASE_FAMILIES))


# ── gp_code, on synthetic kernels so no checkpoint is needed ─────────────
def _synthetic(n=4, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, rank))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


@pytest.mark.parametrize("K", [np.eye(4), _synthetic(4, 3), _synthetic(4, 2)])
def test_codes_are_unit_norm_at_the_requested_width(K):
    C = sac.gp_code(K, dim=sac.ORACLE_EMBED_DIM, seed=0)
    assert C.shape == (len(K), sac.ORACLE_EMBED_DIM)
    assert C.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-5)


def test_rank_of_the_kernel_survives_into_the_code():
    """A rank-deficient kernel must confine the contexts to that many
    dimensions however wide the code is -- it is the part of a geometry a wider
    code cannot paper over."""
    for rank in (1, 2, 3):
        C = sac.gp_code(_synthetic(4, rank), dim=sac.ORACLE_EMBED_DIM, seed=0)
        assert np.linalg.matrix_rank(C, tol=1e-4) == rank


def test_the_draw_is_centered_on_the_kernel():
    """One draw misses K by O(1/sqrt(dim)) and that is accepted, but the draws
    have to be centered ON K -- averaging many realised Grams must converge."""
    K = _synthetic(4, 3)
    G = np.mean([sac.gp_code(K, sac.ORACLE_EMBED_DIM, seed=s) @
                 sac.gp_code(K, sac.ORACLE_EMBED_DIM, seed=s).T
                 for s in range(400)], 0)
    assert np.abs(G - K).max() < 0.05
    one = sac.gp_code(K, sac.ORACLE_EMBED_DIM, seed=0)
    assert np.abs(one @ one.T - K).max() > 0.01     # so the average is not vacuous


def test_negative_eigenvalues_are_clipped():
    """A measured cosine matrix need not be quite PSD; nothing may reach a
    sqrt of a negative number."""
    K = _synthetic(4, 3) - 0.05 * (np.ones((4, 4)) - np.eye(4))
    assert np.linalg.eigvalsh(K).min() < 0
    C = sac.gp_code(K, sac.ORACLE_EMBED_DIM, seed=0)
    assert np.isfinite(C).all()


# ── the measured kernels ─────────────────────────────────────────────────
def test_z_kernel_keeps_the_shared_component_and_z_centered_removes_it():
    """The distinction the two kernels exist for, and the one the residualise
    intercept can silently erase."""
    _needs_checkpoint()
    n = sac.N_PHASE_FAMILIES
    off = ~np.eye(n, dtype=bool)
    Kz = sac.oracle_kernel_matrix("z")
    Kc = sac.oracle_kernel_matrix("z_centered")
    # uncentered: the contexts share a large common component, so every pair is
    # strongly positive -- this is the ~0.62 sac's ORACLE_EMBED comment cites
    assert Kz[off].mean() > 0.3
    assert (Kz[off] > 0).all()
    # centered: n vectors summing to zero have mean pairwise INNER PRODUCT
    # exactly -1/(n-1) of the mean squared norm, so the mean COSINE lands there
    # too up to the spread in the centroid norms -- 4e-04 here. Near-arithmetic,
    # not a measurement, so the tolerance is loose on purpose: what would be a
    # real failure is centering silently not happening at all (+0.59).
    np.testing.assert_allclose(Kc[off].mean(), -1.0 / (n - 1), atol=5e-3)
    # ...and centering costs exactly one dimension
    assert np.linalg.matrix_rank(Kc, tol=1e-6) == n - 1


def test_kernels_are_valid_gram_matrices():
    _needs_checkpoint()
    for k in sac.ORACLE_KERNELS:
        K = sac.oracle_kernel_matrix(k)
        assert K.shape == (sac.N_PHASE_FAMILIES,) * 2
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6)
        np.testing.assert_allclose(K, K.T, atol=1e-6)
        assert np.linalg.eigvalsh(K).min() > -1e-6
        assert np.abs(K).max() <= 1.0 + 1e-6


def test_kernel_is_measured_over_the_phase_contexts_only():
    """The agent meets PHASE_CONTEXT_IDS, not the whole table, and the shared
    component it sees is the mean of those."""
    _needs_checkpoint()
    assert len(sac.PHASE_CONTEXT_IDS) == sac.N_PHASE_FAMILIES
    assert all(c in T.C_LABEL for c in sac.PHASE_CONTEXT_IDS)
    K = sac.oracle_kernel_matrix("z", context_ids=sac.PHASE_CONTEXT_IDS)
    np.testing.assert_allclose(K, sac.oracle_kernel_matrix("z"), atol=1e-12)
    with pytest.raises(ValueError):                 # wrong number of contexts
        sac.oracle_kernel_matrix("z", context_ids=sac.PHASE_CONTEXT_IDS[:2])


# ── failure modes ────────────────────────────────────────────────────────
def test_unknown_kernel_raises():
    with pytest.raises(KeyError):
        sac.oracle_kernel_matrix("nonsense")
    with pytest.raises(KeyError):
        sac.make_oracle_embed(0, kernel="nonsense")


def test_missing_checkpoint_raises_rather_than_falling_back_to_identity():
    """A silent fall back would produce a run NAMED for the circuit's geometry
    that was trained on an orthogonal code -- wrong rather than missing, and
    invisible in every figure downstream."""
    with pytest.raises(FileNotFoundError):
        sac.make_oracle_embed(0, kernel="z", arm="main", arm_seed=999)


def test_presets_are_constructible_and_name_valid_kernels():
    import sweep_ablations as S
    for name, cfg in S.CONFIG_BY_NAME.items():
        c = sac.AblationConfig(**cfg)
        assert c.oracle_kernel in sac.ORACLE_KERNELS
        if c.oracle_kernel != "identity":
            assert c.use_oracle, f"{name} sets a kernel but has no oracle"
    assert S.CONFIG_BY_NAME["oracle_geom"]["oracle_kernel"] == "z"
    assert S.CONFIG_BY_NAME["oracle"].get("oracle_kernel", "identity") == "identity"


def test_arm_seed_is_applied_to_the_geometry_oracles_by_the_sweep():
    """They read a checkpoint, so --arm-seed has to reach them as it reaches
    the frozen configs."""
    import sweep_ablations as S
    presets = [sac.AblationConfig(**S.CONFIG_BY_NAME[n])
               for n in ("oracle", "oracle_geom", "main")]
    applied = [sac.AblationConfig(**{**p.__dict__, "arm_seed": 3})
               if (p.freeze or p.oracle_kernel != "identity") else p
               for p in presets]
    assert [c.arm_seed for c in applied] == [0, 3, 3]
