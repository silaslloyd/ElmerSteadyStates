"""ess.predictor.extrapolate: Zs and temperature extrapolated in d at fixed x, the patch unchanged."""
import numpy as np

from ess.predictor import extrapolate
from ess.statefile import SavedState
from test_transfer import mesh

L = 800e3
XGL = 600e3


def state_at(d):
    """a toy state of patch length d: upstream of the head (XGL - d) Zs and T depend on (x, d) quadratically in d,
    downstream (the patch) not on d"""
    head = XGL - d
    s = mesh(np.linspace(0, L, 81), zs_fun=lambda x: 500.0 + 0.001 * x)
    sl = s.layout.slices()
    gT, gS = s.layout.blocks[0].gids, s.layout.blocks[1].gids
    cx = s.diag["coordinate 1"]
    xT, xS = cx[1][gT - 1], cx[1][gS - 1]
    dk = d / 1e3
    f = lambda X: np.where(X < head, 1e-3 * (X / 1e3) * (1 + 0.01 * dk + 1e-4 * dk * dk), 0.0)
    x = s.x.copy()
    x[sl["Zs"]] = 500.0 + 0.001 * xS + f(xS) * 100.0
    x[sl["Temperature"]] = 260.0 + 1e-5 * xT + f(xT)
    return SavedState(s.layout, x, s.p, s.frozen, s.diag, {}), head


def test_extrapolation_exact_for_quadratic_in_d_and_patch_unchanged():
    ds = [500e3, 475e3, 450e3, 425e3]
    sts = [state_at(d) for d in ds]
    s, head_s = sts[-1]
    hist = [(d, st.x, h) for d, (st, h) in zip(ds[:-1], sts[:-1])]
    d_next = 400e3
    x, info = extrapolate(s, ds[-1], head_s, hist, d_next, XGL - d_next, order=2, npts=4)
    ex, head_n = state_at(d_next)
    sl = s.layout.slices()
    cx = s.diag["coordinate 1"]
    for key, name in (("Zs", "Zs"), ("Temperature", "Temperature")):
        b = [b_ for b_ in s.layout.blocks if b_.name == name][0]
        X = cx[1][b.gids - 1]
        up = X < head_s                                  # cold in every history state with a head >= x: exact
        deep = X < XGL - ds[0]                           # cold in all four states
        assert np.allclose(x[sl[name]][deep], ex.x[sl[name]][deep], atol=1e-6)
        pat = X >= head_n
        assert np.array_equal(x[sl[name]][pat], s.x[sl[name]][pat])      # the new patch: unchanged
    assert info["zs"]["max_change"] > 0
