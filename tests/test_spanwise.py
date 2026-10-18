"""Spanwise projection and classification on a synthetic 3D strip (no Elmer)."""
import numpy as np

from ess.linalg import arnoldi, ritz
from ess.spanwise import Spanwise
from ess.statefile import Block, StateLayout


def strip(nx=6, ny=8, nz=5, W=20e3):
    xs = np.linspace(0, 600e3, nx)
    ys = np.linspace(0, W, ny + 1)
    g = 0
    gid, X, Y, Z, top = [], [], [], [], []
    for x in xs:
        for y in ys:
            zs = np.linspace(-1000, 2000 - x / 1e3, nz)     # same column geometry for every y
            for k, z in enumerate(zs):
                g += 1
                gid.append(g); X.append(x); Y.append(y); Z.append(z)
                if k == nz - 1:
                    top.append(g)
    gid, X, Y, Z = map(np.array, (gid, X, Y, Z))
    lay = StateLayout([Block("Temperature", gid, 5.0), Block("Zs", np.array(top), 50.0)])
    coords = {"coordinate 1": (gid, X), "coordinate 2": (gid, Y), "coordinate 3": (gid, Z)}
    yv = np.concatenate([Y, Y[np.array(top) - 1]])
    xv = np.concatenate([X, X[np.array(top) - 1]])
    return lay, coords, yv, xv, W


def test_projection_and_spectrum():
    lay, coords, y, x, W = strip()
    sp = Spanwise(lay, coords)
    assert sp.ny == 9 and abs(sp.W - W) < 1e-6
    f = np.sin(x / 1e5)                                       # y-uniform part
    v = f + 0.3 * np.cos(np.pi * y / W) * np.cos(x / 2e5) + 0.1 * np.cos(3 * np.pi * y / W)
    pv = sp.project(v)
    assert np.allclose(sp.project(pv), pv)                    # a projection
    s = sp.spectrum(pv)
    assert s[0] < 1e-12 and s.argmax() == 1 and s[3] > 0.05
    assert abs(sp.spectrum(np.cos(2 * np.pi * y / W))[2] - 1) < 1e-9


def test_arnoldi_finds_leading_spanwise_mode():
    lay, coords, y, x, W = strip()
    sp = Spanwise(lay, coords)
    n = len(y)
    rng = np.random.default_rng(3)
    # a map that commutes with the across-flow symmetry: a different decay factor for each n, and the
    # 2D (n = 0) part is the least damped (0.99) -- the projection must hide it
    basis = [np.cos(m * np.pi * y / W) for m in range(sp.ny)]
    mu_n = {0: 0.99, 1: 0.5, 2: 0.85, 3: 0.7}

    def M(v):
        out = np.zeros(n)
        for idx, yy in sp.lines:
            w = sp._weights(yy)
            for m, b in enumerate(basis):
                c = b[idx]
                a = np.sum(w * c * v[idx]) / np.sum(w * c * c)
                out[idx] += mu_n.get(m, 0.2) * a * c
        return out

    st = arnoldi(lambda v: sp.project(M(sp.project(v))), sp.project(rng.standard_normal(n)), 12)
    mu, X, res = ritz(st)
    assert abs(abs(mu[0]) - 0.85) < 1e-8
    assert sp.spectrum(X[:, 0]).argmax() == 2
