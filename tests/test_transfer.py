"""ess transfer: column-wise interpolation between two extruded meshes of the same flowline."""
import numpy as np

from ess.statefile import Block, SavedState, StateLayout
from ess.transfer import transfer

D = 1000.0          # bedrock depth


def bed(x):
    return -0.01 * x / 1e3


def surf(x, L):
    return bed(x) + 30 + 2000 * np.sqrt(np.clip(1 - x / L, 0, 1))


def Tfield(x, z, zs):
    b = bed(x)
    zeta = np.clip((z - b) / (zs - b), 0, 1)
    return np.where(z >= b - 1e-3, 260 + 10 * (1 - zeta) + 1e-5 * x, 270 + 0.02 * (b - z) + 1e-5 * x)


def mesh(xs, nrock=6, nice=11, zs_fun=None):
    """Two rows (y = 0, W) of columns; returns a SavedState whose x holds the exact field."""
    g = 0
    gid, X, Z, B, isurf, ibed = [], [], [], [], [], []
    for xc in xs:
        for y in (0, 1):
            b = bed(xc)
            zs = zs_fun(xc)
            zr = b - D * np.linspace(1, 0, nrock)[:-1]
            zi = b + (zs - b) * np.linspace(0, 1, nice)
            for k, zz in enumerate(np.concatenate([zr, zi])):
                g += 1
                gid.append(g); X.append(xc); Z.append(zz); B.append(b)
                if k == len(zr) + nice - 1:
                    isurf.append(g)
                if k == len(zr):
                    ibed.append(g)
    gid = np.array(gid); X = np.array(X); Z = np.array(Z); B = np.array(B)
    isurf, ibed = np.array(isurf), np.array(ibed)
    zsurf = Z[isurf - 1]
    zs_at = np.interp(X, X[isurf - 1], zsurf)
    T = Tfield(X, Z, zs_at)
    lay = StateLayout([Block("Temperature", gid, 5.0), Block("Zs", isurf, 50.0)])
    x = np.concatenate([T, zsurf])
    frozen = {"bedrock": (gid, B), "ColdTempMask": (ibed, np.where(X[ibed - 1] < 300e3, 1.0, -1.0)),
              "GroundedMask": (ibed, np.ones(len(ibed)))}
    diag = {"coordinate 1": (gid, X), "coordinate 3": (gid, Z)}
    return SavedState(lay, x, 268.15, frozen, diag, {})


def test_transfer_to_cut_mesh():
    L = 800e3
    src = mesh(np.linspace(0, L, 161), zs_fun=lambda x: surf(x, 0.8 * L))
    xe = 0.8 * L
    dst = mesh(np.concatenate([np.linspace(0, 0.7 * xe, 30), np.linspace(0.7 * xe, xe, 77)[1:]]),
               nrock=4, nice=21, zs_fun=lambda x: bed(x) + 1000.0)      # a different initial surface
    x, frozen = transfer(src, dst, x_end_m=xe, edge=30.0)
    sl = dst.layout.slices()
    gZ = dst.layout.blocks[1].gids
    X = dst.diag["coordinate 1"][1][gZ - 1]
    Z = x[sl["Zs"]]
    end = X >= xe - 1
    assert np.allclose(Z[end], bed(X[end]) + 30)
    assert np.allclose(Z[~end], surf(X[~end], xe), atol=40)          # linear interpolation of a sqrt profile
    # temperature: exact on the new nodes once they sit at the same zeta / depth under the new surface
    gT = dst.layout.blocks[0].gids
    Xt = dst.diag["coordinate 1"][1][gT - 1]
    Zt = dst.diag["coordinate 3"][1][gT - 1]
    zs_init = bed(Xt) + 1000.0
    b = bed(Xt)
    zeta = (Zt - b) / (zs_init - b)
    Zphys = np.where(Zt >= b - 1e-3, b + zeta * (np.interp(Xt, X, Z) - b), Zt)
    Tex = Tfield(Xt, Zphys, np.interp(Xt, X, Z))
    assert np.max(np.abs(x[sl["Temperature"]] - Tex)) < 0.05
    ct = frozen["ColdTempMask"][1]
    xb = dst.diag["coordinate 1"][1][frozen["ColdTempMask"][0] - 1]
    assert np.all(ct[xb < 290e3] > 0) and np.all(ct[xb > 310e3] < 0)
