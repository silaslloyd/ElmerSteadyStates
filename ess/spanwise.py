"""
Spanwise (across-flow) structure for the stability of a y-uniform steady state on a 3D strip.

Walls (periodic=False): the strip 0 <= y <= W between free-slip side walls (no flow through them, no heat flux);
the eigenmodes are, up to discretisation, delta(x, y, z) = delta_hat(x, z) cos(n pi y / W), n = 0, 1, 2, ...,
spanwise wavenumber k_n = n pi / W, wavelength 2 W / n.
Periodic sides (periodic=True, [case] periodic_sides): delta_hat(x, z) cos or sin(2 pi n y / W), k_n = 2 pi n / W,
wavelength W / n; each n >= 1 twice (the cos and sin modes, any phase across: the same eigenvalue).
n = 0 are the 2D (flowline) modes.

Nodes are grouped into lines across the flow: same x and the same position in the column (rank in z
within the column at that (x, y)), one node per y. `project` removes the across-flow mean of every
line (kills n = 0), `spectrum` gives the energy in each n by a discrete cosine transform (DCT-I on
the uniform y grid, weights 1/2 at the walls; periodic: the Fourier series, the two seam copies weight 1/2 each).
The node lines need the same x on every row across: a mesh structured across the flow ([case] mesh_across =
"structured", ess/case.py structured_y_geo), not the mirrored unstructured triangles.
"""
from __future__ import annotations

import numpy as np

from .statefile import StateLayout, find_name


class Spanwise:
    def __init__(self, layout: StateLayout, coords: dict, xtol=1.0, ytol=1.0, periodic=False):
        """coords: records with 'coordinate 1/2/3' (gids, values) covering the state nodes."""
        self.periodic = bool(periodic)
        def look(name, g):
            gc, vc = coords[find_name(coords, name)]
            j = np.clip(np.searchsorted(gc, g), 0, len(gc) - 1)
            if not np.array_equal(gc[j], g):
                raise KeyError(f"{name} does not cover all state nodes")
            return vc[j]

        self.lines = []           # per line: (indices into the state vector sorted by y, y values)
        off = 0
        y_all = []
        for b in layout.blocks:
            g = b.gids
            x, y, z = look("coordinate 1", g), look("coordinate 2", g), look("coordinate 3", g)
            y_all.append(y)
            xk = np.round(x / xtol).astype(np.int64)
            yk = np.round(y / ytol).astype(np.int64)
            # rank of each node within its column (x, y), by z
            col = xk * 10 ** 9 + yk
            order = np.lexsort((z, col))
            rank = np.empty(len(g), np.int64)
            cs = col[order]
            start = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
            for a, e in zip(start, np.r_[start[1:], len(cs)]):
                rank[order[a:e]] = np.arange(e - a)
            # lines across the flow: same x and same rank
            key = xk * 10 ** 6 + rank
            o = np.lexsort((y, key))
            ks = key[o]
            st = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
            for a, e in zip(st, np.r_[st[1:], len(ks)]):
                idx = o[a:e]
                self.lines.append((off + idx, y[idx]))
            off += len(g)
        self.n = off
        y_all = np.concatenate(y_all)
        self.y0, self.W = float(y_all.min()), float(y_all.max() - y_all.min())
        ny = {len(i) for i, _ in self.lines}
        if self.W <= 0 or ny == {1}:
            raise ValueError("the mesh has a single node across the flow: no spanwise structure")
        self.ny = max(ny)
        self.irregular = len(ny) > 1      # lines with differing node counts (should not happen)
        if self.irregular:
            raise ValueError(f"node lines across the flow with {sorted(ny)[:5]} ... nodes: the mesh is not structured "
                             "across the flow ([case] mesh_across = \"structured\")")
        # modes resolved: walls n = 0 .. ny - 1 (DCT-I); periodic n = 0 .. (ny - 1) // 2 (ny - 1 distinct nodes)
        self.nmax = (self.ny - 1) // 2 if self.periodic else self.ny - 1

    def project(self, v):
        """Remove the across-flow mean of every line (the n = 0 part)."""
        v = np.array(v, dtype=complex if np.iscomplexobj(v) else float, copy=True)
        for idx, y in self.lines:
            w = self._weights(y)
            v[idx] -= np.sum(w * v[idx]) / np.sum(w)
        return v

    def _weights(self, y):
        """Trapezoid weights on the line's y nodes (DCT-I weights on a uniform grid)."""
        if len(y) == 1:
            return np.ones(1)
        dy = np.diff(y)
        w = np.zeros(len(y))
        w[:-1] += dy / 2
        w[1:] += dy / 2
        return w

    def spectrum(self, v, nmax=None):
        """Energy fraction in each n = 0..nmax, summed over all lines (complex v: |re|^2 + |im|^2): walls the
        cos(n pi y / W) part, periodic the cos and sin(2 pi n y / W) parts together."""
        nmax = self.nmax if nmax is None else nmax
        E = np.zeros(nmax + 1)
        f = 2.0 if self.periodic else 1.0
        for idx, y in self.lines:
            w = self._weights(y)
            eta = (y - self.y0) / self.W
            for nn in range(nmax + 1):
                bases = [np.cos(f * nn * np.pi * eta)]
                if self.periodic and nn > 0:
                    bases.append(np.sin(f * nn * np.pi * eta))
                for c in bases:
                    cc = np.sum(w * c * c)
                    if cc <= 1e-9 * np.sum(w):        # the sin at the Nyquist n: zero on the nodes
                        continue
                    a = np.sum(w * c * v[idx]) / cc
                    E[nn] += np.abs(a) ** 2 * cc
        tot = E.sum()
        return E / tot if tot > 0 else E

    def wavenumber(self, n):
        """k_n (1/m) and the wavelength (m): walls n pi / W and 2 W / n, periodic 2 pi n / W and W / n."""
        f = 2.0 if self.periodic else 1.0
        n = np.asarray(n, dtype=float)
        with np.errstate(divide="ignore"):
            lam = np.where(n > 0, 2 * self.W / (f * np.maximum(n, 1e-300)), np.inf)
        return f * n * np.pi / self.W, lam
