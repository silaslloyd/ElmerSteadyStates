"""
Matrix-free GMRES and Arnoldi. Every matvec is one or two model runs, so these keep the Krylov basis
(cheap) and minimise products (expensive): modified Gram-Schmidt with one re-orthogonalisation pass.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("ess")


@dataclass
class GmresInfo:
    converged: bool
    iterations: int
    residual: float            # relative residual estimate ||b - A x|| / ||b||
    history: list = field(default_factory=list)
    stalled: bool = False


def _orth(V, w, k):
    """Orthogonalise w against V[0..k-1]; returns w, coefficients (two passes of MGS)."""
    h = np.zeros(k)
    for _ in range(2):
        for i in range(k):
            c = V[i] @ w
            h[i] += c
            w = w - c * V[i]
    return w, h


def gmres(matvec, b, x0=None, tol=1e-3, restart=40, maxiter=80, callback=None,
          stall_window=10, stall_factor=0.97) -> tuple[np.ndarray, GmresInfo]:
    """Solve A x = b to ||b - A x|| <= tol ||b||. matvec(v) returns A v.
    Stops early (not converged) when the residual has fallen by less than stall_factor over the last
    stall_window iterations: with expensive products, a stagnating solve is better cut short and
    its (best so far) step used."""
    b = np.asarray(b, float)
    n = b.size
    bnorm = np.linalg.norm(b)
    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
    info = GmresInfo(False, 0, 1.0)
    if bnorm == 0:
        info.converged, info.residual = True, 0.0
        return np.zeros(n), info
    total = 0
    while True:
        if np.any(x):
            r = b - matvec(x)
            total += 1
        else:
            r = b.copy()
        beta = np.linalg.norm(r)
        info.residual = beta / bnorm
        if beta <= tol * bnorm:
            info.converged = True
            break
        if total >= maxiter:
            break
        m = min(restart, maxiter - total)
        V = [r / beta]
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        k = 0
        for j in range(m):
            w = matvec(V[j])
            total += 1
            w, h = _orth(V, w, j + 1)
            H[: j + 1, j] = h
            H[j + 1, j] = np.linalg.norm(w)
            for i in range(j):                       # previous rotations
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            den = np.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = (1.0, 0.0) if den == 0 else (H[j, j] / den, H[j + 1, j] / den)
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            res = abs(g[j + 1]) / bnorm
            info.history.append(res)
            if callback:
                callback(total, res)
            breakdown = np.linalg.norm(w) <= 1e-14 * max(beta, 1e-300)
            h = info.history
            stalled = stall_window and len(h) > stall_window and h[-1] > stall_factor * h[-1 - stall_window]
            if stalled:
                info.stalled = True
            if res <= tol or breakdown or total >= maxiter or stalled:
                break
            V.append(w / np.linalg.norm(w))
        yk = np.linalg.solve(np.triu(H[:k, :k]), g[:k])
        x = x + np.array(V[:k]).T @ yk
        info.residual = abs(g[k]) / bnorm
        if info.residual <= tol:
            info.converged = True
            break
        if total >= maxiter or info.stalled:
            break
    info.iterations = total
    return x, info


@dataclass
class ArnoldiState:
    V: np.ndarray      # (m+1, n) orthonormal basis (rows)
    H: np.ndarray      # (m+1, m) Hessenberg
    k: int             # number of completed steps


def arnoldi(matvec, v0, m, checkpoint: str | None = None, callback=None, key: str = "") -> ArnoldiState:
    """m steps of Arnoldi; resumes from (and saves to) an .npz checkpoint after every product.
    `key` identifies the operator (state, map settings): a checkpoint with a different key is ignored."""
    n = len(v0)
    st = None
    if checkpoint and os.path.exists(checkpoint):
        with np.load(checkpoint) as z:
            same = z["V"].shape[1] == n and ("key" in z.files and str(z["key"]) == key)
            if not same:
                log.info(f"arnoldi: checkpoint {checkpoint} is for a different state or map, starting afresh")
            if same:
                V0, H0, k0 = z["V"], z["H"], int(z["k"])
                V = np.zeros((m + 1, n))
                H = np.zeros((m + 1, m))
                kk = min(k0, m)
                V[: kk + 1] = V0[: kk + 1]
                H[: kk + 1, :kk] = H0[: kk + 1, :kk]
                st = ArnoldiState(V, H, kk)
                log.info(f"arnoldi: resumed {kk} steps from {checkpoint}")
    if st is None:
        V = np.zeros((m + 1, n))
        H = np.zeros((m + 1, m))
        V[0] = v0 / np.linalg.norm(v0)
        st = ArnoldiState(V, H, 0)
    for j in range(st.k, m):
        w = matvec(st.V[j])
        w, h = _orth(st.V, w, j + 1)
        st.H[: j + 1, j] = h
        hn = np.linalg.norm(w)
        st.H[j + 1, j] = hn
        st.k = j + 1
        if hn <= 1e-14 * np.abs(h).max():
            log.info("arnoldi: invariant subspace found")
            st.V[j + 1] = 0
            if checkpoint:
                np.savez(checkpoint, V=st.V, H=st.H, k=st.k, key=key)
            break
        st.V[j + 1] = w / hn
        if checkpoint:
            np.savez(checkpoint + ".tmp.npz", V=st.V, H=st.H, k=st.k, key=key)
            os.replace(checkpoint + ".tmp.npz", checkpoint)
        if callback:
            callback(st)
    return st


def ritz(st: ArnoldiState):
    """Ritz values (sorted by decreasing modulus), Ritz vectors (columns) and residual norms."""
    k = st.k
    Hk = st.H[:k, :k]
    mu, Y = np.linalg.eig(Hk)
    res = np.abs(st.H[k, k - 1] * Y[-1, :]) if k < st.H.shape[0] else np.zeros(k)
    order = np.argsort(-np.abs(mu))
    mu, Y, res = mu[order], Y[:, order], res[order]
    X = st.V[:k].T @ Y
    return mu, X, res


class Recycler:
    """Deflated GMRES with a recycled subspace (matrix-free, no extra products).

    The slow modes of the map (D Phi ~ 1, J ~ 0: thermal, with a time scale far over the horizon) are what costs GMRES its
    iterations. Every product A v the solves make is kept; the harmonic Ritz vectors of the smallest |theta| over everything
    kept (the k that were deflated before and the new products) span the next deflation space U, with C = A U known without a
    product. A solve starts from x0 = U (C^T b) and runs GMRES on (I - C C^T) A (r0 = (I - C C^T) b), x = x0 + z - U (C^T A z):
    exact in the residual, every iteration one product as before. A stale U (the operator moved on) only makes the start
    worse than it could be: the residual is always the true one.  k = 0: plain GMRES.
    """

    def __init__(self, k: int, max_store: int | None = None):
        self.k = int(k)
        self.max_store = int(max_store) if max_store else max(3 * self.k, 20)
        self.U = self.C = None            # (n, m) with A U = C, C orthonormal
        self.n = None

    def reset(self):
        self.U = self.C = None

    def solve(self, matvec, b, tol=1e-3, restart=40, maxiter=80, callback=None, **kw):
        b = np.asarray(b, float)
        if self.k <= 0:
            return gmres(matvec, b, tol=tol, restart=restart, maxiter=maxiter, callback=callback, **kw)
        if self.n is not None and self.n != b.size:
            self.reset()
        self.n = b.size
        pairs_v, pairs_w = [], []

        def A(v):                                  # the raw product, kept
            w = matvec(v)
            pairs_v.append(np.array(v, float))
            pairs_w.append(np.array(w, float))
            return w

        bnorm = np.linalg.norm(b)
        if self.U is not None and self.U.shape[1]:
            U, C = self.U, self.C
            x0 = U @ (C.T @ b)
            r0 = b - C @ (C.T @ b)
            log.info(f"  gmres: deflating {U.shape[1]} recycled vectors, start residual {np.linalg.norm(r0) / bnorm:.2e} of ||b||")

            def PA(v):
                w = A(v)
                return w - C @ (C.T @ w)

            if np.linalg.norm(r0) <= tol * bnorm:
                from_info = GmresInfo(True, 0, float(np.linalg.norm(r0) / bnorm))
                return x0, from_info
            z, info = gmres(PA, r0, tol=tol * bnorm / np.linalg.norm(r0), restart=restart, maxiter=maxiter, callback=callback, **kw)
            # x = x0 + z - U (C^T A z): A z from the kept products is not available for an arbitrary combination, but
            # P A z = A z - C C^T A z and C^T A z is needed: one more product unless z came out of the Krylov basis
            # whose products are kept; rebuild A z from them (A linear: A z = sum c_i A v_i is not directly known), so
            # take the product (counted in info.iterations)
            Az = A(z)
            x = x0 + z - U @ (C.T @ Az)
            info.iterations += 1
            info.residual = float(np.linalg.norm(b - C @ (C.T @ b) - (Az - C @ (C.T @ Az))) / bnorm)
        else:
            x, info = gmres(A, b, tol=tol, restart=restart, maxiter=maxiter, callback=callback, **kw)
        self._update(pairs_v, pairs_w)
        return x, info

    def _update(self, vs, ws):
        """harmonic Ritz vectors of the smallest |theta| over the old (U, C) and the new products"""
        V = list(vs[-self.max_store:])
        W = list(ws[-self.max_store:])
        if self.U is not None:
            V = [self.U[:, i] for i in range(self.U.shape[1])] + V
            W = [self.C[:, i] for i in range(self.C.shape[1])] + W
        if len(V) < 2:
            return
        V = np.array(V).T
        W = np.array(W).T
        Q, R = np.linalg.qr(V)
        if np.min(np.abs(np.diag(R))) <= 1e-12 * np.max(np.abs(np.diag(R))):      # dependent columns: drop the tail
            keep = np.abs(np.diag(R)) > 1e-12 * np.max(np.abs(np.diag(R)))
            V, W = V[:, keep], W[:, keep]
            Q, R = np.linalg.qr(V)
        AQ = np.linalg.solve(R.T, W.T).T                     # A Q = W R^-1
        # harmonic Ritz: (AQ)^T (AQ) g = theta (AQ)^T Q g
        G1 = AQ.T @ AQ
        G2 = AQ.T @ Q
        try:
            theta, g = np.linalg.eig(np.linalg.solve(G2, G1))
        except np.linalg.LinAlgError:
            return
        o = np.argsort(np.abs(theta))[: self.k]
        g = g[:, o]
        # real basis of the chosen (possibly complex-pair) vectors
        Gr = np.concatenate([g.real, g.imag], axis=1)
        Gr = Gr[:, np.linalg.norm(Gr, axis=0) > 1e-12]
        if Gr.shape[1] == 0:
            return
        Gq, _ = np.linalg.qr(Gr)
        Gq = Gq[:, : self.k + (self.k % 2)]
        U = Q @ Gq
        AU = AQ @ Gq
        C, Rc = np.linalg.qr(AU)
        if np.min(np.abs(np.diag(Rc))) <= 1e-12 * np.max(np.abs(np.diag(Rc))):
            return
        self.U = np.linalg.solve(Rc.T, U.T).T                # U R^-1: A U' = C
        self.C = C
        log.info(f"  gmres: recycling {self.U.shape[1]} vectors (smallest |theta| {np.sort(np.abs(theta))[0]:.2e} .. "
                 f"{np.sort(np.abs(theta))[min(self.k, len(theta)) - 1]:.2e})")
