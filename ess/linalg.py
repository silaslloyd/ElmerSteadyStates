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


def _orth(V, w, k):
    """Orthogonalise w against V[0..k-1]; returns w, coefficients (two passes of MGS)."""
    h = np.zeros(k)
    for _ in range(2):
        for i in range(k):
            c = V[i] @ w
            h[i] += c
            w = w - c * V[i]
    return w, h


def gmres(matvec, b, x0=None, tol=1e-3, restart=40, maxiter=80, callback=None) -> tuple[np.ndarray, GmresInfo]:
    """Solve A x = b to ||b - A x|| <= tol ||b||. matvec(v) returns A v."""
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
            if res <= tol or breakdown or total >= maxiter:
                break
            V.append(w / np.linalg.norm(w))
        yk = np.linalg.solve(np.triu(H[:k, :k]), g[:k])
        x = x + np.array(V[:k]).T @ yk
        info.residual = abs(g[k]) / bnorm
        if info.residual <= tol:
            info.converged = True
            break
        if total >= maxiter:
            break
    info.iterations = total
    return x, info


@dataclass
class ArnoldiState:
    V: np.ndarray      # (m+1, n) orthonormal basis (rows)
    H: np.ndarray      # (m+1, m) Hessenberg
    k: int             # number of completed steps


def arnoldi(matvec, v0, m, checkpoint: str | None = None, callback=None) -> ArnoldiState:
    """m steps of Arnoldi; resumes from (and saves to) an .npz checkpoint after every product."""
    n = len(v0)
    st = None
    if checkpoint and os.path.exists(checkpoint):
        with np.load(checkpoint) as z:
            if z["V"].shape[1] == n:
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
                np.savez(checkpoint, V=st.V, H=st.H, k=st.k)
            break
        st.V[j + 1] = w / hn
        if checkpoint:
            np.savez(checkpoint + ".tmp.npz", V=st.V, H=st.H, k=st.k)
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
