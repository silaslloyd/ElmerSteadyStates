"""Tests on the stand-in model (exact steady states and fold) and of the Elmer plumbing (fake Elmer)."""
import os
import sys
import textwrap

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ess.continuation import BranchStore, ContinuationOptions, continuation  # noqa: E402
from ess.elmer import override_lua_assignment  # noqa: E402
from ess.linalg import arnoldi, gmres, ritz  # noqa: E402
from ess.model import FixedPointProblem, MapSpec, rms  # noqa: E402
from ess.newton import NewtonOptions, newton_krylov  # noqa: E402
from ess.stability import StabilityOptions, stability  # noqa: E402
from ess.statefile import (Block, StateLayout, load_state, merge_partitions, read_state_file, save_state,  # noqa: E402
                           write_state_file)
from ess.toy import ToyColumn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- linear algebra

def test_gmres_and_arnoldi():
    rng = np.random.default_rng(0)
    A = np.eye(60) + 0.3 * rng.standard_normal((60, 60)) / np.sqrt(60)
    b = rng.standard_normal(60)
    x, info = gmres(lambda v: A @ v, b, tol=1e-10, restart=15, maxiter=200)
    assert info.converged and np.linalg.norm(A @ x - b) < 1e-9 * np.linalg.norm(b)
    D = np.diag(np.r_[3.0, 2.0, 1.5, rng.uniform(0, 0.5, 57)])
    st = arnoldi(lambda v: D @ v, rng.standard_normal(60), 25)
    mu, X, res = ritz(st)
    assert np.allclose(mu[:3].real, [3.0, 2.0, 1.5], atol=1e-8)


# ---------------------------------------------------------------- state files

def test_state_file_roundtrip_and_merge(tmp_path):
    g = np.arange(1, 11, dtype=np.int64)
    v = np.random.default_rng(1).standard_normal(10)
    write_state_file(str(tmp_path / "a.0"), {"Temperature": (g[:6], v[:6]), "Zs": (g[:2], v[:2])})
    write_state_file(str(tmp_path / "a.1"), {"Temperature": (g[4:], v[4:]), "Zs": (g[1:3], v[1:3])})
    rec, mism = merge_partitions([str(tmp_path / "a.0"), str(tmp_path / "a.1")])
    assert np.array_equal(rec["Temperature"][0], g) and np.array_equal(rec["Temperature"][1], v)
    assert mism["Temperature"] == 0.0
    r2 = read_state_file(str(tmp_path / "a.0"))
    assert np.array_equal(r2["Zs"][1], v[:2])
    lay = StateLayout([Block("Temperature", g, 5.0), Block("Zs", g[:3], 50.0)])
    x = lay.to_vector(rec)
    save_state(str(tmp_path / "s.npz"), lay, x, 3.5, {"m": (g, v)}, None, note="x")
    s = load_state(str(tmp_path / "s.npz"))
    assert np.array_equal(s.x, x) and s.p == 3.5 and s.layout.scale_vector()[0] == 5.0


def test_lua_override():
    txt = "T0 = 273.15\nTscale = 5\nTsurf = T0 - Tscale    -- surface\nTsurfX = 1\n"
    out = override_lua_assignment(txt, "Tsurf", 265.0)
    assert "Tsurf = 265.0" in out and "TsurfX = 1" in out and "T0 = 273.15" in out
    with pytest.raises(KeyError):
        override_lua_assignment(txt, "nothere", 1)


# ---------------------------------------------------------------- solvers on the toy model

@pytest.mark.parametrize("coupling", ["staggered", "implicit"])
def test_newton_finds_unstable_middle_branch(coupling):
    m = ToyColumn(coupling=coupling)
    # implicit Euler is singular when sigma dt = 1 on the unstable branch (sigma ~ 17 here): keep dt small
    prob = FixedPointProblem(m, MapSpec(0.05, 4) if coupling == "staggered" else MapSpec(0.01, 10), fd_eps=1e-6)
    Ts = -1.0
    Tb = m.steady_Tb(Ts, "warm")
    x, _ = m.steady_state(Tb)
    r = newton_krylov(prob, x + 0.02 * np.cos(np.arange(len(x))), Ts, NewtonOptions(tol=1e-11))
    assert r.converged
    assert abs(r.y[0] - Tb) < 1e-8
    sig = np.sort(np.linalg.eigvals(m.jac(r.y, Ts)).real)[::-1]
    if coupling == "implicit":
        # backward Euler: sigma_be inverts mu = (1 - sigma dt)^-n exactly
        st = stability(prob, r.y, Ts, MapSpec(0.002, 10), StabilityOptions(m=10, nev=3))
        assert abs(st.sigma_be[0].real - sig[0]) < 1e-3 * abs(sig[0])
    else:
        # the staggered (lagged friction) step has an O(dt) splitting error in the growth rate
        prob.fd_scheme = "central"
        st = stability(prob, r.y, Ts, MapSpec(0.0005, 40), StabilityOptions(m=10, nev=3))
        assert abs(st.sigma[0].real - sig[0]) < 0.03 * abs(sig[0])
    assert st.n_unstable == 1


def test_stability_matches_exact_map_jacobian():
    m = ToyColumn()
    Ts = -1.0
    x, _ = m.steady_state(m.steady_Tb(Ts, "cold"))
    spec = MapSpec(0.01, 10)
    prob = FixedPointProblem(m, spec, fd_eps=1e-7, fd_scheme="central")
    st = stability(prob, x, Ts, spec, StabilityOptions(m=15, nev=3))
    ev = np.linalg.eigvals(m.map_jacobian(x, Ts, spec))
    ev = ev[np.argsort(-np.abs(ev))][:3]
    assert np.allclose(st.mu, ev, rtol=1e-5)
    assert st.n_unstable == 0


def test_continuation_through_fold(tmp_path):
    m = ToyColumn()
    prob = FixedPointProblem(m, MapSpec(0.05, 4), fd_eps=1e-6, p_scale=0.03)
    Ts0 = -1.0
    x0, _ = m.steady_state(m.steady_Tb(Ts0, "cold"))
    store = BranchStore(str(tmp_path), prob)
    continuation(prob, x0, Ts0, store, ContinuationOptions(ds=0.3, dp0=0.05, ds_max=1.0, ds_min=1e-4,
                                                            max_points=40, tol=1e-9, p_min=-1.0),
                 NewtonOptions(tol=1e-10))
    folds = [r for r in store.rows if str(r["fold"]) == "1"]
    assert len(folds) == 1
    assert abs(float(folds[0]["p_fold_estimate"]) - m.fold()[1]) < 1e-5
    for r in store.rows:                     # every point is an exact steady state
        y, p, _ = store.load(int(r["index"]))
        assert abs(m.Ts_of_Tb(y[0]) - p) < 1e-7
    y, p, _ = store.load(len(store) - 1)
    assert y[0] > m.fold()[0]                # ended on the warm (middle) branch
    # resume: a new store on the same directory continues from the last two points
    store2 = BranchStore(str(tmp_path), prob)
    n0 = len(store2)
    continuation(prob, None, None, store2, ContinuationOptions(ds=0.3, ds_max=1.0, ds_min=1e-4,
                                                               max_points=n0 + 2, tol=1e-9))
    assert len(store2) == n0 + 2


# ---------------------------------------------------------------- Elmer plumbing with a fake Elmer

def test_elmer_driver_with_fake_elmer(tmp_path):
    from ess.cli import main
    case = tmp_path / "case"
    (case / "footprint").mkdir(parents=True)
    (case / "parameters.lua").write_text("T0 = 273.15\nTscale = 5\nTsurf = T0 - Tscale\n")
    work = tmp_path / "work"
    cfg = tmp_path / "fake.toml"
    cfg.write_text(textwrap.dedent(f"""
        [paths]
        work_dir = "{work}"
        case_dir = "{case}"
        [case]
        parameters = "{case}/parameters.lua"
        [elmer]
        n_procs = 3
        mpirun = "{sys.executable} {HERE}/fake_elmer.py {{n}}"
        solver = "ElmerSolver"
        ess_lib = "{tmp_path}/SteadyStateIO"
        diagnostics = ["Temperature Loads"]
        [state]
        variables = ["Temperature", "Zs"]
        scales = {{ Temperature = 5.0, Zs = 50.0 }}
        frozen = ["ColdTempMask", "GroundedMask", "Zb", "bedrock"]
        [extract]
        restart_file = "fake.result"
        [parameter]
        name = "Tsurf"
        value = 265.0
        scale = 1.0
        [map]
        dt = 1.0
        n_steps = 3
        [fd]
        eps = 1e-5
        [newton]
        tol = 1e-10
        """))
    assert main(["extract", str(cfg)]) == 0
    s0 = load_state(str(work / "state0.npz"))
    assert s0.layout.names == ["Temperature", "Zs"] and len(s0.frozen) == 4
    assert main(["check", str(cfg), "--eps", "1e-3", "1e-5"]) == 0
    assert main(["newton", str(cfg)]) == 0
    s = load_state(str(work / "steady_p265.npz"))
    from tests.fake_elmer import GID, TOP, tstar  # noqa: E402
    zs = 100 + 0.01 * TOP
    T = s.layout.split(s.x)["Temperature"]
    assert np.allclose(s.layout.split(s.x)["Zs"], zs, atol=1e-8)
    assert np.allclose(T, tstar(265.0, zs), atol=1e-8)
    # the cache: repeating the newton solve needs no new runs
    n_runs = len(os.listdir(work / "cache"))
    assert main(["newton", str(cfg), "--state", str(work / "steady_p265.npz")]) == 0
    assert len(os.listdir(work / "cache")) <= n_runs + 1


def test_case_setup_initial_conditions(tmp_path):
    from ess.case import eval_lua_parameters, initial_geometry, write_ics
    P = eval_lua_parameters(os.path.join(os.path.dirname(HERE), "case", "parameters.lua"))
    assert abs(P["Tsurf"] - 268.15) < 1e-12 and P["p0"] == 0.0
    g = initial_geometry(P)
    xm = g["x"][g["gm"] > 0].max()
    assert abs(xm - 0.8 * P["L"]) < 0.01 * P["L"]
    assert 1000 < g["H"].max() < 4000 and np.all(g["Tb"] <= P["T0"])
    write_ics(g, str(tmp_path))
    assert (tmp_path / "ics.lua").read_text().count("function") == 3
