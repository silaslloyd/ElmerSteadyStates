"""Configuration (TOML) and model construction."""
from __future__ import annotations

import logging
import os
import sys

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

from .continuation import ContinuationOptions
from .model import MapSpec
from .newton import NewtonOptions
from .stability import StabilityOptions

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger("ess")


def load_config(path: str) -> dict:
    if tomllib is None:
        raise SystemExit("reading the config needs Python >= 3.11 or `pip install tomli`")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("paths", {})
    cfg.setdefault("model", {"type": "elmer"})
    cfg.setdefault("elmer", {})
    cfg.setdefault("state", {"variables": ["Temperature", "Zs"]})
    cfg.setdefault("parameter", {"name": "Tsurf", "value": 0.0, "scale": 1.0})
    cfg["_repo"] = REPO
    cfg["_config"] = os.path.abspath(path)
    work = cfg["paths"].get("work_dir", "work/" + os.path.splitext(os.path.basename(path))[0])
    case = cfg["paths"].get("case_dir", "")
    fmt = dict(repo=REPO, case_dir=case)

    def p(s):
        s = str(s).format(**fmt, work=cfg.get("_work", ""))
        s = os.path.expanduser(os.path.expandvars(s))
        return s if os.path.isabs(s) else os.path.normpath(os.path.join(REPO, s))

    cfg["_path"] = p
    cfg["paths"]["case_dir"] = p(case) if case else ""
    fmt["case_dir"] = cfg["paths"]["case_dir"]
    cfg["_work"] = p(work)
    os.makedirs(cfg["_work"], exist_ok=True)
    return cfg


def setup_logging(cfg, verbose=False):
    lg = logging.getLogger("ess")
    lg.setLevel(logging.DEBUG)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(cfg["_work"], "ess.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg


def _opts(cls, d):
    names = set(cls.__dataclass_fields__)
    unknown = set(d) - names
    if unknown:
        raise SystemExit(f"unknown {cls.__name__} keys in config: {', '.join(sorted(unknown))}")
    return cls(**d)


def map_spec(cfg, section="map") -> MapSpec:
    m = cfg.get(section, cfg.get("map", {}))
    return MapSpec(float(m.get("dt", 1.0)), int(m.get("n_steps", 1)), int(m.get("coupled_iters", 1)))


def newton_options(cfg) -> NewtonOptions:
    return _opts(NewtonOptions, dict(cfg.get("newton", {})))


def continuation_options(cfg) -> ContinuationOptions:
    d = dict(cfg.get("continuation", {}))
    d.pop("stability_every", None)
    d.pop("name", None)
    d.pop("stability_m", None)
    return _opts(ContinuationOptions, d)


def stability_options(cfg) -> StabilityOptions:
    return _opts(StabilityOptions, dict(cfg.get("stability", {})))


def fd_settings(cfg):
    fd = cfg.get("fd", {})
    return float(fd.get("eps", 1e-4)), fd.get("scheme", "forward")


def build_model(cfg, saved=None):
    kind = cfg["model"].get("type", "elmer")
    if kind == "toy":
        from .toy import ToyColumn
        kw = {k: v for k, v in cfg["model"].items() if k != "type"}
        return ToyColumn(**kw)
    if kind == "elmer":
        from .elmer import ElmerModel
        return ElmerModel(cfg, saved.layout if saved else None, saved.frozen if saved else None)
    raise SystemExit(f"unknown model type {kind}")
