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
    _sector_derived(cfg.get("case", {}))
    cfg["_repo"] = REPO
    cfg["_config"] = os.path.abspath(path)
    work = cfg["paths"].get("work_dir", "work/" + os.path.splitext(os.path.basename(path))[0])
    fmt = dict(repo=REPO, case_dir="", work="")

    def p(s):
        s = str(s).format(**fmt)
        s = os.path.expanduser(os.path.expandvars(s))
        return s if os.path.isabs(s) else os.path.normpath(os.path.join(REPO, s))

    cfg["_path"] = p
    cfg["_work"] = fmt["work"] = p(work)
    cfg["paths"]["case_dir"] = fmt["case_dir"] = p(cfg["paths"].get("case_dir", "{work}/case"))
    os.makedirs(cfg["_work"], exist_ok=True)
    return cfg


def _sector_derived(c: dict) -> None:
    """[case] sector_copies = N: a sector that tiles the full circle, angle 360 / N degrees (sector_angle derived).
    With sector_inverted = true (the divide the outer arc) and sector_front_arc_km (default 100): the front arc that
    long, so the front at r = front_arc / angle and the divide at that + front_km (sector_divide_km derived); or
    sector_divide_arc_km instead: the divide at r = divide_arc / angle, the front front_km inside it.
    accumulation_factor = "match": the factor that gives the sector the strip's flux per unit arc at the grounding line
    s = sector_gl_km (default 791) from the divide (a uniform accumulation: sector flux a f (r_d^2 - r^2) / (2 r) in an
    inverted sector, a f (r^2 - r0^2) / (2 r) in a normal one, against the strip's a s)."""
    import math
    n = c.get("sector_copies")
    if n:
        n = int(n)
        if "sector_angle" in c and abs(float(c["sector_angle"]) - 360.0 / n) > 1e-9:
            raise SystemExit(f"[case] sector_copies = {n} and sector_angle = {c['sector_angle']} disagree")
        c["sector_angle"] = 360.0 / n
        th = 2 * math.pi / n
        if c.get("sector_inverted"):
            if "front_km" not in c:
                raise SystemExit("[case] sector_inverted needs front_km (the front's distance from the divide)")
            if c.get("sector_divide_arc_km") and c.get("sector_front_arc_km"):
                raise SystemExit("[case] give sector_front_arc_km OR sector_divide_arc_km (with sector_copies and "
                                 "front_km the other one follows)")
            if c.get("sector_divide_arc_km"):      # the divide arc that long: the divide at r = arc / angle
                c["sector_divide_km"] = float(c["sector_divide_arc_km"]) / th
            else:
                rf = float(c.get("sector_front_arc_km", 100.0)) / th
                c["sector_divide_km"] = rf + float(c["front_km"])
            if c["sector_divide_km"] <= float(c["front_km"]):
                raise SystemExit("[case] inverted sector: the front would be at r <= 0 (divide arc too short)")
    if str(c.get("accumulation_factor", "")).lower() == "match":
        if not c.get("sector_angle"):
            raise SystemExit('[case] accumulation_factor = "match" needs a sector')
        s_gl = float(c.get("sector_gl_km", 791.0))
        if c.get("sector_divide_km"):
            rd = float(c["sector_divide_km"])
            r = rd - s_gl
            c["accumulation_factor"] = 2 * r * s_gl / (rd ** 2 - r ** 2)
        else:
            r0 = float(c.get("sector_r0_km", 15.0))
            c["accumulation_factor"] = 2 * s_gl ** 2 / (s_gl ** 2 - r0 ** 2)


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
    return MapSpec(float(m.get("dt", 1.0)), int(m.get("n_steps", 1)), int(m.get("coupled_iters", 1)),
                   float(m.get("rock_heat_capacity", 1.0)))


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
