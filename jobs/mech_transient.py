#!/usr/bin/env python3
"""A purely MECHANICAL free transient from a converged steady state: Stokes + free surface + the grounding line by
contact every step, and nothing else. No heat solve, no strain/friction heating, no water sheet, no remasking; the
whole bed stays temperate (ess run --bed temperate --stokes-only --no-remask). It is jobs/free_transient.py (same mesh:
250 m from --fine-from-km to 1065 km, same transfer of the state, same shelf break, same --report) with only the final
`ess run` command rewritten, so every option of free_transient.py works here too.

    python3 jobs/mech_transient.py configs/flowline_rect_coarse.toml --from-state work/flowline_rect_coarse/steady_temperate.npz \
        --tag T1039 --fine-from-km 700 [--dt 5] [--steps 40] [--save-every 5]
    python3 jobs/mech_transient.py configs/flowline_rect_coarse.toml --from-state ... --tag T1039 --report
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import free_transient as ft  # noqa: E402

_run = ft.run


def mechanical(cmd):
    """The `ess run` call of free_transient with the mechanical flags; every other command unchanged."""
    if len(cmd) > 5 and cmd[3:5] == ["ess", "run"]:
        cmd = [("--no-remask" if c == "--remask" else c) for c in cmd]
        if "--no-remask" not in cmd:
            cmd.append("--no-remask")
        cmd += ["--bed", "temperate", "--stokes-only"]
    return cmd


def run(cmd):
    return _run(mechanical(list(cmd)))


if __name__ == "__main__":
    if "--thermal" not in sys.argv:             # the frozen thermal option adds none of ElmerStreams' thermal config keys
        sys.argv += ["--thermal", "frozen"]
    ft.run = run
    ft.main()
