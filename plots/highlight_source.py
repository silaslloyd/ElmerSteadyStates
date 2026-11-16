#!/usr/bin/env python3
"""A source file as one self-contained HTML page with chosen line ranges highlighted, one colour per change (the
ColdTempMaskSolver page of 2026-11-16): line numbers, a legend, "jump to" links to every highlighted block, light and
dark mode, no external files.

    python3 plots/highlight_source.py FILE OUT.html --mark "LABEL:RANGES" [--mark "LABEL:RANGES" ...] [--title T]

LABEL: the change's description (shown in the legend); RANGES: line numbers, e.g. 73-76,151-161,207-215. Up to six
changes (orange, blue, green, purple, red, teal); a line in two changes takes the first.
E.g.
    python3 plots/highlight_source.py elmer/ColdTempMaskSolver.F90 ctmask.html \\
        --mark "the shelf: Floating Nodes Temperate (b8b309b):73-76,151-161,207-215" \\
        --mark "the 1e-4 reset at the previous level + once per call (c21718b, 2bfa84f):190-195,237-241"
Line ranges of a commit's changes: git show -U0 COMMIT -- FILE | grep '^@@' (the +start,count of each hunk).
"""
import argparse
import html
import os

# (fill, text/bar) light, (fill, text/bar) dark
COLOURS = [("#fde7c8", "#b85c00", "#4a3416", "#ffb35c"), ("#d6ecfa", "#0b5c8f", "#13344a", "#7cc4f2"),
           ("#dcf2d8", "#2e7d32", "#1f3a1d", "#8fd18a"), ("#ece0f7", "#6a3d9a", "#33244a", "#c7a3ef"),
           ("#fadcdc", "#b3261e", "#4a1d1b", "#f2938d"), ("#d4f1ee", "#00796b", "#123a36", "#6fd6c9")]


def ranges(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--mark", action="append", default=[], help='"LABEL:RANGES"')
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    if not a.mark or len(a.mark) > len(COLOURS):
        raise SystemExit(f"give 1..{len(COLOURS)} --mark LABEL:RANGES")
    marks = []
    for m in a.mark:
        label, _, spec = m.rpartition(":")
        if not label:
            raise SystemExit(f"--mark {m!r}: LABEL:RANGES")
        marks.append((label, ranges(spec)))
    lines = open(a.src, errors="replace").read().split("\n")
    tag = [None] * (len(lines) + 1)
    for k, (_, rs) in enumerate(marks):
        for n in rs:
            if 1 <= n <= len(lines) and tag[n] is None:
                tag[n] = k
    name = os.path.basename(a.src)
    title = a.title or f"{name}: the changes"
    css = [":root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#8a877f;--line:#e8e5de;"
           + "".join(f"--c{k}:{c[0]};--t{k}:{c[1]};" for k, c in enumerate(COLOURS)) + "}",
           "@media (prefers-color-scheme:dark){:root:not([data-theme=\"light\"]){--bg:#17171a;--fg:#e8e6e1;"
           "--mut:#77756f;--line:#2a2a2e;" + "".join(f"--c{k}:{c[2]};--t{k}:{c[3]};" for k, c in enumerate(COLOURS))
           + "}}",
           "body{background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,sans-serif;margin:0;"
           "padding:16px}",
           "h1{font-size:18px;margin:0 0 6px} p{margin:4px 0 12px;color:var(--mut)}",
           ".legend span{display:inline-block;padding:2px 8px;border-radius:4px;margin:0 8px 4px 0}",
           "nav{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2}",
           "nav a{text-decoration:none;font-weight:600;padding:2px 8px;border-radius:4px;margin-right:6px}",
           ".wrap{overflow-x:auto} table{border-collapse:collapse;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,"
           "monospace;width:100%}",
           "td{padding:0 8px;white-space:pre;vertical-align:top} td.n{color:var(--mut);text-align:right;"
           "user-select:none;width:3.5em}",
           "tr.tag td{font:600 12px ui-sans-serif,system-ui,sans-serif;padding-top:10px} tr.tag{scroll-margin-top:48px}"]
    for k in range(len(marks)):
        css.append(f".k{k}{{background:var(--c{k});color:var(--t{k})}} tr.m{k} td{{background:var(--c{k})}} "
                   f"tr.m{k} td:first-child{{border-left:4px solid var(--t{k})}} tr.tag.m{k} td{{color:var(--t{k});"
                   f"background:none}} tr.tag.m{k} td:first-child{{border-left:none}}")
    letters = "ABCDEF"
    legend = "".join(f'<span class="k{k}">{letters[k]} · {html.escape(lb)}</span>' for k, (lb, _) in enumerate(marks))
    rows, jumps, count = [], [], [0] * len(marks)
    for n in range(1, len(lines) + 1):
        k = tag[n]
        if k is not None and (n == 1 or tag[n - 1] != k):     # a block starts: a label row and a jump link
            count[k] += 1
            ident = f"L{n}"
            jumps.append(f'<a class="k{k}" href="#{ident}">{letters[k]}{count[k]}</a>')
            rows.append(f'<tr class="tag m{k}" id="{ident}"><td class="n"></td><td>{letters[k]}{count[k]} · '
                        f'{html.escape(marks[k][0])}</td></tr>')
        cls = f' class="m{k}"' if k is not None else ""
        rows.append(f'<tr{cls}><td class="n">{n}</td><td><code>{html.escape(lines[n - 1])}</code></td></tr>')
    page = ("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,"
            f"initial-scale=1\">\n<title>{html.escape(title)}</title><style>\n" + "\n".join(css) + "\n</style></head>"
            f"<body>\n<h1>{html.escape(title)}</h1>\n<p class=\"legend\">{legend}</p>\n"
            f"<nav>Jump to: {' '.join(jumps)}</nav>\n<div class=\"wrap\"><table>" + "".join(rows)
            + "</table></div>\n</body></html>\n")
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out}: {sum(count)} blocks in {len(marks)} colour(s), {len(lines)} lines")


if __name__ == "__main__":
    main()
