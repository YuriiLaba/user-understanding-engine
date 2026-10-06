"""Paper figure: F1 vs. average number of propositions, with the Pareto frontier.

One point per (modality subset x merging arm): x = propositions kept after merging,
y = F1 against the ideal. Values are averaged over extraction models within a session,
then over sessions, so uneven per-session coverage cannot tilt the mean. The frontier
(nothing is both smaller and more faithful) is drawn as a single polyline through the
non-dominated points -- straight segments, not a staircase.

Writes a vector PDF for \\includegraphics plus a PNG preview, and prints a ready
\\begin{figure} snippet.

Run: .venv/bin/python scripts/plot_pareto.py
     .venv/bin/python scripts/plot_pareto.py --user hlib --scale linear
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from build_comparison import collect

METHOD_ORDER = ["none", "embeddings_reranker", "embeddings_reranker_llm", "sage_time_window", "moving_window",
                "merged_observations"]
METHOD_LABEL = {
    "none": "1. no merging",
    "embeddings_reranker": "2. emb. + rerank",
    "embeddings_reranker_llm": "3.  + LLM",
    "sage_time_window": "4. SAGE + window",
    "moving_window": "5. moving window",
    "merged_observations": "6. merge obs. + propose",
}
ARM_NUM = {m: i + 1 for i, m in enumerate(METHOD_ORDER)}
METHOD_STYLE = {  # colour, marker  (colours validated for CVD separation; markers repeat the identity)
    "none": ("#6E7C79", "o"),
    "embeddings_reranker": ("#2a78d6", "s"),
    "embeddings_reranker_llm": ("#eb6834", "^"),
    "sage_time_window": ("#1baf7a", "D"),
    "moving_window": ("#4a3aa7", "v"),
    "merged_observations": ("#c2185b", "P"),
}
MOD_CODE = {"ax": "A", "metadata": "M", "ocr": "O", "screen": "S"}
short = lambda mods: "+".join(MOD_CODE.get(m, m[:1].upper()) for m in mods.split("+"))
label = lambda p: f"{short(p[0])}$^{{{ARM_NUM.get(p[1], 0)}}}$"  # subset + arm number, as in the table


def aggregate(rows, users, model, xkey, ykey="f1_score"):
    """[(modalities, method, x, y, n)] — mean over models within a session, then over sessions."""
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] != "ok" or r.get(ykey) is None or r.get(xkey) is None:
            continue
        if r["user"] not in users or (model != "all" and r["model"] != model):
            continue
        per[(r["modalities"], r["method"])][r["user"]].append((r[xkey], r[ykey]))
    out = []
    for (mods, method), by_user in per.items():
        xs, ys, n = [], [], 0
        for vals in by_user.values():
            xs.append(sum(v[0] for v in vals) / len(vals))
            ys.append(sum(v[1] for v in vals) / len(vals))
            n += len(vals)
        out.append((mods, method, sum(xs) / len(xs), sum(ys) / len(ys), n))
    return out


def frontier(points):
    """Non-dominated: no other point has x <= this x and y >= this y (with one strict)."""
    keep = [p for p in points if not any(q is not p and q[2] <= p[2] and q[3] >= p[3]
                                         and (q[2] < p[2] or q[3] > p[3]) for q in points)]
    return sorted(keep, key=lambda p: p[2])


def _annotate(ax, p, dx, dy, ha, va, fontsize):
    """Place one label; add a hairline leader when it had to sit far from its marker."""
    kw = {}
    if (dx * dx + dy * dy) ** 0.5 >= 15:
        kw["arrowprops"] = dict(arrowstyle="-", linewidth=0.4, color="#8A8A85", shrinkA=0, shrinkB=3)
    return ax.annotate(label(p), (p[2], p[3]), textcoords="offset points", xytext=(dx, dy),
                       ha=ha, va=va, fontsize=fontsize, zorder=6, **kw)


def place_labels(ax, fig, pts, front, fontsize):
    """Greedy label placement: first candidate offset that clears every marker and placed label."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for p in pts:
        x, y = ax.transData.transform((p[2], p[3]))
        boxes.append((x - 5, y - 5, x + 5, y + 5))
    cands = [(7, 0, "left", "center"), (-7, 0, "right", "center"),
             (0, 8, "center", "bottom"), (0, -8, "center", "top"),
             (6, 6, "left", "bottom"), (-6, 6, "right", "bottom"),
             (6, -6, "left", "top"), (-6, -6, "right", "top"),
             (0, 17, "center", "bottom"), (0, -17, "center", "top"),
             (14, 10, "left", "bottom"), (-14, 10, "right", "bottom"),
             (14, -10, "left", "top"), (-14, -10, "right", "top")]
    ax_box = ax.get_window_extent()
    for p in sorted(front, key=lambda p: -p[3]):
        best = None
        for dx, dy, ha, va in cands:
            t = _annotate(ax, p, dx, dy, ha, va, fontsize)
            bb = t.get_window_extent(rend).expanded(1.34, 1.40)
            r = (bb.x0, bb.y0, bb.x1, bb.y1)
            inside = r[0] >= ax_box.x0 and r[2] <= ax_box.x1 and r[1] >= ax_box.y0 and r[3] <= ax_box.y1
            clash = any(not (r[2] < b[0] or r[0] > b[2] or r[3] < b[1] or r[1] > b[3]) for b in boxes)
            if inside and not clash:
                best = r
                break
            t.remove()
        if best is None:  # nothing clear — take the slot that overlaps the least
            scored = []
            for dx, dy, ha, va in cands:
                t = _annotate(ax, p, dx, dy, ha, va, fontsize)
                bb = t.get_window_extent(rend)
                r = (bb.x0, bb.y0, bb.x1, bb.y1)
                pad = 3.0  # keep a visible gap even in the least-bad slot
                area = sum(max(0, min(r[2] + pad, b[2] + pad) - max(r[0] - pad, b[0] - pad))
                           * max(0, min(r[3] + pad, b[3] + pad) - max(r[1] - pad, b[1] - pad))
                           for b in boxes)
                area += 400 * (r[0] < ax_box.x0 or r[2] > ax_box.x1 or r[1] < ax_box.y0 or r[3] > ax_box.y1)
                scored.append((area, (dx, dy, ha, va), r))
                t.remove()
            _, (dx, dy, ha, va), best = min(scored, key=lambda t: t[0])
            _annotate(ax, p, dx, dy, ha, va, fontsize)
        boxes.append(best)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results")
    p.add_argument("--user", default="all", help="'all' (mean over sessions) or one session name")
    p.add_argument("--model", default="all", help="'all' (mean over models) or one extraction model")
    p.add_argument("--y", default="f1_score", choices=["f1_score", "soft_f1", "soft_jaccard", "soft_recall", "soft_precision"],
                   help="y metric (default: reranker F1; soft_* need results from run_soft_metrics.sh)")
    p.add_argument("--x", default="eval_count", choices=["eval_count", "keep"],
                   help="propositions after merging (default) or the kept fraction")
    p.add_argument("--scale", default="log", choices=["log", "linear"])
    p.add_argument("--width", type=float, default=3.4, help="inches (3.4 = single column, 7.0 = full width)")
    p.add_argument("--height", type=float, default=2.7)
    p.add_argument("--fontsize", type=float, default=7.5)
    p.add_argument("--annotate", default="above", choices=["above", "key", "frontier", "all", "none"],
                   help="'above' (default): label only the points sitting above the fitted frontier line; "
                        "'key': the smallest, the knee and the most faithful frontier point; "
                        "'frontier': every non-dominated point; 'all': every point")
    p.add_argument("--frontier", default="loglinear", choices=["loglinear", "linear", "segments"],
                   help="'loglinear' (default): least-squares fit of F1 on log10(x) through the non-dominated "
                        "points; 'linear': the same fit on raw x; 'segments': the exact piecewise frontier")
    p.add_argument("--show-segments", action="store_true",
                   help="also draw the exact frontier polyline, dotted, behind the fit")
    p.add_argument("--out", default=None, help="default: results/pareto.pdf")
    args = p.parse_args()

    rows = collect(Path(args.results_root))
    for r in rows:
        r["keep"] = r["eval_count"] / r["candidate_count"] if r.get("candidate_count") and r.get("eval_count") is not None else None
    sessions = sorted({r["user"] for r in rows})
    users = set(sessions) if args.user == "all" else {args.user}

    pts = aggregate(rows, users, args.model, args.x, args.y)
    if not pts:
        raise SystemExit("no matching runs")
    front = frontier(pts)
    methods = [m for m in METHOD_ORDER if any(p[1] == m for p in pts)]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": args.fontsize,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    ax.grid(True, which="major", color="#DDDDD8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # frontier first, so markers sit on top
    fit_txt, fit_line = "", None
    if args.frontier == "segments":
        ax.plot([p[2] for p in front], [p[3] for p in front], color="#111111", linewidth=0.9,
                alpha=0.75, zorder=2, solid_joinstyle="round")
        front_label = "Pareto frontier"
    else:
        # least-squares trend through the non-dominated points: F1 = a + b*log10(x)  (or a + b*x)
        logx = args.frontier == "loglinear"
        tx = (lambda v: math.log10(v)) if logx else (lambda v: v)
        xs = [tx(p[2]) for p in front]
        ys = [p[3] for p in front]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((v - mx) ** 2 for v in xs)
        b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx if sxx else 0.0
        a = my - b * mx
        ss_res = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
        ss_tot = sum((v - my) ** 2 for v in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        x0, x1 = min(p[2] for p in front), max(p[2] for p in front)
        grid = [x0 + (x1 - x0) * i / 200 for i in range(201)] if not logx else \
               [10 ** (math.log10(x0) + (math.log10(x1) - math.log10(x0)) * i / 200) for i in range(201)]
        fit_line = lambda v: a + b * tx(v)
        ax.plot(grid, [fit_line(v) for v in grid], color="#111111", linewidth=0.9, alpha=0.8, zorder=2)
        if args.show_segments:
            ax.plot([p[2] for p in front], [p[3] for p in front], color="#111111", linewidth=0.5,
                    alpha=0.30, linestyle=":", zorder=2)
        unit = r"$\log_{10}$" if logx else ""
        front_label = f"frontier fit ($R^2={r2:.2f}$)"
        fit_txt = (f"F1 = {a:.3f} + {b:.4f}·{'log10(x)' if logx else 'x'}   R^2 = {r2:.3f}   "
                   f"(n = {n} non-dominated points)")
    on_front = {id(p) for p in front}
    for m in methods:
        colour, marker = METHOD_STYLE[m]
        for dominated in (True, False):
            sel = [p for p in pts if p[1] == m and (id(p) not in on_front) == dominated]
            if not sel:
                continue
            ax.scatter([p[2] for p in sel], [p[3] for p in sel], s=15 if dominated else 26,
                       marker=marker, facecolor=colour, edgecolor="white" if dominated else "#111111",
                       linewidth=0.4 if dominated else 0.7, alpha=0.55 if dominated else 1.0,
                       zorder=3 if dominated else 4)

    if args.scale == "log" and args.x == "eval_count":
        ax.set_xscale("log")
        ax.set_xlabel("avg. propositions after merging (log scale)")
        lo, hi = min(p[2] for p in pts), max(p[2] for p in pts)
        ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if lo * 0.85 <= t <= hi * 1.15]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_xlabel("kept fraction $k/n$" if args.x == "keep" else "avg. propositions after merging")
    ax.set_ylabel({"f1_score": r"$\mathrm{F}_1$ vs. ideal", "soft_f1": r"soft $\mathrm{F}_1$ vs. ideal",
                   "soft_jaccard": "soft Jaccard vs. ideal", "soft_recall": "soft recall vs. ideal",
                   "soft_precision": "soft precision vs. ideal"}[args.y])
    ax.margins(x=0.16, y=0.10)

    handles = [Line2D([], [], linestyle="none", marker=METHOD_STYLE[m][1], markersize=4,
                      markerfacecolor=METHOD_STYLE[m][0], markeredgecolor="#111111", markeredgewidth=0.5,
                      label=METHOD_LABEL.get(m, m)) for m in methods]
    handles.append(Line2D([], [], color="#111111", linewidth=0.9, alpha=0.8, label=front_label))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=args.fontsize - 0.7,
              handletextpad=0.4, labelspacing=0.28, borderpad=0.1)

    if args.annotate != "none":
        if args.annotate == "all":
            targets = pts
        elif args.annotate == "frontier":
            targets = front
        elif args.annotate == "above":
            targets = [p for p in pts if fit_line(p[2]) < p[3]] if fit_line else front
        else:  # 'key': the extremes of the frontier plus its knee
            a, b = front[0], front[-1]
            lx = lambda p: math.log10(p[2]) if p[2] > 0 else 0.0
            dx, dy = lx(b) - lx(a), b[3] - a[3]
            knee = max(front, key=lambda p: ((p[3] - a[3]) / dy if dy else 0) - ((lx(p) - lx(a)) / dx if dx else 0))
            targets = list({id(q): q for q in (a, knee, b)}.values())
        place_labels(ax, fig, pts, targets, args.fontsize - 0.7)

    fig.tight_layout(pad=0.25)
    out = Path(args.out) if args.out else Path(args.results_root) / "pareto.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out}\nSaved: {png}")
    if fit_txt:
        print(f"Frontier fit: {fit_txt}")

    ann_txt = {
        "above": "Labelled are the configurations that beat the fitted trend, i.e.\\ sit above the line,",
        "key": "Labelled are the smallest, the knee and the most faithful frontier configurations,",
        "frontier": "Labelled are the non-dominated configurations,",
        "all": "Points are labelled",
        "none": "Configurations are named",
    }[args.annotate]
    codes = ", ".join(f"{v}~=~\\textsc{{{k}}}" for k, v in MOD_CODE.items())
    esc = lambda t: t.replace("_", "\\_")
    scope = f"session {esc(args.user)}" if args.user != "all" else "sessions " + ", ".join(esc(s) for s in sessions)
    print(f"""
% ---- paste into the paper ----
\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=\\columnwidth]{{{out.name}}}
  \\caption{{Quality against memory size: each point is one modality subset merged by one arm,
  averaged over the extraction models within each session and then over {scope}.
  The line is a least-squares fit of $\\mathrm{{F}}_1$ on $\\log_{{10}}$ of the memory size through the
  non-dominated points---those for which no other configuration is both smaller and more faithful.
  {ann_txt} named by modality subset ({codes}) with the arm number in superscript.}}
  \\label{{fig:pareto}}
\\end{{figure}}""")


if __name__ == "__main__":
    main()
