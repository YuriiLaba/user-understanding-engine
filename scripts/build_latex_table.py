"""Paper-ready LaTeX table: merging method (columns) x input-modality subset (rows).

Columns are grouped into the no-merging baseline, proposition-level merging (merge the
propositions each modality produced) and observation-level merging (merge the modality
summaries before proposing). Rows mark the modalities used with check marks.

Each cell is the metric averaged over extraction models (a model's value is first averaged
over the sessions it ran), with a 95% Student-t interval over those models. By default only
(session, model, modality subset) runs present in every column are used, so all columns of a
row are computed on identical runs; --unpaired lifts that. Runs whose candidate is empty for
one of the requested modalities (e.g. a text-only model asked for "screen") are dropped, since
they silently degrade to a smaller subset.

Run: python3 scripts/build_latex_table.py
       -> results_soft/table_merging_soft_f1.tex
     python3 scripts/build_latex_table.py --metric soft_jaccard --no-ci
     python3 scripts/build_latex_table.py --results-root results_soft --metric soft_f1

--results-root takes one or more roots; ROOT=METHOD relabels every run under ROOT as METHOD
(both observation-level arms write merged_observations__*.csv, so one of them needs it).

Overleaf preamble: \\usepackage{booktabs,graphicx,amssymb}
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOTS = ["results_soft", "results_soft_merged_observation",
                 "results_soft_merged_summary=merged_observations_keep"]

# (group header, [(method key, column header)]) in print order; empty header = no group line.
GROUPS = [
    ("", [("none", "No Merge")]),
    ("Proposition level", [
        ("embeddings_reranker", "Dedup"),
        ("embeddings_reranker_llm", "Synthesis"),
        ("sage_time_window", "Temporal"),
        ("moving_window", "Moving window"),
    ]),
    ("Observation level", [
        ("merged_observations", "Obs-Summary"),
        ("merged_observations_keep", "Obs-Transcript"),
    ]),
]
METHOD_NOTE = {
    "none": "\\emph{No merging} keeps every proposition",
    "sage_time_window": "\\emph{SAGE window} drops propositions that add nothing new within a time window",
    "embeddings_reranker": "\\emph{Rerank\\,+\\,medoid} clusters propositions by embedding and cross-encoder "
                           "similarity and keeps each cluster's medoid",
    "embeddings_reranker_llm": "\\emph{Rerank\\,+\\,LLM} rewrites each cluster with an LLM instead",
    "moving_window": "\\emph{Moving window} de-duplicates propositions over a sliding window",
    "merged_observations": "\\emph{Summaries} de-duplicates the modality summaries on wall-clock windows "
                           "before propositions are generated",
    "merged_observations_keep": "\\emph{\\,+\\,transcripts} does the same but also passes each kept "
                                "summary's transcription to the proposer",
}
MODALITIES = [("ax", "AX"), ("metadata", "Meta"), ("ocr", "OCR"), ("screen", "Screen")]
MODALITY_NOTE = ("AX: accessibility tree; Meta: app, mouse and keyboard events; "
                 "OCR: text recognised in the screen recording; Screen: the recording's frames")
METRIC_LABEL = {
    "soft_f1": "Soft $\\mathrm{F}_1$", "soft_jaccard": "Soft Jaccard", "soft_precision": "Soft precision",
    "soft_recall": "Soft recall", "f1_score": "$\\mathrm{F}_1$", "precision_rate": "Precision",
    "recall_rate": "Recall", "symmetric_best_mean": "$\\mu_{\\text{sym}}$", "semantic_jaccard": "Semantic Jaccard",
}
# two-sided 95% Student-t critical values, df = 1..30
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(specs: list[str]) -> list[dict]:
    """One dict per ok run: user, model, method, modalities, metric columns, eval_count."""
    rows = []
    for spec in specs:
        root, _, relabel = spec.partition("=")
        for path in sorted(Path(root).glob("*/*/*__*.csv")):
            rec = next(csv.DictReader(path.open(encoding="utf-8")), None)
            if not rec or rec.get("status") != "ok":
                continue
            method, _, mods = path.stem.partition("__")
            mods = mods.removeprefix("merged+")
            # A baseline candidate is the union of per-modality folders; an empty one means that
            # modality did not exist for this model, so the run is really the smaller subset of
            # the modalities it does have (e.g. a text-only model's all-four run is AX+Meta+OCR).
            present = [m for m in mods.split("+") if _num(rec.get(f"count_{m}")) != 0]
            if not present:
                continue
            row = {"user": path.parts[-3], "model": path.parts[-2], "method": relabel or method,
                   "modalities": "+".join(present), "native": len(present) == len(mods.split("+"))}
            row.update({k: _num(rec.get(k)) for k in METRIC_LABEL})
            row["eval_count"] = _num(rec.get("eval_count")) or _num(rec.get("candidate_count"))
            rows.append(row)
    # A reduced run only stands in for a subset that was never evaluated directly; when a native
    # run of the same subset exists (e.g. a text-only model's ax+screen vs. its ax), keep that.
    native = {(r["user"], r["model"], r["method"], r["modalities"]) for r in rows if r["native"]}
    return [r for r in rows
            if r["native"] or (r["user"], r["model"], r["method"], r["modalities"]) not in native]


def mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, float("nan")
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    t = T95[n - 2] if n - 1 <= len(T95) else 1.960
    return mu, t * sd / math.sqrt(n)


def per_model(runs: list[dict], key: str) -> list[float]:
    """Each model's value, averaged over the sessions it ran."""
    by_model = defaultdict(list)
    for r in runs:
        if r[key] is not None:
            by_model[r["model"]].append(r[key])
    return [sum(v) / len(v) for v in by_model.values()]


def mod_sort_key(mods: str):
    order = [m for m, _ in MODALITIES]
    idx = [order.index(m) for m in mods.split("+")]
    return len(idx), idx


def fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def build(rows: list[dict], args) -> str:
    present = {r["method"] for r in rows}
    groups = [(g, [(k, h) for k, h in cols if k in present]) for g, cols in GROUPS]
    groups = [(g, cols) for g, cols in groups if cols]
    methods = [k for _, cols in groups for k, _ in cols]
    unknown = present - set(methods)
    if unknown:
        raise SystemExit(f"no column defined for method(s): {', '.join(sorted(unknown))}")

    if not args.unpaired:
        runs_of = defaultdict(set)
        for r in rows:
            runs_of[r["method"]].add((r["user"], r["model"], r["modalities"]))
        common = set.intersection(*runs_of.values())
        rows = [r for r in rows if (r["user"], r["model"], r["modalities"]) in common]
    if not rows:
        raise SystemExit("no runs left (with pairing, every column must share the same runs; try --unpaired)")

    cell_runs = defaultdict(list)
    for r in rows:
        cell_runs[(r["modalities"], r["method"])].append(r)
    subsets = sorted({r["modalities"] for r in rows}, key=mod_sort_key)

    cell = {k: mean_ci(per_model(v, args.metric)) for k, v in cell_runs.items() if per_model(v, args.metric)}
    size = {k: mean_ci(per_model(v, "eval_count"))[0] for k, v in cell_runs.items()}
    n_models = {s: len({r["model"] for r in rows if r["modalities"] == s}) for s in subsets}

    # best / second-best per column (which modality subset suits a method) or per row (which method
    # suits a subset); ties share the mark.
    mark = {}
    lines = ([[(s, m) for s in subsets] for m in methods] if args.highlight == "column"
             else [[(s, m) for m in methods] for s in subsets] if args.highlight == "row" else [])
    for line in lines:
        vals = {k: round(cell[k][0], args.decimals) for k in line if k in cell}
        levels = sorted(set(vals.values()), reverse=True)
        for k, v in vals.items():
            if len(levels) > 1 and v == levels[0]:
                mark[k] = "best"
            elif len(levels) > 2 and v == levels[1]:
                mark[k] = "second"

    def value(s, m):
        if (s, m) not in cell:
            return "---"
        mu, ci = cell[(s, m)]
        txt = fmt(mu, args.decimals)
        if (s, m) in mark:
            txt = f"\\{mark[(s, m)]}{{{txt}}}"
        if not args.no_ci and not math.isnan(ci):
            txt += f"\\ci{{{fmt(ci, args.decimals)}}}"
        return txt

    n_mod, n_lead = len(MODALITIES), len(MODALITIES) + 1
    metric = METRIC_LABEL[args.metric]
    users = {r["user"] for r in rows}
    models = {r["model"] for r in rows}
    hl = {"column": "In each column the best modality subset is in \\best{bold} and the second-best "
                    "\\second{underlined}.",
          "row": "In each row the best method is in \\best{bold} and the second-best \\second{underlined}.",
          "none": ""}[args.highlight]
    pairing = ("All columns of a row are computed on the same runs. " if not args.unpaired else "")
    caption = args.caption or (
        f"{metric} of the inferred user profile against the reference, by input modalities (rows) and "
        f"merging method (columns). Each cell is the mean over extraction models, each model first "
        f"averaged over its sessions"
        + ("" if args.no_ci else ", $\\pm$ the 95\\% confidence interval over models")
        + f"; $n$ is the number of models ({len(models)} models, {len(users)} sessions in total). "
        + pairing + "; ".join(METHOD_NOTE[m] for m in methods) + f". {MODALITY_NOTE}. {hl}"
    )

    L = [
        "% Generated by scripts/build_latex_table.py; regenerate rather than edit by hand.",
        "% Preamble: \\usepackage{booktabs,graphicx,amssymb}",
        f"\\begin{{{args.env}}}[t]",
        "\\centering",
        "\\providecommand{\\best}[1]{\\textbf{#1}}",
        "\\providecommand{\\second}[1]{\\underline{#1}}",
        "\\providecommand{\\ci}[1]{{\\scriptsize$\\,\\pm\\,$#1}}",
        "\\providecommand{\\yes}{\\ensuremath{\\checkmark}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{args.label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}" + "c" * n_mod + "r" + "".join("c" * len(c) for _, c in groups) + "@{}}",
        "\\toprule",
    ]
    head1 = [f"\\multicolumn{{{n_mod}}}{{c}}{{\\textbf{{Input modalities}}}}", ""]
    rules, col = [f"\\cmidrule(r){{1-{n_mod}}}"], n_lead + 1
    for g, cols in groups:
        if g and len(cols) > 1:
            head1.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{\\textbf{{{g}}}}}")
            rules.append(f"\\cmidrule(lr){{{col}-{col + len(cols) - 1}}}")
        elif g:
            head1.append(f"\\textbf{{{g}}}")
            rules.append(f"\\cmidrule(lr){{{col}-{col}}}")
        else:
            head1 += [""] * len(cols)
        col += len(cols)
    L.append(" & ".join(head1) + " \\\\")
    L.append(" ".join(rules))
    head2 = [h for _, h in MODALITIES] + ["$n$"] + [h for _, cols in groups for _, h in cols]
    L.append(" & ".join(head2) + " \\\\")
    L.append("\\midrule")

    prev = None
    for s in subsets:
        k = len(s.split("+"))
        if prev is not None and k != prev:
            L.append("\\addlinespace")
        prev = k
        ticks = ["\\yes" if m in s.split("+") else "" for m, _ in MODALITIES]
        L.append(" & ".join(ticks + [str(n_models[s])] + [value(s, m) for m in methods]) + " \\\\")

    L.append("\\midrule")
    means = {m: [cell[(s, m)][0] for s in subsets if (s, m) in cell] for m in methods}
    mean = {m: sum(v) / len(v) for m, v in means.items() if v}
    label = lambda t: f"\\multicolumn{{{n_lead}}}{{@{{}}l}}{{{t}}}"
    L.append(" & ".join([label(f"Mean {metric}")] + [fmt(mean[m], args.decimals) if m in mean else "---"
                                                      for m in methods]) + " \\\\")
    if "none" in mean:
        rel = [("---" if m == "none" else f"${100 * (mean[m] / mean['none'] - 1):+.0f}$\\%") if m in mean else "---"
               for m in methods]
        L.append(" & ".join([label("vs.\\ No Merge")] + rel) + " \\\\")
    if not args.no_size_row:
        avg_size = {m: [size[(s, m)] for s in subsets if (s, m) in size] for m in methods}
        L.append(" & ".join([label("Propositions per profile")] +
                            [f"{sum(v) / len(v):.0f}" if v else "---" for v in avg_size.values()]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", f"\\end{{{args.env}}}"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", nargs="+", default=DEFAULT_ROOTS,
                   help="result roots; ROOT=METHOD relabels every run under ROOT (default: the three soft roots)")
    p.add_argument("--metric", default="soft_f1", choices=list(METRIC_LABEL))
    p.add_argument("--models", default=None, help="comma-separated extraction models to keep")
    p.add_argument("--unpaired", action="store_true",
                   help="use every run of each column instead of only runs shared by all columns")
    p.add_argument("--highlight", default="column", choices=["column", "row", "none"])
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--no-ci", action="store_true", help="means only, no confidence intervals")
    p.add_argument("--no-size-row", action="store_true", help="omit the propositions-per-profile row")
    p.add_argument("--env", default="table*", choices=["table*", "table"],
                   help="table* spans both columns of a two-column paper (default)")
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="default: <first root>/table_merging_<metric>.tex; '-' = stdout only")
    args = p.parse_args()
    args.label = args.label or f"tab:merging_{args.metric}"

    rows = load(args.results_root)
    if args.models:
        keep = set(args.models.split(","))
        rows = [r for r in rows if r["model"] in keep]
    tex = build(rows, args)
    print(tex)
    if args.out != "-":
        out = Path(args.out) if args.out else \
            Path(args.results_root[0].partition("=")[0]) / f"table_merging_{args.metric}.tex"
        out.write_text(tex, encoding="utf-8")
        print(f"% Saved: {out}")


if __name__ == "__main__":
    main()
