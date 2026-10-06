"""Aggregate results/<user>/<model>/<method>__<mods>.csv across ALL users.

Writes (at the results root):
  results/comparison.csv    - tidy long-form table (one row per run, incl. user)
  results/comparison.json   - same, for the HTML
  results/comparison.html   - interactive: per-run table + session comparison

Run: python scripts/build_comparison.py
"""

import argparse
import csv
import json
from pathlib import Path

METHOD_ORDER = ["none", "embeddings_reranker", "embeddings_reranker_llm", "sage_time_window", "moving_window",
                "merged_observations"]

COLUMNS = [
    "user", "model", "method", "modalities", "status",
    "candidate_count", "eval_count", "ideal_count",
    "symmetric_best_mean", "semantic_jaccard",
    "soft_jaccard", "soft_precision", "soft_recall", "soft_f1",
    "soft_source_regions", "soft_reference_regions",
    "precision_rate", "recall_rate", "f1_score",
]
SOFT_COLUMNS = ["soft_jaccard", "soft_precision", "soft_recall", "soft_f1",
                "soft_source_regions", "soft_reference_regions"]


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect(results_root: Path) -> list[dict]:
    rows = []
    for user_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for csv_path in sorted(user_dir.glob("*/*.csv")):
            if csv_path.name == "comparison.csv":
                continue
            model = csv_path.parent.name
            try:
                record = next(iter(csv.DictReader(csv_path.open(encoding="utf-8"))), None)
            except Exception:
                record = None
            if not record:
                continue
            method, _, mods = csv_path.stem.partition("__")
            rows.append(
                {
                    "user": user_dir.name,
                    "model": model,
                    "method": method,
                    "modalities": mods or record.get("modalities", ""),
                    "status": record.get("status", ""),
                    "candidate_count": _num(record.get("candidate_count")),
                    "eval_count": _num(record.get("eval_count")) or _num(record.get("candidate_count")),
                    "ideal_count": _num(record.get("ideal_count")),
                    "symmetric_best_mean": _num(record.get("symmetric_best_mean")),
                    "semantic_jaccard": _num(record.get("semantic_jaccard")),
                    **{k: _num(record.get(k)) for k in SOFT_COLUMNS},
                    "precision_rate": _num(record.get("precision_rate")),
                    "recall_rate": _num(record.get("recall_rate")),
                    "f1_score": _num(record.get("f1_score")),
                }
            )
    rows.sort(
        key=lambda r: (
            r["user"], r["modalities"], r["model"],
            METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99,
        )
    )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in COLUMNS})


def write_html(rows: list[dict], path: Path) -> None:
    path.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = collect(results_root)
    users = sorted({r["user"] for r in rows})
    print(f"Collected {len(rows)} runs across {len(users)} sessions: {', '.join(users)}")

    write_csv(rows, results_root / "comparison.csv")
    (results_root / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    write_html(rows, results_root / "comparison.html")
    for name in ("comparison.csv", "comparison.json", "comparison.html"):
        print(f"Saved: {results_root / name}")


HTML_TEMPLATE = r"""<title>Memory Merge — Baseline Comparison (all sessions)</title>
<style>
  :root{
    --bg:#F5F7F6;--surface:#fff;--surface-2:#EEF2F1;--ink:#14201E;--muted:#566462;
    --faint:#8A9995;--hair:#DCE3E1;--accent:#0E8C86;--accent-soft:#0e8c8618;--good:#2E8B57;--warn:#B07503;
    --none:#8A9995;--emb:#B07503;--embllm:#7A5AB0;--sage:#0E8C86;
    --font-sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0D1413;--surface:#141D1B;--surface-2:#1B2523;--ink:#E7EDEB;--muted:#93A29F;--faint:#6A7876;
    --hair:#26302E;--accent:#35C9BE;--accent-soft:#35c9be1f;--good:#4FBF83;--warn:#E0A43B;
    --none:#6A7876;--emb:#E0A43B;--embllm:#B79AE6;--sage:#35C9BE;}}
  :root[data-theme="dark"]{--bg:#0D1413;--surface:#141D1B;--surface-2:#1B2523;--ink:#E7EDEB;--muted:#93A29F;--faint:#6A7876;--hair:#26302E;--accent:#35C9BE;--accent-soft:#35c9be1f;--good:#4FBF83;--warn:#E0A43B;--none:#6A7876;--emb:#E0A43B;--embllm:#B79AE6;--sage:#35C9BE;}
  :root[data-theme="light"]{--bg:#F5F7F6;--surface:#fff;--surface-2:#EEF2F1;--ink:#14201E;--muted:#566462;--faint:#8A9995;--hair:#DCE3E1;--accent:#0E8C86;--accent-soft:#0e8c8618;--good:#2E8B57;--warn:#B07503;--none:#8A9995;--emb:#B07503;--embllm:#7A5AB0;--sage:#0E8C86;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-sans);line-height:1.5;-webkit-font-smoothing:antialiased;padding:clamp(20px,4vw,44px)}
  .wrap{max-width:1180px;margin:0 auto}
  .eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
  h1{font-size:clamp(24px,3.5vw,34px);letter-spacing:-.02em;margin:0 0 8px;font-weight:800}
  h2{font-size:19px;letter-spacing:-.01em;margin:34px 0 12px;font-weight:750}
  .sub{color:var(--muted);font-size:15px;margin:0 0 20px;max-width:80ch}
  .sub code{font-family:var(--font-mono);font-size:13px;color:var(--ink)}
  .controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
  .controls .lbl{font-family:var(--font-mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:0 4px}
  button.chip{font-family:var(--font-mono);font-size:12.5px;border:1px solid var(--hair);background:var(--surface);color:var(--muted);border-radius:999px;padding:6px 13px;cursor:pointer}
  button.chip[aria-pressed="true"]{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
  .twrap{overflow-x:auto;border:1px solid var(--hair);border-radius:14px;background:var(--surface)}
  table{border-collapse:collapse;width:100%;min-width:820px}
  th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--hair);font-size:13.5px;font-variant-numeric:tabular-nums}
  th:first-child,td:first-child,th.l,td.l{text-align:left}
  thead th{position:sticky;top:0;background:var(--surface-2);font-family:var(--font-mono);font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);font-weight:600;cursor:pointer;white-space:nowrap}
  thead th:hover{color:var(--ink)} thead th.sorted{color:var(--accent)}
  td.mono{font-family:var(--font-mono);color:var(--ink)}
  tr.dim td{color:var(--faint)} tbody tr:hover{background:var(--accent-soft)}
  tr.best td.f1{font-weight:800;color:var(--good)} tr.best td.f1::after{content:" \2605";font-size:10px;color:var(--good)}
  .pill{display:inline-block;font-family:var(--font-mono);font-size:11px;padding:2px 9px;border-radius:999px;font-weight:600}
  .pill.none{color:var(--none);background:color-mix(in srgb,var(--none) 15%,transparent)}
  .pill.embeddings_reranker{color:var(--emb);background:color-mix(in srgb,var(--emb) 15%,transparent)}
  .pill.embeddings_reranker_llm{color:var(--embllm);background:color-mix(in srgb,var(--embllm) 18%,transparent)}
  .pill.sage_time_window{color:var(--sage);background:var(--accent-soft)}
  .pill.moving_window{color:var(--accent);background:var(--accent-soft)}
  td.spread{color:var(--warn);font-weight:600}
  .note{color:var(--muted);font-size:13px;margin-top:14px;max-width:82ch}.note b{color:var(--ink)}
</style>
<div class="wrap">
  <p class="eyebrow">Memory-merge baselines · all sessions</p>
  <h1>Merge-method comparison vs. the ideal</h1>
  <p class="sub">Each run: a candidate model's propositions (for the chosen modalities) merged by a method, scored against that session's ideal (all four gpt_5.5 modalities, unioned). <code>Keep%</code>=merged/candidate (lower=more compression); <code>Sym</code>/<code>SemJac</code>=embedding overlap; <code>Prec/Rec/F1</code>=reranker-judge.</p>

  <h2>Session comparison — averages &amp; spread</h2>
  <p class="sub">Per method, the metric averaged over all models, shown for each session; <code>Avg</code> is the mean across sessions and <code>Δ</code> is the between-session spread (max−min) — how much the method's result changes from one session to another.</p>
  <div class="controls">
    <span class="lbl">Modalities</span><span id="modBtns"></span>
    <span class="lbl">Metric</span><span id="metricBtns"></span>
  </div>
  <div class="twrap"><table><thead><tr id="sumHead"></tr></thead><tbody id="sumBody"></tbody></table></div>

  <h2>All runs</h2>
  <div class="controls">
    <span class="lbl">Session</span><span id="userBtns"></span>
  </div>
  <div class="twrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>

  <p class="note"><b>Reading it:</b> <span class="pill none">none</span> = no-merge upper bound. <span class="pill sage_time_window">sage_time_window</span> keeps F1 near it with moderate compression. <span class="pill embeddings_reranker_llm">embeddings_reranker_llm</span> / <span class="pill embeddings_reranker">embeddings_reranker</span> compress hard, cost F1. A small <code>Δ</code> means the method behaves consistently across sessions.</p>
</div>
<script>
const ROWS=__DATA__;
const METHODS=["none","embeddings_reranker","embeddings_reranker_llm","sage_time_window","moving_window"].filter(m=>ROWS.some(r=>r.method===m));
const USERS=[...new Set(ROWS.map(r=>r.user))].sort();
const MODS=[...new Set(ROWS.map(r=>r.modalities))].sort((a,b)=>a.split("+").length-b.split("+").length||a.localeCompare(b));
ROWS.forEach(r=>{r.keep=(r.candidate_count&&r.eval_count!=null)?r.eval_count/r.candidate_count:null;});
const METRICS={f1_score:{label:"F1",fmt:v=>v.toFixed(3)},symmetric_best_mean:{label:"Sym",fmt:v=>v.toFixed(3)},keep:{label:"Keep%",fmt:v=>(v*100).toFixed(0)+"%"}};
let curMod=MODS.includes("ax+metadata+ocr+screen")?"ax+metadata+ocr+screen":MODS[0];
let curMetric="f1_score", curUser="all", sortKey="f1_score", sortDir=-1, userSorted=false;

const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
function btns(id,items,cur,on,labeler){document.getElementById(id).innerHTML=items.map(x=>`<button class="chip" data-x="${x}" aria-pressed="${x===cur}">${labeler?labeler(x):x}</button>`).join(" ");
  document.querySelectorAll(`#${id} .chip`).forEach(b=>b.onclick=()=>on(b.dataset.x));}

function renderSummary(){
  const metric=curMetric;
  // per method per user: mean over models (status ok, current modality)
  const cell={};
  METHODS.forEach(m=>{cell[m]={};USERS.forEach(u=>{
    const rs=ROWS.filter(r=>r.method===m&&r.user===u&&r.modalities===curMod&&r.status==="ok"&&r[metric]!=null);
    cell[m][u]=mean(rs.map(r=>r[metric]));
  });});
  const cols=["Method",...USERS,"Avg","Δ"];
  document.getElementById("sumHead").innerHTML=cols.map((c,i)=>`<th class="${i===0?'l':''}">${c}</th>`).join("");
  document.getElementById("sumBody").innerHTML=METHODS.map(m=>{
    const vals=USERS.map(u=>cell[m][u]).filter(v=>v!=null);
    const avg=mean(vals);
    const spread=vals.length>1?Math.max(...vals)-Math.min(...vals):null;
    const f=METRICS[metric].fmt;
    let tds=`<td class="l"><span class="pill ${m}">${m}</span></td>`;
    USERS.forEach(u=>{tds+=`<td class="mono">${cell[m][u]!=null?f(cell[m][u]):"—"}</td>`;});
    tds+=`<td class="mono"><b>${avg!=null?f(avg):"—"}</b></td>`;
    tds+=`<td class="spread">${spread!=null?f(spread):"—"}</td>`;
    return `<tr>${tds}</tr>`;
  }).join("");
}

const DCOLS=[
  {k:"user",label:"Session",t:"str",cls:"l"},
  {k:"model",label:"Model",t:"str",cls:"mono l"},
  {k:"method",label:"Method",t:"method",cls:"l"},
  {k:"candidate_count",label:"Cand",t:"int"},
  {k:"eval_count",label:"Merged",t:"int"},
  {k:"keep",label:"Keep%",t:"pct"},
  {k:"symmetric_best_mean",label:"Sym",t:"num"},
  {k:"precision_rate",label:"Prec",t:"num"},
  {k:"recall_rate",label:"Rec",t:"num"},
  {k:"f1_score",label:"F1",t:"num",cls:"f1"},
];
const MORD={none:0,embeddings_reranker:1,embeddings_reranker_llm:2,sage_time_window:3,moving_window:4};
function fmt(v,t){if(v==null||v==="")return "—";if(t==="int")return Math.round(v);if(t==="pct")return (v*100).toFixed(0)+"%";if(t==="num")return Number(v).toFixed(3);return v;}
function renderDetail(){
  let rows=ROWS.filter(r=>r.modalities===curMod&&(curUser==="all"||r.user===curUser));
  if(!userSorted){rows.sort((a,b)=>a.user.localeCompare(b.user)||a.model.localeCompare(b.model)||(MORD[a.method]-MORD[b.method]));}
  else{rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];if(x==null)return 1;if(y==null)return -1;return typeof x==="string"?sortDir*x.localeCompare(y):sortDir*(x-y);});}
  const bestKey={},best={};
  ROWS.filter(r=>r.modalities===curMod&&(curUser==="all"||r.user===curUser)&&r.f1_score!=null).forEach(r=>{const k=r.user+"|"+r.model;if(best[k]==null||r.f1_score>best[k])best[k]=r.f1_score;});
  document.getElementById("head").innerHTML=DCOLS.map(c=>{const s=(userSorted&&c.k===sortKey)?" sorted":"";const ar=(userSorted&&c.k===sortKey)?(sortDir<0?" ↓":" ↑"):"";return `<th class="${(c.cls||'').includes('l')?'l':''}${s}" data-k="${c.k}" data-t="${c.t}">${c.label}${ar}</th>`;}).join("");
  document.getElementById("body").innerHTML=rows.map(r=>{
    const ok=r.status==="ok";const isBest=ok&&r.f1_score===best[r.user+"|"+r.model];
    const tds=DCOLS.map(c=>{if(c.k==="method")return `<td class="l"><span class="pill ${r.method}">${r.method}</span></td>`;return `<td class="${c.cls||'mono'}">${fmt(r[c.k],c.t)}</td>`;}).join("");
    return `<tr class="${ok?'':'dim'}${isBest?' best':''}">${tds}</tr>`;
  }).join("");
  document.querySelectorAll("#head th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortKey===k&&userSorted)sortDir*=-1;else{sortKey=k;sortDir=(th.dataset.t==="str")?1:-1;}userSorted=true;renderDetail();});
}

function renderAll(){renderSummary();renderDetail();}
function drawModBtns(){btns("modBtns",MODS,curMod,x=>{curMod=x;userSorted=false;drawModBtns();renderAll();},m=>m.split("+").length===4?"all 4":m);}
function drawMetricBtns(){btns("metricBtns",Object.keys(METRICS),curMetric,x=>{curMetric=x;drawMetricBtns();renderSummary();},k=>METRICS[k].label);}
function drawUserBtns(){btns("userBtns",["all",...USERS],curUser,x=>{curUser=x;userSorted=false;drawUserBtns();renderDetail();});}
drawModBtns();drawMetricBtns();drawUserBtns();renderAll();
</script>
"""


if __name__ == "__main__":
    main()
