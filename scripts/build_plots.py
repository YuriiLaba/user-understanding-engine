"""Plot F1 / semantic Jaccard across modalities and the metric-vs-size Pareto frontier, for all sessions.

Reads the same runs as scripts/build_comparison.py (results/<user>/<model>/<method>__<mods>.csv)
and writes:
  results/plots.html   - interactive: F1 or Jaccard by modality (per session + all), Pareto scatter
                         (one pair of charts per metric: F1, then Jaccard)

Run: python scripts/build_plots.py
"""

import argparse
import json
from pathlib import Path

from build_comparison import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default=None, help="output html path (default: <results-root>/plots.html)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = collect(results_root)
    users = sorted({r["user"] for r in rows})
    print(f"Collected {len(rows)} runs across {len(users)} sessions: {', '.join(users)}")

    out = Path(args.out) if args.out else results_root / "plots.html"
    out.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
    print(f"Saved: {out}")


HTML_TEMPLATE = r"""<title>Modality &amp; Pareto Plots</title>
<style>
  :root{
    --bg:#F5F7F6;--surface:#fff;--surface-2:#EEF2F1;--ink:#14201E;--muted:#566462;
    --faint:#8A9995;--hair:#DCE3E1;--accent:#0E8C86;--accent-soft:#0e8c8618;
    --grid:#E1E0D9;--axis:#C3C2B7;
    /* methods - blue / orange / aqua validated all-pairs; "none" is chrome gray, not a hue */
    --m-none:#8A9995;--m-emb:#2a78d6;--m-embllm:#eb6834;--m-sage:#1baf7a;--m-move:#4a3aa7;
    /* sessions - violet / magenta / yellow, validated all-pairs */
    --s1:#4a3aa7;--s2:#e87ba4;--s3:#eda100;
    --bar:#C7D2CF;
    --font-sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#0D1413;--surface:#141D1B;--surface-2:#1B2523;--ink:#E7EDEB;--muted:#93A29F;--faint:#6A7876;
    --hair:#26302E;--accent:#35C9BE;--accent-soft:#35c9be1f;--grid:#243230;--axis:#33403E;
    --m-none:#6A7876;--m-emb:#3987e5;--m-embllm:#d95926;--m-sage:#199e70;--m-move:#9085e9;
    --s1:#9085e9;--s2:#d55181;--s3:#c98500;
    --bar:#2C3A37;}}
  :root[data-theme="dark"]{
    --bg:#0D1413;--surface:#141D1B;--surface-2:#1B2523;--ink:#E7EDEB;--muted:#93A29F;--faint:#6A7876;
    --hair:#26302E;--accent:#35C9BE;--accent-soft:#35c9be1f;--grid:#243230;--axis:#33403E;
    --m-none:#6A7876;--m-emb:#3987e5;--m-embllm:#d95926;--m-sage:#199e70;--m-move:#9085e9;
    --s1:#9085e9;--s2:#d55181;--s3:#c98500;--bar:#2C3A37;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-sans);line-height:1.5;
       -webkit-font-smoothing:antialiased;padding:clamp(20px,4vw,44px)}
  .wrap{max-width:1180px;margin:0 auto}
  .eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 8px}
  h1{font-size:clamp(24px,3.5vw,34px);letter-spacing:-.02em;margin:0 0 8px;font-weight:800}
  h2{font-size:19px;letter-spacing:-.01em;margin:0 0 6px;font-weight:750}
  .sub{color:var(--muted);font-size:14.5px;margin:0 0 18px;max-width:84ch}
  .sub code,code{font-family:var(--font-mono);font-size:12.5px;color:var(--ink)}
  .card{background:var(--surface);border:1px solid var(--hair);border-radius:16px;padding:clamp(14px,2vw,22px);margin:26px 0}
  .controls{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:12px 0 6px}
  .controls .lbl{font-family:var(--font-mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:0 3px 0 6px}
  .controls .lbl:first-child{margin-left:0}
  button.chip{font-family:var(--font-mono);font-size:12px;border:1px solid var(--hair);background:var(--surface);
              color:var(--muted);border-radius:999px;padding:5px 12px;cursor:pointer}
  button.chip:hover{color:var(--ink)}
  button.chip[aria-pressed="true"]{background:var(--accent-soft);color:var(--accent);border-color:var(--accent)}
  select.chip{font-family:var(--font-mono);font-size:12px;border:1px solid var(--hair);background:var(--surface);
              color:var(--muted);border-radius:999px;padding:5px 10px;cursor:pointer}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:14px 0 4px}
  .tile{background:var(--surface-2);border-radius:12px;padding:11px 14px}
  .tile .k{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
  .tile .v{font-size:20px;font-weight:750;letter-spacing:-.01em;margin-top:2px}
  .tile .d{font-size:12.5px;color:var(--muted);font-family:var(--font-mono)}
  .legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:10px 0 2px;font-size:12.5px;color:var(--muted)}
  .legend .it{display:flex;align-items:center;gap:6px}
  .chart{width:100%;overflow-x:auto}
  svg{display:block;max-width:100%}
  svg text{font-family:var(--font-sans)}
  .axis-t{font-size:11px;fill:var(--faint);font-variant-numeric:tabular-nums}
  .cat-t{font-size:12.5px;fill:var(--ink)}
  .val-t{font-size:11.5px;fill:var(--muted);font-variant-numeric:tabular-nums}
  .lab-t{font-size:11px;fill:var(--ink);stroke:var(--surface);stroke-width:3.5px;paint-order:stroke;stroke-linejoin:round}
  .ax-title{font-size:11px;fill:var(--faint);font-family:var(--font-mono);letter-spacing:.06em;text-transform:uppercase}
  #tip{position:fixed;pointer-events:none;z-index:9;opacity:0;transition:opacity .08s;background:var(--surface);
       border:1px solid var(--hair);border-radius:10px;padding:8px 11px;font-size:12.5px;color:var(--ink);
       box-shadow:0 6px 22px rgba(0,0,0,.14);max-width:280px}
  #tip .t{font-weight:700;margin-bottom:3px}
  #tip .r{color:var(--muted);font-family:var(--font-mono);font-size:11.5px;white-space:nowrap}
  #tip .r b{color:var(--ink);font-weight:650}
  details{margin-top:16px;border-top:1px solid var(--hair);padding-top:10px}
  summary{cursor:pointer;font-family:var(--font-mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
  summary:hover{color:var(--ink)}
  .twrap{overflow-x:auto;margin-top:10px}
  table{border-collapse:collapse;width:100%;min-width:520px}
  th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--hair);font-size:13px;font-variant-numeric:tabular-nums}
  th:first-child,td:first-child{text-align:left}
  thead th{background:var(--surface-2);font-family:var(--font-mono);font-size:10.5px;letter-spacing:.05em;
           text-transform:uppercase;color:var(--faint);font-weight:600;white-space:nowrap}
  td.mono{font-family:var(--font-mono)}
  tr.front td{color:var(--accent);font-weight:650}
  .note{color:var(--muted);font-size:13px;margin:14px 0 0;max-width:84ch}.note b{color:var(--ink)}
</style>
<div class="wrap">
  <p class="eyebrow">Memory-merge baselines · modality &amp; cost/quality</p>
  <h1>Which modality wins, and which merge is worth its size</h1>
  <p class="sub">Every run scores a candidate model's propositions against the session's ideal, as <b>F1</b> (semantic
  precision/recall) and as <b>semantic Jaccard</b> (matched propositions over the union). The first pair of charts is F1,
  the second pair is Jaccard. Runs are averaged
  over models first, then over sessions (each session weighs the same, so uneven per-session coverage can't skew the mean).
  Only <code>status=ok</code> runs count.</p>

  <div id="cards"></div>
</div>
<div id="tip"></div>
<script>
const ROWS=__DATA__;
const CARD_TPL=(m,ML)=>`
  <div class="card">
    <h2>${ML} by modality</h2>
    <p class="sub">The tick is the mean over all three sessions; the dots are each session on its own, joined by
    their range. Every modality scores high, so the axis is <b>zoomed to the observed range</b> — read the gaps,
    not the distance from zero. Wide-apart dots mean the modality is unstable across sessions, not reliably weak.</p>
    <div class="controls">
      <span class="lbl">Merge</span><span id="c1Method_${m}"></span>
      <span class="lbl">Model</span><span id="c1Model_${m}"></span>
      <span class="lbl">Order</span><span id="c1Sort_${m}"></span>
    </div>
    <div class="tiles" id="c1Tiles_${m}"></div>
    <div class="legend" id="c1Legend_${m}"></div>
    <div class="chart"><svg id="c1_${m}"></svg></div>
    <details><summary>Table view</summary><div class="twrap"><table id="c1Tab_${m}"></table></div></details>
  </div>

  <div class="card">
    <h2>Pareto frontier — ${ML} vs. candidates after merge</h2>
    <p class="sub">One point per <b>modality × merge method</b>. Up is more faithful, left is fewer propositions to
    store and re-read. Ringed points are on the frontier: nothing else is both smaller and better.</p>
    <div class="controls">
      <span class="lbl">Session</span><span id="c2User_${m}"></span>
      <span class="lbl">Model</span><span id="c2Model_${m}"></span>
      <span class="lbl">X axis</span><span id="c2X_${m}"></span>
      <span class="lbl">Scale</span><span id="c2Scale_${m}"></span>
    </div>
    <div class="tiles" id="c2Tiles_${m}"></div>
    <div class="legend" id="c2Legend_${m}"></div>
    <div class="chart"><svg id="c2_${m}"></svg></div>
    <details><summary>Table view — frontier first</summary><div class="twrap"><table id="c2Tab_${m}"></table></div></details>
    <p class="note"><b>Reading it:</b> <code>none</code> is the no-merge upper bound — it sits far right by definition.
    A method is worth it when it moves left much faster than it falls. Points on the frontier are the only combinations
    that aren't strictly beaten by something smaller and more faithful.</p>
  </div>
`;
const OK=ROWS.filter(r=>r.status==="ok"&&(r.f1_score!=null||r.semantic_jaccard!=null)&&r.eval_count!=null);
const METRIC={f1_score:{label:"F1",title:"F1 vs. ideal"},semantic_jaccard:{label:"Jaccard",title:"semantic Jaccard vs. ideal"}};
const METRICS=Object.keys(METRIC).filter(k=>OK.some(r=>r[k]!=null));
OK.forEach(r=>{r.keep=r.candidate_count?r.eval_count/r.candidate_count:null;});
const USERS=[...new Set(OK.map(r=>r.user))].sort();
const MODELS=[...new Set(OK.map(r=>r.model))].sort();
const MORD=["none","embeddings_reranker","embeddings_reranker_llm","sage_time_window","moving_window"];
const METHODS=MORD.filter(m=>OK.some(r=>r.method===m));
const MODS=[...new Set(OK.map(r=>r.modalities))]
  .sort((a,b)=>a.split("+").length-b.split("+").length||a.localeCompare(b));
const MCOLOR={none:"--m-none",embeddings_reranker:"--m-emb",embeddings_reranker_llm:"--m-embllm",
              sage_time_window:"--m-sage",moving_window:"--m-move"};
const MSHAPE={none:"circle",embeddings_reranker:"square",embeddings_reranker_llm:"triangle",
              sage_time_window:"diamond",moving_window:"circle"};
const UCOLOR={};USERS.forEach((u,i)=>UCOLOR[u]="--s"+(i%3+1));
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const shortMod=m=>m.replace(/metadata/g,"meta");
const f3=v=>v==null?"—":v.toFixed(3);
const fInt=v=>v==null?"—":Math.round(v).toLocaleString();

const S1={},S2={};
METRICS.forEach(m=>{S1[m]={metric:m,method:"none",model:"all",sort:"metric"};S2[m]={metric:m,user:"all",model:"all",x:"eval_count",scale:"log"};});
const $=(id,m)=>document.getElementById(id+"_"+m);
document.getElementById("cards").innerHTML=METRICS.map(m=>CARD_TPL(m,METRIC[m].label)).join("");

function btns(id,m,items,cur,on,labeler){
  const el=$(id,m);
  el.innerHTML=items.map(x=>`<button class="chip" data-x="${esc(x)}" aria-pressed="${x===cur}">${esc(labeler?labeler(x):x)}</button>`).join(" ");
  el.querySelectorAll(".chip").forEach(b=>b.onclick=()=>on(b.dataset.x));
}
function sel(id,m,items,cur,on,labeler){
  const el=$(id,m);
  el.innerHTML=`<select class="chip">${items.map(x=>`<option value="${esc(x)}"${x===cur?" selected":""}>${esc(labeler?labeler(x):x)}</option>`).join("")}</select>`;
  el.querySelector("select").onchange=e=>on(e.target.value);
}

/* ---------- aggregation: mean over models within a session, then over sessions ---------- */
function agg(pred,key,users){
  const per={};let n=0;
  users.forEach(u=>{
    const rs=OK.filter(r=>r.user===u&&pred(r));
    n+=rs.length;
    const vs=rs.map(r=>r[key]).filter(v=>v!=null);
    per[u]=vs.length?mean(vs):null;
  });
  const vals=users.map(u=>per[u]).filter(v=>v!=null);
  return {per,macro:mean(vals),n,sessions:vals.length};
}

/* ---------- tooltip ---------- */
const tip=document.getElementById("tip");
function showTip(e,html){
  tip.innerHTML=html;tip.style.opacity=1;
  const b=tip.getBoundingClientRect();
  let x=e.clientX+14,y=e.clientY+14;
  if(x+b.width>innerWidth-8)x=e.clientX-b.width-14;
  if(y+b.height>innerHeight-8)y=e.clientY-b.height-14;
  tip.style.left=x+"px";tip.style.top=y+"px";
}
const hideTip=()=>{tip.style.opacity=0;};
function bindTips(root){
  root.querySelectorAll("[data-tip]").forEach(el=>{
    el.style.cursor="crosshair";
    el.onmousemove=e=>showTip(e,el.dataset.tip);
    el.onmouseleave=hideTip;
  });
}

/* ---------- chart 1: F1 by modality ---------- */
function c1Data(m){const c1=S1[m];
  const pred=r=>(c1.model==="all"||r.model===c1.model)&&(c1.method==="all"||r.method===c1.method);
  const d=MODS.map(m=>({mod:m,...agg(r=>r.modalities===m&&pred(r),c1.metric,USERS)}))
              .filter(d=>d.macro!=null);
  d.sort(c1.sort==="metric"?(a,b)=>b.macro-a.macro
                       :(a,b)=>a.mod.split("+").length-b.mod.split("+").length||a.mod.localeCompare(b.mod));
  return d;
}
function renderC1(m){const c1=S1[m];
  const data=c1Data(m);const ML=METRIC[m].label;
  const W=Math.max(680,$("c1",m).parentNode.clientWidth);
  const L=178,R=62,T=30,ROW=34,H=T+data.length*ROW+34;
  const x0=L,x1=W-R;
  const all=data.flatMap(d=>[d.macro,...USERS.map(u=>d.per[u])]).filter(v=>v!=null);
  let lo=Math.min(...all),hi=Math.max(...all);
  const pad=Math.max(0.01,(hi-lo)*0.07);lo=Math.max(0,lo-pad);hi=Math.min(1,hi+pad);
  const X=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
  const step=(hi-lo)>0.3?0.1:((hi-lo)>0.12?0.05:0.02);
  const s=[];
  for(let t=Math.ceil(lo/step)*step;t<=hi+1e-9;t+=step){
    s.push(`<line x1="${X(t).toFixed(1)}" y1="${T-8}" x2="${X(t).toFixed(1)}" y2="${T+data.length*ROW}" stroke="var(--grid)" stroke-width="1"/>`);
    s.push(`<text class="axis-t" x="${X(t).toFixed(1)}" y="${T-14}" text-anchor="middle">${t.toFixed(2)}</text>`);
  }
  s.push(`<text class="ax-title" x="${x0}" y="${H-8}">${esc(METRIC[c1.metric].title)} — axis zoomed to observed range</text>`);
  data.forEach((d,i)=>{
    const cy=T+i*ROW+ROW/2-1;
    const vals=USERS.map(u=>d.per[u]).filter(v=>v!=null);
    const tipAll=esc(`<div class='t'>${shortMod(d.mod)}</div><div class='r'>all sessions <b>${f3(d.macro)}</b></div>`+
      USERS.map(u=>`<div class='r'>${u} <b>${f3(d.per[u])}</b></div>`).join("")+`<div class='r'>${d.n} runs</div>`);
    s.push(`<rect x="${x0}" y="${cy-ROW/2}" width="${x1-x0}" height="${ROW}" fill="transparent" data-tip="${tipAll}"/>`);
    s.push(`<text class="cat-t" x="${L-12}" y="${cy+4}" text-anchor="end">${esc(shortMod(d.mod))}</text>`);
    if(vals.length>1)
      s.push(`<line x1="${X(Math.min(...vals)).toFixed(1)}" y1="${cy}" x2="${X(Math.max(...vals)).toFixed(1)}" y2="${cy}" stroke="var(--bar)" stroke-width="3" stroke-linecap="round"/>`);
    USERS.forEach(u=>{
      if(d.per[u]==null)return;
      s.push(`<circle cx="${X(d.per[u]).toFixed(1)}" cy="${cy}" r="4.6" fill="var(${UCOLOR[u]})"
              stroke="var(--surface)" stroke-width="2"
              data-tip="${esc(`<div class='t'>${u}</div><div class='r'>${shortMod(d.mod)} <b>${f3(d.per[u])}</b></div>`)}"/>`);
    });
    s.push(`<rect x="${(X(d.macro)-1.5).toFixed(1)}" y="${cy-11}" width="3" height="22" rx="1.5" fill="var(--ink)" stroke="var(--surface)" stroke-width="1.5"/>`);
    s.push(`<text class="val-t" x="${x1+8}" y="${cy+4}">${f3(d.macro)}</text>`);
  });
  s.push(`<line x1="${x0}" y1="${T-8}" x2="${x0}" y2="${T+data.length*ROW}" stroke="var(--axis)" stroke-width="1"/>`);
  const svg=$("c1",m);
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.setAttribute("width",W);svg.setAttribute("height",H);
  svg.innerHTML=s.join("");
  bindTips(svg);

  $("c1Legend",m).innerHTML=
    `<span class="it"><svg width="12" height="14"><rect x="4.5" y="0" width="3" height="14" rx="1.5" fill="var(--ink)"/></svg> all sessions (mean)</span>`+
    USERS.map(u=>`<span class="it"><svg width="12" height="12"><circle cx="6" cy="6" r="4.6" fill="var(${UCOLOR[u]})" stroke="var(--surface)" stroke-width="2"/></svg> ${esc(u)}</span>`).join("")+
    `<span class="it"><svg width="20" height="12"><line x1="1" y1="6" x2="19" y2="6" stroke="var(--bar)" stroke-width="3" stroke-linecap="round"/></svg> session range</span>`;

  const best=data[0],worst=data[data.length-1];
  const spread=data.map(d=>({m:d.mod,s:Math.max(...USERS.map(u=>d.per[u]).filter(v=>v!=null))-Math.min(...USERS.map(u=>d.per[u]).filter(v=>v!=null))}))
                   .sort((a,b)=>b.s-a.s)[0];
  $("c1Tiles",m).innerHTML=
    tile("Best modality",shortMod(best.mod),`${ML} ${f3(best.macro)} · ${best.n} runs`)+
    tile("Weakest modality",shortMod(worst.mod),`${ML} ${f3(worst.macro)}`)+
    tile("Least stable across sessions",shortMod(spread.m),`spread ${f3(spread.s)} ${ML}`);

  $("c1Tab",m).innerHTML=
    `<thead><tr><th>Modality</th>${USERS.map(u=>`<th>${esc(u)}</th>`).join("")}<th>All</th><th>Runs</th></tr></thead>`+
    `<tbody>${data.map(d=>`<tr><td class="mono">${esc(shortMod(d.mod))}</td>${USERS.map(u=>`<td class="mono">${f3(d.per[u])}</td>`).join("")}<td class="mono"><b>${f3(d.macro)}</b></td><td class="mono">${d.n}</td></tr>`).join("")}</tbody>`;
}
const tile=(k,v,d)=>`<div class="tile"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="d">${esc(d)}</div></div>`;

/* ---------- chart 2: Pareto ---------- */
function c2Data(m){const c2=S2[m];
  const users=c2.user==="all"?USERS:[c2.user];
  const pred=r=>(c2.model==="all"||r.model===c2.model);
  const pts=[];
  MODS.forEach(m=>METHODS.forEach(me=>{
    const f=agg(r=>r.modalities===m&&r.method===me&&pred(r),c2.metric,users);
    if(f.macro==null)return;
    const xv=agg(r=>r.modalities===m&&r.method===me&&pred(r),c2.x,users);
    const kp=agg(r=>r.modalities===m&&r.method===me&&pred(r),"keep",users);
    const cd=agg(r=>r.modalities===m&&r.method===me&&pred(r),"candidate_count",users);
    if(xv.macro==null)return;
    pts.push({mod:m,method:me,y:f.macro,x:xv.macro,n:f.n,keep:kp.macro,cand:cd.macro});
  }));
  pts.forEach(p=>{p.front=!pts.some(q=>q!==p&&q.x<=p.x&&q.y>=p.y&&(q.x<p.x||q.y>p.y));});
  return pts;
}
function renderC2(m){const c2=S2[m];
  const pts=c2Data(m);const ML=METRIC[m].label;
  const W=Math.max(680,$("c2",m).parentNode.clientWidth);
  const H=Math.min(600,Math.max(420,W*0.52));
  const L=62,R=26,T=22,B=52;
  const isLog=c2.scale==="log"&&c2.x==="eval_count";
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  let xlo=Math.min(...xs),xhi=Math.max(...xs);
  const ylo=Math.max(0,Math.min(...ys)-0.03),yhi=Math.min(1,Math.max(...ys)+0.03);
  let X;
  if(isLog){const a=Math.log10(xlo*0.85),b=Math.log10(xhi*1.15);X=v=>L+(Math.log10(v)-a)/(b-a)*(W-L-R);}
  else{const pad=(xhi-xlo)*0.06||1;const a=Math.max(0,xlo-pad),b=xhi+pad;X=v=>L+(v-a)/(b-a)*(W-L-R);}
  const Y=v=>H-B-(v-ylo)/(yhi-ylo)*(H-T-B);
  const s=[];
  // y grid
  const ystep=(yhi-ylo)>0.4?0.1:0.05;
  for(let t=Math.ceil(ylo/ystep)*ystep;t<=yhi+1e-9;t+=ystep){
    s.push(`<line x1="${L}" y1="${Y(t).toFixed(1)}" x2="${W-R}" y2="${Y(t).toFixed(1)}" stroke="var(--grid)" stroke-width="1"/>`);
    s.push(`<text class="axis-t" x="${L-10}" y="${(Y(t)+4).toFixed(1)}" text-anchor="end">${t.toFixed(2)}</text>`);
  }
  // x ticks
  let ticks=[];
  if(isLog){for(let e=0;e<=4;e++)[1,2,5].forEach(m=>{const v=m*Math.pow(10,e);if(v>=xlo*0.85&&v<=xhi*1.15)ticks.push(v);});}
  else{const n=6;for(let i=0;i<=n;i++)ticks.push(xlo+(xhi-xlo)*i/n);}
  ticks.forEach(t=>{
    s.push(`<line x1="${X(t).toFixed(1)}" y1="${T}" x2="${X(t).toFixed(1)}" y2="${H-B}" stroke="var(--grid)" stroke-width="1"/>`);
    s.push(`<text class="axis-t" x="${X(t).toFixed(1)}" y="${H-B+18}" text-anchor="middle">${c2.x==="keep"?(t*100).toFixed(0)+"%":fInt(t)}</text>`);
  });
  s.push(`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="var(--axis)" stroke-width="1"/>`);
  s.push(`<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="var(--axis)" stroke-width="1"/>`);
  s.push(`<text class="ax-title" x="${W-R}" y="${H-14}" text-anchor="end">${c2.x==="keep"?"kept after merge (% of candidates)":"candidates after merge"+(isLog?" (log)":"")}</text>`);
  s.push(`<text class="ax-title" transform="translate(14,${T+6}) rotate(-90)" text-anchor="end">${esc(METRIC[c2.metric].title)}</text>`);
  // frontier staircase
  const fr=pts.filter(p=>p.front).sort((a,b)=>a.x-b.x);
  if(fr.length>1){
    let d=`M${X(fr[0].x).toFixed(1)},${Y(fr[0].y).toFixed(1)}`;
    for(let i=1;i<fr.length;i++)d+=` L${X(fr[i].x).toFixed(1)},${Y(fr[i-1].y).toFixed(1)} L${X(fr[i].x).toFixed(1)},${Y(fr[i].y).toFixed(1)}`;
    s.push(`<path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" opacity=".55"/>`);
  }
  // marks
  const placedLabs=[];
  pts.slice().sort((a,b)=>a.front-b.front).forEach(p=>{
    const cx=X(p.x),cy=Y(p.y),c=`var(${MCOLOR[p.method]})`;
    const t=esc(`<div class='t'>${shortMod(p.mod)} · ${p.method}</div>`+
      `<div class='r'>${ML} <b>${f3(p.y)}</b></div>`+
      `<div class='r'>after merge <b>${fInt(p.cand!=null&&c2.x==="keep"?p.x*p.cand:p.x)}</b>${p.keep!=null?` of ${fInt(p.cand)} (${(p.keep*100).toFixed(0)}%)`:""}</div>`+
      `<div class='r'>${p.n} runs${p.front?" · on frontier":""}</div>`);
    s.push(mark(MSHAPE[p.method],cx,cy,p.front?6.5:5,c,t,p.front));
  });
  // frontier labels: first candidate slot that clears every mark and every placed label
  const boxes=pts.map(p=>({x0:X(p.x)-9,y0:Y(p.y)-9,x1:X(p.x)+9,y1:Y(p.y)+9}));
  const hit=(a,b)=>!(a.x1<b.x0||a.x0>b.x1||a.y1<b.y0||a.y0>b.y1);
  const inb=a=>a.x0>=L+2&&a.x1<=W-R-2&&a.y0>=T+2&&a.y1<=H-B-2;
  fr.slice().sort((a,b)=>b.y-a.y).forEach(p=>{
    const cx=X(p.x),cy=Y(p.y),lab=shortMod(p.mod),w=lab.length*6.2;
    const cands=[[cx+12,cy+4,"start"],[cx-12,cy+4,"end"],[cx,cy-13,"middle"],[cx,cy+19,"middle"],
                 [cx+11,cy-9,"start"],[cx-11,cy-9,"end"],[cx+11,cy+17,"start"],[cx-11,cy+17,"end"]];
    let pick=null;
    for(const [lx,ly,an] of cands){
      const x0=an==="start"?lx:an==="end"?lx-w:lx-w/2;
      const r={x0,x1:x0+w,y0:ly-10,y1:ly+3};
      if(!inb(r)||boxes.some(b=>hit(r,b))||placedLabs.some(b=>hit(r,b)))continue;
      pick=[lx,ly,an,r];break;
    }
    if(!pick){const x0=cx+12;pick=[cx+12,cy+4,"start",{x0,x1:x0+w,y0:cy-6,y1:cy+7}];}
    placedLabs.push(pick[3]);
    s.push(`<text class="lab-t" x="${pick[0].toFixed(1)}" y="${pick[1].toFixed(1)}" text-anchor="${pick[2]}">${esc(lab)}</text>`);
  });
  const svg=$("c2",m);
  svg.setAttribute("viewBox",`0 0 ${W} ${H}`);svg.setAttribute("width",W);svg.setAttribute("height",H);
  svg.innerHTML=s.join("");
  bindTips(svg);

  $("c2Legend",m).innerHTML=
    METHODS.map(m=>`<span class="it"><svg width="14" height="14">${mark(MSHAPE[m],7,7,5,`var(${MCOLOR[m]})`,"",false)}</svg> ${esc(m)}</span>`).join("")+
    `<span class="it"><svg width="20" height="12"><line x1="0" y1="6" x2="20" y2="6" stroke="var(--accent)" stroke-width="2" opacity=".55"/></svg> Pareto frontier</span>`;

  const bestF1=pts.slice().sort((a,b)=>b.y-a.y)[0];
  const knee=(()=>{
    if(fr.length<3)return fr[fr.length-1];
    const tx=p=>isLog?Math.log10(p.x):p.x;
    const a=fr[0],b=fr[fr.length-1],dx=tx(b)-tx(a),dy=b.y-a.y;
    let best=a,bd=-Infinity;
    fr.forEach(p=>{const d=(dy?(p.y-a.y)/dy:0)-(dx?(tx(p)-tx(a))/dx:0);if(d>bd){bd=d;best=p;}});
    return best;
  })();
  const fmtX=v=>c2.x==="keep"?(v*100).toFixed(0)+"% kept":fInt(v)+" left";
  $("c2Tiles",m).innerHTML=
    tile("On the frontier",fr.length+" of "+pts.length,"combinations not strictly beaten")+
    tile("Highest "+ML,shortMod(bestF1.mod),`${bestF1.method} · ${ML} ${f3(bestF1.y)} · ${fmtX(bestF1.x)}`)+
    tile("Knee of the frontier",shortMod(knee.mod),`${knee.method} · ${ML} ${f3(knee.y)} · ${fmtX(knee.x)}`);

  const tab=pts.slice().sort((a,b)=>(b.front-a.front)||b.y-a.y);
  $("c2Tab",m).innerHTML=
    `<thead><tr><th>Modality</th><th>Method</th><th>${ML}</th><th>${c2.x==="keep"?"Keep%":"After merge"}</th><th>Candidates</th><th>Runs</th><th>Frontier</th></tr></thead>`+
    `<tbody>${tab.map(p=>`<tr class="${p.front?"front":""}"><td class="mono">${esc(shortMod(p.mod))}</td><td class="mono">${esc(p.method)}</td><td class="mono">${f3(p.y)}</td><td class="mono">${c2.x==="keep"?(p.x*100).toFixed(0)+"%":fInt(p.x)}</td><td class="mono">${fInt(p.cand)}</td><td class="mono">${p.n}</td><td class="mono">${p.front?"✓":""}</td></tr>`).join("")}</tbody>`;
}
function mark(shape,cx,cy,r,fill,tipHtml,ring){
  const a=`fill="${fill}" stroke="var(--surface)" stroke-width="2"${tipHtml?` data-tip="${tipHtml}"`:""}`;
  let el;
  if(shape==="square")el=`<rect x="${(cx-r).toFixed(1)}" y="${(cy-r).toFixed(1)}" width="${2*r}" height="${2*r}" rx="1.5" ${a}/>`;
  else if(shape==="triangle")el=`<path d="M${cx.toFixed(1)},${(cy-r*1.15).toFixed(1)} L${(cx+r*1.1).toFixed(1)},${(cy+r*0.85).toFixed(1)} L${(cx-r*1.1).toFixed(1)},${(cy+r*0.85).toFixed(1)} Z" ${a}/>`;
  else if(shape==="diamond")el=`<path d="M${cx.toFixed(1)},${(cy-r*1.25).toFixed(1)} L${(cx+r*1.15).toFixed(1)},${cy.toFixed(1)} L${cx.toFixed(1)},${(cy+r*1.25).toFixed(1)} L${(cx-r*1.15).toFixed(1)},${cy.toFixed(1)} Z" ${a}/>`;
  else el=`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}" ${a}/>`;
  return (ring?`<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r+4.5}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>`:"")+el;
}

/* ---------- controls ---------- */
function drawControls(){
  METRICS.forEach(m=>{const c1=S1[m],c2=S2[m];
  btns("c1Method",m,["all",...METHODS],c1.method,v=>{c1.method=v;drawControls();renderC1(m);},v=>v==="all"?"mean of all":v);
  sel("c1Model",m,["all",...MODELS],c1.model,v=>{c1.model=v;renderC1(m);},v=>v==="all"?"all models (mean)":v);
  btns("c1Sort",m,["metric","mods"],c1.sort,v=>{c1.sort=v;drawControls();renderC1(m);},v=>v==="metric"?"by "+METRIC[m].label:"by modality");
  btns("c2User",m,["all",...USERS],c2.user,v=>{c2.user=v;drawControls();renderC2(m);});
  sel("c2Model",m,["all",...MODELS],c2.model,v=>{c2.model=v;renderC2(m);},v=>v==="all"?"all models (mean)":v);
  btns("c2X",m,["eval_count","keep"],c2.x,v=>{c2.x=v;drawControls();renderC2(m);},v=>v==="keep"?"keep %":"count after merge");
  btns("c2Scale",m,["log","linear"],c2.scale,v=>{c2.scale=v;drawControls();renderC2(m);},v=>c2.x==="keep"?v+" (n/a)":v);
  });
}
function renderAll(){METRICS.forEach(m=>{renderC1(m);renderC2(m);});}
drawControls();renderAll();
let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(renderAll,120);});
</script>
"""


if __name__ == "__main__":
    main()
