"""
build_dashboard.py — Generate a self-contained dashboard.html from results.json.

No external dependencies: data is embedded as JSON and all charts are drawn
with vanilla JS on <canvas>, so the file works offline by double-clicking.
"""
import os, json, html, datetime

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")
OUT = os.path.join(HERE, "dashboard.html")

OD_MAX = 64.0
ID_MIN = 55.0
F_MHZ = 10.64


def build():
    with open(RESULTS) as f:
        store = json.load(f)
    raw = [d for d in store.get("designs", []) if "max_error" in d]
    # dedupe identical geometries (repeated-region sweeps), keep best homogeneity
    seen = {}
    for d in raw:
        key = (d.get("coil_length_mm"), d.get("coil_radius_mm"), d.get("M"),
               d.get("n_max"), d.get("allow_bucking"), d.get("former_mm"),
               d.get("awg"), d.get("symmetric"))
        cur = d.get("off_mag_pm", d.get("max_error", 9e9))
        if key not in seen or cur < seen[key].get("off_mag_pm", seen[key].get("max_error", 9e9)):
            seen[key] = d
    designs = list(seen.values())
    batches = store.get("batches", [])

    fitting = [d for d in designs if d.get("fits")]
    # champion = best TRUE 3D homogeneity among designs with a STABLE, buildable
    # tuning cap (C >= 25 pF so stray capacitance doesn't dominate).
    def hom(d):
        return d.get("off_mag_pm", d.get("max_error", 1e9))
    practical = [d for d in fitting
                 if 25.0 <= (d.get("C_pF") or 0) <= 300.0 and (d.get("Q") or 0) >= 350]
    pool = practical if practical else fitting
    fitting_sorted = sorted(pool, key=hom)
    best = fitting_sorted[0] if fitting_sorted else None

    # compact payload for the browser
    payload = {
        "designs": [
            {k: d.get(k) for k in (
                "id", "batch", "max_error", "off_mag_pm", "ripple_pm", "flat5_mm", "flat1_mm",
                "total_turns", "L_uH", "C_pF", "Q", "OD_mm", "ID_mm", "max_layers",
                "coil_length_mm", "coil_radius_mm", "M", "n_max", "allow_bucking",
                "former_mm", "awg", "fov_half_mm", "n_forward", "n_bucking",
                "fits", "c_status", "windings")}
            for d in designs
        ],
        "best_id": best["id"] if best else None,
        "batches": batches,
        "od_max": OD_MAX, "id_min": ID_MIN, "f_mhz": F_MHZ,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    n_exp = len(designs)
    n_fit = len(fitting)
    best_err = f"{hom(best)*100:.2f}%" if best else "—"
    data_json = json.dumps(payload)

    doc = TEMPLATE.replace("/*DATA*/", data_json) \
                  .replace("{{N_EXP}}", str(n_exp)) \
                  .replace("{{N_FIT}}", str(n_fit)) \
                  .replace("{{BEST_ERR}}", html.escape(best_err)) \
                  .replace("{{GEN}}", payload["generated"])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    return OUT


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RF Coil Optimization Dashboard</title>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --panel2:#1c2230; --line:#2a3240;
    --tx:#e6edf3; --mut:#9aa7b4; --acc:#4dabf7; --good:#37b24d; --bad:#f03e3e;
    --warn:#f59f00; --fwd:#4dabf7; --buck:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);
    font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:20px 24px;border-bottom:1px solid var(--line);
    display:flex;flex-wrap:wrap;align-items:baseline;gap:12px 20px}
  h1{font-size:20px;margin:0;font-weight:650}
  .sub{color:var(--mut);font-size:13px}
  .wrap{padding:20px 24px;max-width:1180px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .kpi .v{font-size:26px;font-weight:680;letter-spacing:-.5px}
  .kpi .l{color:var(--mut);font-size:12px;margin-top:2px}
  .kpi .v.good{color:var(--good)} .kpi .v.acc{color:var(--acc)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
  .card h2{font-size:14px;margin:0 0 4px;font-weight:640}
  .card .note{color:var(--mut);font-size:12px;margin:0 0 10px}
  canvas{width:100%;display:block}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
  tr:hover td{background:var(--panel2)}
  .tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600}
  .tag.ok{background:rgba(55,178,77,.16);color:#69db7c}
  .tag.no{background:rgba(240,62,62,.16);color:#ff8787}
  .tablewrap{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:10px}
  .best dl{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;margin:0;font-size:13px}
  .best dt{color:var(--mut)} .best dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-top:6px}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
  .constraints{font-size:12px;color:var(--mut)}
  .constraints b{color:var(--tx)}
  code{background:var(--panel2);padding:1px 5px;border-radius:5px;font-size:12px}
</style>
</head>
<body>
<header>
  <h1>RF Coil Optimization</h1>
  <span class="sub">DIY-MRI solenoid @ {{GEN}}</span>
  <span class="sub constraints">Fixed: <b>f=10.64 MHz</b> · <b>OD ≤ 64 mm</b> · <b>ID ≥ 55 mm</b> — optimizing the winding pattern + geometry for <b>3D B₁ uniformity</b> over the bore volume (±20 mm axial, r ≤ 0.65·R), with a buildable tuning cap.</span>
</header>
<div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="v">{{N_EXP}}</div><div class="l">Experiments run</div></div>
    <div class="kpi"><div class="v acc">{{N_FIT}}</div><div class="l">Fit OD≤64 &amp; ID≥55</div></div>
    <div class="kpi"><div class="v good">{{BEST_ERR}}</div><div class="l">Best 3D B₁ error (fitting)</div></div>
    <div class="kpi" id="kpiC"><div class="v">—</div><div class="l">Best design tuning C</div></div>
    <div class="kpi" id="kpiQ"><div class="v">—</div><div class="l">Best design Q</div></div>
  </div>

  <div class="grid">
    <div class="card best">
      <h2>Best fitting design <span id="bestid" class="sub"></span></h2>
      <p class="note">Lowest on-axis B₁ error among designs that satisfy all fixed constraints.</p>
      <dl id="bestdl"></dl>
    </div>
    <div class="card">
      <h2>On-axis B₁ profile — best design</h2>
      <p class="note">Achieved vs. ideal-uniform over the ±20 mm FoV (per amp, computed from winding pattern).</p>
      <canvas id="fieldCanvas" height="220"></canvas>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Winding pattern — best design</h2>
      <p class="note">Integer turns per groove along the coil axis.</p>
      <canvas id="windCanvas" height="200"></canvas>
      <div class="legend">
        <span><span class="dot" style="background:var(--fwd)"></span>Forward</span>
        <span><span class="dot" style="background:var(--buck)"></span>Bucking</span>
      </div>
    </div>
    <div class="card">
      <h2>Homogeneity vs. turns — all fitting designs</h2>
      <p class="note">Each point a design. Lower-left = flat field with few turns (practical). Hover for details.</p>
      <canvas id="scatterCanvas" height="220"></canvas>
      <div id="scTip" class="sub"></div>
    </div>
  </div>

  <div class="card">
    <h2>Top designs</h2>
    <p class="note">Click a header to sort. ✓ = satisfies OD/ID. Click a row to load it into the plots above.</p>
    <div class="tablewrap">
      <table id="tbl"><thead><tr>
        <th data-k="id">#</th><th data-k="fits">fit</th>
        <th data-k="max_error">3D err</th><th data-k="off_mag_pm">off-val</th>
        <th data-k="flat5_mm">flat&lt;5% (mm)</th>
        <th data-k="total_turns">turns</th><th data-k="L_uH">L (µH)</th>
        <th data-k="C_pF">C (pF)</th><th data-k="Q">Q</th>
        <th data-k="OD_mm">OD</th><th data-k="ID_mm">ID</th>
        <th data-k="coil_length_mm">len</th><th data-k="M">M</th>
        <th data-k="n_max">n±</th><th data-k="allow_bucking">buck</th>
        <th data-k="awg">AWG</th>
      </tr></thead><tbody></tbody></table>
    </div>
  </div>

  <div class="card">
    <h2>Experiment log</h2>
    <div class="tablewrap" style="max-height:200px">
      <table id="logtbl"><thead><tr><th>batch</th><th>designs kept</th><th>seconds</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>

<script>
const DATA = /*DATA*/;
const MU0 = 4e-7*Math.PI;
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();

// ---- physics: on-axis field per amp from a winding pattern ----
function fieldProfile(d, xs){
  const L = d.coil_length_mm/1000, R = d.coil_radius_mm/1000, M = d.M;
  const w = d.windings || [];
  return xs.map(x=>{
    let B=0;
    for(let m=0;m<M;m++){
      if(!w[m]) continue;
      const xm = -L/2 + L*m/(M-1);
      const dx = x - xm;
      B += w[m]*MU0*R*R/(2*Math.pow(R*R+dx*dx,1.5));
    }
    return B;
  });
}

function fmt(x,dp=2){ return (x==null||!isFinite(x))?'—':Number(x).toFixed(dp); }

// ---- canvas helpers ----
function setup(cv){
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth||cv.parentElement.clientWidth;
  const h=cv.height;
  cv.width=w*dpr; cv.height=h*dpr;
  const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
  return {ctx,w,h};
}
function axes(ctx,w,h,pad){
  ctx.strokeStyle=css('--line'); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,h-pad.b); ctx.lineTo(w-pad.r,h-pad.b); ctx.stroke();
}

function drawField(d){
  const cv=document.getElementById('fieldCanvas'); const {ctx,w,h}=setup(cv);
  ctx.clearRect(0,0,w,h);
  if(!d){return;}
  const fov=d.fov_half_mm/1000;
  const xs=[]; const N=160; for(let i=0;i<N;i++) xs.push(-fov+2*fov*i/(N-1));
  const B=fieldProfile(d,xs);
  const mean=B.reduce((a,b)=>a+b,0)/B.length;
  const pad={l:52,r:12,t:12,b:30};
  const mn=Math.min(...B), mx=Math.max(...B);
  const lo=mn-(mx-mn)*0.3-1e-12, hi=mx+(mx-mn)*0.3+1e-12;
  const X=x=>pad.l+(x+fov)/(2*fov)*(w-pad.l-pad.r);
  const Y=y=>h-pad.b-(y-lo)/(hi-lo)*(h-pad.t-pad.b);
  axes(ctx,w,h,pad);
  // mean line
  ctx.strokeStyle=css('--warn'); ctx.setLineDash([4,4]); ctx.beginPath();
  ctx.moveTo(X(-fov),Y(mean)); ctx.lineTo(X(fov),Y(mean)); ctx.stroke(); ctx.setLineDash([]);
  // achieved
  ctx.strokeStyle=css('--acc'); ctx.lineWidth=2; ctx.beginPath();
  xs.forEach((x,i)=>{ const px=X(x),py=Y(B[i]); i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
  ctx.stroke();
  ctx.fillStyle=css('--mut'); ctx.font='11px sans-serif';
  ctx.fillText((mean*1e6).toFixed(2)+' µT/A mean', pad.l+6, pad.t+12);
  ctx.fillText('−'+d.fov_half_mm+' mm', pad.l, h-10);
  ctx.textAlign='right'; ctx.fillText('+'+d.fov_half_mm+' mm', w-pad.r, h-10); ctx.textAlign='left';
}

function drawWind(d){
  const cv=document.getElementById('windCanvas'); const {ctx,w,h}=setup(cv);
  ctx.clearRect(0,0,w,h);
  if(!d){return;}
  const wnd=d.windings||[]; const M=d.M; const L=d.coil_length_mm;
  const pad={l:36,r:12,t:14,b:26};
  const nmax=Math.max(1,...wnd.map(Math.abs));
  const X=m=>pad.l+m/(M-1)*(w-pad.l-pad.r);
  const y0=h-pad.b - (h-pad.t-pad.b)*(nmax)/(2*nmax); // zero baseline centered
  const Y=n=>y0 - n/nmax*(h-pad.t-pad.b)/2;
  ctx.strokeStyle=css('--line'); ctx.beginPath(); ctx.moveTo(pad.l,y0); ctx.lineTo(w-pad.r,y0); ctx.stroke();
  const bw=Math.max(2,(w-pad.l-pad.r)/M*0.7);
  wnd.forEach((n,m)=>{
    if(!n) return;
    ctx.fillStyle= n>0?css('--fwd'):css('--buck');
    const x=X(m), yt=Y(n);
    ctx.fillRect(x-bw/2, Math.min(y0,yt), bw, Math.abs(yt-y0));
  });
  ctx.fillStyle=css('--mut'); ctx.font='11px sans-serif';
  ctx.fillText('+'+nmax, 6, Y(nmax)+4); ctx.fillText('0', 6, y0+4); ctx.fillText('−'+nmax,6,Y(-nmax)+4);
  ctx.fillText('groove position ('+L.toFixed(0)+' mm span)', pad.l, h-8);
}

let scatterPts=[];
function drawScatter(designs){
  const cv=document.getElementById('scatterCanvas'); const {ctx,w,h}=setup(cv);
  ctx.clearRect(0,0,w,h);
  const pts=designs.filter(d=>d.fits && isFinite(d.max_error));
  if(!pts.length){return;}
  const pad={l:52,r:12,t:12,b:32};
  const xs=pts.map(d=>d.total_turns), ys=pts.map(d=>d.max_error*100);
  const xmn=Math.min(...xs), xmx=Math.max(...xs);
  const ymn=Math.max(1e-3,Math.min(...ys)), ymx=Math.max(...ys);
  const lx=Math.log10(ymn), hx=Math.log10(ymx*1.1);
  const X=x=>pad.l+(x-xmn)/((xmx-xmn)||1)*(w-pad.l-pad.r);
  const Y=y=>h-pad.b-(Math.log10(Math.max(y,1e-3))-lx)/((hx-lx)||1)*(h-pad.t-pad.b);
  axes(ctx,w,h,pad);
  ctx.fillStyle=css('--mut'); ctx.font='11px sans-serif';
  ctx.fillText('B₁ err % (log)', 4, pad.t+2);
  ctx.fillText('total turns', w/2-24, h-8);
  // gridlines at decades
  for(let e=Math.ceil(lx);e<=Math.floor(hx);e++){
    const yy=Y(Math.pow(10,e)); ctx.strokeStyle=css('--line');
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();
    ctx.fillText(Math.pow(10,e)+'%',6,yy+3);
  }
  scatterPts=[];
  pts.forEach(d=>{
    const px=X(d.total_turns), py=Y(d.max_error*100);
    const inC=d.c_status==='in_range';
    ctx.beginPath(); ctx.arc(px,py,4,0,7);
    ctx.fillStyle= inC?css('--good'):css('--mut'); ctx.globalAlpha=inC?0.9:0.5;
    ctx.fill(); ctx.globalAlpha=1;
    scatterPts.push({px,py,d});
  });
  ctx.fillStyle=css('--mut');
  ctx.fillText('● practical C   ● other', pad.l, pad.t+2);
}

// ---- best design panel ----
function showBest(d){
  if(!d) return;
  document.getElementById('bestid').textContent = '#'+d.id;
  const rows=[
    ['3D B₁ error (bore vol.)', fmt(d.max_error*100,2)+' %'],
    ['Off-axis validation', d.off_mag_pm==null?'—':fmt(d.off_mag_pm*100,2)+' %'],
    ['Flat &lt;5% length', fmt(d.flat5_mm,1)+' mm'],
    ['Inner diameter', fmt(d.ID_mm,1)+' mm'],
    ['Outer diameter', fmt(d.OD_mm,1)+' mm'],
    ['Coil length', fmt(d.coil_length_mm,0)+' mm'],
    ['Grooves M', d.M],
    ['Layers (n±)', d.max_layers+' / '+d.n_max],
    ['Former', fmt(d.former_mm,2)+' mm · AWG '+d.awg],
    ['Total turns', d.total_turns+'  (fwd '+d.n_forward+', buck '+d.n_bucking+')'],
    ['Inductance', fmt(d.L_uH,2)+' µH'],
    ['Tuning C', fmt(d.C_pF,1)+' pF'],
    ['Q factor', fmt(d.Q,0)],
  ];
  document.getElementById('bestdl').innerHTML =
    rows.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');
  document.querySelector('#kpiC .v').textContent = fmt(d.C_pF,0)+' pF';
  document.querySelector('#kpiQ .v').textContent = fmt(d.Q,0);
  drawField(d); drawWind(d);
}

// ---- table ----
let sortK='off_mag_pm', sortAsc=true;
function fitCols(d){return d.fits;}
function renderTable(){
  const tb=document.querySelector('#tbl tbody');
  const arr=DATA.designs.slice().sort((a,b)=>{
    let x=a[sortK], y=b[sortK];
    if(typeof x==='boolean'){x=x?1:0;y=y?1:0;}
    if(x==null)x=Infinity; if(y==null)y=Infinity;
    return sortAsc?(x-y):(y-x);
  }).slice(0,60);
  tb.innerHTML=arr.map(d=>`<tr data-id="${d.id}" style="cursor:pointer">
    <td>${d.id}</td>
    <td><span class="tag ${d.fits?'ok':'no'}">${d.fits?'✓':'✕'}</span></td>
    <td>${fmt(d.max_error*100,2)}%</td>
    <td>${d.off_mag_pm==null?'—':fmt(d.off_mag_pm*100,2)+'%'}</td>
    <td>${fmt(d.flat5_mm,0)}</td>
    <td>${d.total_turns}</td>
    <td>${fmt(d.L_uH,2)}</td>
    <td>${fmt(d.C_pF,0)}</td>
    <td>${fmt(d.Q,0)}</td>
    <td>${fmt(d.OD_mm,1)}</td>
    <td>${fmt(d.ID_mm,1)}</td>
    <td>${fmt(d.coil_length_mm,0)}</td>
    <td>${d.M}</td>
    <td>${d.n_max}</td>
    <td>${d.allow_bucking?'yes':'no'}</td>
    <td>${d.awg}</td>
  </tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr=>tr.onclick=()=>{
    const d=DATA.designs.find(x=>x.id==tr.dataset.id); showBest(d);
    window.scrollTo({top:0,behavior:'smooth'});
  });
}
document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; if(k===sortK) sortAsc=!sortAsc; else {sortK=k; sortAsc=(k==='max_error'||k==='OD_mm'||k==='total_turns');}
  renderTable();
});

// ---- scatter hover ----
const scv=document.getElementById('scatterCanvas');
scv.addEventListener('mousemove',e=>{
  const r=scv.getBoundingClientRect(); const mx=e.clientX-r.left, my=e.clientY-r.top;
  let near=null,best=1e9;
  scatterPts.forEach(p=>{const dd=(p.px-mx)**2+(p.py-my)**2; if(dd<best){best=dd;near=p;}});
  const tip=document.getElementById('scTip');
  if(near && best<200){const d=near.d;
    tip.textContent=`#${d.id}: err ${fmt(d.max_error*100,3)}% · ${d.total_turns} turns · C ${fmt(d.C_pF,0)}pF · Q ${fmt(d.Q,0)} · L=${fmt(d.coil_length_mm,0)}mm M=${d.M} n±${d.n_max}`;
  } else tip.textContent='';
});
scv.addEventListener('click',e=>{
  const r=scv.getBoundingClientRect(); const mx=e.clientX-r.left, my=e.clientY-r.top;
  let near=null,best=1e9;
  scatterPts.forEach(p=>{const dd=(p.px-mx)**2+(p.py-my)**2; if(dd<best){best=dd;near=p;}});
  if(near&&best<200){showBest(near.d); window.scrollTo({top:0,behavior:'smooth'});}
});

// ---- log ----
document.querySelector('#logtbl tbody').innerHTML =
  DATA.batches.map(b=>`<tr><td>${b.name}</td><td>${b.n}</td><td>${b.secs}</td></tr>`).join('')
  || '<tr><td colspan="3" class="sub">no batches yet</td></tr>';

// ---- init ----
const best = DATA.designs.find(d=>d.id===DATA.best_id);
showBest(best);
drawScatter(DATA.designs);
renderTable();
window.addEventListener('resize',()=>{ if(best){drawField(best);drawWind(best);} drawScatter(DATA.designs); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(build())
