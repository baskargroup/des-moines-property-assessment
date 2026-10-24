"""
Build a PORTABLE review package (for Box / sharing) in one folder:
  - 3 self-contained HTML files (images embedded as compressed thumbnails, so they
    work on ANY computer with no broken image links)
  - A short README
  - An index.html landing page
Files cover the annotation-audit set: cases where the AI models disagree with the City.
"""
import json, html, collections, base64, io, os, urllib.parse
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predictions_full.jsonl"
PKG = Path(os.getenv("NOGWH_REVIEW_OUTPUT", ROOT / "generated_review_package"))
os.makedirs(PKG, exist_ok=True)

MODELS=[("gpt52","GPT-5.2","gpt-5.2-2025-12-11"),
        ("gem3fp","Gemini-3-Flash-Preview","gemini-3-flash-preview"),
        ("gpt41","GPT-4.1","gpt-4.1")]
SCORE_LABEL={1:"Excellent",2:"Good",3:"Fair",4:"Poor",5:"Critical"}
SCORE_COLOR={1:"#1F8A65",2:"#7DCAB0",3:"#E8C030",4:"#F0A040",5:"#C04848"}

def esc(s): return html.escape(str(s)) if s is not None else ""
def thumb(path,maxw=680,q=74):
    try:
        im=Image.open(path)
        im=ImageOps.exif_transpose(im)   # honor phone EXIF orientation
        im=im.convert("RGB"); w,h=im.size
        if w>maxw: im=im.resize((maxw,int(h*maxw/w)))
        buf=io.BytesIO(); im.save(buf,"JPEG",quality=q)
        return base64.b64encode(buf.getvalue()).decode()
    except: return ""

def subcards(p):
    if not p or "subscores" not in p: return "<p class='none'>No subscores.</p>"
    out=[]
    for k,d in p["subscores"].items():
        lbl=k.replace("_"," ").title(); v=d.get("score","?") if isinstance(d,dict) else d; ex=d.get("explanation","") if isinstance(d,dict) else ""
        out.append(f"<div class='sub-card'><h4>{esc(lbl)}</h4><div class='val'>{esc(v)}/5</div><p>{esc(ex)}</p></div>")
    return "<div class='subgrid'>"+"".join(out)+"</div>"
def issues(p):
    if not p: return "<p class='none'>—</p>"
    out=[]
    for fld,cls,t in [("red_flags","flag","Red Flags"),("minor_issues","","Minor Issues"),("positive_signs","positive","Positive Signs")]:
        it=p.get(fld) or []
        if it: out.append(f"<h5>{t}</h5>"+"".join(f"<div class='issue {cls}'>{esc(x)}</div>" for x in it))
    return "".join(out) if out else "<p class='none'>None.</p>"
def model_box(k,l,m,pred,gt):
    p=pred.get("parsed") or {}; sc=pred.get("predicted_score")
    if sc is None: bd="#bbb"; mk="no response"
    elif sc==gt: bd="#1F8A65"; mk="exact"
    elif abs(sc-gt)==1: bd="#E8C030"; mk="±1"
    else: bd="#C04848"; mk="miss"
    sccol=SCORE_COLOR.get(sc,"#ccc")
    scval=esc(sc if sc is not None else "—")
    just=esc(p.get("justification",""))
    return f"""<div class='mbox' style='border-top:3px solid {bd}'>
      <div class='mtop'><span class='mname'>{esc(l)}</span><span class='mscore' style='background:{sccol}'>{scval}</span></div>
      <div class='mmatch' style='color:{bd}'>{esc(mk)}</div>
      <details><summary>Subcategory scores</summary>{subcards(p)}</details>
      <details><summary>Red flags &amp; observations</summary>{issues(p)}</details>
      <details><summary>Justification</summary><div class='just'>{just}</div></details></div>"""
def chips(r):
    out=""
    for k,l,_ in MODELS:
        s=r["predictions"][k]["predicted_score"]
        col=SCORE_COLOR.get(s,"#ccc")
        nm=l.split("-")[0].strip()[:6]
        sv=s if s is not None else "—"
        out+=f"<span class='chip' style='background:{col}'>{nm}: {sv}</span>"
    return out
def card(i,r):
    gt=r["gt_score"]
    boxes="".join(model_box(k,l,m,r["predictions"][k],gt) for k,l,m in MODELS)
    rb="".join(f"<button class='rb' data-s='{s}' onclick='setR({i},{s})' style='--c:{SCORE_COLOR[s]}'>{s}</button>" for s in (1,2,3,4,5))
    gtcol=SCORE_COLOR.get(gt,"#888")
    gtlbl=esc(SCORE_LABEL.get(gt,"?"))
    fn=esc(r['filename'])
    return f"""<article class='card' id='p-{i}'>
      <div class='chead'><span class='fn'>#{i+1} · {fn}</span><span class='chips'>{chips(r)}</span>
        <span class='gt' style='background:{gtcol}'>City: {gt} · {gtlbl}</span></div>
      <div class='cbody'><div class='imgcol'><img src='data:image/jpeg;base64,{thumb(r["image_path"])}'/>
        <div class='rater'>Your score:&nbsp; {rb}</div>
        <textarea class='cmt' id='cmt-{i}' placeholder='Comment (optional)...' oninput='saveCmt({i})'></textarea>
        <div class='saverow'><button class='savebtn' onclick='saveCard({i})'>Save</button><span class='savemsg' id='sv-{i}'></span></div></div>
        <div class='mcol'>{boxes}</div></div></article>"""

CSS=r"""body{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:#f2f5fb;color:#2f3640;}
.page{max-width:1240px;margin:0 auto;padding:22px 18px 70px;}
.head{background:#2c5364;color:#fff;padding:22px 28px;border-radius:14px;margin-bottom:18px;}
.head h1{margin:0;font-size:21px;}.head .sub{opacity:.92;margin-top:8px;font-size:14px;line-height:1.5;}
.head .nav{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap;}.head .nav a{background:rgba(255,255,255,.15);color:#fff;text-decoration:none;padding:5px 11px;border-radius:999px;font-size:12px;}
.head button{margin-top:12px;background:#1F8A65;color:#fff;border:none;padding:9px 18px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;}
.card{background:#fff;border-radius:14px;overflow:hidden;margin-bottom:16px;box-shadow:0 6px 16px rgba(0,0,0,.06);}
.chead{padding:11px 16px;border-bottom:1px solid #eef1f7;display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;}
.fn{font-size:13px;color:#456;}.chips{display:flex;gap:5px;flex-wrap:wrap;}.chip{color:#fff;font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;}
.gt{color:#fff;font-weight:600;padding:5px 14px;border-radius:999px;font-size:13px;}
.cbody{display:grid;grid-template-columns:minmax(280px,1fr) 2fr;gap:16px;padding:14px 18px;}
.imgcol img{width:100%;border-radius:10px;box-shadow:0 6px 16px rgba(0,0,0,.1);display:block;}
.rater{margin-top:10px;font-size:13px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.rb{width:34px;height:34px;border:2px solid #d8e0ec;border-radius:8px;cursor:pointer;font-weight:600;background:#fff;}.rb:hover{border-color:var(--c);}.rb.sel{background:var(--c);color:#fff;border-color:var(--c);}
.cmt{width:100%;margin-top:9px;padding:8px 10px;border:1px solid #cfd8e6;border-radius:8px;font-size:13px;font-family:inherit;min-height:44px;resize:vertical;box-sizing:border-box;}
.saverow{margin-top:8px;display:flex;align-items:center;gap:10px;}.savebtn{background:#2E79B5;color:#fff;border:none;padding:8px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;}.savemsg{font-size:12px;color:#1F8A65;font-weight:600;}
.mcol{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}.mbox{border:1px solid #e0e6f1;border-radius:12px;padding:12px;}
.mtop{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;}.mname{font-weight:600;color:#2c3e50;font-size:13px;}
.mscore{width:36px;height:36px;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:18px;flex-shrink:0;}
.mmatch{font-size:11px;font-weight:600;margin-top:3px;}details{margin-top:8px;}summary{cursor:pointer;font-size:11px;color:#456;text-transform:uppercase;letter-spacing:.05em;}
.subgrid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:5px;}.sub-card{background:#f8fbff;padding:6px 8px;border-radius:7px;border:1px solid #e6ecf5;}
.sub-card h4{margin:0;font-size:10px;color:#34495e;font-weight:600;}.sub-card .val{font-size:13px;font-weight:600;margin:2px 0;}.sub-card p{margin:0;font-size:9px;color:#7b8aab;line-height:1.3;}
.issue{background:#fff5e6;border-left:3px solid #f39c12;padding:5px 8px;border-radius:5px;margin-bottom:4px;font-size:10px;}.issue.flag{background:#ffe8ea;border-color:#e74c3c;}.issue.positive{background:#e8f9f2;border-color:#1abc9c;}
h5{margin:6px 0 3px;font-size:10px;}.just{font-size:10px;margin-top:5px;color:#5a6577;font-style:italic;line-height:1.5;}.none{font-size:10px;color:#aab;}
@media(max-width:880px){.cbody{grid-template-columns:1fr;}.mcol{grid-template-columns:1fr;}}"""

def build(subset,fname,title,intro,key,nav):
    cards="\n".join(card(i,r) for i,r in enumerate(subset))
    h=f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'/><meta name='viewport' content='width=device-width,initial-scale=1.0'/>
<title>{title}</title><style>{CSS}</style></head><body><div class='page'>
<div class='head'><h1>{title}</h1><div class='sub'>{intro}</div><div class='nav'>{nav}</div>
<button onclick='downloadCSV()'>Download my reviews (Excel)</button></div>
{cards}<footer style='text-align:center;color:#7b8aab;font-size:12px;padding:20px;'>{len(subset)} properties · your work saves in this browser; click Download when done.</footer></div>
<script>const P='{key}_';
function setR(i,s){{document.querySelectorAll('#p-'+i+' .rb').forEach(b=>b.classList.toggle('sel',Number(b.dataset.s)===s));localStorage.setItem(P+i,String(s));}}
function saveCmt(i){{localStorage.setItem(P+'cmt_'+i,document.getElementById('cmt-'+i).value);}}
function saveCard(i){{saveCmt(i);const sc=localStorage.getItem(P+i)||'';const e=document.getElementById('sv-'+i);e.textContent=sc?'✓ Saved':'✓ Comment saved';setTimeout(()=>e.textContent='',2500);}}
function downloadCSV(){{let rows=[['property_number','filename','city_score','ai_scores','your_score','your_comment']];
document.querySelectorAll('.card').forEach(c=>{{const i=c.id.split('-')[1];const sc=localStorage.getItem(P+i)||'';const cm=localStorage.getItem(P+'cmt_'+i)||'';
if(sc||cm){{const fn=c.querySelector('.fn').textContent.replace(/^#\\d+ · /,'');const gt=(c.querySelector('.gt').textContent.match(/City: (\\d)/)||[])[1]||'';
const ai=c.querySelector('.chips').textContent.replace(/\\s+/g,' ').trim();rows.push([Number(i)+1,fn,gt,'"'+ai+'"',sc,'"'+cm.replace(/"/g,'""')+'"']);}}}});
const b=new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='reviews_{key}.csv';a.click();}}
document.addEventListener('DOMContentLoaded',()=>{{document.querySelectorAll('.card').forEach(c=>{{const i=c.id.split('-')[1];const v=localStorage.getItem(P+i);
if(v)c.querySelectorAll('.rb').forEach(b=>b.classList.toggle('sel',Number(b.dataset.s)===Number(v)));const cm=localStorage.getItem(P+'cmt_'+i);if(cm){{const t=document.getElementById('cmt-'+i);if(t)t.value=cm;}}}});}});
</script></body></html>"""
    with open(os.path.join(PKG,fname),"w",encoding="utf-8") as f: f.write(h)
    print(f"  {fname:<34}{len(subset):>4} props  {os.path.getsize(os.path.join(PKG,fname))/1024/1024:.1f} MB")

# classify by 3-CLASS category (Healthy / Intermediate / Unhealthy)
def class3(s):
    if s in (1,2): return "Healthy"
    if s==3: return "Intermediate"
    if s in (4,5): return "Unhealthy"
with open(PRED,encoding="utf-8") as f: rows=[json.loads(l) for l in f]
rated=[r for r in rows if r["gt_score"] in (1,2,3,4,5)]
def resp(r): return all(r["predictions"][m]["predicted_score"] is not None for m,_,_ in MODELS)
B=[];C2=[];D=[];A=[];C1=[];INC=[]
for r in rated:
    if not resp(r): INC.append(r); continue
    cats=[class3(r["predictions"][m]["predicted_score"]) for m,_,_ in MODELS]
    gt=class3(r["gt_score"])
    cnt=collections.Counter(cats); top,topn=cnt.most_common(1)[0]
    if topn==3: (A if top==gt else B).append(r)
    elif topn==2: (C1 if top==gt else C2).append(r)
    else: D.append(r)
for g in (B,C2,D,A,C1,INC): g.sort(key=lambda r:(r["gt_score"],r["filename"]))
half=len(C2)//2; C2a=C2[:half]; C2b=C2[half:]

nav=("<a href='index.html'>Home</a>"
     "<a href='1_priority_all3_disagree.html'>1</a>"
     "<a href='2a_two_disagree.html'>2a</a><a href='2b_two_disagree.html'>2b</a>"
     "<a href='3_all_different.html'>3</a>"
     "<a href='4_all3_match.html'>4</a><a href='5_two_match.html'>5</a>"
     "<a href='6_incomplete.html'>6</a>")

print("Building FULL portable package (embedding thumbnails, please wait)...")
build(B,"1_priority_all3_disagree.html","File 1 (PRIORITY) — All 3 AI models agree on a category, but the City is different",
      f"{len(B)} properties. All three AI models independently placed this property in the SAME condition category (Healthy / Fair / Distressed), yet the City put it in a different category — so the City annotation is most likely to need correction here. Look at each photo, decide the correct score (1-5), add a comment, and Save.",
      "f1",nav)
build(C2a,"2a_two_disagree.html","File 2a — 2 of 3 AI models agree on a category that differs from the City (part 1)",
      f"{len(C2a)} properties where two of the three models agreed on a category that differs from the City.","f2a",nav)
build(C2b,"2b_two_disagree.html","File 2b — 2 of 3 AI models agree on a category that differs from the City (part 2)",
      f"{len(C2b)} properties where two of the three models agreed on a category that differs from the City.","f2b",nav)
build(D,"3_all_different.html","File 3 — The 3 AI models gave three DIFFERENT categories",
      f"{len(D)} properties the models found genuinely ambiguous (each model chose a different category). Lower priority for annotation review.","f3",nav)
build(A,"4_all3_match.html","File 4 — All 3 AI models agree AND match the City (confirmation)",
      f"{len(A)} properties where all three models agree with the City. Spot-check only — confirm or adjust.","f4",nav)
build(C1,"5_two_match.html","File 5 — 2 of 3 AI models agree AND match the City (confirmation)",
      f"{len(C1)} properties where a 2-model majority matches the City. Spot-check only.","f5",nav)
build(INC,"6_incomplete.html","File 6 — A model did not respond (incomplete)",
      f"{len(INC)} properties where one AI model failed to return a score.","f6",nav)

# index.html
idx=f"""<!DOCTYPE html><html><head><meta charset='UTF-8'/><title>NOGWH Annotation Review</title>
<style>body{{font-family:'Segoe UI',Arial,sans-serif;background:#f2f5fb;margin:0;color:#2f3640;}}
.w{{max-width:780px;margin:40px auto;padding:0 20px;}}
.h{{background:#2c5364;color:#fff;padding:30px 34px;border-radius:16px;}}.h h1{{margin:0;font-size:24px;}}
.c{{background:#fff;border-radius:14px;padding:20px 24px;margin-top:14px;box-shadow:0 8px 20px rgba(0,0,0,.06);}}
.c h2{{margin:0 0 4px;font-size:17px;}}.c p{{margin:4px 0;font-size:14px;color:#556;}}
.c a{{display:inline-block;margin-top:8px;background:#2E79B5;color:#fff;text-decoration:none;padding:8px 16px;border-radius:8px;font-weight:600;font-size:14px;}}
.pri{{border-left:6px solid #C04848;}}.tag{{font-size:12px;color:#C04848;font-weight:700;}}
.sec{{font-size:12px;color:#1F8A65;font-weight:700;}}.opt{{opacity:.75;}}
ol{{line-height:1.7;}} hr{{border:none;border-top:1px solid #e5ebf3;margin:20px 0;}}</style></head><body><div class='w'>
<div class='h'><h1>Property Condition — Annotation Review</h1>
<p style='opacity:.9;margin:8px 0 0;'>Please help us check whether the City's property-condition scores are accurate. Start with File 1.</p></div>

<div class='c pri'><span class='tag'>START HERE — most important</span>
<h2>File 1 — Models disagree with the City ({len(B)} properties)</h2>
<p>All 3 AI models put the property in the SAME condition category, but the City chose a different one. These are the most likely annotation errors.</p>
<a href='1_priority_all3_disagree.html'>Open File 1</a></div>

<div class='c'><span class='sec'>NEXT</span><h2>File 2a — 2 of 3 disagree with the City ({len(C2a)})</h2>
<a href='2a_two_disagree.html'>Open File 2a</a></div>
<div class='c'><span class='sec'>NEXT</span><h2>File 2b — 2 of 3 disagree with the City ({len(C2b)})</h2>
<a href='2b_two_disagree.html'>Open File 2b</a></div>
<div class='c'><h2>File 3 — Models all differ from each other ({len(D)})</h2>
<p>Ambiguous cases; lower priority.</p><a href='3_all_different.html'>Open File 3</a></div>

<hr/>
<div class='c opt'><h2>Optional confirmation files</h2>
<p>These are properties where the AI already agrees with the City — only spot-check if you have time.</p>
<a href='4_all3_match.html'>File 4 — all agree &amp; match ({len(A)})</a>
&nbsp;<a href='5_two_match.html'>File 5 — 2 agree &amp; match ({len(C1)})</a>
&nbsp;<a href='6_incomplete.html'>File 6 — incomplete ({len(INC)})</a></div>

<div class='c'><h2>How to do it</h2><ol>
<li>Open a file (start with File 1).</li>
<li>For each property: look at the photo and the AI models' notes.</li>
<li>Pick the score YOU think is correct (1 = Excellent … 5 = Critical).</li>
<li>Add a short comment if helpful, then click <b>Save</b>.</li>
<li>When finished with a file, click <b>"Download my reviews (Excel)"</b> at the top and send us that file.</li>
</ol><p style='color:#888;font-size:13px;'>Your work saves in the browser automatically, so you can stop and come back. Just remember to Download at the end of each file.</p></div>
</div></body></html>"""
with open(os.path.join(PKG,"index.html"),"w",encoding="utf-8") as f: f.write(idx)

README=f"""PROPERTY CONDITION — ANNOTATION REVIEW
========================================

WHAT THIS IS
We are checking whether the City of Des Moines' property-condition scores
(1 = Excellent ... 5 = Critical) are accurate. For each property we also have
scores from 3 AI models. This package groups the properties by how much the AI
models DISAGREE with the City -- the disagreements are the ones worth a human check.

HOW TO START
1. Open  index.html  in any web browser (Chrome, Edge). Just double-click it.
2. Start with FILE 1 (the priority file -- {len(B)} properties).
3. For each property:
     - Look at the photo and the AI models' notes.
     - Choose the score YOU think is correct using the 1-5 buttons.
     - Add a comment if you like.
     - Click SAVE.
4. When you finish a file, click "Download my reviews (Excel)" at the top.
   It downloads a small spreadsheet -- please send those files back to us.

FILES (priority order)
  index.html                      <- open this first
  PRIORITY:
   1_priority_all3_disagree.html  <- FILE 1 (start here, {len(B)} properties)
  NEXT:
   2a_two_disagree.html           <- FILE 2a ({len(C2a)} properties)
   2b_two_disagree.html           <- FILE 2b ({len(C2b)} properties)
   3_all_different.html           <- FILE 3 ({len(D)} properties, ambiguous)
  OPTIONAL (AI already agrees with the City -- spot-check only):
   4_all3_match.html              <- FILE 4 ({len(A)} properties)
   5_two_match.html               <- FILE 5 ({len(C1)} properties)
   6_incomplete.html              <- FILE 6 ({len(INC)} properties)

NOTES
- Everything works offline; no internet or login needed.
- Your answers save automatically in the browser as you go, so you can stop and
  resume. Just remember to click "Download my reviews" at the end of each file.
- 1 = Excellent, 2 = Good, 3 = Fair, 4 = Poor, 5 = Critical.

Thank you!
"""
with open(os.path.join(PKG,"README.txt"),"w",encoding="utf-8") as f: f.write(README)

print(f"\nPackage folder: {PKG}")
tot=sum(os.path.getsize(os.path.join(PKG,x)) for x in os.listdir(PKG))
print(f"Total package size: {tot/1024/1024:.1f} MB")
print("Contents:", ", ".join(sorted(os.listdir(PKG))))
