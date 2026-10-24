"""
Split ALL images into agreement-based review files (5-class / exact score), same
rich format as nogwh_consensus_wrong.html (photo + 3 models' detail + score
dropdown + comment + Save + Download CSV). Each file uses its own localStorage
namespace so reviews never clash. Buckets are mutually exclusive (no overlap).
"""
import json, html, collections, urllib.parse, os

PRED=r"predictions_full.jsonl"
OUTDIR=r"."
MODELS=[("gpt52","GPT-5.2","gpt-5.2-2025-12-11"),
        ("gem3fp","Gemini-3-Flash-Preview","gemini-3-flash-preview"),
        ("gpt41","GPT-4.1","gpt-4.1")]
SCORE_LABEL={1:"Excellent",2:"Good",3:"Fair",4:"Poor",5:"Critical"}
SCORE_COLOR={1:"#1F8A65",2:"#7DCAB0",3:"#E8C030",4:"#F0A040",5:"#C04848"}

def esc(s): return html.escape(str(s)) if s is not None else ""
def file_url(p): return "file:///"+urllib.parse.quote(os.path.abspath(p).replace("\\","/"))

def subcards(parsed):
    if not parsed or "subscores" not in parsed: return "<p class='none'>No subscores.</p>"
    out=[]
    for key,data in parsed["subscores"].items():
        lbl=key.replace("_"," ").title()
        v=data.get("score","?") if isinstance(data,dict) else data
        ex=data.get("explanation","") if isinstance(data,dict) else ""
        out.append(f"<div class='sub-card'><h4>{esc(lbl)}</h4><div class='val'>{esc(v)}/5</div><p>{esc(ex)}</p></div>")
    return "<div class='subgrid'>"+"".join(out)+"</div>"

def issues(parsed):
    if not parsed: return "<p class='none'>—</p>"
    out=[]
    for fld,cls,title in [("red_flags","flag","Red Flags"),("minor_issues","","Minor Issues"),("positive_signs","positive","Positive Signs")]:
        items=parsed.get(fld) or []
        if items: out.append(f"<h5>{title}</h5>"+"".join(f"<div class='issue {cls}'>{esc(x)}</div>" for x in items))
    return "".join(out) if out else "<p class='none'>None reported.</p>"

def model_box(key,label,mid,pred,gt):
    parsed=pred.get("parsed") or {}
    score=pred.get("predicted_score")
    if score is None: border="#bbb"; mark="no response"
    elif score==gt: border="#1F8A65"; mark="exact"
    elif abs(score-gt)==1: border="#E8C030"; mark="±1"
    else: border="#C04848"; mark="miss"
    return f"""
      <div class='mbox' style='border-top:3px solid {border}'>
        <div class='mtop'><span class='mname'>{esc(label)}</span>
          <span class='mscore' style='background:{SCORE_COLOR.get(score,'#ccc')}'>{esc(score if score is not None else '—')}</span></div>
        <div class='mid'>{esc(mid)}</div>
        <div class='mmatch' style='color:{border}'>{esc(mark)} · {esc(pred.get('predicted_category',''))}</div>
        <details><summary>Subcategory scores</summary>{subcards(parsed)}</details>
        <details><summary>Red flags &amp; observations</summary><div class='iss'>{issues(parsed)}</div></details>
        <details><summary>Justification</summary><div class='just'>{esc(parsed.get('justification',''))}</div></details>
      </div>"""

def chips(r):
    out=""
    for k,l,_ in MODELS:
        s=r["predictions"][k]["predicted_score"]
        c=SCORE_COLOR.get(s,"#ccc")
        out+=f"<span class='chip' style='background:{c}'>{l.split('-')[0].strip()[:6]}: {s if s is not None else '—'}</span>"
    return out

def render_card(gidx,r):
    gt=r["gt_score"]
    boxes="".join(model_box(k,l,m,r["predictions"][k],gt) for k,l,m in MODELS)
    rb="".join(f"<button class='rb' data-s='{s}' onclick='setR({gidx},{s})' style='--c:{SCORE_COLOR[s]}'>{s}</button>" for s in (1,2,3,4,5))
    return f"""
    <article class='card' id='p-{gidx}'>
      <div class='chead'>
        <span class='fn'>#{gidx+1} · {esc(r['filename'])}</span>
        <span class='chips'>{chips(r)}</span>
        <span class='gt' style='background:{SCORE_COLOR.get(gt,'#888')}'>City: {gt} · {esc(SCORE_LABEL.get(gt,'?'))}</span>
      </div>
      <div class='cbody'>
        <div class='imgcol'>
          <img loading='lazy' src='{file_url(r["image_path"])}' alt='{esc(r["filename"])}' />
          <div class='rater'>Your score:&nbsp; {rb}</div>
          <textarea class='cmt' id='cmt-{gidx}' placeholder='Add a comment (optional)...' oninput='saveCmt({gidx})'></textarea>
          <div class='saverow'><button class='savebtn' onclick='saveCard({gidx})'>Save</button><span class='savemsg' id='sv-{gidx}'></span></div>
        </div>
        <div class='mcol'>{boxes}</div>
      </div>
    </article>"""

CSS=r"""
body{margin:0;font-family:'Inter','Segoe UI',sans-serif;background:#f2f5fb;color:#2f3640;}
.page{max-width:1280px;margin:0 auto;padding:24px 20px 80px;}
.head{background:linear-gradient(135deg,#0f2027,#2c5364);color:#fff;padding:24px 30px;border-radius:16px;margin-bottom:20px;}
.head h1{margin:0;font-size:22px;} .head .sub{opacity:.9;margin-top:8px;font-size:14px;line-height:1.5;}
.head .nav{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap;}
.head .nav a{background:rgba(255,255,255,.13);color:#fff;text-decoration:none;padding:5px 11px;border-radius:999px;font-size:12px;}
.head .nav a:hover{background:rgba(255,255,255,.28);}
.head .dl{margin-top:12px;} .head button{background:#1F8A65;color:#fff;border:none;padding:9px 18px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;}
.card{background:#fff;border-radius:14px;overflow:hidden;margin-bottom:18px;box-shadow:0 8px 20px rgba(0,0,0,.05);}
.chead{padding:12px 18px;border-bottom:1px solid #eef1f7;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;}
.fn{font-size:13px;color:#456;} .chips{display:flex;gap:5px;flex-wrap:wrap;}
.chip{color:#fff;font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;}
.gt{color:#fff;font-weight:600;padding:5px 14px;border-radius:999px;font-size:13px;}
.cbody{display:grid;grid-template-columns:minmax(300px,1fr) 2fr;gap:18px;padding:16px 20px;}
.imgcol img{width:100%;border-radius:10px;box-shadow:0 6px 16px rgba(0,0,0,.1);display:block;}
.rater{margin-top:12px;font-size:13px;display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.rb{width:34px;height:34px;border:2px solid #d8e0ec;border-radius:8px;cursor:pointer;font-weight:600;background:#fff;}
.rb:hover{border-color:var(--c);} .rb.sel{background:var(--c);color:#fff;border-color:var(--c);}
.cmt{width:100%;margin-top:10px;padding:8px 10px;border:1px solid #cfd8e6;border-radius:8px;font-size:13px;font-family:inherit;min-height:46px;resize:vertical;box-sizing:border-box;}
.saverow{margin-top:8px;display:flex;align-items:center;gap:10px;}
.savebtn{background:#2E79B5;color:#fff;border:none;padding:8px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;}
.savemsg{font-size:12px;color:#1F8A65;font-weight:600;}
.mcol{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.mbox{border:1px solid #e0e6f1;border-radius:12px;padding:14px;}
.mtop{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;}
.mname{font-weight:600;color:#2c3e50;font-size:14px;} .mid{font-size:10px;color:#8a98ad;margin-top:1px;}
.mscore{width:38px;height:38px;border-radius:50%;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:20px;flex-shrink:0;}
.mmatch{font-size:11px;font-weight:600;margin-top:4px;}
details{margin-top:10px;} summary{cursor:pointer;font-size:11px;color:#456;text-transform:uppercase;letter-spacing:.05em;padding:4px 0;}
details[open] summary{font-weight:600;color:#2c3e50;}
.subgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px;}
.sub-card{background:#f8fbff;padding:7px 9px;border-radius:7px;border:1px solid #e6ecf5;}
.sub-card h4{margin:0;font-size:10px;color:#34495e;font-weight:600;} .sub-card .val{font-size:14px;font-weight:600;margin:2px 0;color:#2c3e50;}
.sub-card p{margin:0;font-size:9px;color:#7b8aab;line-height:1.3;}
.issue{background:#fff5e6;border-left:3px solid #f39c12;padding:5px 9px;border-radius:5px;margin-bottom:4px;font-size:10px;}
.issue.flag{background:#ffe8ea;border-color:#e74c3c;} .issue.positive{background:#e8f9f2;border-color:#1abc9c;}
h5{margin:7px 0 4px;font-size:10px;color:#2c3e50;}
.just{font-size:10px;margin-top:6px;color:#5a6577;font-style:italic;line-height:1.5;} .none{font-size:10px;color:#aab;margin:4px 0;}
@media(max-width:900px){.cbody{grid-template-columns:1fr;}.mcol{grid-template-columns:1fr;}}
"""

def build(subset, fname, title, intro, key, nav):
    cards="\n".join(render_card(i,r) for i,r in enumerate(subset))
    html_out=f"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1.0'/><title>{title}</title>
<style>{CSS}</style></head><body><div class='page'>
<div class='head'><h1>{title}</h1><div class='sub'>{intro}</div>
<div class='nav'>{nav}</div>
<div class='dl'><button onclick='downloadCSV()'>Download my reviews (Excel)</button></div></div>
{cards}
<footer style='text-align:center;color:#7b8aab;font-size:12px;padding:24px 0;'>NOGWH annotation review · {len(subset)} properties · your scores &amp; comments saved in this browser</footer>
</div><script>
const P='{key}_';
function setR(i,s){{document.querySelectorAll('#p-'+i+' .rb').forEach(b=>b.classList.toggle('sel',Number(b.dataset.s)===s));localStorage.setItem(P+i,String(s));}}
function saveCmt(i){{localStorage.setItem(P+'cmt_'+i,document.getElementById('cmt-'+i).value);}}
function saveCard(i){{saveCmt(i);const sc=localStorage.getItem(P+i)||'';const el=document.getElementById('sv-'+i);el.textContent=sc?'✓ Saved':'✓ Comment saved (no score)';setTimeout(()=>{{el.textContent='';}},2500);}}
function downloadCSV(){{let rows=[['property_number','filename','city_score','ai_scores','your_score','your_comment']];
 document.querySelectorAll('.card').forEach(c=>{{const i=c.id.split('-')[1];const sc=localStorage.getItem(P+i)||'';const cm=localStorage.getItem(P+'cmt_'+i)||'';
  if(sc||cm){{const fn=c.querySelector('.fn').textContent.replace(/^#\d+ · /,'');const gt=(c.querySelector('.gt').textContent.match(/City: (\d)/)||[])[1]||'';
   const ai=c.querySelector('.chips').textContent.replace(/\s+/g,' ').trim();
   rows.push([Number(i)+1,fn,gt,'"'+ai+'"',sc,'"'+cm.replace(/"/g,'""')+'"']);}}}});
 const blob=new Blob([rows.map(r=>r.join(',')).join('\n')],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviews_{key}.csv';a.click();}}
document.addEventListener('DOMContentLoaded',()=>{{document.querySelectorAll('.card').forEach(c=>{{const i=c.id.split('-')[1];
 const v=localStorage.getItem(P+i);if(v)c.querySelectorAll('.rb').forEach(b=>b.classList.toggle('sel',Number(b.dataset.s)===Number(v)));
 const cm=localStorage.getItem(P+'cmt_'+i);if(cm){{const t=document.getElementById('cmt-'+i);if(t)t.value=cm;}}}});}});
</script></body></html>"""
    with open(os.path.join(OUTDIR,fname),"w",encoding="utf-8") as f: f.write(html_out)
    print(f"  {fname:<42}{len(subset):>4} properties  {os.path.getsize(os.path.join(OUTDIR,fname))/1024:.0f} KB")

# ---- classify ----
with open(PRED,encoding="utf-8") as f: rows=[json.loads(l) for l in f]
rated=[r for r in rows if r["gt_score"] in (1,2,3,4,5)]
def responded(r): return all(r["predictions"][m]["predicted_score"] is not None for m,_,_ in MODELS)
complete=[r for r in rated if responded(r)]
incomplete=[r for r in rated if not responded(r)]
A=[];B=[];C1=[];C2=[];D=[]
for r in complete:
    s=[r["predictions"][m]["predicted_score"] for m,_,_ in MODELS]; gt=r["gt_score"]
    cnt=collections.Counter(s); top,topn=cnt.most_common(1)[0]
    if topn==3: (A if top==gt else B).append(r)
    elif topn==2: (C1 if top==gt else C2).append(r)
    else: D.append(r)
for grp in (A,B,C1,C2,D,incomplete): grp.sort(key=lambda r:(r["gt_score"],r["filename"]))
# split C2 in half
half=len(C2)//2; C2a=C2[:half]; C2b=C2[half:]

FILES=[
 ("review_file1_all3agree_differ.html","file1","File 1 — All 3 AI models AGREE, but DIFFERENT from City",B),
 ("review_file2a_two_agree_differ.html","file2a","File 2a — 2 of 3 AI models agree, but DIFFERENT from City (part 1)",C2a),
 ("review_file2b_two_agree_differ.html","file2b","File 2b — 2 of 3 AI models agree, but DIFFERENT from City (part 2)",C2b),
 ("review_file3_all_different.html","file3","File 3 — All 3 AI models gave DIFFERENT scores",D),
 ("review_file4_all3agree_match.html","file4","File 4 — All 3 AI models AGREE and MATCH the City",A),
 ("review_file5_two_agree_match.html","file5","File 5 — 2 of 3 AI models agree and MATCH the City",C1),
 ("review_file6_incomplete.html","file6","File 6 — A model did not respond (incomplete)",incomplete),
]
nav="".join(f"<a href='{fn}'>{key}</a>" for fn,key,_,_ in FILES)

print("Building 7 review files (same rich format)...")
for fname,key,title,subset in FILES:
    intro=f"{len(subset)} properties. For each: review the photo and the AI models' reasoning, pick YOUR score (the City annotation may be inaccurate), add a comment, and Save. Then click 'Download my reviews'."
    build(subset,fname,title,intro,key,nav)

# overlap check
allids=[r["image_id"] for _,_,_,sub in FILES for r in sub]
print(f"\nTotal across all files: {len(allids)}   Unique: {len(set(allids))}   Overlap: {'NONE' if len(allids)==len(set(allids)) else 'YES!'}")
print(f"(Original rated images: {len(rated)})")
