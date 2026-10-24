"""Build the final interactive Terra / City / change GIS map."""
import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "citywide_terra"
SOURCE = RUN_DIR / "stage_66701_results.csv"
DELIVERY = RUN_DIR / "delivery"
HOTSPOTS = DELIVERY / "terra_hotspots_multi.json"
OUTPUT = DELIVERY / "terra_citywide_map.html"

points = []
terra_counts = Counter()
baseline_counts = Counter()
delta_counts = Counter()
with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        terra = int(row["terra_score"])
        baseline = int(row["baseline_score"])
        terra_counts[terra] += 1
        baseline_counts[baseline] += 1
        if baseline in range(1, 6):
            delta_counts[terra - baseline] += 1
        if not row["x"] or not row["y"]:
            continue
        points.append(
            [
                round(float(row["y"]), 7),
                round(float(row["x"]), 7),
                terra,
                baseline,
                terra - baseline if baseline in range(1, 6) else None,
                row["property_globalid"],
            ]
        )

hotspots = json.loads(HOTSPOTS.read_text(encoding="utf-8"))
points_json = json.dumps(points, separators=(",", ":"))
hotspots_json = json.dumps(hotspots, separators=(",", ":"))

html = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Des Moines — GPT-5.6 Terra Property Condition Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body{margin:0;height:100%;font-family:"Segoe UI",Arial,sans-serif;background:#15191c;color:#e8edf0}
#header{height:58px;padding:10px 16px;background:#101416;border-bottom:1px solid #30373c}
#header h1{font-size:18px;margin:0}.sub{font-size:11px;color:#9caab3;margin-top:3px}
#wrap{display:flex;height:calc(100vh - 58px)}#map{flex:1;background:#e7ebed}
#side{width:305px;background:#171c1f;border-left:1px solid #30373c;padding:13px;overflow:auto}
h2{font-size:11px;color:#91a0aa;text-transform:uppercase;letter-spacing:.08em;margin:0 0 7px}
.group{margin-bottom:15px}.btn{width:100%;padding:7px 9px;margin:0 0 5px;text-align:left;background:#242b2f;color:#e8edf0;border:1px solid #394249;border-radius:5px;cursor:pointer}
.btn:hover{background:#2d363b}.btn.active{background:#245b78;border-color:#3784ab}
.row{display:flex;gap:8px;align-items:center;padding:5px 7px;border-radius:4px;cursor:pointer;font-size:12px}
.row:hover{background:#252c30}.row.active{outline:1px solid #59666e}.swatch{width:11px;height:11px;border-radius:50%}.label{flex:1}.count{font-size:10px;color:#88969f}
.stat{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #252c30;font-size:11px}.stat b{color:#fff}
.note{font-size:10px;line-height:1.4;color:#88969f;margin-top:6px}
#banner{position:absolute;z-index:1000;left:50%;top:12px;transform:translateX(-50%);background:#15191ce8;color:#fff;border:1px solid #56636b;border-radius:999px;padding:7px 14px;font-size:12px;font-weight:600}
.leaflet-tooltip{background:#15191c;color:#fff;border:1px solid #59666e}
@media(max-width:800px){#side{width:240px}}
</style></head><body>
<div id="header"><h1>Des Moines Property Condition — GPT-5.6 Terra</h1>
<div class="sub">66,701 properties · one photo per parcel · compare Terra, City baseline, and score changes</div></div>
<div id="wrap"><div id="map"><div id="banner">Loading…</div></div><aside id="side">
 <div class="group"><h2>View</h2>
  <button class="btn active mode" data-mode="terra">Terra property scores</button>
  <button class="btn mode" data-mode="baseline">City baseline scores</button>
  <button class="btn mode" data-mode="delta">Terra − City score change</button>
  <button class="btn mode" data-mode="avg">Terra average by area</button>
  <button class="btn mode" data-mode="change">Average change by area</button>
 </div>
 <div class="group" id="res-group" style="display:none"><h2>Grid resolution</h2>
  <button class="btn res" data-res="coarse">Coarse (~1.1 km)</button>
  <button class="btn res active" data-res="medium">Medium (~550 m)</button>
  <button class="btn res" data-res="fine">Fine (~275 m)</button>
  <button class="btn res" data-res="ultra">Ultra (~140 m)</button>
 </div>
 <div class="group" id="legend"><h2>Legend / filter</h2></div>
 <div class="group"><button class="btn" id="clear">Clear filter</button></div>
 <div class="group"><h2>Campaign summary</h2>
  <div class="stat"><span>Properties scored</span><b>66,701</b></div>
  <div class="stat"><span>Mapped coordinates</span><b>66,685</b></div>
  <div class="stat"><span>Terra score 4–5</span><b>16,798</b></div>
  <div class="stat"><span>Differences ≥2</span><b>7,051</b></div>
  <div class="note">A higher Terra score means worse condition. Change views include only parcels with a City baseline score of 1–5.</div>
  <div class="note">Basemap: Esri World Light Gray Canvas. No API key is used or required; an internet connection is required to load the background map.</div>
 </div>
</aside></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const P=__POINTS__, H=__HOTSPOTS__;
const TCOUNTS=__TCOUNTS__, BCOUNTS=__BCOUNTS__, DCOUNTS=__DCOUNTS__;
const COLORS={0:'#7c8790',1:'#187a5a',2:'#61ad91',3:'#d1aa24',4:'#df8428',5:'#b53c49'};
const LABELS={0:'Unrated',1:'Excellent',2:'Good',3:'Average',4:'Starting Distress',5:'Distressed'};
const DELTA_COLORS={'-4':'#075a3a','-3':'#087d50','-2':'#2a9b67','-1':'#7bc7a7','0':'#8f969b','1':'#e4a35b','2':'#d86b3d','3':'#bd3f35','4':'#8f202d'};
const map=L.map('map',{preferCanvas:true}).setView([41.587,-93.625],12);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',{attribution:'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',maxNativeZoom:16,maxZoom:19}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:16,maxZoom:19}).addTo(map);
const renderer=L.canvas({padding:.5}); let layer=null, mode='terra', active=null, res='medium';
function terraColor(v){return COLORS[v]||'#888'}
function avgColor(v){if(v<1.8)return'#187a5a';if(v<2.3)return'#61ad91';if(v<2.8)return'#d1aa24';if(v<3.4)return'#df8428';return'#b53c49'}
function deltaColor(v){if(v==null)return'#777';return DELTA_COLORS[String(Math.max(-4,Math.min(4,Math.round(v))))]}
function clearLayer(){if(layer)map.removeLayer(layer);layer=null}
function tooltip(p){const d=p[4]==null?'n/a':(p[4]>0?'+':'')+p[4];return `<b>${p[5]}</b><br>Terra: ${p[2]} (${LABELS[p[2]]})<br>City: ${p[3]} (${LABELS[p[3]]||'Unrated'})<br>Change: ${d}`}
function dots(){clearLayer();const g=L.layerGroup();for(const p of P){let value=mode==='terra'?p[2]:mode==='baseline'?p[3]:p[4];if(mode==='delta'&&value==null)continue;if(active!==null&&value!==active)continue;const color=mode==='delta'?deltaColor(value):terraColor(value);L.circleMarker([p[0],p[1]],{radius:2.4,fillColor:color,color,weight:0,fillOpacity:.78,renderer}).bindTooltip(tooltip(p),{sticky:true}).addTo(g)}layer=g.addTo(map)}
function grid(){clearLayer();const data=H[res],g=L.layerGroup(),half=data.size/2;for(const c of data.cells){const v=mode==='avg'?c.terra_mean:c.mean_score_delta;if(v==null)continue;const color=mode==='avg'?avgColor(v):deltaColor(v);const r=L.rectangle([[c.latitude-half,c.longitude-half],[c.latitude+half,c.longitude+half]],{weight:0,fillColor:color,fillOpacity:.66});r.bindTooltip(`<b>${c.property_count.toLocaleString()} properties</b><br>Terra mean: ${c.terra_mean.toFixed(2)}<br>City mean: ${c.baseline_mean==null?'n/a':c.baseline_mean.toFixed(2)}<br>Mean change: ${c.mean_score_delta==null?'n/a':(c.mean_score_delta>0?'+':'')+c.mean_score_delta.toFixed(2)}<br>Terra score 4–5: ${c.terra_poor_or_distressed_pct.toFixed(1)}%`,{sticky:true});r.addTo(g)}layer=g.addTo(map)}
function render(){document.querySelectorAll('.mode').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));document.getElementById('res-group').style.display=(mode==='avg'||mode==='change')?'block':'none';buildLegend();if(mode==='terra'||mode==='baseline'||mode==='delta')dots();else grid();document.getElementById('banner').textContent=bannerText()}
function bannerText(){if(mode==='terra')return'Terra scores · '+(active==null?'all':`score ${active}`);if(mode==='baseline')return'City baseline · '+(active==null?'all':`score ${active}`);if(mode==='delta')return'Terra − City change · '+(active==null?'all rated':`${active>0?'+':''}${active}`);if(mode==='avg')return`Terra area averages · ${res}`;return`Average score change · ${res}`}
function buildLegend(){const el=document.getElementById('legend');el.innerHTML='<h2>Legend / filter</h2>';if(mode==='terra'||mode==='baseline'){const counts=mode==='terra'?TCOUNTS:BCOUNTS;for(const v of (mode==='terra'?[1,2,3,4,5]:[0,1,2,3,4,5]))addLegend(el,v,COLORS[v],`${v} — ${LABELS[v]}`,counts[String(v)]||0)}else if(mode==='delta'){for(const v of [-4,-3,-2,-1,0,1,2,3,4])addLegend(el,v,DELTA_COLORS[String(v)],v===0?'No change':`${v>0?'+':''}${v} (${v>0?'Terra worse':'Terra better'})`,DCOUNTS[String(v)]||0)}else{const note=document.createElement('div');note.className='note';note.textContent=mode==='avg'?'Green = better average; red = worse average':'Green = Terra lower; red = Terra higher';el.appendChild(note)}}
function addLegend(el,v,color,label,count){const r=document.createElement('div');r.className='row'+(active===v?' active':'');r.innerHTML=`<span class="swatch" style="background:${color}"></span><span class="label">${label}</span><span class="count">${count.toLocaleString()}</span>`;r.onclick=()=>{active=active===v?null:v;render()};el.appendChild(r)}
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;active=null;render()});
document.querySelectorAll('.res').forEach(b=>b.onclick=()=>{res=b.dataset.res;document.querySelectorAll('.res').forEach(x=>x.classList.toggle('active',x===b));render()});
document.getElementById('clear').onclick=()=>{active=null;render()};render();
</script></body></html>"""

html = html.replace("__POINTS__", points_json)
html = html.replace("__HOTSPOTS__", hotspots_json)
html = html.replace("__TCOUNTS__", json.dumps({str(k): v for k, v in terra_counts.items()}))
html = html.replace("__BCOUNTS__", json.dumps({str(k): v for k, v in baseline_counts.items()}))
html = html.replace("__DCOUNTS__", json.dumps({str(k): v for k, v in delta_counts.items()}))
OUTPUT.write_text(html, encoding="utf-8")
print(f"points={len(points)} output={OUTPUT} size={OUTPUT.stat().st_size/1024/1024:.2f} MB")
