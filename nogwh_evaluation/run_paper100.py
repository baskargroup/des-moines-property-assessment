"""Run OUR ensemble (GPT-5.2 + Gemini-3-Flash + GPT-4.1, P6 prompt, our 5 examples)
on the paper's 100-image set -> predictions_paper100.jsonl. Resume-safe.
This makes the paper's set directly comparable to our NOGWH run and to the paper's 93.3%."""
import os, sys, csv, json, time, base64, asyncio
from pathlib import Path
import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
sys.path.insert(0, r".")
from prompt import get_prompt_and_examples

MANIFEST=r"paper100_manifest.csv"
OUT=r"predictions_paper100.jsonl"
prompt_text, examples = get_prompt_and_examples()
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
OPENAI_KEY=os.getenv("OPENAI_API_KEY"); GOOGLE_KEY=os.getenv("GOOGLE_API_KEY")

def encode(p):
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
EX_OAI,EX_GEM=[],[]
for path,score,desc in examples:
    dt=f"\n--- EXAMPLE (Score {score}) ---\n{desc}"
    EX_OAI.append(("text",dt)); EX_GEM.append({"text":dt})
    mime="image/png" if path.endswith(".png") else "image/jpeg"; b=encode(path)
    EX_OAI.append(("image",f"data:{mime};base64,{b}")); EX_GEM.append({"inline_data":{"mime_type":mime,"data":b}})
def openai_content(t):
    c=[{"type":"text","text":prompt_text}]
    for k,v in EX_OAI: c.append({"type":"text","text":v} if k=="text" else {"type":"image_url","image_url":{"url":v}})
    c.append({"type":"text","text":"\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
    c.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{encode(t)}"}})
    return [{"role":"user","content":c}]
def gemini_parts(t):
    p=[{"text":prompt_text}]+EX_GEM
    p.append({"text":"\n--- NOW SCORE THIS TARGET IMAGE ---\nOutput ONLY the JSON response."})
    p.append({"inline_data":{"mime_type":"image/jpeg","data":encode(t)}}); return p
client=AsyncOpenAI(api_key=OPENAI_KEY)
async def call_openai(model,t):
    reasoning=("gpt-5" in model and "chat-latest" not in model) or model.startswith(("o1","o3","o4"))
    p={"model":model,"messages":openai_content(t)}
    if reasoning:
        p["max_completion_tokens"]=8000; p["temperature"]=1.0
        if any(v in model for v in ["5.2","5.4","5.5","o3","o4"]): p["reasoning_effort"]="low"
    else: p["max_tokens"]=8000; p["temperature"]=0.0
    for a in range(3):
        try:
            resp=await client.chat.completions.create(**p); return resp.choices[0].message.content,None
        except Exception as e:
            err=str(e)
            if "rate" in err.lower() or "429" in err: await asyncio.sleep([3,10,25][a]); continue
            return None,err[:120]
    return None,"retries_exhausted"
async def call_gemini(session,model,t):
    payload={"contents":[{"role":"user","parts":gemini_parts(t)}],
        "generationConfig":{"temperature":0.0,"maxOutputTokens":8000},
        "safetySettings":[{"category":c,"threshold":"BLOCK_NONE"} for c in
            ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_KEY}"
    for a in range(5):
        try:
            async with session.post(url,json=payload,timeout=aiohttp.ClientTimeout(total=180)) as r:
                if r.status==200:
                    data=await r.json(); cands=data.get("candidates",[])
                    if cands:
                        txt=" ".join(x.get("text","") for x in cands[0].get("content",{}).get("parts",[]) if "text" in x).strip()
                        if txt: return txt,None
                    return None,"no_candidates"
                elif r.status in (429,503): await asyncio.sleep([3,10,30,60,90][a]); continue
                else: return None,f"http_{r.status}"
        except Exception: await asyncio.sleep([3,10,30,60,90][a])
    return None,"retries_exhausted"
def parse(raw):
    if not raw: return None
    s=raw.strip()
    if "```json" in s: s=s.split("```json",1)[1].split("```",1)[0].strip()
    elif "```" in s: s=s.split("```",1)[1].split("```",1)[0].strip()
    if not s.startswith("{"):
        i=s.find("{"); j=s.rfind("}")
        if i!=-1 and j>i: s=s[i:j+1]
    try: return json.loads(s)
    except: return None
ENSEMBLE={"gpt52":"gpt-5.2-2025-12-11","gem3fp":"gemini-3-flash-preview","gpt41":"gpt-4.1"}
_lock=asyncio.Lock()
async def write_record(rec):
    async with _lock:
        with open(OUT,"a",encoding="utf-8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
def load_done():
    if not os.path.exists(OUT): return set()
    s=set()
    for line in open(OUT,encoding="utf-8"):
        try: s.add(json.loads(line)["image_id"])
        except: pass
    return s
async def process(row,session,sem,counter,total):
    async with sem:
        t=row["image_path"]
        tasks={"gpt52":asyncio.create_task(call_openai(ENSEMBLE["gpt52"],t)),
               "gem3fp":asyncio.create_task(call_gemini(session,ENSEMBLE["gem3fp"],t)),
               "gpt41":asyncio.create_task(call_openai(ENSEMBLE["gpt41"],t))}
        preds={}
        for k,task in tasks.items():
            raw,err=await task; pr=parse(raw)
            preds[k]={"model":ENSEMBLE[k],"raw_response":raw,"parsed":pr,
                "predicted_score":pr.get("score") if pr else None,
                "predicted_category":pr.get("category") if pr else None,"error":err}
        rec={"image_id":row["image_id"],"filename":row["filename"],"image_path":row["image_path"],
             "neighborhood":row.get("neighborhood",""),"source":row.get("source",""),
             "gt_score":int(row["gt_score"]) if row["gt_score"].isdigit() else None,
             "gt_category":row["gt_category"],"human_rater_score":row.get("human_rater_score",""),
             "predictions":preds}
        await write_record(rec)
        counter[0]+=1
        if counter[0]%10==0 or counter[0]==total: print(f"  [{counter[0]}/{total}]",flush=True)
async def main():
    rows=list(csv.DictReader(open(MANIFEST,encoding="utf-8")))
    done=load_done(); pending=[r for r in rows if r["image_id"] not in done]
    print(f"paper100: total={len(rows)} done={len(done)} pending={len(pending)} -> {OUT}",flush=True)
    if not pending: print("nothing to do."); return
    sem=asyncio.Semaphore(6); counter=[0]; t0=time.time()
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*(process(r,s,sem,counter,len(pending)) for r in pending))
    print(f"DONE in {(time.time()-t0)/60:.1f} min -> {OUT}",flush=True)
if __name__=="__main__":
    asyncio.run(main())
