import base64
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

PROVIDERS = [
    {"key": "openai", "title": "Zero-Shot Report – OpenAI", "csv": Path("data/outputs/zeroshot_scores_openai.csv")},
    {"key": "google", "title": "Zero-Shot Report – Google", "csv": Path("data/outputs/zeroshot_scores_google.csv")},
    {"key": "together", "title": "Zero-Shot Report – Together AI", "csv": Path("data/outputs/zeroshot_scores_together.csv")},
    {"key": "local", "title": "Zero-Shot Report – Local Model", "csv": Path("data/outputs/zeroshot_scores_local.csv")},
]

OUTPUT_DIR = Path("reports/zero_shot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_examples(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "raw_response" not in df.columns:
        raise ValueError(f"CSV {csv_path} missing 'raw_response' column")
    return df

def parse_response(raw: str) -> dict:
    data = raw.strip()
    if "```json" in data:
        data = data.split("```json")[1].split("```")[0].strip()
    elif "```" in data:
        data = data.split("```")[1].split("```")[0].strip()
    return json.loads(data)

def build_report_html(title: str, rows: list[dict]) -> str:
    cards_html = []
    for row in rows:
        parsed = row["parsed"]
        subscores = parsed.get("subscores", {})
        red_flags = parsed.get("red_flags", [])
        minor_issues = parsed.get("minor_issues", [])
        pos_signs = parsed.get("positive_signs", [])
        justification = parsed.get("justification", "")

        subs_html_parts = []
        for label, data in subscores.items():
            pretty = label.replace("_", " ").title()
            val = data.get("score") if isinstance(data, dict) else data
            expl = data.get("explanation", "") if isinstance(data, dict) else ""
            subs_html_parts.append(
                f"""
                <div class='sub-card'>
                    <h4>{pretty}</h4>
                    <div class='sub-value'>{val}/5</div>
                    <p class='sub-expl'>{expl}</p>
                </div>
                """
            )
        subs_html = "".join(subs_html_parts)

        def build_list(items: list[str], cls: str) -> str:
            if not items:
                return ""
            blocks = [f"<div class='issue {cls}'>{txt}</div>" for txt in items]
            return "".join(blocks)

        cards_html.append(
            f"""
            <article class='entry'>
              <div class='entry-image'>
                <img src='data:image/jpeg;base64,{row['image_b64']}' alt='{row['file_name']}' />
              </div>
              <div class='score-row'>
                <div class='score-box'>
                  <span class='label'>Expert Score</span>
                  <strong>{row['expert_score']}</strong>
                </div>
                <div class='score-box'>
                  <span class='label'>Predicted Score</span>
                  <strong>{row.get('predicted_score','')}</strong>
                </div>
              </div>
              <div class='entry-body'>
                <h3 class='entry-title'>{row['file_name']}</h3>
                <p class='justification'>{justification}</p>
                <div class='sub-grid'>
                  {subs_html}
                </div>
                <div class='issue-group'>
                  {build_list(red_flags, 'flag')}
                  {build_list(minor_issues, 'minor')}
                  {build_list(pos_signs, 'positive')}
                </div>
              </div>
            </article>
            """
        )

    cards = "\n".join(cards_html)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:'Inter',sans-serif; background:#f3f6fb; color:#2f3640; }}
    header {{ background:linear-gradient(120deg,#0F2027,#2C5364); color:#fff; padding:40px 60px; }}
    header h1 {{ margin:0; font-size:32px; }}
    header p {{ margin:8px 0 0; opacity:.8; }}
    .container {{ max-width:1200px; margin:40px auto; padding:0 20px 60px; }}
    .entry {{ background:#fff; border-radius:18px; margin-bottom:32px; box-shadow:0 18px 40px rgba(0,0,0,.08); overflow:hidden; }}
    .entry-image img {{ width:100%; height:auto; display:block; }}
    .score-row {{ display:flex; gap:16px; justify-content:center; padding:18px; flex-wrap:wrap; background:#f8fafc; }}
    .score-box {{ flex:1; min-width:140px; background:#fff; border-radius:10px; padding:12px 18px; text-align:center; box-shadow:0 8px 18px rgba(0,0,0,.06); }}
    .score-box .label {{ display:block; font-size:12px; letter-spacing:.08em; color:#6c7a91; }}
    .score-box strong {{ font-size:26px; }}
    .entry-body {{ padding:32px; }}
    .entry-title {{ margin:0 0 12px; font-size:20px; color:#1f2a44; }}
    .justification {{ margin-bottom:24px; color:#4b556a; line-height:1.5; }}
    .sub-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin-bottom:20px; }}
    .sub-card {{ background:#f8fbff; border-radius:12px; padding:14px; border:1px solid #e1e7f5; }}
    .sub-card h4 {{ margin:0; font-size:14px; color:#344666; }}
    .sub-card .sub-value {{ font-size:24px; margin:6px 0; font-weight:600; }}
    .sub-card .sub-expl {{ margin:0; color:#7a859f; font-size:13px; }}
    .issue-group {{ display:grid; gap:10px; }}
    .issue {{ padding:12px 16px; border-radius:10px; font-size:14px; }}
    .issue.flag {{ background:#ffe6e8; border-left:4px solid #e74c3c; }}
    .issue.minor {{ background:#fff4e2; border-left:4px solid #f39c12; }}
    .issue.positive {{ background:#e8f9f3; border-left:4px solid #1abc9c; }}
    footer {{ text-align:center; color:#7b859a; font-size:13px; padding:20px 0 40px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Generated on {timestamp} – {len(rows)} examples</p>
  </header>
  <div class='container'>
    {cards}
  </div>
  <footer>Zero-shot model outputs rendered for review.</footer>
</body>
</html>
"""

def main():
    print("=" * 70)
    print("GENERATING ZERO-SHOT REPORTS - DETAILED LOG")
    print("=" * 70)
    
    for spec in PROVIDERS:
        print(f"\n{'=' * 70}")
        print(f"Processing: {spec['key'].upper()}")
        print(f"CSV: {spec['csv']}")
        print(f"{'=' * 70}")
        
        try:
            df = load_examples(spec["csv"])
            print(f"✅ Loaded CSV: {len(df)} total rows")
        except Exception as exc:
            print(f"❌ Failed to load CSV: {exc}")
            continue

        # Count valid data
        total_rows = len(df)
        missing_raw = len(df[df['raw_response'].isna() | (df['raw_response'].astype(str).str.strip() == '')])
        missing_img = len(df[~df['image_path'].apply(lambda p: isinstance(p, str) and Path(p).exists())])
        
        print(f"📊 Data quality:")
        print(f"   - Total rows: {total_rows}")
        print(f"   - Missing raw_response: {missing_raw}")
        print(f"   - Missing image files: {missing_img}")
        
        rows = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            # Check raw response
            raw = str(row.get("raw_response", ""))
            if not raw or not raw.strip():
                skipped_count += 1
                continue
            
            # Check image exists
            img_path = Path(row["image_path"])
            if not img_path.exists():
                print(f"   ⚠️  Row {idx}: Image missing: {img_path}")
                skipped_count += 1
                continue
            
            # Parse JSON
            try:
                parsed = parse_response(raw)
            except Exception as exc:
                print(f"   ⚠️  Row {idx}: JSON parse failed for {row.get('file_name')}: {str(exc)[:100]}")
                skipped_count += 1
                continue

            # Encode image
            try:
                with img_path.open("rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as exc:
                print(f"   ⚠️  Row {idx}: Failed to read image {img_path}: {exc}")
                skipped_count += 1
                continue

            rows.append({
                "file_name": row.get("file_name", ""),
                "image_b64": b64,
                "expert_score": row.get("expert_score", ""),
                "predicted_score": row.get("predicted_score", ""),
                "parsed": parsed,
            })

        print(f"\n📝 Processing complete:")
        print(f"   - Valid cards: {len(rows)}")
        print(f"   - Skipped: {skipped_count}")
        print(f"   - Total: {total_rows}")
        
        if not rows:
            print(f"❌ No valid rows to render for {spec['key']}")
            continue

        html = build_report_html(spec["title"], rows)
        out_path = OUTPUT_DIR / f"{spec['key']}_report.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"✅ Wrote {out_path} ({len(rows)} cards)")

    print(f"\n{'=' * 70}")
    print("DONE! All reports generated.")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
