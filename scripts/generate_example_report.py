import base64
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    csv_path = Path("data/outputs/zeroshot_scores_openai.csv")
    output_html = Path("reports/zeroshot_example_report.html")
    output_html.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    print(f"Loading data from {csv_path}…")
    df = pd.read_csv(csv_path)
    df["raw_response"] = df["raw_response"].fillna("")
    df = df[df["raw_response"].str.strip().astype(bool)]
    df = df[df["image_path"].apply(lambda p: Path(p).exists() if isinstance(p, str) else False)]

    if df.empty:
        raise RuntimeError("No valid example with image and raw response.")

    def delta(row):
        try:
            return abs(float(row.get("expert_score", 0)) - float(row.get("predicted_score", 0)))
        except Exception:
            return 999

    df = df.assign(delta=df.apply(lambda r: delta(r), axis=1)).sort_values(["delta"]).reset_index(drop=True)
    row = df.iloc[0]

    image_path = Path(row["image_path"])
    with image_path.open("rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    raw_response = str(row["raw_response"]).strip()
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0].strip()

    parsed = json.loads(raw_response)

    subscores = parsed.get("subscores", {})
    red_flags = parsed.get("red_flags", [])
    minor_issues = parsed.get("minor_issues", [])
    pos_signs = parsed.get("positive_signs", [])
    score = parsed.get("score")
    category = parsed.get("category", "Unknown")
    justification = parsed.get("justification", "")

    subcards = []
    for key, data in subscores.items():
        label = key.replace("_", " ").title()
        value = data.get("score") if isinstance(data, dict) else data
        expl = data.get("explanation", "") if isinstance(data, dict) else ""
        subcards.append((label, value, expl))

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <title>Zero-Shot Property Assessment – {row['file_name']}</title>
  <style>
    body {{ margin:0; font-family:'Inter','Segoe UI',sans-serif; background:#f2f5fb; color:#2f3640; }}
    .container {{ max-width:1150px; margin:40px auto; background:#fff; border-radius:18px; box-shadow:0 20px 45px rgba(0,0,0,.08); overflow:hidden; }}
    header {{ padding:36px 44px; background:linear-gradient(135deg,#0f2027,#2c5364); color:#fff; }}
    header h1 {{ margin:0; font-size:30px; letter-spacing:.4px; }}
    header .meta {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:16px; }}
    header .meta span {{ background:rgba(255,255,255,.15); padding:8px 14px; border-radius:999px; font-size:14px; }}
    main {{ padding:40px 45px 50px; }}
    .image-panel {{ text-align:center; margin-bottom:32px; }}
    .image-panel img {{ width:100%; border-radius:16px; box-shadow:0 15px 35px rgba(0,0,0,.12); max-height:520px; object-fit:cover; }}
    .score-row {{ display:flex; justify-content:center; gap:24px; margin:22px 0 10px; flex-wrap:wrap; }}
    .score-box {{ background:#fff; border-radius:14px; padding:18px 30px; box-shadow:0 12px 30px rgba(0,0,0,.08); min-width:230px; }}
    .score-box .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:#8693a6; }}
    .score-box .value {{ font-size:36px; font-weight:600; margin-top:6px; }}
    .section {{ margin-bottom:34px; }}
    .section h2 {{ font-size:22px; margin-bottom:12px; color:#2c3e50; }}
    .summary {{ display:flex; gap:20px; flex-wrap:wrap; }}
    .summary-card {{ flex:1; min-width:260px; background:#f8fbff; border-radius:14px; padding:24px; border:1px solid #e4ebf5; box-shadow:0 10px 24px rgba(0,0,0,.06); }}
    .summary-card h3 {{ margin:0 0 10px; font-size:15px; letter-spacing:.1em; color:#7f8fa6; text-transform:uppercase; }}
    .sub-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; }}
    .sub-card {{ background:#fff; border-radius:12px; padding:18px; border:1px solid #e6ecf5; box-shadow:0 8px 18px rgba(0,0,0,.05); }}
    .sub-card h4 {{ margin:0; font-size:16px; color:#34495e; }}
    .sub-card .val {{ font-size:28px; font-weight:600; margin:8px 0; }}
    .issue {{ background:#fff5e6; border-left:4px solid #f39c12; padding:12px 16px; border-radius:10px; margin-bottom:12px; }}
    .issue.flag {{ background:#ffe8ea; border-color:#e74c3c; }}
    .issue.positive {{ background:#e8f9f2; border-color:#1abc9c; }}
  </style>
</head>
<body>
  <div class='container'>
    <header>
      <h1>Zero-Shot Property Assessment</h1>
      <div class='meta'>
        <span>{row['file_name']}</span>
        <span>Provider: {row['provider']} ({row['model']})</span>
        <span>Expert Score: {row['expert_score']}</span>
        <span>Predicted: {score}</span>
      </div>
    </header>
    <main>
      <section class='image-panel'>
        <img src='data:image/jpeg;base64,{img_base64}' alt='Property photo' />
        <div class='score-row'>
          <div class='score-box'>
            <span class='label'>Expert Score</span>
            <span class='value'>{row['expert_score']}</span>
          </div>
          <div class='score-box'>
            <span class='label'>Predicted Score</span>
            <span class='value'>{score}</span>
          </div>
        </div>
      </section>

      <section class='section'>
        <h2>Overall Summary</h2>
        <div class='summary'>
          <div class='summary-card'>
            <h3>Final Score</h3>
            <p style='font-size: 40px; margin:0;'>{score}/5</p>
            <p style='margin-top:6px;'>Category: {category}</p>
          </div>
          <div class='summary-card'>
            <h3>Justification</h3>
            <p>{justification}</p>
          </div>
        </div>
      </section>

      <section class='section'>
        <h2>Subcategory Ratings</h2>
        <div class='sub-grid'>
"""

    for label, value, expl in subcards:
        html += f"""
          <div class='sub-card'>
            <h4>{label}</h4>
            <div class='val'>{value}/5</div>
            <p style='color:#7b8aab;'>{expl}</p>
          </div>
"""

    html += """
        </div>
      </section>

      <section class='section'>
        <h2>Issues & Highlights</h2>
"""

    if red_flags:
        html += "<h3>Red Flags</h3>"
        for item in red_flags:
            html += f"<div class='issue flag'>{item}</div>"
    if minor_issues:
        html += "<h3>Minor Issues</h3>"
        for item in minor_issues:
            html += f"<div class='issue'>{item}</div>"
    if pos_signs:
        html += "<h3>Positive Signs</h3>"
        for item in pos_signs:
            html += f"<div class='issue positive'>{item}</div>"

    html += """
      </section>
    </main>
  </div>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")
    print(f"Report written to {output_html}")


if __name__ == "__main__":
    main()
