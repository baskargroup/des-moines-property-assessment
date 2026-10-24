#!/usr/bin/env python3
"""
Run predictions with full details (subscores, red_flags, minor_issues)
"""

import asyncio
import base64
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import Config

import google.generativeai as genai

EXAMPLES_DIR = Path("../data/raw/documentation/dsm_criteria_images")

FEWSHOT_EXAMPLES = [
    {"file": "example_3.png", "score": 1, "category": "Excellent"},
    {"file": "example_6.jpg", "score": 2, "category": "Good"},
    {"file": "example_2.jpg", "score": 3, "category": "Average"},
    {"file": "example_1.jpg", "score": 4, "category": "Starting Distress"},
    {"file": "example_4.jpg", "score": 5, "category": "Distressed"},
]

# The prompt text is not distributed with this repository; see data/prompts/README.md.
PROMPT = (Path(__file__).parent.parent / "data" / "prompts" / "run_with_details_prompt.txt").read_text(encoding="utf-8")


def load_image(path):
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(path).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return {"mime_type": mime, "data": data}


async def analyze_image(model, target_path):
    contents = [PROMPT, "\n=== EXAMPLE IMAGES ===\n"]
    
    for ex in FEWSHOT_EXAMPLES:
        img_path = EXAMPLES_DIR / ex["file"]
        contents.append(f"\nExample Score {ex['score']} ({ex['category']}):")
        contents.append(load_image(img_path))
    
    contents.append("\n=== NOW SCORE THIS IMAGE ===\n")
    contents.append(load_image(target_path))
    contents.append("\nOutput ONLY JSON:")
    
    try:
        response = await asyncio.to_thread(model.generate_content, contents)
        text = response.text.strip()
        
        # Parse JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        
        result = json.loads(text)
        return result
    except Exception as e:
        return {"score": None, "error": str(e)}


async def main():
    print("="*70)
    print("RUNNING PREDICTIONS WITH FULL DETAILS")
    print("="*70)
    
    # Load data
    gt = pd.read_csv('../data/datasets/ground_truth_100.csv')
    print(f"\nLoaded {len(gt)} images")
    
    # Init model
    config = Config()
    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Process
    semaphore = asyncio.Semaphore(5)
    results = []
    
    async def process(row):
        async with semaphore:
            r = await analyze_image(model, row['path'])
            
            # Extract subscores
            subs = r.get('subscores', {})
            
            return {
                "filename": row['filename'],
                "ground_truth": row['ground_truth'],
                "predicted_score": r.get('score'),
                "category": r.get('category', ''),
                "reason": r.get('reason', ''),
                # Subscores from model
                "pred_exterior": subs.get('exterior'),
                "pred_porch": subs.get('porch'),
                "pred_landscaping": subs.get('landscaping'),
                "pred_roof": subs.get('roof'),
                "pred_windows": subs.get('windows'),
                "pred_personal": subs.get('personal_touches'),
                # Counts
                "num_red_flags": len(r.get('red_flags', [])),
                "num_minor_issues": len(r.get('minor_issues', [])),
                "num_positive_signs": len(r.get('positive_signs', [])),
                # Raw lists
                "red_flags": str(r.get('red_flags', [])),
                "minor_issues": str(r.get('minor_issues', [])),
                "positive_signs": str(r.get('positive_signs', [])),
            }
    
    start = time.time()
    tasks = [process(row) for _, row in gt.iterrows()]
    
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 10 == 0:
            print(f"  Progress: {done}/100")
    
    print(f"\n✓ Completed in {(time.time()-start)/60:.1f} minutes")
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv('predictions_with_details.csv', index=False)
    print(f"✓ Saved predictions_with_details.csv")
    
    # Quick stats
    valid = df[df['predicted_score'].notna()]
    print(f"\n{'='*70}")
    print("QUICK STATS")
    print(f"{'='*70}")
    print(f"Valid predictions: {len(valid)}")
    print(f"\nRed flags distribution:")
    print(valid['num_red_flags'].value_counts().sort_index())
    print(f"\nMinor issues distribution:")
    print(valid['num_minor_issues'].value_counts().sort_index())


if __name__ == "__main__":
    asyncio.run(main())
