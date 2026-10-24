import os
import json
import random
import pandas as pd
from src.data_loader import DataLoader
from src.providers import get_provider
from src.config import Config

def load_quality_prompt():
    prompt_path = os.path.join(Config.PROMPTS_DIR, "quality_check.txt")
    with open(prompt_path, "r") as f:
        return f.read()

def run_quality_check(image_paths, provider_name="local", sample_size=None):
    """
    Run quality check on a list of images.
    
    Args:
        image_paths: List of image file paths
        provider_name: Which VLM provider to use ("local", "openai", "google", "together")
        sample_size: If specified, randomly sample this many images
        
    Returns:
        DataFrame with quality check results
    """
    if sample_size and len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
    
    provider = get_provider(provider_name)
    prompt = load_quality_prompt()
    
    results = []
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            results.append({
                "image_path": img_path,
                "error": "File not found"
            })
            continue
            
        print(f"Processing: {os.path.basename(img_path)}")
        
        try:
            response = provider.analyze(img_path, prompt)
            
            # Try to parse JSON from response
            if response:
                # Clean response (remove markdown code blocks if present)
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                try:
                    parsed = json.loads(clean_response)
                    parsed["image_path"] = img_path
                    results.append(parsed)
                except json.JSONDecodeError:
                    results.append({
                        "image_path": img_path,
                        "raw_response": response,
                        "error": "Failed to parse JSON"
                    })
            else:
                results.append({
                    "image_path": img_path,
                    "error": "Empty response"
                })
        except Exception as e:
            results.append({
                "image_path": img_path,
                "error": str(e)
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load annotations
    loader = DataLoader()
    df = loader.load_annotations()
    
    # Get all image paths
    all_images = df['image_path'].tolist()
    
    # Filter to only existing images
    existing_images = [img for img in all_images if os.path.exists(img)]
    
    print(f"Found {len(existing_images)} existing images out of {len(all_images)} total")
    
    # Run pilot test on 10 random images
    print("\n=== Running Pilot Test (10 images) ===")
    pilot_results = run_quality_check(existing_images, sample_size=10)
    
    # Save results
    output_path = os.path.join(Config.OUTPUTS_DIR, "quality_check_pilot.csv")
    os.makedirs(Config.OUTPUTS_DIR, exist_ok=True)
    pilot_results.to_csv(output_path, index=False)
    print(f"\n✅ Pilot results saved to: {output_path}")
    
    # Show summary
    print("\n=== Pilot Summary ===")
    print(pilot_results.describe())
    print("\n=== Sample Results ===")
    print(pilot_results.head())

