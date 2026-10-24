import os
import json
import pandas as pd
from src.data_loader import DataLoader
from src.providers import get_provider
from src.config import Config

def load_scoring_prompt():
    prompt_path = os.path.join(Config.PROMPTS_DIR, "prompt_zero_shot.txt")
    with open(prompt_path, "r") as f:
        return f.read()

def score_images(image_paths, provider_name="openai", batch_size=10):
    """
    Score property images using zero-shot VLM.
    
    Args:
        image_paths: List of image file paths
        provider_name: Which VLM provider to use
        batch_size: Process in batches (for progress tracking)
        
    Returns:
        DataFrame with scoring results
    """
    provider = get_provider(provider_name)
    prompt = load_scoring_prompt()
    
    results = []
    total = len(image_paths)
    
    for idx, img_path in enumerate(image_paths, 1):
        if not os.path.exists(img_path):
            results.append({
                "image_path": img_path,
                "provider": provider_name,
                "error": "File not found"
            })
            continue
        
        if idx % batch_size == 0:
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
        
        try:
            response = provider.analyze(img_path, prompt)
            
            if response:
                # Try to extract score from response
                # The prompt should return a score, but we need to parse it
                # This is a simple extraction - may need refinement based on actual responses
                score = None
                try:
                    # Try to find score in JSON format
                    if "```json" in response:
                        json_str = response.split("```json")[1].split("```")[0].strip()
                        parsed = json.loads(json_str)
                        score = parsed.get("score") or parsed.get("overall_score")
                    elif "score" in response.lower():
                        # Try to extract number after "score"
                        import re
                        score_match = re.search(r'score[:\s]+(\d)', response, re.IGNORECASE)
                        if score_match:
                            score = int(score_match.group(1))
                except:
                    pass
                
                results.append({
                    "image_path": img_path,
                    "provider": provider_name,
                    "model": provider.model_name,
                    "raw_response": response,
                    "predicted_score": score
                })
            else:
                results.append({
                    "image_path": img_path,
                    "provider": provider_name,
                    "error": "Empty response"
                })
        except Exception as e:
            results.append({
                "image_path": img_path,
                "provider": provider_name,
                "error": str(e)
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load annotations
    loader = DataLoader()
    df = loader.load_annotations()
    
    # Get scored images for testing
    scored_images = [img for img in df[df['expert_score'].notna()]['image_path'].tolist() 
                     if os.path.exists(img)]
    
    print(f"Found {len(scored_images)} scored images")
    
    # Test with OpenAI (you can change provider)
    print("\n=== Running Zero-Shot Scoring (OpenAI) ===")
    results = score_images(scored_images[:10], provider_name="openai")  # Test on 10 first
    
    # Save results
    output_path = os.path.join(Config.OUTPUTS_DIR, "zeroshot_scores_openai.csv")
    os.makedirs(Config.OUTPUTS_DIR, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

