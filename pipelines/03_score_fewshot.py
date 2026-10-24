import os
import json
import pandas as pd
from src.data_loader import DataLoader
from src.providers import get_provider
from src.config import Config
from src.data_loader import DataLoader

def select_gold_standard_examples(df_annotations, examples_per_score=1):
    """
    Select representative images for each score (1-5) to use as examples.
    
    Args:
        df_annotations: DataFrame with annotations and expert scores
        examples_per_score: How many examples to select per score
        
    Returns:
        Dictionary mapping score -> list of image paths
    """
    gold_standards = {}
    
    for score in [1, 2, 3, 4, 5]:
        score_images = df_annotations[
            (df_annotations['expert_score'] == str(score)) & 
            (df_annotations['image_path'].apply(os.path.exists))
        ]
        
        if len(score_images) > 0:
            # Select first N examples (or random sample)
            selected = score_images.head(examples_per_score)
            gold_standards[score] = selected['image_path'].tolist()
        else:
            gold_standards[score] = []
    
    return gold_standards

def build_fewshot_prompt(base_prompt, gold_standards, provider):
    """
    Build a few-shot prompt by adding example images and their scores (text version for fallback).
    
    Args:
        base_prompt: The zero-shot scoring prompt
        gold_standards: Dictionary mapping score -> list of image paths
        provider: VLM provider instance (for encoding images if needed)
        
    Returns:
        Enhanced prompt string (text-based fallback)
    """
    fewshot_text = base_prompt + "\n\n## Examples:\n\n"
    
    for score in [1, 2, 3, 4, 5]:
        if score in gold_standards and gold_standards[score]:
            example_path = gold_standards[score][0]
            fewshot_text += f"Example Score {score}: See image '{os.path.basename(example_path)}'\n"
    
    fewshot_text += "\nNow analyze the target image using the same criteria as the examples above."
    
    return fewshot_text

def analyze_with_fewshot_examples(provider, test_image_path, base_prompt, gold_standards_dict, provider_name="openai"):
    """
    Analyze image with few-shot examples - sends actual example images to the API.
    
    For OpenAI: Uses conversation format with example image-score pairs
    For other providers: Falls back to text-based few-shot prompt
    
    Args:
        provider: VLM provider instance
        test_image_path: Path to the test image to score
        base_prompt: Base scoring prompt
        gold_standards_dict: Dictionary mapping score -> list of image paths
        provider_name: Name of the provider
        
    Returns:
        API response string
    """
    # Try to use image-based few-shot for OpenAI
    if provider_name == "openai" and hasattr(provider, 'client'):
        try:
            messages = []
            
            # Add example pairs (user shows image, assistant responds with score)
            for score in [1, 2, 3, 4, 5]:
                if score in gold_standards_dict and gold_standards_dict[score]:
                    example_image_path = gold_standards_dict[score][0]  # Use first example per score
                    
                    # Encode example image
                    base64_example = DataLoader.encode_image(example_image_path)
                    if base64_example:
                        # User message: show example image
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Here is an example property image (filename: {os.path.basename(example_image_path)})."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_example}"
                                    }
                                }
                            ]
                        })
                        
                        # Assistant message: provide the score
                        messages.append({
                            "role": "assistant",
                            "content": f"This property received a score of {score}."
                        })
            
            # Final user message: test image with scoring prompt
            base64_test = DataLoader.encode_image(test_image_path)
            if base64_test:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": base_prompt + "\n\nNow analyze this image and assign a score following the same criteria as the examples above."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_test}"
                            }
                        }
                    ]
                })
                
                # Call API with conversation history
                response = provider.client.chat.completions.create(
                    model=provider.model_name,
                    messages=messages,
                    max_tokens=8000
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Image-based few-shot failed, falling back to text: {e}")
            # Fall back to text-based few-shot
            pass
    
    # Fallback: use text-based few-shot prompt (for non-OpenAI or if image-based fails)
    fewshot_prompt = build_fewshot_prompt(base_prompt, gold_standards_dict, provider)
    return provider.analyze(test_image_path, fewshot_prompt)

def score_with_fewshot(image_paths, provider_name="openai", examples_per_score=1):
    """
    Score images using few-shot learning with gold standard examples.
    
    Args:
        image_paths: List of target image paths to score
        provider_name: VLM provider to use
        examples_per_score: Number of examples per score category
        
    Returns:
        DataFrame with scoring results
    """
    # Load annotations to get gold standards
    loader = DataLoader()
    df_annotations = loader.load_annotations()
    df_scored = df_annotations[df_annotations['expert_score'].notna()]
    
    # Select gold standard examples
    gold_standards = select_gold_standard_examples(df_scored, examples_per_score)
    
    # Load base prompt
    prompt_path = os.path.join(Config.PROMPTS_DIR, "prompt_zero_shot.txt")
    with open(prompt_path, "r") as f:
        base_prompt = f.read()
    
    provider = get_provider(provider_name)
    
    # Build few-shot prompt
    fewshot_prompt = build_fewshot_prompt(base_prompt, gold_standards, provider)
    
    results = []
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            results.append({
                "image_path": img_path,
                "provider": provider_name,
                "error": "File not found"
            })
            continue
        
        try:
            # Use image-based few-shot (sends actual example images to API)
            response = analyze_with_fewshot_examples(
                provider, img_path, base_prompt, gold_standards, provider_name
            )
            
            if response:
                # Parse JSON response
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                try:
                    parsed = json.loads(clean_response)
                    parsed["image_path"] = img_path
                    parsed["provider"] = provider_name
                    parsed["method"] = "fewshot"
                    results.append(parsed)
                except json.JSONDecodeError:
                    results.append({
                        "image_path": img_path,
                        "provider": provider_name,
                        "error": "JSON parse error",
                        "raw_response": response
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
    
    # Test few-shot scoring
    print("\n=== Running Few-Shot Scoring ===")
    results = score_with_fewshot(scored_images[:10], provider_name="openai")  # Test on 10 first
    
    # Save results
    output_path = os.path.join(Config.OUTPUTS_DIR, "fewshot_scores.csv")
    os.makedirs(Config.OUTPUTS_DIR, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

