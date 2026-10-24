#!/usr/bin/env python3
"""
Experiment Runner - Config-based experiment execution

Usage:
    python experiments/run_experiment.py experiments/config_zeroshot_test.json
    
Change the config file to run different experiments with different prompts/models.
"""

import sys
import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.providers import get_provider
from src.providers.openai import OpenAIVLM
from src.providers.google import GoogleVLM
from src.providers.together import TogetherVLM
from src.providers.local import LocalVLM


def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_ground_truth(gt_file):
    """Load ground truth CSV into a dict keyed by filename."""
    gt_dict = {}
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_dict[row['filename']] = {
                'id': row['id'],
                'ground_truth': int(row['ground_truth']),
                'neighborhood': row.get('neighborhood', ''),
            }
    return gt_dict


def load_prompt(prompt_file):
    """Load prompt text from file."""
    with open(prompt_file, 'r') as f:
        return f.read()


def get_images(images_dir, ground_truth):
    """Get list of image paths that exist and have ground truth."""
    images = []
    for filename, gt_data in ground_truth.items():
        img_path = os.path.join(images_dir, filename)
        if os.path.exists(img_path):
            images.append({
                'path': img_path,
                'filename': filename,
                'id': gt_data['id'],
                'ground_truth': gt_data['ground_truth']
            })
    return images


def create_provider(model_config):
    """Create a provider instance based on config."""
    provider_type = model_config['provider']
    model_id = model_config['model_id']
    
    if provider_type == 'openai':
        return OpenAIVLM(model_name=model_id)
    elif provider_type == 'google':
        return GoogleVLM(model_name=model_id)
    elif provider_type == 'together':
        return TogetherVLM(model_name=model_id)
    elif provider_type == 'local':
        return LocalVLM(model_name=model_id)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


def process_single_image(provider, image_info, prompt, max_retries=3, retry_delay=5):
    """Process a single image and return result with timing."""
    result = {
        'image_id': image_info['id'],
        'filename': image_info['filename'],
        'ground_truth': image_info['ground_truth'],
        'predicted_score': None,
        'success': False,
        'error': None,
        'start_time': None,
        'end_time': None,
        'duration_sec': None,
        'raw_response': None
    }
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            result['start_time'] = datetime.now().isoformat()
            
            response = provider.analyze(image_info['path'], prompt)
            
            end_time = time.time()
            result['end_time'] = datetime.now().isoformat()
            result['duration_sec'] = round(end_time - start_time, 2)
            
            if response:
                result['raw_response'] = response
                result['success'] = True
                
                # Try to extract score from JSON response
                try:
                    # Clean response (remove markdown code blocks if present)
                    clean_response = response.strip()
                    if clean_response.startswith('```'):
                        clean_response = clean_response.split('\n', 1)[1]
                    if clean_response.endswith('```'):
                        clean_response = clean_response.rsplit('```', 1)[0]
                    clean_response = clean_response.strip()
                    
                    response_json = json.loads(clean_response)
                    result['predicted_score'] = response_json.get('score')
                except (json.JSONDecodeError, KeyError) as e:
                    result['error'] = f"JSON parse error: {str(e)[:50]}"
                
                return result
            else:
                result['error'] = "Empty response"
                
        except Exception as e:
            result['error'] = str(e)[:100]
            result['end_time'] = datetime.now().isoformat()
            result['duration_sec'] = round(time.time() - start_time, 2) if 'start_time' in locals() else None
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    return result


def run_model_experiment(model_config, images, prompt, output_dir, settings):
    """Run experiment for a single model with parallel processing."""
    model_name = model_config['name']
    batch_size = model_config.get('batch_size', 5)
    max_retries = settings.get('max_retries', 3)
    retry_delay = settings.get('retry_delay_sec', 5)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_config['model_id']})")
    print(f"Provider: {model_config['provider']}, Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Create provider
    try:
        provider = create_provider(model_config)
    except Exception as e:
        print(f"  ERROR: Failed to create provider: {e}")
        return None
    
    results = []
    total = len(images)
    completed = 0
    start_total = time.time()
    
    # Process in parallel batches
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(
                process_single_image, 
                provider, 
                img, 
                prompt, 
                max_retries, 
                retry_delay
            ): img for img in images
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Progress update
            status = "✓" if result['success'] else "✗"
            pred = result['predicted_score'] if result['predicted_score'] else "?"
            gt = result['ground_truth']
            match = "=" if pred == gt else "≠"
            duration = result['duration_sec'] if result['duration_sec'] else 0
            
            print(f"  [{completed:3d}/{total}] {status} {result['image_id']}: "
                  f"pred={pred} {match} gt={gt} ({duration:.1f}s)")
    
    total_time = time.time() - start_total
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    correct = [r for r in successful if r['predicted_score'] == r['ground_truth']]
    
    print(f"\n  Summary for {model_name}:")
    print(f"  - Total time: {total_time:.1f}s")
    print(f"  - Success rate: {len(successful)}/{total} ({100*len(successful)/total:.1f}%)")
    if successful:
        print(f"  - Accuracy: {len(correct)}/{len(successful)} ({100*len(correct)/len(successful):.1f}%)")
        avg_time = sum(r['duration_sec'] for r in successful) / len(successful)
        print(f"  - Avg response time: {avg_time:.2f}s")
    
    # Save results to CSV
    model_output_dir = os.path.join(output_dir, model_name.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)
    
    csv_path = os.path.join(model_output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['image_id', 'filename', 'ground_truth', 'predicted_score', 
                      'success', 'duration_sec', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    
    # Save raw responses if enabled
    if settings.get('save_raw_responses', True):
        raw_path = os.path.join(model_output_dir, 'raw_responses.json')
        with open(raw_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"  - Results saved to: {csv_path}")
    
    return {
        'model': model_name,
        'total_images': total,
        'successful': len(successful),
        'correct': len(correct),
        'accuracy': len(correct) / len(successful) if successful else 0,
        'total_time': total_time,
        'avg_response_time': avg_time if successful else 0
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.json> [--limit N]")
        print("Example: python experiments/run_experiment.py experiments/config_zeroshot_test.json")
        print("         python experiments/run_experiment.py experiments/config_quick_test.json --limit 5")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Parse --limit argument
    limit = None
    if '--limit' in sys.argv:
        limit_idx = sys.argv.index('--limit')
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])
    
    # Load config
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config['experiment_name']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"{'='*60}")
    
    # Setup paths (relative to project root)
    os.chdir(PROJECT_ROOT)
    
    prompt_file = config['prompt_file']
    gt_file = config['ground_truth_file']
    images_dir = config['images_dir']
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output_dir'], f"{config['experiment_name']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config copy
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print(f"\nLoading prompt: {prompt_file}")
    prompt = load_prompt(prompt_file)
    
    print(f"Loading ground truth: {gt_file}")
    ground_truth = load_ground_truth(gt_file)
    print(f"  Found {len(ground_truth)} entries")
    
    print(f"Loading images from: {images_dir}")
    images = get_images(images_dir, ground_truth)
    print(f"  Found {len(images)} images with ground truth")
    
    if not images:
        print("ERROR: No images found!")
        sys.exit(1)
    
    # Apply limit if specified
    if limit:
        images = images[:limit]
        print(f"  Limited to {len(images)} images (--limit {limit})")
    
    # Get enabled models
    enabled_models = [m for m in config['models'] if m.get('enabled', True)]
    print(f"\nEnabled models: {[m['name'] for m in enabled_models]}")
    
    # Run experiments
    settings = config.get('settings', {})
    all_results = []
    
    for model_config in enabled_models:
        try:
            result = run_model_experiment(model_config, images, prompt, output_dir, settings)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR running {model_config['name']}: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\n{'Model':<20} {'Success':<10} {'Accuracy':<10} {'Time':<10} {'Avg Resp':<10}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['model']:<20} {r['successful']}/{r['total_images']:<7} "
              f"{r['accuracy']*100:>6.1f}%    {r['total_time']:>6.1f}s   {r['avg_response_time']:>6.2f}s")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment': config['experiment_name'],
            'timestamp': timestamp,
            'results': all_results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
