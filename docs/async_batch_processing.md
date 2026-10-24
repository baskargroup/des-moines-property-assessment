# Async Batch Processing Guide

**Date:** January 23, 2026  
**New Feature:** Concurrent/Parallel Image Processing

## Overview

All VLM providers now support **async/concurrent processing** for dramatically faster batch image analysis. Process hundreds of images in parallel while respecting API rate limits.

---

## 🚀 Performance Improvements

### Before (Sequential Processing)
```python
# Process 100 images one at a time
for image in images:
    result = provider.analyze(image, prompt)
    # Wait for each image... 🐌
```
**Time:** ~500 seconds (100 images × 5 sec/image)

### After (Concurrent Processing)
```python
# Process 100 images concurrently
processor = BatchProcessor(provider, max_concurrent=10)
results = await processor.process_batch(images, prompt)
```
**Time:** ~50 seconds (10× speedup!) 🚀

---

## Features

### ✅ All Providers Enhanced

Every provider now has both **sync** and **async** methods:

#### Synchronous (Original)
```python
result = provider.analyze(image_path, prompt)
```

#### Asynchronous (New!)
```python
result = await provider.analyze_async(image_path, prompt)
```

**Supported providers:**
- ✅ Google Gemini (`google.py`)
- ✅ OpenAI GPT-4/5 (`openai.py`)
- ✅ Together AI (`together.py`)
- ✅ Local Ollama (`local.py`)

### ✅ Automatic Retry Logic

All providers now retry failed requests automatically:

```python
# Retries up to 3 times with exponential backoff
result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze this",
    max_retries=3,      # Number of retry attempts
    retry_delay=2       # Base delay in seconds (doubles each retry)
)
```

**Benefits:**
- Handles transient network errors
- Respects rate limits with exponential backoff
- Automatically waits for API cooldown
- No manual retry code needed

### ✅ Rate Limiting

Built-in throttling to stay under API limits:

```python
processor = BatchProcessor(
    provider=provider,
    max_concurrent_requests=10,    # Process 10 images at once
    max_requests_per_minute=50     # Stay under rate limit
)
```

### ✅ Progress Tracking

Real-time statistics on batch processing:

```python
status = processor.get_status()
print(f"Success rate: {status['success_rate']:.1f}%")
print(f"Completed: {status['succeeded']}/{status['total_tasks']}")
print(f"Failed: {status['failed']}")
print(f"Retries: {status['retries']}")
```

---

## Quick Start

### 1. Simple Async Processing

Process a single image asynchronously:

```python
import asyncio
from src.providers import get_provider

async def analyze_image():
    provider = get_provider("google")
    
    result = await provider.analyze_async(
        image_path="image.jpg",
        prompt="Analyze this medical image"
    )
    
    print(result)

# Run it
asyncio.run(analyze_image())
```

### 2. Batch Processing (Easy Way)

Use the convenience function:

```python
import asyncio
from src.providers.batch_processor import process_images_batch

async def main():
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        # ... more images
    ]
    
    prompt = "Analyze this DSM property image..."
    
    results = await process_images_batch(
        provider_name="google",
        image_paths=image_paths,
        prompt=prompt,
        max_concurrent=10,  # Process 10 at once
        max_rpm=50          # 50 requests per minute
    )
    
    # Process results
    for task in results:
        if task.result:
            print(f"Image {task.task_id}: ✅ Success")
            print(f"Result: {task.result}")
        else:
            print(f"Image {task.task_id}: ❌ Failed - {task.error}")

asyncio.run(main())
```

### 3. Advanced Batch Processing

Full control with BatchProcessor:

```python
import asyncio
from src.providers import get_provider
from src.providers.batch_processor import BatchProcessor

async def main():
    # Initialize provider
    provider = get_provider("google")
    
    # Create batch processor
    processor = BatchProcessor(
        provider=provider,
        max_concurrent_requests=10,     # Concurrent requests
        max_requests_per_minute=50,     # Rate limit
        max_retries=3,                  # Retry attempts
        retry_delay=2.0                 # Base retry delay
    )
    
    # Prepare images
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    prompt = "Analyze this image..."
    
    # Optional metadata
    metadata = [
        {"filename": "img1.jpg", "category": "NH-Typ 1"},
        {"filename": "img2.jpg", "category": "NH-Typ 2"},
        {"filename": "img3.jpg", "category": "NH-Typ 3"}
    ]
    
    # Process batch
    results = await processor.process_batch(
        image_paths=image_paths,
        prompt=prompt,
        metadata=metadata
    )
    
    # Get summary statistics
    summary = processor.get_status()
    print("\nProcessing Summary:")
    print(f"Total: {summary['total_tasks']}")
    print(f"Succeeded: {summary['succeeded']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Elapsed Time: {summary['elapsed_time']:.2f}s")
    
    # Save results
    for task in results:
        if task.result:
            # Save to file, database, etc.
            with open(f"results/task_{task.task_id}.json", "w") as f:
                f.write(task.result)

asyncio.run(main())
```

---

## Provider-Specific Notes

### Google Gemini

Already had excellent retry logic - now enhanced with async support:

```python
provider = get_provider("google")

# Async with custom temperature
result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze...",
    temperature=0.0,  # Optional: override config
    max_retries=3,
    retry_delay=2
)
```

**Rate limits:**
- Free tier: ~15 requests/minute
- Paid tier: ~60 requests/minute

**Recommended settings:**
```python
max_concurrent_requests=5   # Conservative
max_requests_per_minute=10  # Leave headroom
```

### OpenAI (GPT-4/GPT-5)

Now includes retry logic and async support:

```python
provider = get_provider("openai")

result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze...",
    max_retries=3,
    retry_delay=2
)
```

**Rate limits vary by tier:**
- See: https://platform.openai.com/account/rate-limits

**Recommended settings:**
```python
max_concurrent_requests=10
max_requests_per_minute=50  # Adjust based on your tier
```

### Together AI

Enhanced with retry logic and async support:

```python
provider = get_provider("together")

result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze...",
    max_retries=3,
    retry_delay=2
)
```

**Rate limits:**
- Generous limits for paid tier
- Check your account for specific limits

**Recommended settings:**
```python
max_concurrent_requests=15
max_requests_per_minute=100
```

### Local Ollama

Now supports async with retry logic:

```python
provider = get_provider("local")

result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze...",
    max_retries=3,
    retry_delay=2
)
```

**No rate limits** (runs locally), but limited by:
- GPU/CPU resources
- Memory
- Model size

**Recommended settings:**
```python
max_concurrent_requests=3   # Don't overwhelm GPU
max_requests_per_minute=None  # No limit
```

---

## Error Handling

### Automatic Retry on Transient Errors

All providers automatically retry on:
- Network errors
- Timeout errors
- Rate limit errors (with exponential backoff)
- API 5xx errors

```python
# Retries happen automatically!
result = await provider.analyze_async(
    image_path="image.jpg",
    prompt="Analyze...",
    max_retries=3       # Try up to 3 times
)
```

### Handling Failed Tasks

Check results after batch processing:

```python
results = await processor.process_batch(image_paths, prompt)

successful = [task for task in results if task.result]
failed = [task for task in results if task.error]

print(f"Success: {len(successful)}/{len(results)}")

# Retry failed tasks
if failed:
    print(f"Retrying {len(failed)} failed tasks...")
    failed_paths = [task.image_path for task in failed]
    retry_results = await processor.process_batch(failed_paths, prompt)
```

---

## Performance Tips

### 1. Choose Optimal Concurrency

Too low = slow  
Too high = rate limits

**Start conservative:**
```python
# Google Gemini (free tier)
max_concurrent_requests=3
max_requests_per_minute=10

# OpenAI (paid tier)
max_concurrent_requests=10
max_requests_per_minute=50

# Local Ollama
max_concurrent_requests=2  # Based on GPU
```

### 2. Monitor Progress

Use logging to track progress:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see:
# INFO: Processing task 0: image1.jpg (attempt 1/3)
# INFO: ✅ Task 0 completed successfully
# INFO: Processing task 1: image2.jpg (attempt 1/3)
# ...
```

### 3. Save Results Incrementally

Don't wait for entire batch:

```python
async def process_with_save(tasks, processor):
    results = await processor.process_batch(tasks, prompt)
    
    # Save each result immediately
    for task in results:
        if task.result:
            save_result(task)  # Save to file/database
    
    return results
```

### 4. Handle Large Batches

Split very large batches:

```python
def chunks(lst, n):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Process in chunks of 100
all_images = [...1000 images...]
all_results = []

for chunk in chunks(all_images, 100):
    results = await processor.process_batch(chunk, prompt)
    all_results.extend(results)
    print(f"Processed {len(all_results)}/{len(all_images)} images")
```

---

## Complete Example: Process Entire Dataset

```python
import asyncio
import json
from pathlib import Path
from src.providers import get_provider
from src.providers.batch_processor import BatchProcessor

async def process_dataset():
    # Load dataset
    data_dir = Path("data/test/random_56")
    image_paths = list(data_dir.glob("**/*.jpg"))
    
    print(f"Found {len(image_paths)} images")
    
    # Load prompt
    with open("data/test/annotated/final_prompt.txt") as f:
        prompt = f.read()
    
    # Initialize provider and processor
    provider = get_provider("google")
    processor = BatchProcessor(
        provider=provider,
        max_concurrent_requests=5,
        max_requests_per_minute=10,
        max_retries=3
    )
    
    # Process all images
    print("Starting batch processing...")
    results = await processor.process_batch(
        image_paths=[str(p) for p in image_paths],
        prompt=prompt
    )
    
    # Save results
    output_dir = Path("results/batch_processing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for task in results:
        if task.result:
            output_file = output_dir / f"{Path(task.image_path).stem}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "image_path": task.image_path,
                    "result": task.result,
                    "attempts": task.attempts
                }, f, indent=2)
    
    # Print summary
    summary = processor.get_status()
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total: {summary['total_tasks']}")
    print(f"Succeeded: {summary['succeeded']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Time: {summary['elapsed_time']:.2f}s")
    print(f"Retries: {summary['retries']}")
    print("=" * 60)

# Run it
if __name__ == "__main__":
    asyncio.run(process_dataset())
```

---

## Troubleshooting

### Issue: Rate Limit Errors

**Solution:** Reduce concurrency and requests per minute:
```python
max_concurrent_requests=3      # Lower
max_requests_per_minute=10     # Lower
```

### Issue: Out of Memory

**Solution:** Process in smaller batches:
```python
# Instead of all 1000 images at once
for chunk in chunks(images, 100):  # 100 at a time
    results = await processor.process_batch(chunk, prompt)
```

### Issue: Too Slow

**Solution:** Increase concurrency (if under rate limits):
```python
max_concurrent_requests=20     # Higher
max_requests_per_minute=100    # Higher
```

### Issue: Connection Timeouts

**Solution:** Increase retry delay:
```python
retry_delay=5.0  # Longer wait between retries
```

---

## Migration Guide

### Migrating Existing Code

**Old sequential code:**
```python
results = []
for image in images:
    result = provider.analyze(image, prompt)
    results.append(result)
```

**New async code:**
```python
results = await process_images_batch(
    provider_name="google",
    image_paths=images,
    prompt=prompt
)
```

**That's it! 10× faster with one change!**

---

## Summary

**All providers now support:**
- ✅ Async/concurrent processing
- ✅ Automatic retry logic with exponential backoff
- ✅ Rate limiting and throttling
- ✅ Progress tracking
- ✅ Batch processing utilities

**Performance gains:**
- 5-10× faster for batch processing
- Automatic error handling
- Stay under rate limits
- Production-ready

**Start using it today!**

```python
import asyncio
from src.providers.batch_processor import process_images_batch

async def main():
    results = await process_images_batch(
        provider_name="google",
        image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
        prompt="Analyze this image...",
        max_concurrent=5,
        max_rpm=10
    )
    
    for task in results:
        print(f"Image {task.task_id}: {'✅' if task.result else '❌'}")

asyncio.run(main())
```
