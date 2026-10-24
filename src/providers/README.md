# VLM Providers

Production-ready Vision Language Model providers with async/concurrent processing.

## Available Providers

### 🔵 Google Gemini (`google.py`)
- Model: Gemini 3 Pro Preview
- Features: Vision analysis, retry logic, rate limiting
- Rate limits: Free tier ~15/min, Paid ~60/min

### 🟢 OpenAI (`openai.py`)
- Models: GPT-4 Vision, GPT-5.2
- Features: Vision analysis, reasoning models
- Rate limits: Varies by tier

### 🔴 Together AI (`together.py`)
- Model: Qwen 2.5 Vision 72B
- Features: Open-source vision models
- Rate limits: Generous for paid tier

### 🟡 Local Ollama (`local.py`)
- Model: Gemma3 27B (configurable)
- Features: Local processing, no API costs
- Rate limits: None (hardware limited)

---

## Quick Start

### Simple Usage (Sync)

```python
from src.providers import get_provider

provider = get_provider("google")
result = provider.analyze(
    image_path="image.jpg",
    prompt="Analyze this medical image"
)
print(result)
```

### Async Processing (Fast!)

```python
import asyncio
from src.providers import get_provider

async def main():
    provider = get_provider("google")
    result = await provider.analyze_async(
        image_path="image.jpg",
        prompt="Analyze this image"
    )
    print(result)

asyncio.run(main())
```

### Batch Processing (10× Faster!)

```python
import asyncio
from src.providers.batch_processor import process_images_batch

async def main():
    results = await process_images_batch(
        provider_name="google",
        image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
        prompt="Analyze...",
        max_concurrent=10,
        max_rpm=50
    )
    
    for task in results:
        print(f"Image {task.task_id}: {'✅' if task.result else '❌'}")

asyncio.run(main())
```

---

## Features

### ✅ All Providers Support:
- Synchronous `analyze()` method
- Asynchronous `analyze_async()` method  
- Automatic retry logic with exponential backoff
- Centralized configuration from `config.py`
- Comprehensive logging
- Temperature = 0.0 for reproducibility

### ✅ Batch Processor:
- Concurrent processing (10× faster!)
- Rate limiting and throttling
- Progress tracking with statistics
- Works with ANY provider
- Memory efficient
- Production-ready error handling

---

## Configuration

All settings in `src/config.py`:

```python
# Generation parameters (all providers)
GENERATION_TEMPERATURE = 0.0      # Reproducible results
GENERATION_MAX_TOKENS = 8000      # Complete responses
GENERATION_TOP_P = 0.95
GENERATION_TOP_K = 40

# Model names
MODEL_GOOGLE = "gemini-3-pro-preview"
MODEL_OPENAI = "gpt-5.2-2025-12-11"
MODEL_TOGETHER = "Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_LOCAL = "gemma3:27b"
```

---

## Files

```
providers/
├── __init__.py              # Provider factory
├── base.py                  # Abstract base class
├── google.py                # Google Gemini
├── openai.py                # OpenAI GPT-4/5
├── together.py              # Together AI
├── local.py                 # Local Ollama
├── batch_processor.py       # Unified batch processor
├── api_request_parallel.py  # Reference implementation
└── README.md                # This file
```

---

## Documentation

📚 **Complete guides in `docs/`:**

- **`async_batch_processing.md`** - Async/concurrent processing guide
- **`configuration_guide.md`** - Config parameters explained
- **`provider_improvements.md`** - All improvements summary
- **`enhancement_summary.md`** - Enhancement overview
- **`quick_config_reference.md`** - Quick reference

---

## Performance

### Sequential (Old)
```python
# Process 100 images
for img in images:
    result = provider.analyze(img, prompt)
# Time: ~500 seconds
```

### Concurrent (New)
```python
# Process 100 images
results = await process_images_batch(
    "google", images, prompt, max_concurrent=10
)
# Time: ~50 seconds (10× faster!)
```

---

## Examples

See `docs/async_batch_processing.md` for complete examples including:
- Single image async processing
- Batch processing with progress tracking
- Error handling and retries
- Provider-specific settings
- Large dataset processing

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `aiohttp>=3.9.0` - Async HTTP
- `openai>=1.0.0` - OpenAI API
- `google-generativeai>=0.3.0` - Google Gemini
- `requests>=2.31.0` - HTTP requests
- `Pillow>=10.0.0` - Image processing

---

## Support

**Issues?** Check documentation:
1. `docs/async_batch_processing.md` - Usage guide
2. `docs/configuration_guide.md` - Settings
3. `docs/provider_improvements.md` - Features

**Still stuck?** The code is well-documented with docstrings and comments!

---

**Happy processing!** 🚀
