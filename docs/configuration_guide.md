# Configuration Guide

**Location:** `DSM_code/src/config.py`  
**Date:** January 23, 2026

## Overview

All VLM provider settings are now centralized in `config.py`. This ensures consistent behavior across all providers (Google Gemini, OpenAI, Together AI, and local Ollama) and makes it easy to tune generation parameters from a single location.

---

## Generation Parameters

### Core Settings

All generation parameters are defined in the `Config` class and used by all providers:

```python
# Temperature: Controls randomness in responses
GENERATION_TEMPERATURE = 0.0  # Most deterministic

# Max output tokens: Maximum length of generated responses  
GENERATION_MAX_TOKENS = 8000  # Sufficient for detailed JSON

# Top-p (nucleus sampling): Focuses on most likely tokens
GENERATION_TOP_P = 0.95  # Consider top 95% probability mass

# Top-k: Limits to top K most likely tokens
GENERATION_TOP_K = 40  # Standard value

# Repetition penalty (Together AI only)
GENERATION_REPETITION_PENALTY = 1.0  # No penalty
```

---

## Temperature Explained

Temperature controls the randomness/creativity of model responses:

### Temperature = 0.0 (Current Setting) ✅ RECOMMENDED
- **Most deterministic** and reproducible
- Always picks the most likely token
- **Perfect for scientific/medical analysis** where consistency is critical
- Same input → same output (approximately)
- Reduces variability between runs

### Temperature = 0.3-0.7
- Slightly more variation
- Still relatively focused
- Good for balanced use cases

### Temperature = 1.0+
- More creative and diverse
- Higher randomness
- Less reproducible
- Better for creative writing, brainstorming

**For DSM image analysis, Temperature = 0.0 is the optimal choice.**

---

## How It Works

### Before (Scattered Configuration)
Each provider file had its own constants:
```python
# google.py
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_TOKENS = 8000

# together.py  
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8000

# openai.py
temperature = 0.0 (hardcoded)
max_tokens = 8000 (hardcoded)
```

**Problem:** Inconsistent settings, hard to maintain, duplication.

### After (Centralized Configuration) ✅
All providers now reference `Config`:
```python
# google.py
temperature=Config.GENERATION_TEMPERATURE
max_output_tokens=Config.GENERATION_MAX_TOKENS

# together.py
temperature=Config.GENERATION_TEMPERATURE  
max_tokens=Config.GENERATION_MAX_TOKENS

# openai.py
temperature=Config.GENERATION_TEMPERATURE
max_tokens=Config.GENERATION_MAX_TOKENS
```

**Benefits:** Single source of truth, consistent behavior, easy to adjust.

---

## Provider-Specific Behavior

### Google Gemini (`google.py`)
- Uses all config parameters
- Supports both old and new API packages
- Includes retry logic for rate limiting

**Applied settings:**
- `GENERATION_TEMPERATURE`
- `GENERATION_MAX_TOKENS` 
- `GENERATION_TOP_P`
- `GENERATION_TOP_K`

### OpenAI (`openai.py`)
- Uses temperature and max tokens from config
- Handles GPT-4 vs GPT-5 differences automatically
- Special handling for reasoning models (GPT-5.2)

**Applied settings:**
- `GENERATION_TEMPERATURE`
- `GENERATION_MAX_TOKENS`

### Together AI (`together.py`)
- Uses all config parameters
- Includes repetition penalty setting

**Applied settings:**
- `GENERATION_TEMPERATURE`
- `GENERATION_MAX_TOKENS`
- `GENERATION_TOP_P`
- `GENERATION_TOP_K`
- `GENERATION_REPETITION_PENALTY`

### Local Ollama (`local.py`)
- Ollama handles generation parameters internally
- No explicit temperature/token settings needed
- Uses JSON format enforcement

---

## How to Adjust Settings

### Change Temperature

Edit `src/config.py`:
```python
# For more deterministic (recommended)
GENERATION_TEMPERATURE = 0.0

# For slightly more variation
GENERATION_TEMPERATURE = 0.3

# For creative responses (not recommended for medical analysis)
GENERATION_TEMPERATURE = 1.0
```

### Change Max Tokens

```python
# Current setting (sufficient for detailed responses)
GENERATION_MAX_TOKENS = 8000

# If you need longer responses
GENERATION_MAX_TOKENS = 16000

# If you want shorter responses (faster, cheaper)
GENERATION_MAX_TOKENS = 4000
```

### Change Sampling Parameters

```python
# More focused responses
GENERATION_TOP_P = 0.8
GENERATION_TOP_K = 20

# More diverse responses  
GENERATION_TOP_P = 0.95
GENERATION_TOP_K = 50
```

---

## Best Practices

### ✅ DO:
- Keep `GENERATION_TEMPERATURE = 0.0` for scientific/medical analysis
- Use `GENERATION_MAX_TOKENS = 8000` for complete JSON responses
- Adjust settings in `config.py` only, not in individual provider files
- Test changes with a small sample before running on full dataset
- Document why you changed a setting (add comment in config.py)

### ❌ DON'T:
- Don't hardcode values in provider files (always use Config)
- Don't use high temperature (>0.5) for medical image analysis
- Don't set max_tokens too low (<4000) or responses may be truncated
- Don't change settings mid-experiment (affects reproducibility)

---

## Reproducibility

With `GENERATION_TEMPERATURE = 0.0`, results should be highly reproducible:

### What's Reproducible:
- Same image + same prompt → same analysis (approximately)
- Consistent scoring across runs
- Deterministic category assignments

### What May Still Vary:
- Minor differences due to model updates
- API-level randomness (minimal with temp=0)
- Different API versions may behave slightly differently

### Tips for Maximum Reproducibility:
1. Use `GENERATION_TEMPERATURE = 0.0` ✅
2. Record model versions and dates
3. Save all responses for verification
4. Use the same API package versions
5. Run experiments in batches with same configuration

---

## Configuration Checklist

Before running experiments:
- [ ] Temperature is set appropriately (0.0 for scientific)
- [ ] Max tokens is sufficient (8000 recommended)
- [ ] All API keys are configured in `.env` file
- [ ] Model names are up to date in `config.py`
- [ ] Settings are documented (commented in config)

---

## Model Configuration

Current model settings (as of Jan 2026):

```python
# OpenAI: GPT-5.2 (latest thinking model)
MODEL_OPENAI = "gpt-5.2-2025-12-11"

# Google: Gemini 3 Pro Preview (latest vision model)
MODEL_GOOGLE = "gemini-3-pro-preview"

# Together AI: Qwen 2.5 Vision (72B parameters)
MODEL_TOGETHER = "Qwen/Qwen2.5-VL-72B-Instruct"

# Local: Gemma 3 (27B parameters)
MODEL_LOCAL = "gemma3:27b"
```

See `config.py` for alternative model options.

---

## Environment Variables

Required in `src/.env`:
```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
```

The config file automatically loads these from `src/.env`.

---

## Summary

**All generation settings are now centralized in `config.py` with:**
- ✅ Temperature = 0.0 (most deterministic)
- ✅ Max tokens = 8000 (complete responses)
- ✅ Consistent across all providers
- ✅ Easy to adjust from one location
- ✅ Well-documented and maintainable

**This ensures reproducible, consistent results across all VLM providers.**
