# Quick Configuration Reference

## 🎯 Where to Change Settings

**File:** `DSM_code/src/config.py`

All generation parameters are in ONE place - scroll to the bottom of the Config class.

---

## ⚙️ Current Settings (Optimized for Scientific Use)

```python
GENERATION_TEMPERATURE = 0.0          # ← Most deterministic ✅
GENERATION_MAX_TOKENS = 8000          # ← Detailed responses ✅
GENERATION_TOP_P = 0.95               # ← Standard
GENERATION_TOP_K = 40                 # ← Standard
GENERATION_REPETITION_PENALTY = 1.0   # ← No penalty
```

---

## 🔥 Most Important Setting

### Temperature = 0.0 ✅

**This is THE most critical setting for reproducible medical/scientific analysis.**

```python
GENERATION_TEMPERATURE = 0.0  # ← DO NOT CHANGE unless you have a good reason
```

**What it does:**
- Makes AI responses **deterministic** (same input → same output)
- Eliminates randomness
- Perfect for scientific experiments
- Ensures reproducibility

**When to change:**
- **Never** for medical/scientific image analysis
- Only if you need creative/diverse responses (not your use case)

---

## 📊 How Providers Use These Settings

### ✅ Google Gemini
Uses ALL settings from config

### ✅ OpenAI (GPT-4/GPT-5)  
Uses temperature + max_tokens from config

### ✅ Together AI
Uses ALL settings from config

### ✅ Local Ollama
Handles parameters internally (no explicit config needed)

---

## 🚀 Quick Start

1. **Check current settings:**
   - Open `DSM_code/src/config.py`
   - Scroll to bottom (around line 68)
   - Verify `GENERATION_TEMPERATURE = 0.0` ✅

2. **Run your experiments:**
   - All providers will use these consistent settings
   - Results will be reproducible
   - No per-provider tweaking needed

3. **If you need to change settings:**
   - Edit `config.py` ONLY (don't touch provider files)
   - Change applies to all providers automatically
   - Document why you changed it (add a comment)

---

## 📚 Documentation

- **Detailed guide:** `docs/configuration_guide.md`
- **All improvements:** `docs/provider_improvements.md`
- **This quick ref:** `docs/quick_config_reference.md`

---

## ✅ Benefits of Centralized Config

**Before:** 
- Temperature 0.3 in Google ❌
- Temperature 0.7 in Together ❌  
- Temperature 0.0 in OpenAI ❌
- **Inconsistent results!**

**After:**
- Temperature 0.0 everywhere ✅
- **Consistent, reproducible results!**

---

## 🎓 Remember

**One file to rule them all:** `src/config.py`

Change it once → affects all providers → consistent results! 🎉
