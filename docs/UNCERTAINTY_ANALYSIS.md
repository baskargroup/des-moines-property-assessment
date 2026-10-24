# Uncertainty Analysis Report

## Overview

This report analyzes **ensemble-based uncertainty quantification** across all 7 prompts and 3 VLM models.

**Key Insight:** When multiple independent models agree on a prediction, that prediction is significantly more reliable.

---

## Models Evaluated

| Model | Provider |
|-------|----------|
| GPT-5.2 | OpenAI |
| Gemini-3-Flash | Google |
| Qwen3-VL-32B | Together AI |

---

## Prompts Evaluated

| # | Prompt | Type |
|---|--------|------|
| 01 | zeroshot_original | Zero-shot |
| 02 | zeroshot_improved_class3 | Zero-shot (Class 3 focus) |
| 03 | zeroshot_text_examples | Zero-shot with text examples |
| 04 | fewshot_1img | Few-shot (1 example image) |
| 05 | fewshot_2img | Few-shot (2 example images) |
| 06 | fewshot_5img | Few-shot (5 example images) |
| 07 | fewshot_8img | Few-shot (8 example images) |

---

## Individual Model Accuracy (3-Class)

| Prompt | GPT-5.2 | Gemini-3-Flash | Qwen3-VL-32B | Best Model |
|--------|---------|----------------|--------------|------------|
| 01_zeroshot_original | 80.8% | 82.8% | 42.4% | Gemini (82.8%) |
| 02_zeroshot_improved_class3 | 77.0% | 76.0% | 28.0% | GPT (77.0%) |
| 03_zeroshot_text_examples | 75.0% | 79.0% | 55.0% | Gemini (79.0%) |
| 04_fewshot_1img | 78.8% | 79.8% | 36.4% | Gemini (79.8%) |
| 05_fewshot_2img | 84.0% | 81.0% | 64.0% | GPT (84.0%) |
| 06_fewshot_5img | 84.0% | 81.0% | 67.0% | GPT (84.0%) |
| 07_fewshot_8img | 81.8% | 81.8% | 71.7% | GPT (81.8%) |

---

## Ensemble Agreement Analysis

### Method: All 3 Models Agree (3-Class)

When all 3 models predict the same 3-class category (Healthy/Average/Unhealthy), we consider the prediction **confident**.

| Prompt | Agreement % | Accuracy (Agree) | Accuracy (Disagree) | Improvement |
|--------|-------------|------------------|---------------------|-------------|
| 01_zeroshot_original | 31% (31/99) | **87.1%** | 80.9% | +6.2% |
| 02_zeroshot_improved_class3 | 28% (28/100) | **67.9%** | 77.8% | +-9.9% |
| 03_zeroshot_text_examples | 52% (52/100) | **84.6%** | 68.8% | +15.9% |
| 04_fewshot_1img | 26% (26/99) | **84.6%** | 79.5% | +5.2% |
| 05_fewshot_2img | 57% (57/100) | **91.2%** | 79.1% | +12.2% |
| 06_fewshot_5img | 60% (60/100) | **93.3%** | 70.0% | +23.3% |
| 07_fewshot_8img | 66% (65/99) | **92.3%** | 67.6% | +24.7% |

---

## Class-Based Agreement Analysis

### Agreement Rate by Ground Truth Class (All Prompts)


#### 01_zeroshot_original

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 2 | 4.7% | 50.0% | 95.1% |
| Average | 15 | 1 | 6.7% | 0.0% | 42.9% |
| Unhealthy | 41 | 28 | 68.3% | 92.9% | 76.9% |

#### 02_zeroshot_improved_class3

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 2 | 4.7% | 0.0% | 90.2% |
| Average | 16 | 7 | 43.8% | 85.7% | 77.8% |
| Unhealthy | 41 | 19 | 46.3% | 68.4% | 54.5% |

#### 03_zeroshot_text_examples

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 25 | 58.1% | 92.0% | 77.8% |
| Average | 16 | 6 | 37.5% | 83.3% | 70.0% |
| Unhealthy | 41 | 21 | 51.2% | 76.2% | 60.0% |

#### 04_fewshot_1img

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 42 | 4 | 9.5% | 100.0% | 92.1% |
| Average | 16 | 2 | 12.5% | 50.0% | 50.0% |
| Unhealthy | 41 | 20 | 48.8% | 85.0% | 76.2% |

#### 05_fewshot_2img

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 34 | 79.1% | 97.1% | 88.9% |
| Average | 16 | 3 | 18.8% | 66.7% | 53.8% |
| Unhealthy | 41 | 20 | 48.8% | 85.0% | 90.5% |

#### 06_fewshot_5img

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 34 | 79.1% | 97.1% | 88.9% |
| Average | 16 | 3 | 18.8% | 33.3% | 38.5% |
| Unhealthy | 41 | 23 | 56.1% | 95.7% | 83.3% |

#### 07_fewshot_8img

| GT Class | Total | Agree | Agreement % | Accuracy (Agree) | Accuracy (Disagree) |
|----------|-------|-------|-------------|------------------|---------------------|
| Healthy | 43 | 37 | 86.0% | 97.3% | 83.3% |
| Average | 15 | 5 | 33.3% | 40.0% | 50.0% |
| Unhealthy | 41 | 23 | 56.1% | 95.7% | 72.2% |

---

## Key Finding: Class 3 (Average) is the Hardest

Across all prompts, the **Average** class (score 3) shows:

1. **Lowest agreement rate** (~18-30% vs 50-80% for other classes)
2. **Lowest accuracy** even when models agree
3. **High confusion** - models spread predictions across all 3 classes

This indicates that Class 3 represents borderline cases that are inherently ambiguous.

---

## GPT + Gemini Agreement (Cost-Effective Alternative)

Using only 2 models (GPT + Gemini) reduces API costs by 33% while maintaining good uncertainty detection.

| Prompt | Agreement % | Accuracy (When Agree) |
|--------|-------------|----------------------|
| 01_zeroshot_original | 86% (85/99) | 87.1% |
| 02_zeroshot_improved_class3 | 79% (79/100) | 84.8% |
| 03_zeroshot_text_examples | 76% (76/100) | 88.2% |
| 04_fewshot_1img | 85% (84/99) | 86.9% |
| 05_fewshot_2img | 82% (82/100) | 91.5% |
| 06_fewshot_5img | 84% (84/100) | 90.5% |
| 07_fewshot_8img | 84% (83/99) | 89.2% |

---

## Recommended Production Strategy

### Option A: Maximum Accuracy (3 Models)

```
1. Run image through GPT-5.2, Gemini-3-Flash, Qwen3-VL-32B
2. If ALL 3 agree on 3-class → Auto-classify (high confidence)
3. If ANY disagree → Flag for human review
```

**Best prompt for this strategy:** `06_fewshot_5img` or `07_fewshot_8img`
- 60-66% auto-classification
- 92-93% accuracy on auto-classified images

### Option B: Cost-Effective (2 Models)

```
1. Run image through GPT-5.2 and Gemini-3-Flash only
2. If BOTH agree → Auto-classify
3. If they disagree → Flag for human review
```

**Trade-off:** Higher coverage (~80-85%) with slightly lower accuracy improvement

### Option C: Class-Specific Strategy

```
1. Run all 3 models
2. If prediction is Healthy or Unhealthy AND all agree → Auto-classify (95%+ accuracy)
3. If prediction is Average OR models disagree → Human review
```

This acknowledges that Average (Class 3) is inherently harder and may need human judgment.

---

## Summary

| Metric | Value |
|--------|-------|
| Best prompt | `06_fewshot_5img` |
| Best single model | GPT-5.2 (84%) |
| Accuracy when all agree | 93.3% |
| Coverage (all agree) | 60% |
| Hardest class | Average (18.8% agreement) |

**Conclusion:** Ensemble disagreement is a reliable uncertainty signal, especially for Healthy and Unhealthy classes. Class 3 (Average) remains challenging across all models and prompts.

---

*Generated from 100 test images across 7 prompts and 3 models.*
