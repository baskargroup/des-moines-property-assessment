# Experiment Plan: All Prompts

## Configs (7 total, 3 cloud models each, NO local)

| # | Config | Prompt | Status |
|---|--------|--------|--------|
| 1 | config_01_zeroshot_original.json | 01_zeroshot_original.txt | ✅ DONE |
| 2 | config_02_zeroshot_improved.json | 02_zeroshot_improved_class3.txt | ✅ DONE |
| 3 | config_03_zeroshot_text_examples.json | 03_zeroshot_with_text_examples.txt | ✅ DONE |
| 4 | config_04_fewshot_1img.json | 04_fewshot_1img.txt | ✅ DONE |
| 5 | config_05_fewshot_2img.json | 05_fewshot_2img.txt | ✅ DONE |
| 6 | config_06_fewshot_5img.json | 06_fewshot_5img.txt | ✅ DONE |
| 7 | config_07_fewshot_8img.json | 07_fewshot_8img.txt | ✅ DONE |

## Models (3 cloud only)
- gpt-5.2 (OpenAI)
- gemini-3-flash (Google)
- qwen3-vl-32b (Together)

## Results Summary

### 5-Class Accuracy
| Prompt | GPT-5.2 | Gemini | Qwen3 |
|--------|---------|--------|-------|
| 01_zeroshot_original | 59.0% | **67.7%** | 35.0% |
| 02_zeroshot_improved | 47.0% | 59.0% | 24.0% |
| 03_zeroshot_text_examples | 48.0% | 62.0% | 40.0% |
| 04_fewshot_1img | 51.0% | 56.6% | 29.0% |
| 05_fewshot_2img | 57.0% | 56.0% | 42.0% |
| 06_fewshot_5img | 59.0% | 59.0% | 44.0% |
| 07_fewshot_8img | 56.0% | 58.0% | 46.5% |

### 3-Class Accuracy
| Prompt | GPT-5.2 | Gemini | Qwen3 |
|--------|---------|--------|-------|
| 01_zeroshot_original | 80.0% | 82.8% | 42.0% |
| 02_zeroshot_improved | 77.0% | 76.0% | 28.0% |
| 03_zeroshot_text_examples | 75.0% | 79.0% | 55.0% |
| 04_fewshot_1img | 79.0% | 79.8% | 36.0% |
| 05_fewshot_2img | **84.0%** | 81.0% | 64.0% |
| 06_fewshot_5img | **84.0%** | 81.0% | 67.0% |
| 07_fewshot_8img | 81.0% | 82.0% | 71.7% |

### Human Rater Baseline
- 5-Class: 65.0%
- 3-Class: 78.0%

## Analysis Output
- Full report: `data/outputs/analysis_results/ANALYSIS_REPORT.md`
- Confusion matrices: `data/outputs/analysis_results/confusion_matrices/` (86 images)
- Raw results: `data/outputs/analysis_results/accuracy_results.json`

## Run Commands
```bash
python experiments/run_experiment.py experiments/config_01_zeroshot_original.json
python experiments/run_experiment.py experiments/config_02_zeroshot_improved.json
python experiments/run_experiment.py experiments/config_03_zeroshot_text_examples.json
python experiments/run_experiment.py experiments/config_04_fewshot_1img.json
python experiments/run_experiment.py experiments/config_05_fewshot_2img.json
python experiments/run_experiment.py experiments/config_06_fewshot_5img.json
python experiments/run_experiment.py experiments/config_07_fewshot_8img.json
```
