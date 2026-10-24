# Prompts

The prompt texts and few-shot example descriptions used in the paper are not
distributed with this repository. The scripts expect them in this folder:

| File | Used by |
|---|---|
| `01_zeroshot_original.txt` … `07_fewshot_8img.txt` | `experiments/config_0*.json` (development-set prompts P1–P7) |
| `quality_check.txt` | `pipelines/01_data_quality.py` |
| `p6_prompt.txt` | `nogwh_evaluation/prompt.py` (P6 rubric and output schema; benchmark and citywide runs) |
| `p6_examples.json` | `nogwh_evaluation/prompt.py` (the five P6 example images) |
| `p6_nogwh_examples.json` | `nogwh_evaluation/prompt_nogwh.py` |
| `run_with_details_prompt.txt` | `uncertainty_analysis/run_with_details.py` |

The two example files are JSON lists of objects with `file`, `score` and
`description` keys, in the order the examples are sent to the model.

Everything in this folder except this README is ignored by git.
