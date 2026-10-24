# NOGWH Test-Set Evaluation

Held-out evaluation of the paper's method on 1,280 property photos from the
North of Grand and Woodland Heights neighborhoods (City of Des Moines 2025
field survey). The ~100-image development set (see `../data/` and
`../experiments/`) was used to select the prompt and example images; this set
was scored once with the final method and never used for tuning.

## Model panel

All 1,280 images were scored by three models with the P6 five-image few-shot
prompt (loaded by `prompt.py`; the prompt files are not distributed, see
`../data/prompts/README.md`):

- `gpt-5.2-2025-12-11` (OpenAI)
- `gemini-3-flash-preview` (Google)
- `gpt-4.1` (OpenAI)

## Workflow and key files

1. `build_manifest.py` -> `nogwh_manifest.csv` — one row per image with the
   city inspector's score (extracted from the survey geodatabase).
2. `run_full_1280.py` -> `predictions_full.jsonl` — scores every image with
   the three-model panel. Resume-safe; skips already-scored images.
3. `analyze_full.py`, `consensus_breakdown.py`, `all_confusion_matrices.py`,
   `generate_summary_report.py` — accuracy, model-vs-city agreement, and
   consensus statistics.
4. `swap_gpt41_for_gemini35.py`, `predictions_full_gemini35.jsonl` — panel
   ablation comparing GPT-4.1 against Gemini 3.5 Flash.
5. `five_buckets.py`, `generate_five_files.py`, `generate_review_package.py` —
   group properties by AI-vs-city disagreement and build the offline HTML
   review packages used by human reviewers.
6. `reviewer/` — reviewer responses (`reviews_*.csv`) and the analysis that
   produces the final metrics and updated ground truth.
7. `build_updated_ground_truth_csv.py` ->
   `nogwh_updated_ground_truth_predictions.csv` — final per-property file
   combining city score, model scores, consensus, and human review.

`run_paper100.py` / `predictions_paper100.jsonl` reproduce the paper's
100-image development-set numbers with the same panel.

## Data note

The NOGWH photos themselves are not distributed in this repository. Scripts
that read images expect a local folder of extracted survey photos
(`build_manifest.py` documents the expected layout). API keys are read from
the repository-root `.env`; copy `.env.example` and supply your own keys.
