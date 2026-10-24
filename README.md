# Des Moines Property Condition Assessment

Research code for vision-language-model assessment of residential property
condition in Des Moines, Iowa.

This repository contains the code for the paper's
development-set prompt comparison, the held-out North of Grand and Woodland
Heights (NOGWH) benchmark, the consensus analysis, and the citywide scoring and
block-group aggregation. City imagery, expert labels and parcel-level results
are not included (see [Data and privacy](#data-and-privacy)), and neither are the
prompt texts (see [`data/prompts/README.md`](data/prompts/README.md)). The Streamlit
deployment application is maintained separately.

## Study design

Each property receives a five-point exterior-condition score from the City of
Des Moines rubric:

1. Excellent
2. Good
3. Average
4. Starting distress
5. Distressed

For policy use, scores are grouped into healthy (1–2), intermediate (3) and
unhealthy (4–5).

The research proceeded in three stages:

1. **Development set (100 images):** seven prompting strategies (P1–P7) were
   compared with GPT-5.6 Terra, GPT-5.2 and Gemini-3-Flash. P6, the five-image
   few-shot prompt, was selected for the final workflow.
2. **Held-out NOGWH benchmark (1,280 images):** the three-model ensemble was
   evaluated against expert labels in three independent runs. Agreement among
   the models was used to separate high-confidence assessments from cases
   routed to human review.
3. **Citywide application (66,701 properties):** the same ensemble scored every
   residential property, and scores were aggregated to 2020 Census block groups
   ([`citywide/`](citywide/README.md)).

## Repository structure

```text
des-moines-property-assessment/
├── experiments/            # P1–P7 development-set experiment configs/runners
├── nogwh_evaluation/       # Held-out 1,280-image benchmark runners and analysis
├── citywide/               # Citywide scoring and block-group aggregation
├── uncertainty_analysis/   # Consensus and review-routing analysis
├── src/                    # Shared research providers and batch processing
├── tests/                  # Tests for the shared research code
├── notebooks/              # Regression and classification analyses
├── data/
│   └── prompts/            # Expected prompt files (not distributed)
├── presentation_figures/   # Aggregate tables and confusion matrices
├── docs/                   # Research configuration and analysis documentation
├── scripts/                # Analysis/report utilities
└── pipelines/              # Earlier pipeline implementations
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For API-backed experiments:

```bash
copy .env.example .env
```

Then provide the required API keys in `.env`. Never commit this file.

## Development-set prompt experiments

The seven experiment configurations are documented in
[`experiments/PLAN.md`](experiments/PLAN.md).

Run an individual configuration from the repository root:

```bash
python experiments/run_experiment.py experiments/config_06_fewshot_5img.json
```

The prompt selected for the final workflow is P6
(`data/prompts/06_fewshot_5img.txt`). Prompt files are not distributed; the
scripts expect them at the paths listed in
[`data/prompts/README.md`](data/prompts/README.md).

The few-shot prompts expect the example images in
`data/download/example_images/` and the development-set images in
`data/download/test_100_images/`; both are City data and must be obtained
separately. Outputs are written to `data/outputs/`.

## Held-out NOGWH evaluation

See [`nogwh_evaluation/README.md`](nogwh_evaluation/README.md) for the complete
workflow and file map.

The runners read the evaluation manifest (`nogwh_manifest.csv`), the expert
labels and the 1,280 source photographs, none of which are distributed here.
Set `NOGWH_IMAGES_DIR` in `.env` to the local image folder (see
[`.env.example`](.env.example)). `run_terra_full.py` and
`reviewer/_paper_repro_terra.py` produce the three-run metrics reported in the
paper.

## Uncertainty and consensus

The uncertainty analysis evaluates model agreement as a routing signal for
human review:

- [`uncertainty_analysis/RESULTS_SUMMARY.md`](uncertainty_analysis/RESULTS_SUMMARY.md)
- [`docs/UNCERTAINTY_ANALYSIS.md`](docs/UNCERTAINTY_ANALYSIS.md)

## Tests

```bash
python tests/run_all_tests.py
```

Some tests call or inspect provider-specific functionality and may require the
corresponding optional environment configuration.

## Data and privacy

Property photographs, expert labels, parcel identifiers, coordinates and
per-image model outputs are City of Des Moines data covered by a municipal
data-use agreement. They are not part of this repository, and `.gitignore`
blocks the folders where scripts read and write them.

Do not commit:

- `.env` files or API credentials
- prompt texts or few-shot example descriptions
- property photographs, including few-shot example images
- manifests, labels, predictions or any file with image names, parcel IDs or
  coordinates
- municipal geodatabases and citywide parcel datasets

## Citation and license

Citation details and the final software/data license will be added when the
manuscript publication information is finalized.
