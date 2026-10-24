# VLM Property Assessment Pipeline

## 1. Project Structuring & Design
- [ ] **Initialize Structure**: Create modular folders (`src/`, `data/`, `pipelines/`, `notebooks/`).
- [ ] **Configuration**: Implement `src/config.py` for centralized API keys and settings.
- [ ] **Core Logic**:
    - [ ] Create `src/data_loader.py` (XML parsing, image loading).
    - [ ] Create `src/providers/` with `BaseVLM` (Strategy Pattern).
    - [ ] Implement provider classes (Local, OpenAI, Google, Together).

## 2. Data Quality Check (Local VLM)
- **Objective**: Assess image clarity, house visibility, and address readability.
- **Tool**: Local VLM via Ollama (Llama 3.2 Vision).
- [ ] **Prompt**: Create `data/prompts/quality_check.txt`.
- [ ] **Pipeline Script**: Implement `pipelines/01_data_quality.py`.
- [ ] **Pilot**: Run on 10 random images.
- [ ] **Hypothesis Test**: Compare "Unscored" vs. "Scored" image quality.
- [ ] **Scale Up**: Run on full dataset.

## 3. Multi-Model Zero-Shot Scoring
- **Objective**: Score properties using state-of-the-art VLMs without examples.
- **Models**: GPT-4o, Gemini 1.5, Llama-3.2-90B.
- [ ] **Prompt**: Migrate `promptWOcontext.txt` to `data/prompts/`.
- [ ] **Pipeline Script**: Implement `pipelines/02_score_zeroshot.py`.
- [ ] **Execution**: Run batch scoring and save results to `data/outputs/`.

## 4. Regression Analysis
- **Objective**: Correlate sub-category scores (roof, windows) with the final class.
- [ ] **Notebook**: Create `notebooks/regression.ipynb`.
- [ ] **Analysis**: Train Linear/RandomForest models on VLM outputs.

## 5. Multiple Judges & Aggregation
- **Objective**: Improve reliability by aggregating diverse model opinions.
- [ ] **Refactor**: Update `02_score_zeroshot.py` to support multiple judges.
- [ ] **Analysis**: Implement voting/averaging logic in notebooks.

## 6. In-Context Learning (Few-Shot)
- **Objective**: Improve alignment with DSM standards using "Gold Standard" examples.
- [ ] **Selection**: Identify 1 representative image per score (1-5) from ground truth.
- [ ] **Pipeline Script**: Implement `pipelines/03_score_fewshot.py`.
- [ ] **Logic**: Dynamically inject examples into the VLM prompt context.

