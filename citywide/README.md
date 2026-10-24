# Citywide scoring and block-group aggregation

Code used to score 66,701 Des Moines residential properties with the P6 prompt
(GPT-5.6 Terra, GPT-5.2 and Gemini-3-Flash) and to aggregate the scores to 2020
Census block groups for the paper's citywide map.

The parcel data, property photographs, manifests and per-property results are
City of Des Moines data and are not included. The folders the scripts read and
write (`manifests/`, `runs/`, `gis/`, `figures/`) are ignored by git.

## Layout

    .env                          API keys for the campaign (not committed)
    config.yaml                   GPT-5.2 / Gemini campaign settings and budget
    terra_config.yaml             GPT-5.6 Terra campaign settings and budget
    build_manifest.py             parcels + one photo each -> manifests/<run_id>.csv
    run_campaign.py               cached, resume-safe GPT-5.2 / Gemini runner
    run_terra_campaign.py         cached, resume-safe GPT-5.6 Terra runner
    export_stage_results.py       flatten JSONL outputs to CSV
    build_consensus_outputs.py    merge the three models and mark consensus cases
    eval_three_class.py           three-class agreement with the 2022 city survey
    build_block_group_scores.py   mean scores per 2020 Census block group
    build_paper_terra_block_group_figure.py   block-group map (paper figure)
    build_paper_terra_grid_figure.py          earlier grid map; shared basemap helpers
    analyze_final_hotspots.py, build_final_gis_outputs.py, build_final_map.py
                                  grid hotspot summaries and GIS exports

The prompt text and calibration images are loaded from `../nogwh_evaluation`.

## Inputs

- `DSM_RAW_DATA`: folder with the City's property-condition geodatabases,
  attachment table and extracted photos (used by `build_manifest.py`).
- `gis/census_2020_block_groups/CensusBlockGroupDSM2020.shp`: block-group
  boundaries provided by the City.

## Commands

    python build_manifest.py --out manifests/full.csv
    python run_terra_campaign.py --dry-run --limit 1000
    python run_terra_campaign.py --limit 66701
    python run_campaign.py
    python build_consensus_outputs.py
    python build_block_group_scores.py
    python build_paper_terra_block_group_figure.py

Runs are resumable: any image without a successful record in
`runs/<run_id>/predictions.jsonl` is retried. Each runner enforces the hard
budget ceiling set in its config file.
