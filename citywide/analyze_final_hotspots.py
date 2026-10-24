"""Create a concise final hotspot analysis from the medium (~550m) grid."""
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "citywide_terra"
DELIVERY = RUN_DIR / "delivery"
SOURCE = RUN_DIR / "stage_66701_results.csv"
GRID = DELIVERY / "terra_hotspots_multi.json"
REPORT_MD = DELIVERY / "terra_hotspot_report.md"
REPORT_JSON = DELIVERY / "terra_hotspot_report.json"


def load_results():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table(title, rows, value_label, value_key):
    lines = [
        f"## {title}",
        "",
        f"| Rank | Latitude | Longitude | Properties | {value_label} | Terra mean | City mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, cell in enumerate(rows, 1):
        value = cell[value_key]
        baseline = cell["baseline_mean"]
        baseline_text = f"{baseline:.2f}" if baseline is not None else "n/a"
        lines.append(
            f"| {rank} | {cell['latitude']:.5f} | {cell['longitude']:.5f} | "
            f"{cell['property_count']:,} | {value:.2f} | {cell['terra_mean']:.2f} | "
            f"{baseline_text} |"
        )
    lines.append("")
    return lines


def main():
    results = load_results()
    grid = json.loads(GRID.read_text(encoding="utf-8"))["medium"]["cells"]
    comparable = [
        cell
        for cell in grid
        if cell["mean_score_delta"] is not None and cell["baseline_rated_count"] >= 10
    ]

    worst = sorted(grid, key=lambda cell: cell["terra_mean"], reverse=True)[:15]
    most_poor = sorted(
        grid,
        key=lambda cell: cell["terra_poor_or_distressed_pct"],
        reverse=True,
    )[:15]
    worsened = sorted(
        comparable, key=lambda cell: cell["mean_score_delta"], reverse=True
    )[:15]
    improved = sorted(comparable, key=lambda cell: cell["mean_score_delta"])[:15]

    terra_scores = [int(row["terra_score"]) for row in results]
    rated = [row for row in results if int(row["baseline_score"]) in range(1, 6)]
    deltas = [int(row["score_delta"]) for row in rated]
    summary = {
        "properties": len(results),
        "mapped_properties": sum(bool(row["x"] and row["y"]) for row in results),
        "terra_mean": round(sum(terra_scores) / len(terra_scores), 3),
        "terra_poor_or_distressed_pct": round(
            100 * sum(score >= 4 for score in terra_scores) / len(terra_scores), 2
        ),
        "baseline_rated_properties": len(rated),
        "mean_score_delta": round(sum(deltas) / len(deltas), 3),
        "terra_worse_pct": round(100 * sum(delta > 0 for delta in deltas) / len(deltas), 2),
        "terra_better_pct": round(100 * sum(delta < 0 for delta in deltas) / len(deltas), 2),
        "large_disagreement_count": sum(abs(delta) >= 2 for delta in deltas),
        "medium_grid_cells": len(grid),
        "worst_condition_cells": worst,
        "highest_poor_share_cells": most_poor,
        "largest_worsening_cells": worsened,
        "largest_improvement_cells": improved,
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# GPT-5.6 Terra Citywide Hotspot Analysis",
        "",
        "Grid resolution: approximately 550 m; cells require at least 10 properties.",
        "",
        "## Citywide summary",
        "",
        f"- Properties scored: **{summary['properties']:,}**",
        f"- Properties mapped: **{summary['mapped_properties']:,}**",
        f"- Terra mean score: **{summary['terra_mean']:.2f}**",
        f"- Terra score 4–5: **{summary['terra_poor_or_distressed_pct']:.1f}%**",
        f"- Mean Terra − City score change: **{summary['mean_score_delta']:+.2f}**",
        f"- Terra higher/worse than City: **{summary['terra_worse_pct']:.1f}%**",
        f"- Terra lower/better than City: **{summary['terra_better_pct']:.1f}%**",
        f"- Differences of at least two points: **{summary['large_disagreement_count']:,}**",
        "",
    ]
    lines += table(
        "Top 15 worst-condition cells",
        worst,
        "Score 4–5 %",
        "terra_poor_or_distressed_pct",
    )
    lines += table(
        "Top 15 cells by score 4–5 share",
        most_poor,
        "Score 4–5 %",
        "terra_poor_or_distressed_pct",
    )
    lines += table(
        "Top 15 cells where Terra is worse than City",
        worsened,
        "Mean delta",
        "mean_score_delta",
    )
    lines += table(
        "Top 15 cells where Terra is better than City",
        improved,
        "Mean delta",
        "mean_score_delta",
    )
    lines += [
        "## Interpretation",
        "",
        "- A higher score means worse property condition.",
        "- Positive delta means Terra assigns a worse score than the City baseline.",
        "- Negative delta means Terra assigns a better score than the City baseline.",
        "- Grid hotspots identify areas for follow-up; they are not administrative neighborhood boundaries.",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if not isinstance(value, list)}, indent=2))
    print(REPORT_MD)
    print(REPORT_JSON)


if __name__ == "__main__":
    main()
