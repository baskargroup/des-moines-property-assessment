"""Build the publication figure: mean GPT-5.6 Terra score per 2020 Census block group.

Needs build_block_group_scores.py (runs/citywide_block_groups/block_groups.geojson) first.
Class breaks follow the City of Des Moines block-group symbology for survey averages.
"""
from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle

from build_paper_terra_grid_figure import (
    REFERENCE_URL, BASE_URL, TILE_SIZE, ZOOM, download_mosaic, world_pixel,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs" / "citywide_block_groups" / "block_groups.geojson"
OUTPUT_DIR = HERE / "figures"
PNG_OUT = OUTPUT_DIR / "citywide_terra_block_groups.png"
PDF_OUT = OUTPUT_DIR / "citywide_terra_block_groups.pdf"
MIN_PROPERTIES = 10

CLASSES = [
    (float("-inf"), 2.0, "< 2.0", "#176B55"),
    (2.0, 2.5, "2.0 – 2.49", "#69A878"),
    (2.5, 3.0, "2.5 – 2.99", "#C9C65A"),
    (3.0, 3.5, "3.0 – 3.49", "#E1A043"),
    (3.5, 4.0, "3.5 – 3.99", "#D06A35"),
    (4.0, float("inf"), "≥ 4.0", "#A93B3F"),
]
SUPPRESSED = "#D9DDDB"


def class_color(score: float) -> str:
    for low, high, _, color in CLASSES:
        if low <= score < high:
            return color
    raise ValueError(score)


def rings(geometry):
    polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        yield list(polygon.exterior.coords)


def main() -> None:
    zones = gpd.read_file(SOURCE)
    zones["shown"] = (zones["n"] >= MIN_PROPERTIES) & zones["terra_mean"].notna()

    west, south, east, north = zones.loc[zones["shown"]].total_bounds
    margin = 0.008
    crop_left, crop_top = world_pixel(west - margin, north + margin)
    crop_right, crop_bottom = world_pixel(east + margin, south - margin)
    x_min, x_max = math.floor(crop_left / TILE_SIZE), math.floor(crop_right / TILE_SIZE)
    y_min, y_max = math.floor(crop_top / TILE_SIZE), math.floor(crop_bottom / TILE_SIZE)
    origin_x, origin_y = x_min * TILE_SIZE, y_min * TILE_SIZE
    crop_box = (round(crop_left - origin_x), round(crop_top - origin_y),
                round(crop_right - origin_x), round(crop_bottom - origin_y))
    base = download_mosaic(BASE_URL, x_min, x_max, y_min, y_max).crop(crop_box)
    reference = download_mosaic(REFERENCE_URL, x_min, x_max, y_min, y_max).crop(crop_box)
    width, height = base.size

    def to_axes(coords):
        out = []
        for lon, lat, *_ in coords:
            px, py = world_pixel(lon, lat)
            out.append((px - crop_left, py - crop_top))
        return out

    fig, ax = plt.subplots(figsize=(9.4, 6.4), facecolor="white")
    ax.imshow(base, extent=(0, width, height, 0), zorder=0)
    for _, zone in zones.iterrows():
        face = class_color(zone["terra_mean"]) if zone["shown"] else SUPPRESSED
        for ring in rings(zone.geometry):
            ax.add_patch(MplPolygon(to_axes(ring), closed=True, facecolor=face, edgecolor="white",
                                    linewidth=0.45, alpha=0.78 if zone["shown"] else 0.6, zorder=2))
    ax.imshow(reference, extent=(0, width, height, 0), zorder=3)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0, 1.065, "Citywide property condition", transform=ax.transAxes, fontsize=19,
            fontweight="bold", color="#202725", ha="left")
    ax.text(0, 1.022, "Des Moines, Iowa · mean GPT-5.6 Terra score by 2020 Census block group",
            transform=ax.transAxes, fontsize=10.5, color="#52605C", ha="left")

    ax.annotate("N", xy=(0.955, 0.91), xytext=(0.955, 0.82), xycoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#28312E",
                arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#28312E"}, zorder=5)

    mean_latitude = (north + south) / 2
    metres_per_pixel = (math.cos(math.radians(mean_latitude)) * 2 * math.pi * 6_378_137
                        / (TILE_SIZE * 2**ZOOM))
    x0 = width * 0.06
    scale_bars = [(1_000 / metres_per_pixel, height * 0.915, "1 km"),
                  (1_609.344 / metres_per_pixel, height * 0.958, "1 mile")]
    ax.add_patch(Rectangle((x0 - 16, height * 0.875), scale_bars[1][0] + 32, height * 0.11,
                           facecolor="white", edgecolor="none", alpha=0.8, zorder=4))
    for pixels, y0, label in scale_bars:
        ax.plot([x0, x0 + pixels], [y0, y0], color="#202725", linewidth=2.2, zorder=5)
        for x in (x0, x0 + pixels):
            ax.plot([x, x], [y0 - 5, y0 + 5], color="#202725", linewidth=1.2, zorder=5)
        ax.text(x0 + pixels / 2, y0 - 10, label, ha="center", va="bottom", fontsize=8.5,
                color="#202725", zorder=5)

    counts = {label: 0 for _, _, label, _ in CLASSES}
    for score in zones.loc[zones["shown"], "terra_mean"]:
        for low, high, label, _ in CLASSES:
            if low <= score < high:
                counts[label] += 1
    handles = [Patch(facecolor=color, edgecolor="white", alpha=0.85, label=f"{label}  ({counts[label]})")
               for _, _, label, color in CLASSES]
    handles.append(Patch(facecolor=SUPPRESSED, edgecolor="white",
                         label=f"< {MIN_PROPERTIES} properties  ({int((~zones['shown']).sum())})"))
    legend = ax.legend(handles=handles, title="Mean score", loc="lower left", bbox_to_anchor=(1.01, 0.0),
                       fontsize=8.5, title_fontsize=9, frameon=True, framealpha=0.9,
                       edgecolor="none", borderpad=0.8)
    legend.set_zorder(6)

    ax.text(0, -0.03,
             "1 = very healthy, 5 = unhealthy · "
             f"{int(zones['shown'].sum())} of {len(zones)} block groups shown · "
             "Basemap © Esri, HERE, Garmin and OpenStreetMap contributors",
             transform=ax.transAxes, fontsize=8, color="#66726E", va="top")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
