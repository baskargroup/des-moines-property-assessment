"""Build the publication figure for the GPT-5.6 Terra citywide grid scores."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs" / "citywide_terra" / "delivery" / "terra_hotspots_multi.json"
OUTPUT_DIR = HERE / "figures"
PNG_OUT = OUTPUT_DIR / "citywide_terra_grid.png"
PDF_OUT = OUTPUT_DIR / "citywide_terra_grid.pdf"
ZOOM = 13
TILE_SIZE = 256
BASE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
REFERENCE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
)


def world_pixel(longitude: float, latitude: float) -> tuple[float, float]:
    scale = TILE_SIZE * (2**ZOOM)
    x = (longitude + 180) / 360 * scale
    sin_latitude = math.sin(math.radians(latitude))
    y = (
        0.5
        - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)
    ) * scale
    return x, y


def download_mosaic(
    url_template: str, x_min: int, x_max: int, y_min: int, y_max: int
) -> Image.Image:
    mosaic = Image.new(
        "RGBA",
        ((x_max - x_min + 1) * TILE_SIZE, (y_max - y_min + 1) * TILE_SIZE),
        (255, 255, 255, 0),
    )
    session = requests.Session()
    session.headers["User-Agent"] = "DataCity-paper-figure/1.0"
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            response = session.get(
                url_template.format(z=ZOOM, x=x, y=y), timeout=30
            )
            response.raise_for_status()
            tile = Image.open(io.BytesIO(response.content)).convert("RGBA")
            mosaic.alpha_composite(
                tile, ((x - x_min) * TILE_SIZE, (y - y_min) * TILE_SIZE)
            )
    return mosaic


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))["medium"]
    size = data["size"]
    cells = data["cells"]

    cmap = LinearSegmentedColormap.from_list(
        "condition",
        ["#176B55", "#69A878", "#E1C84B", "#DC8434", "#A93B3F"],
        N=256,
    )
    norm = Normalize(vmin=1, vmax=5)

    longitudes = [cell["longitude"] for cell in cells]
    latitudes = [cell["latitude"] for cell in cells]
    margin = size * 1.6
    west = min(longitudes) - margin
    east = max(longitudes) + margin
    south = min(latitudes) - margin
    north = max(latitudes) + margin
    crop_left, crop_top = world_pixel(west, north)
    crop_right, crop_bottom = world_pixel(east, south)
    x_min = math.floor(crop_left / TILE_SIZE)
    x_max = math.floor(crop_right / TILE_SIZE)
    y_min = math.floor(crop_top / TILE_SIZE)
    y_max = math.floor(crop_bottom / TILE_SIZE)
    tile_origin_x = x_min * TILE_SIZE
    tile_origin_y = y_min * TILE_SIZE

    base = download_mosaic(BASE_URL, x_min, x_max, y_min, y_max)
    reference = download_mosaic(REFERENCE_URL, x_min, x_max, y_min, y_max)
    crop_box = (
        round(crop_left - tile_origin_x),
        round(crop_top - tile_origin_y),
        round(crop_right - tile_origin_x),
        round(crop_bottom - tile_origin_y),
    )
    base = base.crop(crop_box)
    reference = reference.crop(crop_box)
    width, height = base.size

    fig, ax = plt.subplots(figsize=(8.2, 7.4), facecolor="white")
    ax.imshow(base, extent=(0, width, height, 0), zorder=0)

    for cell in cells:
        left, top = world_pixel(
            cell["longitude"] - size / 2, cell["latitude"] + size / 2
        )
        right, bottom = world_pixel(
            cell["longitude"] + size / 2, cell["latitude"] - size / 2
        )
        ax.add_patch(
            Rectangle(
                (left - crop_left, top - crop_top),
                right - left,
                bottom - top,
                facecolor=cmap(norm(cell["terra_mean"])),
                edgecolor="white",
                linewidth=0.32,
                alpha=0.72,
                zorder=2,
            )
        )
    ax.imshow(reference, extent=(0, width, height, 0), zorder=3)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0,
        1.065,
        "Citywide property condition",
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        color="#202725",
        ha="left",
    )
    ax.text(
        0,
        1.022,
        "Des Moines, Iowa · mean GPT-5.6 Terra score by ~550 m grid cell",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#52605C",
        ha="left",
    )

    # North arrow.
    ax.annotate(
        "N",
        xy=(0.955, 0.91),
        xytext=(0.955, 0.82),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#28312E",
        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#28312E"},
        zorder=5,
    )

    # Dual metric/imperial scale bars in Web Mercator pixels.
    mean_latitude = sum(latitudes) / len(latitudes)
    metres_per_pixel = (
        math.cos(math.radians(mean_latitude))
        * 2
        * math.pi
        * 6_378_137
        / (TILE_SIZE * 2**ZOOM)
    )
    x0 = width * 0.08
    scale_bars = [
        (1_000 / metres_per_pixel, height * 0.895, "1 km"),
        (1_609.344 / metres_per_pixel, height * 0.94, "1 mile"),
    ]
    ax.add_patch(
        Rectangle(
            (x0 - 16, height * 0.85),
            scale_bars[1][0] + 32,
            height * 0.125,
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            zorder=4,
        )
    )
    for scale_pixels, y0, label in scale_bars:
        ax.plot(
            [x0, x0 + scale_pixels],
            [y0, y0],
            color="#202725",
            linewidth=2.2,
            zorder=5,
        )
        ax.plot(
            [x0, x0],
            [y0 - 5, y0 + 5],
            color="#202725",
            linewidth=1.2,
            zorder=5,
        )
        ax.plot(
            [x0 + scale_pixels, x0 + scale_pixels],
            [y0 - 5, y0 + 5],
            color="#202725",
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            x0 + scale_pixels / 2,
            y0 - 10,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#202725",
            zorder=5,
        )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.038,
        pad=0.035,
        aspect=38,
    )
    colorbar.set_ticks([1, 2, 3, 4, 5])
    colorbar.set_ticklabels(
        ["1\nVery healthy", "2\nHealthy", "3\nIntermediate", "4\nSlipping", "5\nUnhealthy"]
    )
    colorbar.ax.tick_params(labelsize=8.5, length=0, pad=5)
    colorbar.outline.set_visible(False)
    colorbar.set_label(
        "Mean property-condition score within each grid cell",
        fontsize=9.5,
        color="#303936",
        labelpad=8,
    )

    fig.text(
        0.105,
        0.018,
        f"Cells contain at least {data['min_n']} properties · {len(cells):,} cells shown · "
        "Basemap © Esri, HERE, Garmin and OpenStreetMap contributors",
        fontsize=8,
        color="#66726E",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_OUT)
    print(PDF_OUT)


if __name__ == "__main__":
    main()
