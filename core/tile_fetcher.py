"""Fetch and stitch satellite imagery tiles for a bounding box."""

from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

from config import BING_TILE_URL, ESRI_TILE_URL, MAPBOX_TOKEN, MAPBOX_TILE_URL, TEXTURE_SIZE, TILE_ZOOM

_TILE_SIZE = 256  # pixels per tile


# ---------------------------------------------------------------------------
# Tile coordinate helpers (Web Mercator / EPSG:3857)
# ---------------------------------------------------------------------------

def _lat_lon_to_tile(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 to tile XY at given zoom level."""
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile XY (top-left corner) to WGS84 lat/lon."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_r)
    return lat, lon


# ---------------------------------------------------------------------------
# Single-tile fetcher with retry
# ---------------------------------------------------------------------------

def _to_quadkey(x: int, y: int, z: int) -> str:
    """Convert tile XY + zoom to a Bing Maps quadkey string."""
    quadkey = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


def _fetch_tile(x: int, y: int, zoom: int, source: str = "esri") -> Image.Image | None:
    if source == "mapbox" and MAPBOX_TOKEN:
        url = MAPBOX_TILE_URL.format(z=zoom, x=x, y=y, token=MAPBOX_TOKEN)
    elif source == "bing":
        url = BING_TILE_URL.format(s=(x + y) % 4, quadkey=_to_quadkey(x, y, zoom))
    else:
        url = ESRI_TILE_URL.format(z=zoom, y=y, x=x)

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_satellite_image(
    bbox_padded: tuple,
    zoom: int = TILE_ZOOM,
    texture_size: int = TEXTURE_SIZE,
    output_path: str | None = None,
    source: str = "esri",
) -> tuple[Image.Image, dict]:
    """
    Fetch and stitch satellite tiles for bbox_padded.

    Returns (image, geo_info) where geo_info holds the pixel→latlon mapping.
    image is an RGB PIL Image at texture_size × texture_size.
    If output_path is provided the JPEG is saved there.
    """
    south, north, west, east = (
        bbox_padded[0], bbox_padded[1], bbox_padded[2], bbox_padded[3]
    )

    # Tile range
    x_min, y_max = _lat_lon_to_tile(south, west, zoom)  # y_max because south → higher tile Y
    x_max, y_min = _lat_lon_to_tile(north, east, zoom)
    x_min = max(0, x_min)
    x_max = min(2 ** zoom - 1, x_max)
    y_min = max(0, y_min)
    y_max = min(2 ** zoom - 1, y_max)

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    canvas = Image.new("RGB", (cols * _TILE_SIZE, rows * _TILE_SIZE), color=(128, 128, 128))

    # Fetch concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                fut = pool.submit(_fetch_tile, tx, ty, zoom, source)
                futures[fut] = (tx - x_min, ty - y_min)

        for fut in as_completed(futures):
            tile_img = fut.result()
            col_idx, row_idx = futures[fut]
            if tile_img is not None:
                canvas.paste(tile_img, (col_idx * _TILE_SIZE, row_idx * _TILE_SIZE))

    # Pixel extent of the stitched canvas in lat/lon
    canvas_north, canvas_west = _tile_to_lat_lon(x_min, y_min, zoom)
    canvas_south, canvas_east = _tile_to_lat_lon(x_max + 1, y_max + 1, zoom)

    canvas_w_px = cols * _TILE_SIZE
    canvas_h_px = rows * _TILE_SIZE

    # Crop to bbox
    def _lon_to_px(lon: float) -> int:
        return int((lon - canvas_west) / (canvas_east - canvas_west) * canvas_w_px)

    def _lat_to_px(lat: float) -> int:
        # Latitude is mercator-projected; use linear approx at this scale
        return int((canvas_north - lat) / (canvas_north - canvas_south) * canvas_h_px)

    left = max(0, _lon_to_px(west))
    right = min(canvas_w_px, _lon_to_px(east))
    top = max(0, _lat_to_px(north))
    bottom = min(canvas_h_px, _lat_to_px(south))

    cropped = canvas.crop((left, top, right, bottom))
    resized = cropped.resize((texture_size, texture_size), Image.LANCZOS)

    if output_path:
        resized.save(output_path, "JPEG", quality=90)

    geo_info = {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "width_px": texture_size,
        "height_px": texture_size,
    }

    return resized, geo_info


def compute_uv(lats: np.ndarray, lons: np.ndarray, geo_info: dict) -> np.ndarray:
    """
    Compute UV texture coordinates for a set of lat/lon vertices.

    Returns array of shape (N, 2) with u in [0,1] and v in [0,1].
    v=0 is bottom (south), v=1 is top (north) in OpenGL convention.
    """
    u = (lons - geo_info["west"]) / (geo_info["east"] - geo_info["west"])
    # V is inverted: Three.js loads textures with flipY=true (OpenGL convention),
    # so v=0 maps to the TOP of the source image (north) and v=1 to the bottom (south).
    v = 1.0 - (lats - geo_info["south"]) / (geo_info["north"] - geo_info["south"])
    uv = np.stack([u, v], axis=-1)
    return np.clip(uv, 0.0, 1.0)
