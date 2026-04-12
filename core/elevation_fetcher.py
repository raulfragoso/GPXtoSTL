"""Fetch SRTM elevation data for a bounding box and return a regular grid."""

from __future__ import annotations

import io
import struct
import time

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from config import TERRAIN_GRID_RESOLUTION, VERTICAL_EXAGGERATION

_OPENTOPO_URL = (
    "https://portal.opentopography.org/API/globaldem"
    "?demtype=SRTMGL1&south={south}&north={north}&west={west}&east={east}"
    "&outputFormat=GTiff&API_Key=demoapikeyot2022"
)
_OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
_OPENTOPODATA_BATCH = 100   # max locations per request
_OPENTOPODATA_DELAY = 1.1   # seconds between requests (rate limit: 1 req/s)


# ---------------------------------------------------------------------------
# GeoTIFF minimal reader (no rasterio / GDAL required)
# ---------------------------------------------------------------------------

def _read_geotiff_elevation(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a single-band float/int16/int32 GeoTIFF and return
    (elevation_2d, lat_1d, lon_1d) ordered south→north, west→east.

    This is a minimal parser that handles SRTM TIFF output from OpenTopography.
    Falls back to Pillow for any format it can't parse natively.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(io.BytesIO(data))
        arr = np.array(img, dtype=float)

        # OpenTopography GeoTIFF: check for geotransform in TIFF tags
        # Tag 33922 = ModelTiepointTag, Tag 33550 = ModelPixelScaleTag
        tiff_tags = img.tag_v2 if hasattr(img, "tag_v2") else {}
        scale_tag = tiff_tags.get(33550)
        tie_tag = tiff_tags.get(33922)

        if scale_tag and tie_tag:
            pixel_scale = scale_tag  # (dx, dy, dz)
            tiepoint = tie_tag       # (i, j, k, x, y, z)
            lon0 = tiepoint[3]
            lat0 = tiepoint[4]
            dx = pixel_scale[0]
            dy = pixel_scale[1]  # positive value; latitude decreases downward

            rows, cols = arr.shape
            lons = lon0 + np.arange(cols) * dx
            lats = lat0 - np.arange(rows) * dy  # north → south

            # Flip so lats go south→north
            arr = arr[::-1, :]
            lats = lats[::-1]
            return arr, lats, lons

        # No geotransform found — return with dummy coords
        rows, cols = arr.shape
        return arr, np.arange(rows, dtype=float), np.arange(cols, dtype=float)

    except Exception:
        raise ValueError("Could not parse GeoTIFF elevation data.")


# ---------------------------------------------------------------------------
# OpenTopography primary source
# ---------------------------------------------------------------------------

def _fetch_opentopo(south: float, north: float, west: float, east: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    url = _OPENTOPO_URL.format(south=south, north=north, west=west, east=east)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.content[:4] in (b"II*\x00", b"MM\x00*"):
            return _read_geotiff_elevation(resp.content)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# OpenTopoData fallback  (free, no auth, SRTM 30m)
# ---------------------------------------------------------------------------

def _fetch_opentopodata(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Query api.opentopodata.org for a flat list of (lat, lon) pairs.
    Returns a 1-D elevation array of the same length.
    Max 100 locations per request; enforces 1-second delay between requests.
    """
    elevations: list[float] = []
    n = len(lats)

    for i in range(0, n, _OPENTOPODATA_BATCH):
        batch_lats = lats[i : i + _OPENTOPODATA_BATCH]
        batch_lons = lons[i : i + _OPENTOPODATA_BATCH]
        loc_str = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in zip(batch_lats, batch_lons))

        batch_elevs: list[float] | None = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"{_OPENTOPODATA_URL}?locations={loc_str}",
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK":
                        batch_elevs = [
                            r["elevation"] if r["elevation"] is not None else 0.0
                            for r in data["results"]
                        ]
                        break
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1)

        if batch_elevs is None:
            elevations.extend([0.0] * len(batch_lats))
        else:
            # Pad if the response is shorter than expected
            if len(batch_elevs) < len(batch_lats):
                fill = float(np.mean(batch_elevs)) if batch_elevs else 0.0
                batch_elevs.extend([fill] * (len(batch_lats) - len(batch_elevs)))
            elevations.extend(batch_elevs[:len(batch_lats)])

        # Respect rate limit between batches
        if i + _OPENTOPODATA_BATCH < n:
            time.sleep(_OPENTOPODATA_DELAY)

    return np.array(elevations, dtype=float)


def _fallback_grid(south: float, north: float, west: float, east: float, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a coarse grid via OpenTopoData and interpolate to target resolution."""
    coarse = 32
    lats_1d = np.linspace(south, north, coarse)
    lons_1d = np.linspace(west, east, coarse)
    grid_lats, grid_lons = np.meshgrid(lats_1d, lons_1d, indexing="ij")

    elev_flat = _fetch_opentopodata(grid_lats.ravel(), grid_lons.ravel())
    elev_coarse = elev_flat.reshape(coarse, coarse)
    return elev_coarse, lats_1d, lons_1d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_elevation_grid(
    bbox_padded: tuple,
    resolution: int = TERRAIN_GRID_RESOLUTION,
    vertical_exaggeration: float = VERTICAL_EXAGGERATION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (elevation_grid, lat_coords, lon_coords) for the padded bbox.

    elevation_grid shape: (resolution, resolution), lats south→north, lons west→east.
    Values are in metres × vertical_exaggeration.
    lat_coords / lon_coords are 1-D arrays of length `resolution`.
    """
    south, north, west, east = (
        bbox_padded[0], bbox_padded[1], bbox_padded[2], bbox_padded[3]
    )

    raw_result = _fetch_opentopo(south, north, west, east)

    if raw_result is not None:
        raw_elev, raw_lats, raw_lons = raw_result
    else:
        raw_elev, raw_lats, raw_lons = _fallback_grid(south, north, west, east, resolution)

    # Interpolate to target resolution
    interp = RegularGridInterpolator(
        (raw_lats, raw_lons),
        raw_elev,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    out_lats = np.linspace(south, north, resolution)
    out_lons = np.linspace(west, east, resolution)
    grid_ll, grid_lo = np.meshgrid(out_lats, out_lons, indexing="ij")
    elevation_grid = interp((grid_ll, grid_lo)).astype(float)

    # Smooth SRTM noise
    elevation_grid = gaussian_filter(elevation_grid, sigma=1.5)

    # Apply vertical exaggeration
    elevation_grid *= vertical_exaggeration

    return elevation_grid, out_lats, out_lons
