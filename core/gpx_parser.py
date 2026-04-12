"""Parse a GPX file into trackpoints, bounding box, and a resampled path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gpxpy
import numpy as np

from config import BBOX_PADDING, MAX_RESAMPLED_POINTS, RESAMPLE_STEP_M


@dataclass
class GPXData:
    raw_points: np.ndarray          # shape (N, 3): lat, lon, ele
    resampled_points: np.ndarray    # shape (M, 3): evenly spaced every ~RESAMPLE_STEP_M
    bbox: tuple                     # (min_lat, max_lat, min_lon, max_lon)
    bbox_padded: tuple              # expanded bbox
    total_distance_m: float
    ele_gain_m: float
    name: str
    has_elevation: bool = True


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length array for a (N,3) lat/lon/ele array."""
    dists = [0.0]
    for i in range(1, len(points)):
        d = _haversine_m(points[i - 1, 0], points[i - 1, 1],
                         points[i, 0], points[i, 1])
        dists.append(dists[-1] + d)
    return np.array(dists)


def _resample(points: np.ndarray, step_m: float, max_points: int) -> np.ndarray:
    """Resample track to evenly spaced points every step_m metres."""
    cum = _cumulative_distances(points)
    total = cum[-1]
    if total == 0:
        return points[:1]

    n = min(int(total / step_m) + 1, max_points)
    target = np.linspace(0, total, n)

    resampled = np.empty((n, 3))
    for col in range(3):
        resampled[:, col] = np.interp(target, cum, points[:, col])
    return resampled


def parse_gpx(filepath: str) -> GPXData:
    """Parse a GPX file and return a GPXData instance."""
    with open(filepath, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)

    points: list[tuple[float, float, float]] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                ele = pt.elevation if pt.elevation is not None else float("nan")
                points.append((pt.latitude, pt.longitude, ele))

    # Also collect waypoints / routes if no tracks
    if not points:
        for route in gpx.routes:
            for pt in route.points:
                ele = pt.elevation if pt.elevation is not None else float("nan")
                points.append((pt.latitude, pt.longitude, ele))

    if not points:
        raise ValueError("GPX file contains no track points.")

    raw = np.array(points, dtype=float)

    has_elevation = not np.all(np.isnan(raw[:, 2]))
    if not has_elevation:
        raw[:, 2] = 0.0  # will be filled from SRTM later

    # Bounding box
    min_lat, max_lat = raw[:, 0].min(), raw[:, 0].max()
    min_lon, max_lon = raw[:, 1].min(), raw[:, 1].max()
    lat_pad = (max_lat - min_lat) * BBOX_PADDING
    lon_pad = (max_lon - min_lon) * BBOX_PADDING
    # Ensure minimum padding so very short tracks still get reasonable terrain
    lat_pad = max(lat_pad, 0.005)
    lon_pad = max(lon_pad, 0.005)
    bbox = (min_lat, max_lat, min_lon, max_lon)
    bbox_padded = (
        min_lat - lat_pad,
        max_lat + lat_pad,
        min_lon - lon_pad,
        max_lon + lon_pad,
    )

    # Total distance
    total_distance_m = _cumulative_distances(raw)[-1]

    # Elevation gain
    ele_diffs = np.diff(raw[:, 2])
    ele_gain_m = float(ele_diffs[ele_diffs > 0].sum()) if has_elevation else 0.0

    # Track name
    name = "GPX Track"
    if gpx.tracks and gpx.tracks[0].name:
        name = gpx.tracks[0].name

    # Resample
    resampled = _resample(raw, RESAMPLE_STEP_M, MAX_RESAMPLED_POINTS)

    return GPXData(
        raw_points=raw,
        resampled_points=resampled,
        bbox=bbox,
        bbox_padded=bbox_padded,
        total_distance_m=total_distance_m,
        ele_gain_m=ele_gain_m,
        name=name,
        has_elevation=has_elevation,
    )
