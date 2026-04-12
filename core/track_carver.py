"""Carve a groove into the terrain mesh along the GPX track path."""

from __future__ import annotations

import numpy as np
import trimesh
from pyproj import Transformer
from scipy.spatial import cKDTree

from config import GROOVE_DEPTH_MM, GROOVE_WIDTH_MM, TERRAIN_GRID_RESOLUTION


def _point_to_segment_distance_2d(px: np.ndarray, py: np.ndarray,
                                   ax: float, ay: float,
                                   bx: float, by: float) -> np.ndarray:
    """
    Vectorised minimum distance from points (px,py) to segment AB in 2D.
    Returns 1-D array of distances.
    """
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / seg_len_sq, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    return np.hypot(px - cx, py - cy)


def carve_groove(
    mesh: trimesh.Trimesh,
    track_pts: np.ndarray,               # (M, 2) in the same coordinate space as mesh vertices
    transformer: Transformer,            # kept for API consistency
    groove_width: float = GROOVE_WIDTH_MM,
    groove_depth: float = GROOVE_DEPTH_MM,
    resolution: int = TERRAIN_GRID_RESOLUTION,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    Displace terrain vertices downward to create a groove along the track.

    ``mesh`` vertices and ``track_pts`` must be in the same coordinate space
    (e.g. both in print-space mm after the UTM→mm scaling step in app.py).

    Returns (modified_mesh, groove_floor_z) where groove_floor_z is a 1-D
    array of Z values at each track point (the groove floor elevation).
    """
    verts = mesh.vertices.copy()
    half_width = groove_width / 2.0

    # Surface vertices are the first resolution² entries (mesh_builder convention).
    n_surface = resolution * resolution
    surface_verts = verts[:n_surface]

    vx = surface_verts[:, 0]
    vy = surface_verts[:, 1]

    track_xy = track_pts[:, :2].astype(np.float64)

    # Derive a safe candidate radius from the actual track point spacing.
    # This avoids the hard-coded 1.5× multiplier that failed when track spacing
    # exceeded half_width (segments would be missed entirely).
    if len(track_xy) > 1:
        avg_step = float(np.mean(np.linalg.norm(np.diff(track_xy, axis=0), axis=1)))
    else:
        avg_step = 0.0
    # candidate_radius must be large enough so that for any point within half_width
    # of a segment, at least one segment endpoint falls inside the radius.
    # Worst case: point is perpendicular to segment midpoint → endpoint dist = avg_step/2.
    candidate_radius = half_width + avg_step

    tree = cKDTree(track_xy)
    indices_near = tree.query_ball_point(np.column_stack([vx, vy]), r=candidate_radius)

    displacements = np.zeros(n_surface)

    for vi in range(n_surface):
        near = indices_near[vi]
        if not near:
            continue

        # Check distance to each segment whose start index is in `near`.
        # Also check the preceding segment (seg_i - 1) so that the last point
        # of a segment is not missed when only the end vertex is in `near`.
        seg_candidates: set[int] = set()
        for pt_i in near:
            if pt_i > 0:
                seg_candidates.add(pt_i - 1)   # segment ending at pt_i
            if pt_i + 1 < len(track_xy):
                seg_candidates.add(pt_i)        # segment starting at pt_i

        min_dist = half_width + 1.0
        for seg_i in seg_candidates:
            ax, ay = track_xy[seg_i]
            bx, by = track_xy[seg_i + 1]
            d = float(_point_to_segment_distance_2d(
                np.array([vx[vi]]), np.array([vy[vi]]), ax, ay, bx, by
            )[0])
            if d < min_dist:
                min_dist = d

        if min_dist < half_width:
            # Cosine-squared cross-section profile → smooth groove edges
            t = min_dist / half_width   # 0 at centre, 1 at edge
            depth = groove_depth * (np.cos(np.pi / 2 * t) ** 2)
            displacements[vi] = depth

    verts[:n_surface, 2] -= displacements

    mesh_copy = trimesh.Trimesh(vertices=verts, faces=mesh.faces,
                                 visual=mesh.visual, process=False)

    # Compute groove floor Z: for each track point, take the minimum Z of all
    # surface vertices within groove_width/2.  This correctly finds the deepest
    # point of the carved groove rather than a smooth average that is pulled up
    # by the many uncarved vertices surrounding the narrow channel.
    surf_v = mesh_copy.vertices[:n_surface]
    surface_xy = surf_v[:, :2]
    surface_z  = surf_v[:, 2]

    surf_tree = cKDTree(surface_xy)
    groove_floor_z = np.empty(len(track_xy))
    for i, pt in enumerate(track_xy):
        neighbours = surf_tree.query_ball_point(pt, r=half_width)
        if neighbours:
            groove_floor_z[i] = surface_z[neighbours].min()
        else:
            # No vertex inside groove radius — fall back to nearest vertex
            _, idx = surf_tree.query(pt)
            groove_floor_z[i] = surface_z[idx]

    return mesh_copy, groove_floor_z
