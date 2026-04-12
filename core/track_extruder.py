"""Extrude the GPX track into a 3D ribbon mesh using a parallel transport frame."""

from __future__ import annotations

import numpy as np
import trimesh
import trimesh.repair

from config import TRACK_RAISE_MM as TRACK_RAISE_M, GROOVE_WIDTH_MM

TRACK_WIDTH_M = GROOVE_WIDTH_MM - 0.4  # groove - 2×0.2mm tolerance


def _parallel_transport_frame(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-point normal and binormal using the parallel transport method.

    Returns (normals, binormals) arrays of shape (N, 3).
    """
    n = len(tangents)
    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))

    # Bootstrap: find a vector not parallel to tangents[0]
    t0 = tangents[0]
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    normals[0] = np.cross(t0, up)
    normals[0] /= np.linalg.norm(normals[0]) + 1e-12
    binormals[0] = np.cross(t0, normals[0])
    binormals[0] /= np.linalg.norm(binormals[0]) + 1e-12

    for i in range(1, n):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        axis = np.cross(t_prev, t_curr)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-10:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
        else:
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Rodrigues rotation
            def _rotate(v: np.ndarray) -> np.ndarray:
                return (cos_a * v
                        + sin_a * np.cross(axis, v)
                        + (1 - cos_a) * np.dot(axis, v) * axis)
            normals[i] = _rotate(normals[i - 1])
            normals[i] /= np.linalg.norm(normals[i]) + 1e-12
            binormals[i] = np.cross(t_curr, normals[i])
            binormals[i] /= np.linalg.norm(binormals[i]) + 1e-12

    return normals, binormals


def _octagon_offsets(half_width: float, half_height: float, n_sides: int = 8) -> np.ndarray:
    """
    Return (n_sides, 2) local cross-section offsets in the (normal, binormal) plane.
    The cross-section is elliptical to give a domed top and flat-ish bottom.
    """
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    # n-axis = lateral (width), b-axis = vertical (height/up)
    offsets = np.column_stack([
        half_width * np.cos(angles),
        half_height * np.sin(angles),
    ])
    return offsets


def extrude_track(
    track_utm: np.ndarray,          # (M, 2+) UTM XY[Z] of resampled track
    groove_floor_z: np.ndarray,     # (M,) Z of groove floor at each track point
    track_width_m: float = TRACK_WIDTH_M,
    track_raise_m: float = TRACK_RAISE_M,
    n_sides: int = 8,
) -> trimesh.Trimesh:
    """
    Sweep an octagonal cross-section along the track using a parallel transport frame.

    The bottom of the cross-section sits at groove_floor_z + track_raise_m.
    Returns a watertight trimesh.Trimesh.
    """
    pts = track_utm[:, :2].astype(np.float64)
    n = len(pts)

    # 3-D spine: XY from track, Z from groove floor
    spine = np.column_stack([pts[:, 0], pts[:, 1], groove_floor_z.astype(np.float64)])

    # Tangent vectors (central differences)
    tangents = np.zeros((n, 3))
    tangents[1:-1] = spine[2:] - spine[:-2]
    tangents[0] = spine[1] - spine[0]
    tangents[-1] = spine[-1] - spine[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents /= norms

    normals, binormals = _parallel_transport_frame(tangents)

    # Cross-section: lateral = normal direction (horizontal), vertical = up
    half_w = track_width_m / 2.0
    half_h = track_raise_m / 2.0 + 0.5  # dome height above groove floor

    offsets = _octagon_offsets(half_w, half_h, n_sides)  # (n_sides, 2)

    # Build all cross-section vertices
    # At each spine point, cross-section centre is at spine + Z offset for raise
    # The flat bottom sits at groove_floor_z; raise the centre by half_h
    all_verts = []
    for i in range(n):
        centre = spine[i].copy()
        centre[2] += half_h  # lift centre so bottom of cross-section ≈ groove_floor_z

        ring = []
        for off in offsets:
            v = centre + off[0] * normals[i] + off[1] * binormals[i]
            ring.append(v)
        all_verts.append(ring)

    verts_arr = np.array(all_verts, dtype=np.float64)  # (n, n_sides, 3)
    verts_flat = verts_arr.reshape(-1, 3)               # (n * n_sides, 3)

    # Build tube faces
    faces = []

    def _vidx(section: int, corner: int) -> int:
        return section * n_sides + (corner % n_sides)

    for i in range(n - 1):
        for j in range(n_sides):
            a = _vidx(i, j)
            b = _vidx(i, j + 1)
            c = _vidx(i + 1, j)
            d = _vidx(i + 1, j + 1)
            faces += [[a, b, c], [b, d, c]]

    # End caps (triangulated fan from centroid)
    # Start cap
    start_centre_idx = len(verts_flat)
    start_centre = spine[0].copy()
    start_centre[2] += half_h
    cap_verts = [start_centre]

    end_centre_idx = len(verts_flat) + 1
    end_centre = spine[-1].copy()
    end_centre[2] += half_h
    cap_verts.append(end_centre)

    extra_verts = np.array(cap_verts, dtype=np.float64)
    verts_flat = np.concatenate([verts_flat, extra_verts], axis=0)

    for j in range(n_sides):
        a = _vidx(0, j)
        b = _vidx(0, j + 1)
        faces.append([start_centre_idx, b, a])  # reversed for outward normal

    for j in range(n_sides):
        a = _vidx(n - 1, j)
        b = _vidx(n - 1, j + 1)
        faces.append([end_centre_idx, a, b])

    faces_arr = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts_flat, faces=faces_arr, process=True)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)

    return mesh
