"""Build a watertight terrain mesh from an elevation grid and satellite image."""

from __future__ import annotations

import math

import numpy as np
import trimesh
import trimesh.repair
from PIL import Image
from pyproj import CRS, Transformer

from config import BASE_THICKNESS_M
from core.tile_fetcher import compute_uv


def _get_utm_transformer(bbox_padded: tuple) -> Transformer:
    """Create a WGS84 → UTM transformer auto-selected from the bbox centre."""
    centre_lat = (bbox_padded[0] + bbox_padded[1]) / 2
    centre_lon = (bbox_padded[2] + bbox_padded[3]) / 2
    zone = int((centre_lon + 180) / 6) + 1
    south_hemi = centre_lat < 0
    utm_crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": south_hemi})
    return Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)


def apply_hex_shape(
    mesh: trimesh.Trimesh,
    satellite_image: Image.Image,
) -> trimesh.Trimesh:
    """
    Clip the mesh to a regular flat-top hexagon fitted inside the XY bounding box.

    UV texture coordinates are recomputed after slicing because trimesh's
    slice_mesh_plane discards TextureVisuals.
    """
    verts = mesh.vertices
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # Largest flat-top regular hexagon that fits inside the rectangular bbox:
    #   flat-to-flat height = 2r  ≤  y_extent  →  r ≤ y_extent / 2
    #   vertex-to-vertex width = 4r/√3  ≤  x_extent  →  r ≤ x_extent × √3/4
    # Leave a 2 % margin so the walls don't sit exactly on the mesh boundary.
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    inradius = min(y_extent / 2.0, x_extent * math.sqrt(3) / 4.0) * 0.98

    # Drop visuals before slicing (they won't survive and would cause errors)
    plain = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)

    for i in range(6):
        angle = math.radians(30.0 + i * 60.0)
        nx, ny = math.cos(angle), math.sin(angle)
        plain = trimesh.intersections.slice_mesh_plane(
            plain,
            plane_normal=np.array([-nx, -ny, 0.0]),   # inward → keep interior
            plane_origin=np.array([cx + inradius * nx, cy + inradius * ny, 0.0]),
            cap=True,
        )

    # Reattach texture: UV is linear in XY so we can recover it from vertex positions
    v = plain.vertices
    u   =       (v[:, 0] - x_min) / (x_max - x_min)
    v_t = 1.0 - (v[:, 1] - y_min) / (y_max - y_min)
    uv  = np.clip(np.column_stack([u, v_t]), 0.0, 1.0)
    plain.visual = trimesh.visual.TextureVisuals(uv=uv, image=satellite_image)

    return plain


def build_terrain_mesh(
    elevation_grid: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    bbox_padded: tuple,
    satellite_image: Image.Image,
    base_thickness_m: float = BASE_THICKNESS_M,
) -> tuple[trimesh.Trimesh, Transformer]:
    """
    Build a watertight textured terrain mesh.

    Returns (mesh, utm_transformer) so callers can reuse the projection.
    The mesh is in UTM metric space (metres).
    """
    R = len(lat_coords)
    C = len(lon_coords)

    transformer = _get_utm_transformer(bbox_padded)

    # --- Build vertex positions in UTM ---
    grid_lons, grid_lats = np.meshgrid(lon_coords, lat_coords)  # shape (R, C)
    flat_lons = grid_lons.ravel()
    flat_lats = grid_lats.ravel()

    utm_x, utm_y = transformer.transform(flat_lons, flat_lats)
    utm_x = utm_x.astype(np.float64)
    utm_y = utm_y.astype(np.float64)
    utm_z = elevation_grid.ravel().astype(np.float64)

    n_surface = R * C
    vertices_surface = np.column_stack([utm_x, utm_y, utm_z])

    # --- UV coordinates ---
    uv_surface = compute_uv(flat_lats, flat_lons, {
        "south": bbox_padded[0],
        "north": bbox_padded[1],
        "west": bbox_padded[2],
        "east": bbox_padded[3],
    })

    # --- Surface faces (vectorised quad → 2 triangles) ---
    i_idx, j_idx = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()

    v0 = i_idx * C + j_idx
    v1 = (i_idx + 1) * C + j_idx
    v2 = i_idx * C + (j_idx + 1)
    v3 = (i_idx + 1) * C + (j_idx + 1)

    faces_surface = np.concatenate([
        np.stack([v0, v2, v1], axis=1),
        np.stack([v1, v2, v3], axis=1),
    ], axis=0)

    # --- Solid base ---
    z_min = utm_z.min()
    z_base = z_min - abs(base_thickness_m)

    # Bottom face vertices: same XY as surface, Z = z_base
    vertices_bottom = np.column_stack([utm_x, utm_y, np.full(n_surface, z_base)])
    uv_bottom = uv_surface.copy()
    base_offset = n_surface  # index offset for bottom vertices

    # Bottom faces (reversed winding so normals point -Z / outward downward)
    faces_bottom = np.stack([
        v0 + base_offset,
        v1 + base_offset,
        v2 + base_offset,
    ], axis=1)
    faces_bottom2 = np.stack([
        v1 + base_offset,
        v3 + base_offset,
        v2 + base_offset,
    ], axis=1)
    faces_bottom_all = np.concatenate([faces_bottom, faces_bottom2], axis=0)

    # --- Side walls (perimeter) ---
    # Four edges: north, south, west, east
    side_verts = []
    side_faces = []

    def _add_wall(top_indices: np.ndarray, reverse: bool = False) -> None:
        """Connect a sequence of surface top_indices to their bottom counterparts."""
        base_v_start = len(vertices_surface) + len(vertices_bottom) + sum(len(s) for s in side_verts)
        n = len(top_indices)
        top_pts = vertices_surface[top_indices]
        bot_pts = top_pts.copy()
        bot_pts[:, 2] = z_base

        side_verts.append(np.concatenate([top_pts, bot_pts], axis=0))
        uv_side = np.zeros((2 * n, 2))
        side_verts.append(uv_side)  # placeholder UV

        for k in range(n - 1):
            t0 = base_v_start + k
            t1 = base_v_start + k + 1
            b0 = base_v_start + n + k
            b1 = base_v_start + n + k + 1
            if reverse:
                side_faces.append([t0, b0, t1])
                side_faces.append([t1, b0, b1])
            else:
                side_faces.append([t0, t1, b0])
                side_faces.append([t1, b1, b0])

    # Build perimeter index sequences
    south_row = np.arange(C)                         # row 0
    north_row = np.arange((R - 1) * C, R * C)       # row R-1
    west_col = np.arange(0, R * C, C)               # col 0
    east_col = np.arange(C - 1, R * C, C)           # col C-1

    # Simpler approach: stitch walls directly without per-wall vertex duplication
    # Use a flat wall approach: top surface edge + base at z_base
    def _wall_faces(top_idx_seq: np.ndarray, out_offset: int, reverse: bool = False):
        n = len(top_idx_seq)
        bot_idx_seq = top_idx_seq + out_offset
        wf = []
        for k in range(n - 1):
            t0, t1 = int(top_idx_seq[k]), int(top_idx_seq[k + 1])
            b0, b1 = int(bot_idx_seq[k]), int(bot_idx_seq[k + 1])
            if reverse:
                wf += [[t0, b0, t1], [t1, b0, b1]]
            else:
                wf += [[t0, t1, b0], [t1, b1, b0]]
        return wf

    # Bottom vertices are at base_offset = n_surface
    wall_faces = []
    wall_faces += _wall_faces(south_row, base_offset, reverse=True)
    wall_faces += _wall_faces(north_row, base_offset, reverse=False)
    wall_faces += _wall_faces(west_col, base_offset, reverse=False)
    wall_faces += _wall_faces(east_col, base_offset, reverse=True)

    wall_faces_arr = np.array(wall_faces, dtype=np.int64) if wall_faces else np.empty((0, 3), dtype=np.int64)

    # --- Assemble full mesh ---
    all_vertices = np.concatenate([vertices_surface, vertices_bottom], axis=0)
    all_uv = np.concatenate([uv_surface, uv_bottom], axis=0)
    all_faces = np.concatenate([
        faces_surface,
        faces_bottom_all,
        wall_faces_arr,
    ], axis=0)

    mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_faces,
        process=False,
    )

    # Attach texture
    texture_vis = trimesh.visual.TextureVisuals(
        uv=all_uv,
        image=satellite_image,
    )
    mesh.visual = texture_vis

    # Repair
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)

    return mesh, transformer
