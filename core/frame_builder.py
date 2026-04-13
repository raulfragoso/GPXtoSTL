"""Build a display frame STL that the terrain model slots into, with embossed text."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _render_heightmap(lines: list[str], width_px: int, height_px: int) -> np.ndarray:
    """Render text lines centred in a grayscale height map (float32, 0–1)."""
    img = Image.new("L", (width_px, height_px), 0)
    draw = ImageDraw.Draw(img)

    n = max(len(lines), 1)
    font_size = max(12, height_px // n - 6)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

    line_h = height_px // (n + 1)
    for i, line in enumerate(lines):
        y = (i + 1) * line_h
        draw.text((width_px // 2, y), line, font=font, fill=255, anchor="mm")

    return np.array(img, dtype=np.float32) / 255.0


def _build_text_relief(
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    y_face: float,
    lines: list[str],
    text_depth_mm: float,
    nx: int = 200,
    nz: int = 50,
) -> "trimesh.Trimesh":
    """
    Closed mesh of raised text on the plane y=y_face, protruding in -Y.

    Vertex layout:
      indices 0 .. nx*nz-1        → front surface (varying Y)
      indices nx*nz .. 2*nx*nz-1  → back surface (flat at y_face)
    """
    import trimesh

    hmap = _render_heightmap(lines, nx, nz)  # (nz, nx)

    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    XX, ZZ = np.meshgrid(xs, zs)  # (nz, nx)

    Y_front = y_face - hmap * text_depth_mm        # protrudes toward viewer
    Y_back = np.full((nz, nx), y_face, dtype=np.float32)

    verts_f = np.column_stack([XX.ravel(), Y_front.ravel(), ZZ.ravel()])
    verts_b = np.column_stack([XX.ravel(), Y_back.ravel(),  ZZ.ravel()])
    nv = nx * nz
    all_verts = np.concatenate([verts_f, verts_b], axis=0)

    faces = []

    # Front surface (normal -Y): CCW from -Y viewpoint
    for i in range(nz - 1):
        for j in range(nx - 1):
            v00 = i * nx + j
            v01 = i * nx + j + 1
            v10 = (i + 1) * nx + j
            v11 = (i + 1) * nx + j + 1
            faces += [[v00, v01, v10], [v01, v11, v10]]

    # Back surface (normal +Y): reversed
    for i in range(nz - 1):
        for j in range(nx - 1):
            b00 = nv + i * nx + j
            b01 = nv + i * nx + j + 1
            b10 = nv + (i + 1) * nx + j
            b11 = nv + (i + 1) * nx + j + 1
            faces += [[b00, b10, b01], [b01, b10, b11]]

    # Left edge (j=0, normal -X)
    for i in range(nz - 1):
        f0 = i * nx;         f1 = (i + 1) * nx
        b0 = nv + i * nx;   b1 = nv + (i + 1) * nx
        faces += [[f0, f1, b0], [f1, b1, b0]]

    # Right edge (j=nx-1, normal +X)
    for i in range(nz - 1):
        f0 = i * nx + nx - 1;       f1 = (i + 1) * nx + nx - 1
        b0 = nv + i * nx + nx - 1; b1 = nv + (i + 1) * nx + nx - 1
        faces += [[f0, b0, f1], [f1, b0, b1]]

    # Bottom edge (i=0, normal -Z)
    for j in range(nx - 1):
        f0 = j;       f1 = j + 1
        b0 = nv + j; b1 = nv + j + 1
        faces += [[f0, b0, f1], [f1, b0, b1]]

    # Top edge (i=nz-1, normal +Z)
    for j in range(nx - 1):
        f0 = (nz - 1) * nx + j;       f1 = (nz - 1) * nx + j + 1
        b0 = nv + (nz - 1) * nx + j; b1 = nv + (nz - 1) * nx + j + 1
        faces += [[f0, f1, b0], [f1, b1, b0]]

    mesh = trimesh.Trimesh(
        vertices=all_verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    trimesh.repair.fix_winding(mesh)
    return mesh


def build_display_frame(
    terrain_width_mm: float,
    terrain_depth_mm: float,
    frame_height_mm: float = 15.0,
    wall_thickness_mm: float = 5.0,
    text_lines: list[str] | None = None,
    text_depth_mm: float = 0.4,
    clearance_mm: float = 0.3,
) -> "trimesh.Trimesh":
    """
    Build a watertight display frame that the terrain model drops into.

    Coordinate system (matches terrain after terrain is centred in the opening):
      Inner opening:  x=[0 .. terrain_width],  y=[0 .. terrain_depth]
      Outer shell:    x=[-t .. W_in+t],         y=[-t .. D_in+t]
      Height:         z=[0 .. frame_height]

    The front face (y = -t) has the track info embossed in raised text.

    Returns a trimesh.Trimesh in mm.
    """
    import trimesh
    from shapely.geometry import Polygon

    t = wall_thickness_mm
    c = clearance_mm
    W_in = terrain_width_mm + 2 * c
    D_in = terrain_depth_mm + 2 * c
    H = frame_height_mm

    # Frame ring in XY (plan view), extruded along Z to height H
    outer = Polygon([(-t, -t), (W_in + t, -t), (W_in + t, D_in + t), (-t, D_in + t)])
    inner = Polygon([(0, 0), (W_in, 0), (W_in, D_in), (0, D_in)])
    ring = outer.difference(inner)

    frame_mesh = trimesh.creation.extrude_polygon(ring, H)

    # Emboss text on front face (y = -t, faces -Y direction)
    if text_lines:
        relief = _build_text_relief(
            x0=-t,
            x1=W_in + t,
            z0=0.0,
            z1=H,
            y_face=-t,
            lines=text_lines,
            text_depth_mm=text_depth_mm,
        )
        frame_mesh = trimesh.util.concatenate([frame_mesh, relief])

    return frame_mesh
