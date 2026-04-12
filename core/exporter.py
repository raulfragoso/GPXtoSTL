"""Export terrain as OBJ+MTL and track as colour-encoded binary STL."""

from __future__ import annotations

import os
import struct

import numpy as np
import trimesh

from config import TRACK_COLOR_RGB


# ---------------------------------------------------------------------------
# Terrain OBJ + MTL export
# ---------------------------------------------------------------------------

def export_terrain_obj(mesh: trimesh.Trimesh, output_dir: str) -> dict[str, str]:
    """
    Export the terrain mesh as OBJ + MTL + JPEG texture.

    Returns a dict with keys 'obj', 'mtl', 'texture' pointing to written files.
    """
    import io
    from PIL import Image as PILImage

    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, "terrain.obj")
    mtl_path = os.path.join(output_dir, "terrain.mtl")
    tex_path = os.path.join(output_dir, "terrain_texture.jpg")

    # return_texture=True → (obj_str, {filename: bytes, ...})
    # The dict contains e.g. "material.mtl" and "material_0.png" as keys.
    obj_str, tex_dict = trimesh.exchange.obj.export_obj(
        mesh, include_texture=True, return_texture=True
    )

    # ── Find MTL and texture content in the returned dict ──────────────────
    mtl_bytes = b""
    tex_bytes = None
    orig_tex_name = ""

    for key, val in tex_dict.items():
        if key.endswith(".mtl") and isinstance(val, bytes):
            mtl_bytes = val
        elif any(key.endswith(ext) for ext in (".png", ".jpg", ".jpeg")) and isinstance(val, bytes):
            tex_bytes = val
            orig_tex_name = key

    # ── Write texture as JPEG ───────────────────────────────────────────────
    if tex_bytes:
        img = PILImage.open(io.BytesIO(tex_bytes)).convert("RGB")
        img.save(tex_path, "JPEG", quality=90)

    # ── Write MTL with full-brightness coefficients ────────────────────────
    # Trimesh's default Ka/Kd of 0.4 multiplies the texture to ~40% brightness
    # (near-black in the viewer). Always write clean values regardless of what
    # trimesh exported.
    with open(mtl_path, "w", encoding="utf-8") as fh:
        fh.write(
            "newmtl material_0\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\n"
            "Ns 0.0\n"
            "map_Kd terrain_texture.jpg\n"
        )

    # ── Write OBJ patched to reference terrain.mtl ─────────────────────────
    lines = obj_str.splitlines()
    new_lines = []
    wrote_mtllib = False
    for line in lines:
        if line.startswith("mtllib"):
            new_lines.append("mtllib terrain.mtl")
            wrote_mtllib = True
        else:
            new_lines.append(line)
    if not wrote_mtllib:
        new_lines.insert(0, "mtllib terrain.mtl")

    with open(obj_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(new_lines))

    return {"obj": obj_path, "mtl": mtl_path, "texture": tex_path}


# ---------------------------------------------------------------------------
# Track STL with RGB555 colour in attribute bytes
# ---------------------------------------------------------------------------

def _rgb_to_rgb555(r: int, g: int, b: int) -> int:
    """Pack an RGB triplet into a 16-bit RGB555 value with validity bit set."""
    r5 = (r >> 3) & 0x1F
    g5 = (g >> 3) & 0x1F
    b5 = (b >> 3) & 0x1F
    return 0x8000 | (r5 << 10) | (g5 << 5) | b5


def export_track_stl(
    mesh: trimesh.Trimesh,
    output_dir: str,
    color_rgb: tuple[int, int, int] = TRACK_COLOR_RGB,
) -> str:
    """
    Write the track mesh as a binary STL with RGB555 colour in attribute bytes.

    Compatible with Materialise Magics and Windows 3D Builder.
    Returns the path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    stl_path = os.path.join(output_dir, "track.stl")

    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals

    attr = _rgb_to_rgb555(*color_rgb)
    n_tris = len(faces)

    with open(stl_path, "wb") as fh:
        # 80-byte header
        header = b"GPXtoSTL track mesh - orange                                                    "
        fh.write(header[:80])
        # Triangle count
        fh.write(struct.pack("<I", n_tris))
        # Per-triangle records
        for i in range(n_tris):
            nx, ny, nz = face_normals[i]
            fh.write(struct.pack("<fff", nx, ny, nz))
            for vi in faces[i]:
                x, y, z = vertices[vi]
                fh.write(struct.pack("<fff", x, y, z))
            fh.write(struct.pack("<H", attr))

    return stl_path


# ---------------------------------------------------------------------------
# Viewer HTML generation
# ---------------------------------------------------------------------------

def generate_viewer(
    output_dir: str,
    track_name: str,
    total_distance_m: float,
    ele_gain_m: float,
    template_path: str = "viewer/viewer_template.html",
) -> str:
    """
    Fill the Three.js viewer template and write output/viewer.html.
    Returns the path of the written file.
    """
    viewer_path = os.path.join(output_dir, "viewer.html")

    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    distance_km = total_distance_m / 1000
    html = html.replace("{{TERRAIN_OBJ_PATH}}", "terrain.obj")
    html = html.replace("{{TRACK_STL_PATH}}", "track.stl")
    html = html.replace("{{SCENE_TITLE}}", track_name)
    html = html.replace("{{DISTANCE_KM}}", f"{distance_km:.1f}")
    html = html.replace("{{ELE_GAIN_M}}", f"{ele_gain_m:.0f}")

    with open(viewer_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return viewer_path
