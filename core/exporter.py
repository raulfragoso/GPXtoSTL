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


def _write_binary_stl(
    mesh: trimesh.Trimesh,
    stl_path: str,
    header_text: bytes,
    color_rgb: tuple[int, int, int] | None = None,
) -> None:
    """Write a binary STL; optionally embed RGB555 colour in the attribute bytes."""
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals
    attr = _rgb_to_rgb555(*color_rgb) if color_rgb else 0
    n_tris = len(faces)

    with open(stl_path, "wb") as fh:
        fh.write(header_text[:80])
        fh.write(struct.pack("<I", n_tris))
        for i in range(n_tris):
            nx, ny, nz = face_normals[i]
            fh.write(struct.pack("<fff", nx, ny, nz))
            for vi in faces[i]:
                x, y, z = vertices[vi]
                fh.write(struct.pack("<fff", x, y, z))
            fh.write(struct.pack("<H", attr))


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
    header = b"GPXtoSTL track mesh - orange" + b" " * 52
    _write_binary_stl(mesh, stl_path, header, color_rgb)
    return stl_path


def export_frame_stl(mesh: trimesh.Trimesh, output_dir: str) -> str:
    """
    Write the display frame mesh as a binary STL (no colour encoding).
    Returns the path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    stl_path = os.path.join(output_dir, "frame.stl")
    header = b"GPXtoSTL display frame" + b" " * 58
    _write_binary_stl(mesh, stl_path, header, color_rgb=None)
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
    Fill the Three.js viewer template with embedded asset data and write output/viewer.html.
    Assets (OBJ, texture, STL) are base64-embedded so the file is self-contained and
    works on Streamlit Cloud without a local file server.
    Returns the path of the written file.
    """
    import base64
    import json

    viewer_path = os.path.join(output_dir, "viewer.html")

    with open(template_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    # Read and embed the generated assets
    obj_path = os.path.join(output_dir, "terrain.obj")
    tex_path = os.path.join(output_dir, "terrain_texture.jpg")
    stl_path = os.path.join(output_dir, "track.stl")

    with open(obj_path, "r", encoding="utf-8") as f:
        obj_text = f.read()
    with open(tex_path, "rb") as f:
        tex_b64 = base64.b64encode(f.read()).decode("ascii")
    with open(stl_path, "rb") as f:
        stl_b64 = base64.b64encode(f.read()).decode("ascii")

    tex_data_uri = f"data:image/jpeg;base64,{tex_b64}"

    distance_km = total_distance_m / 1000
    html = html.replace("{{SCENE_TITLE}}", track_name)
    html = html.replace("{{DISTANCE_KM}}", f"{distance_km:.1f}")
    html = html.replace("{{ELE_GAIN_M}}", f"{ele_gain_m:.0f}")
    html = html.replace("{{OBJ_JSON}}", json.dumps(obj_text))
    html = html.replace("{{TEX_URI_JSON}}", json.dumps(tex_data_uri))
    html = html.replace("{{STL_B64_JSON}}", json.dumps(stl_b64))

    with open(viewer_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return viewer_path
