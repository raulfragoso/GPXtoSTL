"""GPXtoSTL — Streamlit application."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st

import config as cfg
from core.elevation_fetcher import fetch_elevation_grid
from core.exporter import export_terrain_obj, export_track_stl, export_frame_stl, generate_viewer
from core.frame_builder import build_display_frame
from core.gpx_parser import parse_gpx
from core.mesh_builder import build_terrain_mesh, apply_hex_shape
from core.tile_fetcher import fetch_satellite_image
from core.track_carver import carve_groove
from core.track_extruder import extrude_track

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPX → 3D STL",
    page_icon="🗺️",
    layout="wide",
)



# ── Sidebar parameters ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")

    resolution = st.selectbox(
        "Terrain resolution",
        options=[128, 256, 512],
        index=1,
        help="Grid points per axis. 512 produces ~524k triangles — slow but detailed.",
    )

    v_exag = st.slider(
        "Vertical exaggeration",
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help="Multiplier applied to Z (elevation) values.",
    )

    groove_width = st.slider(
        "Groove width (mm)",
        min_value=1.0, max_value=10.0, value=float(cfg.GROOVE_WIDTH_MM), step=0.5,
        help="Width of the carved channel on the printed model.",
    )

    groove_depth = st.slider(
        "Groove depth (mm)",
        min_value=0.5, max_value=5.0, value=float(cfg.GROOVE_DEPTH_MM), step=0.5,
        help="Depth of the carved channel on the printed model.",
    )

    track_height = st.slider(
        "Track height above groove (mm)",
        min_value=0.5, max_value=10.0, value=float(cfg.TRACK_RAISE_MM), step=0.5,
        help="How much the track ribbon protrudes above the terrain surface when fitted into the groove.",
    )

    track_color_hex = st.color_picker(
        "Track colour",
        value="#E66414",
    )

    model_shape = st.radio(
        "Model shape",
        options=["square", "hexagon"],
        format_func={"square": "⬜ Square", "hexagon": "⬡ Hexagon"}.get,
        horizontal=True,
        help="Hexagon clips the terrain to a regular hex tile — great for modular displays.",
    )

    base_size_mm = st.number_input(
        "Target base size (mm)",
        min_value=50, max_value=500, value=200, step=10,
        help="Longest physical dimension of the printed terrain base in mm.",
    )

    st.divider()
    st.subheader("🛰️ Satellite imagery")

    tile_source = st.selectbox(
        "Map source",
        options=["esri", "bing", "mapbox"],
        format_func={
            "esri":   "ESRI World Imagery (free)",
            "bing":   "Bing Aerial (free)",
            "mapbox": "Mapbox Satellite (API key required)",
        }.get,
        help="Bing often has sharper imagery in mountainous regions. "
             "Mapbox requires MAPBOX_TOKEN in your .env file (free at mapbox.com).",
    )

    tile_zoom = st.slider(
        "Tile zoom level",
        min_value=12, max_value=17, value=cfg.TILE_ZOOM,
        help="Higher = sharper texture but more tiles to fetch. "
             "14 ≈ 9 m/px · 15 ≈ 4.5 m/px · 16 ≈ 2.4 m/px · 17 ≈ 1.2 m/px",
    )

    st.divider()
    st.subheader("🖼️ Display frame")

    generate_frame = st.checkbox(
        "Generate display frame",
        value=True,
        help="A separate STL frame the terrain model slots into, with track info embossed on the front.",
    )

    frame_height = st.slider(
        "Frame height (mm)",
        min_value=5, max_value=30, value=15, step=1,
        help="How tall the frame walls are.",
    )

    frame_wall = st.slider(
        "Wall thickness (mm)",
        min_value=3, max_value=10, value=5, step=1,
        help="Thickness of the frame walls.",
    )

    frame_text_depth = st.slider(
        "Text emboss depth (mm)",
        min_value=0.2, max_value=1.0, value=0.4, step=0.1,
        help="How much the text protrudes from the front face of the frame.",
    )

    st.divider()
    st.caption(
        "**Terrain**: OBJ + MTL with satellite texture.\n\n"
        "**Track**: Binary STL with RGB555 colour (orange).\n\n"
        "**Frame**: Binary STL — print separately, then slot the terrain into it.\n\n"
        "STL natively has no colour support — the track uses the Materialise Magics "
        "convention (validity bit + RGB555 in the 2-byte attribute field).\n\n"
        "For full colour in MeshLab/Blender, use PLY."
    )


# ── Helper: hex → rgb ─────────────────────────────────────────────────────────
def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("GPX → 3D STL")
st.markdown(
    "Upload a GPX file to generate a 3D-printable terrain model with a carved track groove "
    "and a matching raised track insert."
)

uploaded = st.file_uploader("Choose a GPX file", type=["gpx"])

if uploaded is not None:
    # Save upload to a temp file so gpxpy can open it
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        gpx_data = parse_gpx(tmp_path)
    except Exception as exc:
        st.error(f"Failed to parse GPX file: {exc}")
        st.stop()
    finally:
        os.unlink(tmp_path)

    # ── Track summary ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Distance", f"{gpx_data.total_distance_m / 1000:.2f} km")
    col2.metric("Elevation gain", f"{gpx_data.ele_gain_m:.0f} m")
    col3.metric("Track points", f"{len(gpx_data.raw_points):,}")

    if not gpx_data.has_elevation:
        st.warning(
            "This GPX file contains no elevation data. "
            "Elevation will be fetched from SRTM (30 m resolution)."
        )

    bbox = gpx_data.bbox
    st.caption(
        f"Bounding box: {bbox[0]:.4f}°N – {bbox[1]:.4f}°N, "
        f"{bbox[2]:.4f}°E – {bbox[3]:.4f}°E"
    )

    # ── Generate button ───────────────────────────────────────────────────────
    if st.button("Generate 3D Models", type="primary"):
        output_dir = os.path.abspath(cfg.OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)

        progress = st.progress(0, text="Starting…")

        try:
            # 1. Elevation
            progress.progress(5, text="Fetching elevation data…")
            elev_grid, lat_coords, lon_coords = fetch_elevation_grid(
                gpx_data.bbox_padded,
                resolution=resolution,
                vertical_exaggeration=v_exag,
            )
            progress.progress(25, text="Elevation data fetched.")

            # 2. Satellite tiles
            progress.progress(26, text="Fetching satellite imagery…")
            sat_path = os.path.join(output_dir, "terrain_texture.jpg")
            satellite_img, geo_info = fetch_satellite_image(
                gpx_data.bbox_padded,
                output_path=sat_path,
                zoom=tile_zoom,
                source=tile_source,
            )
            progress.progress(50, text="Satellite imagery fetched.")

            # 3. Build terrain mesh
            progress.progress(51, text="Building terrain mesh…")
            terrain_mesh, utm_transformer = build_terrain_mesh(
                elev_grid, lat_coords, lon_coords,
                gpx_data.bbox_padded,
                satellite_img,
            )
            progress.progress(65, text="Terrain mesh built.")

            # 3b. Scale from UTM metres → print-space mm
            # The mesh is built in real-world UTM metres. Groove/track dimensions
            # are specified in mm on the physical print. We must scale the mesh
            # (and all subsequent coordinates) to mm so that the carved groove
            # dimensions are meaningful relative to the terrain grid spacing.
            v = terrain_mesh.vertices
            utm_extent = max(
                v[:, 0].max() - v[:, 0].min(),
                v[:, 1].max() - v[:, 1].min(),
            )
            mm_per_utm = base_size_mm / utm_extent  # mm per UTM metre
            terrain_mesh.apply_scale(mm_per_utm)

            # 4. Project track to UTM, then scale to mm
            raw_lons = gpx_data.resampled_points[:, 1]
            raw_lats = gpx_data.resampled_points[:, 0]
            utm_x, utm_y = utm_transformer.transform(raw_lons, raw_lats)
            track_mm = np.column_stack([utm_x, utm_y]) * mm_per_utm

            # 5. Carve groove (all coordinates now in mm)
            progress.progress(66, text="Carving groove into terrain…")
            terrain_carved, groove_floor_z = carve_groove(
                terrain_mesh,
                track_mm,
                utm_transformer,
                groove_width=groove_width,
                groove_depth=groove_depth,
                resolution=resolution,
            )
            progress.progress(80, text="Groove carved.")

            # 5b. Clip track to hexagon, then apply hexagonal shape to terrain
            if model_shape == "hexagon":
                v = terrain_carved.vertices
                xmin, xmax = float(v[:, 0].min()), float(v[:, 0].max())
                ymin, ymax = float(v[:, 1].min()), float(v[:, 1].max())
                hex_cx = (xmin + xmax) / 2.0
                hex_cy = (ymin + ymax) / 2.0
                x_ext = xmax - xmin
                y_ext = ymax - ymin
                hex_r = min(y_ext / 2.0, x_ext * np.sqrt(3) / 4.0) * 0.98
                # Half-plane test: keep only track points inside all 6 hex faces
                inside = np.ones(len(track_mm), dtype=bool)
                for i in range(6):
                    ang = np.radians(30.0 + i * 60.0)
                    nx_h, ny_h = float(np.cos(ang)), float(np.sin(ang))
                    ox, oy = hex_cx + hex_r * nx_h, hex_cy + hex_r * ny_h
                    dot = (track_mm[:, 0] - ox) * nx_h + (track_mm[:, 1] - oy) * ny_h
                    inside &= dot <= 1e-6
                track_mm = track_mm[inside]
                groove_floor_z = groove_floor_z[inside]
                terrain_carved = apply_hex_shape(terrain_carved, satellite_img)

            # 6. Extrude track (coordinates in mm)
            progress.progress(81, text="Extruding track mesh…")
            track_width_mm = groove_width - 2 * cfg.PRINT_TOLERANCE_MM
            track_mesh = extrude_track(
                track_mm,
                groove_floor_z,
                track_width_m=track_width_mm,
                track_raise_m=track_height,
            )
            progress.progress(90, text="Track mesh extruded.")

            # 7. Export
            progress.progress(91, text="Exporting files…")
            track_color = _hex_to_rgb(track_color_hex)

            export_terrain_obj(terrain_carved, output_dir)
            export_track_stl(track_mesh, output_dir, color_rgb=track_color)

            # 8. Display frame
            if generate_frame:
                progress.progress(92, text="Building display frame…")
                tv = terrain_carved.vertices
                frame_mesh = build_display_frame(
                    terrain_width_mm=float(tv[:, 0].max() - tv[:, 0].min()),
                    terrain_depth_mm=float(tv[:, 1].max() - tv[:, 1].min()),
                    frame_height_mm=frame_height,
                    wall_thickness_mm=frame_wall,
                    text_lines=[
                        gpx_data.name,
                        f"{gpx_data.total_distance_m / 1000:.1f} km  ·  +{gpx_data.ele_gain_m:.0f} m",
                    ],
                    text_depth_mm=frame_text_depth,
                )
                export_frame_stl(frame_mesh, output_dir)
                progress.progress(97, text="Frame built.")

            generate_viewer(
                output_dir,
                track_name=gpx_data.name,
                total_distance_m=gpx_data.total_distance_m,
                ele_gain_m=gpx_data.ele_gain_m,
                template_path=os.path.join(os.path.dirname(__file__), "viewer", "viewer_template.html"),
            )

            progress.progress(100, text="Done!")
            st.session_state["output_dir"] = output_dir
            st.session_state["output_ready"] = True

        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.exception(exc)
            st.stop()

    # ── Downloads + viewer — shown after generation and across re-runs ────────
    # Rendered outside the button block so download clicks don't clear them.
    if st.session_state.get("output_ready") and st.session_state.get("output_dir"):
        output_dir = st.session_state["output_dir"]

        st.success("3D models generated successfully!")
        st.subheader("Download Files")

        def _read(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        obj_path   = os.path.join(output_dir, "terrain.obj")
        mtl_path   = os.path.join(output_dir, "terrain.mtl")
        tex_path   = os.path.join(output_dir, "terrain_texture.jpg")
        stl_path   = os.path.join(output_dir, "track.stl")
        frame_path = os.path.join(output_dir, "frame.stl")

        dl_col1, dl_col2, dl_col3, dl_col4, dl_col5 = st.columns(5)
        if os.path.exists(obj_path):
            dl_col1.download_button("terrain.obj", _read(obj_path), "terrain.obj", "text/plain", key="dl_obj")
        if os.path.exists(mtl_path):
            dl_col2.download_button("terrain.mtl", _read(mtl_path), "terrain.mtl", "text/plain", key="dl_mtl")
        if os.path.exists(tex_path):
            dl_col3.download_button("terrain_texture.jpg", _read(tex_path), "terrain_texture.jpg", "image/jpeg", key="dl_tex")
        if os.path.exists(stl_path):
            dl_col4.download_button("track.stl", _read(stl_path), "track.stl", "application/octet-stream", key="dl_stl")
        if os.path.exists(frame_path):
            dl_col5.download_button("frame.stl", _read(frame_path), "frame.stl", "application/octet-stream", key="dl_frame")

        viewer_html_path = os.path.join(output_dir, "viewer.html")
        if os.path.exists(viewer_html_path):
            st.subheader("3D Preview")
            with open(viewer_html_path, "r", encoding="utf-8") as _f:
                viewer_html_content = _f.read()
            st.components.v1.html(viewer_html_content, height=600, scrolling=False)

else:
    st.info("Upload a GPX file above to get started.")
