import os
from dotenv import load_dotenv

load_dotenv()

# --- Terrain mesh ---
TERRAIN_GRID_RESOLUTION = 256   # grid points per axis (256×256 = ~131k triangles)
VERTICAL_EXAGGERATION = 2.0     # Z-scale multiplier for visual drama
BASE_THICKNESS_M = 5.0          # solid base height below lowest terrain point (meters)

# --- Satellite tiles ---
TILE_ZOOM = 14                  # Web Mercator zoom level (14 = ~9m/px at equator)
TEXTURE_SIZE = 1024             # output texture resolution in pixels
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")
MAPBOX_TILE_URL = (
    "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90"
    "?access_token={token}"
)
BING_TILE_URL = "https://ecn.t{s}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1"

# --- Groove geometry (all in mm — model/print space) ---
# The terrain is scaled from UTM metres to the target print size before carving,
# so these values refer to millimetres on the physical print.
GROOVE_WIDTH_MM = 3.0           # channel width in mm
GROOVE_DEPTH_MM = 2.0           # channel carve depth in mm
BBOX_PADDING = 0.10             # fractional padding on each side of bounding box

# --- Track geometry (mm — model/print space) ---
TRACK_RAISE_MM = 1.0            # height track protrudes above groove floor in mm
PRINT_TOLERANCE_MM = 0.2        # FDM clearance between track and groove in mm
TRACK_COLOR_RGB = (230, 100, 20)  # orange

# --- Processing limits ---
MAX_RESAMPLED_POINTS = 5000     # cap for groove carving performance
RESAMPLE_STEP_M = 5.0           # target spacing between resampled track points

# --- Output ---
OUTPUT_DIR = "output"
