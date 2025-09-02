#!/usr/bin/env python3
"""
make_map_poster.py
Create a printable minimalist map poster from OpenStreetMap data.

USAGE:
  python make_map_poster.py path/to/config.yaml

The YAML file must contain:
- user_inputs: one (and only one) of {place | center+radius | bbox}
- map_settings: style + figure + output options (see example below)
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import numpy as np
import osmnx as ox
from shapely.ops import unary_union
from typing import Optional, Tuple, Dict, Any
from models import MapPosterConfig, _center_from_bbox, clip_layers_to_bbox
import geopandas as gpd
from osmnx._errors import InsufficientResponseError


# ----------------------------
# Helpers that don’t depend on CLI flags
# ----------------------------
def _format_dms_component(value: float, is_lat: bool) -> str:
    hemi_pos = "N" if is_lat else "E"
    hemi_neg = "S" if is_lat else "W"
    hemi = hemi_pos if value >= 0 else hemi_neg
    v = abs(value)
    deg = int(v)
    minutes_float = (v - deg) * 60.0
    mins = int(minutes_float)
    secs = int(round((minutes_float - mins) * 60.0, 0))
    # Two decimals looks nice for posters; tweak if you prefer 0-decimal
    return f"{deg}°{mins:02d}'{secs:02d}\"{hemi}"

def _format_coords(lat: float, lon: float, mode: str) -> str:
    mode = (mode or "none").lower()
    if mode == "dd":
        # 3 decimals, include hemispheres
        lat_hemi = "N" if lat >= 0 else "S"
        lon_hemi = "E" if lon >= 0 else "W"
        return f"{abs(lat):.3f}°{lat_hemi}, {abs(lon):.3f}°{lon_hemi}"
    if mode == "dms":
        return f"{_format_dms_component(lat, is_lat=True)}, {_format_dms_component(lon, is_lat=False)}"
    return None  # "none"


def _features_from_bbox(bbox, tags):
    """Return features, or an empty GeoDataFrame if none are found."""
    try:
        gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
        # Normalize Nones to empty GDF too
        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gdf
    except InsufficientResponseError:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def classify_road(highway_val: Optional[str]) -> str:
    if not highway_val:
        return "other"
    val = highway_val if isinstance(highway_val, str) else highway_val[0]
    if val in {"motorway", "motorway_link"}: return "motorway"
    if val in {"trunk", "trunk_link"}: return "trunk"
    if val in {"primary", "primary_link"}: return "primary"
    if val in {"secondary", "secondary_link"}: return "secondary"
    if val in {"tertiary", "tertiary_link"}: return "tertiary"
    if val in {"residential", "living_street"}: return "residential"
    if val in {"service", "unclassified"}: return "service"
    if any(key in val for key in ("foot", "path", "cycle", "pedestrian")): return "footway"
    return "other"

def apply_margin_to_bbox(bbox_wsen, margin_frac):
    w, s, e, n = bbox_wsen
    dx = (e - w) * margin_frac
    dy = (n - s) * margin_frac
    return (w - dx, s - dy, e + dx, n + dy)


def fetch_layers_bbox(bbox, draw_flags: Dict[str, bool]):
    # bbox must be (west, south, east, north)
    layers = {}

    if draw_flags.get("road_edges", True):
        # G = ox.graph.graph_from_bbox(bbox=bbox, network_type="drive", simplify=True)
        G = ox.graph.graph_from_bbox(bbox=bbox, network_type="all", simplify=True, retain_all=True, truncate_by_edge=True)
        edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        layers["roads"] = edges

    if draw_flags.get("buildings", True):
        buildings = _features_from_bbox(bbox, tags={"building": True})
        if not buildings.empty:
            layers["buildings"] = buildings

    if draw_flags.get("landuse_green", False):
        green = _features_from_bbox(
            bbox,
            tags={
                "landuse": ["park", "forest", "recreation_ground", "grass"],
                "leisure": ["park", "garden", "golf_course"],
            },
        )
        if not green.empty:
            layers["green"] = green

    if draw_flags.get("water", True):
        water = _features_from_bbox(
            bbox,
            tags={
                "natural": ["water"],
                "water": True,
                "waterway": True,
            },
        )
        if not water.empty:
            layers["water"] = water

    if draw_flags.get("rail", True):
        rail = _features_from_bbox(bbox, tags={"railway": True})
        if not rail.empty:
            layers["rail"] = rail

    # Make sure only roads within the bbox are shown
    layers = clip_layers_to_bbox(layers, bbox)

    return layers

# Adapter to keep models decoupled from osmnx:
def _geocode_place_with_osmnx(place: str):
    """Returns (bbox_wsen, center_lat, center_lon) for a place."""
    gdf_place = ox.geocode_to_gdf(place)
    polygon = gdf_place.geometry.iloc[0]
    if polygon.geom_type == "MultiPolygon":
        polygon = unary_union([geom for geom in polygon.geoms])
    minx, miny, maxx, maxy = gdf_place.total_bounds  # (west, south, east, north)
    bbox_wsen = (minx, miny, maxx, maxy)
    try:
        c = polygon.centroid
        lat, lon = float(c.y), float(c.x)
    except Exception:
        lat, lon = _center_from_bbox(bbox_wsen)
    return bbox_wsen, lat, lon

def set_mpl_defaults(font: Optional[str], colors: Dict[str, str], fig_w: float, fig_h: float, dpi: int):
    mpl.rcParams["figure.figsize"] = (fig_w, fig_h)
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["savefig.dpi"] = dpi
    mpl.rcParams["axes.facecolor"] = colors["background"]
    mpl.rcParams["figure.facecolor"] = colors["background"]
    if font:
        mpl.rcParams["font.family"] = font

def add_bottom_gradient(ax, background_color: str, height_frac=0.1):
    """
    Add a background → transparent gradient overlay at the bottom of the axes.
    height_frac: fraction of the axes height covered by the gradient.
    """
    # Convert hex (or named) color to RGB
    rgb = np.array(mcolors.to_rgb(background_color))

    n = 256
    alpha = np.linspace(1.0, 0.0, n)[:, None]            # opaque at bottom → transparent at top
    rgba  = np.ones((n, 2, 4), dtype=float)
    rgba[..., :3] = rgb                                  # apply chosen RGB
    rgba[..., 3] = np.repeat(alpha, 2, axis=1)           # alpha channel

    overlay = ax.inset_axes([0, 0, 1, height_frac], transform=ax.transAxes, zorder=1_000)
    overlay.imshow(rgba, origin="lower", aspect="auto", interpolation="bicubic")
    overlay.set_facecolor("none")
    overlay.set_axis_off()

def draw_text_footer(ax, title: Optional[str], coords: Optional[str]):
    if not title and not coords:
        return
    y = 0.075
    if title:
        ax.text(0.5, y, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=112, color="#111111", zorder=2_000)
        y -= 0.035
    if coords:
        ax.text(0.5, y, coords, transform=ax.transAxes, ha="center", va="bottom", fontsize=42, color="#555555", zorder=2_000)

def make_poster(
    ms: Dict[str, Any],
    bbox_wsen: Tuple[float, float, float, float],
    center_lat: Optional[float],
    center_lon: Optional[float],
):
    # Matplotlib defaults
    set_mpl_defaults(ms["font"], ms["colors"], ms["fig_w"], ms["fig_h"], ms["dpi"])

    # OSMnx settings
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.overpass_rate_limit = True

    # Fetch layers
    layers = fetch_layers_bbox(bbox_wsen, ms["draw"])

    # Figure + axes
    fig = plt.figure()
    ax = plt.gca()
    ax.set_facecolor(ms["colors"]["background"])
    ax.set_aspect("equal", adjustable="box")

    # Draw order: water -> green -> buildings -> rail -> roads
    if ms["draw"].get("water") and "water" in layers:
        try:
            layers["water"].plot(ax=ax, color=ms["colors"]["water"], linewidth=0.8)
        except Exception:
            pass

    if ms["draw"].get("landuse_green") and "green" in layers:
        try:
            layers["green"].plot(ax=ax, color=ms["colors"]["green"], linewidth=0.0)
        except Exception:
            pass

    if ms["draw"].get("buildings") and "buildings" in layers:
        try:
            b = layers["buildings"]
            # fill
            b.plot(ax=ax, facecolor=ms["colors"].get("buildings_fill", "#FFFFFF"),
                edgecolor="none", linewidth=0.0)
            
            # outline
            b.plot(ax=ax, facecolor="none",
                edgecolor=ms["colors"].get("buildings_outline", "#BEBEBE"),
                linewidth=ms["widths"].get("buildings_outline", 0.3))
        except Exception:
            pass

    if ms["draw"].get("rail") and "rail" in layers:
        try:
            layers["rail"].plot(ax=ax, color=ms["colors"]["rail"], linewidth=0.6)
        except Exception:
            pass

    if ms["draw"].get("road_edges") and "roads" in layers and not layers["roads"].empty:
        roads = layers["roads"].copy()
        class_col = "_class"
        roads[class_col] = roads["highway"].apply(classify_road)
        classes = ["motorway","trunk","primary","secondary","tertiary","residential","service","footway","other"]
        for cls in classes:
            sub = roads[roads[class_col] == cls]
            if len(sub) == 0:
                continue
            lw = ms["widths"].get(cls, ms["widths"]["other"])
            color = ms["colors"].get(cls, ms["colors"]["other"])
            sub.plot(ax=ax, linewidth=lw, color=color, zorder=10)

        # Apply_wsen_margin
        w, s, e, n = apply_margin_to_bbox(bbox_wsen, ms["margin"])
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.autoscale(False)

    if "icon" in ms:
        try:
            icon_path = ms["icon"]
            img = mpimg.imread(icon_path)

            # Control size with zoom; make configurable via ms.get("icon_zoom", ...)
            imagebox = OffsetImage(img, zoom=ms.get("icon_zoom", 0.05))

            # Place using data coordinates at (center_lon, center_lat)
            # box_alignment=(0.5, 0.0) => bottom-center of the image sits on that point
            ab = AnnotationBbox(
                imagebox,
                (center_lon, center_lat),
                xycoords="data",
                frameon=False,
                box_alignment=(0.5, 0.0),
                zorder=10000000,  # ensure it draws on top of the map
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"Could not load icon: {e}")

    # Axes toggle
    if ms["hide_axes"]:
        ax.set_axis_off()

    # Minimal padding
    margin = ms["margin"]
    plt.subplots_adjust(left=margin, right=1 - margin, top=1 - margin, bottom=margin)

    # Footer gradient
    add_bottom_gradient(ax, ms['colors']['background'], height_frac=0.2)

    # Footer text
    coords_text = ms["coords"]
    if not coords_text:
        auto = ms["coords_auto"]
        if auto in {"dd", "dms"} and center_lat is not None and center_lon is not None:
            coords_text = _format_coords(center_lat, center_lon, auto)

    draw_text_footer(ax, ms["title"], coords_text)

    # Save
    out_path = ms["out_path"]
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved PNG: {out_path}")
    if ms["save_svg"]:
        root, _ = os.path.splitext(out_path)
        svg_path = root + ".svg"
        plt.savefig(svg_path, format="svg", bbox_inches=0)
        print(f"Saved SVG: {svg_path}")

    if ms["show_fig"]:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a minimalist OSM poster from a YAML config file.")
    p.add_argument("config", type=str, help="Path to config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    cfg = MapPosterConfig.from_yaml(args.config)

    # Optional: seed may be used by OSMnx caching/random retries; we just set it if provided
    seed_val = cfg.map_settings.output.seed
    try:
        # Not strictly necessary, but harmless if used elsewhere
        import random
        random.seed(seed_val)
    except Exception:
        pass

    # Resolve bbox + center via model
    bbox_wsen, (center_lat, center_lon) = cfg.resolve_bbox_and_center(_geocode_place_with_osmnx, cfg.map_settings.figure.width_in, cfg.map_settings.figure.height_in)

    # Flattened, validated, defaulted map settings via model
    ms = cfg.map_settings_runtime()

    make_poster(ms, bbox_wsen, center_lat, center_lon)



if __name__ == "__main__":
    main()
