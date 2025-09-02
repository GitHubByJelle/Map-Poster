# models.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, Literal, Union, Sequence
from pydantic import BaseModel, Field, model_validator, PrivateAttr
import math
import pathlib
import yaml
import geopandas as gpd
from shapely.geometry import box


# ----------------------------
# Geometry helpers (pure, re-usable)
# ----------------------------
def meters_to_bbox(lat: float, lon: float, radius_m: int, fig_w: float, fig_h: float) -> Tuple[float, float, float, float]:
    """Return bbox as (west, south, east, north) for a circle of radius_m around (lat, lon)."""
    # meters -> degrees
    coslat = max(1e-8, math.cos(math.radians(lat)))
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * coslat

    half_width_m  = float(radius_m)                  # E–W half span
    half_height_m = (fig_h / fig_w) * float(radius_m)  # N–S half span

    dlon = half_width_m  / meters_per_deg_lon
    dlat = half_height_m / meters_per_deg_lat

    west, east = lon - dlon, lon + dlon
    south, north = lat - dlat, lat + dlat
    return (west, south, east, north)


def _center_from_bbox(bbox_wsen: Tuple[float, float, float, float]) -> Tuple[float, float]:
    w, s, e, n = bbox_wsen
    return ((s + n) / 2.0, (w + e) / 2.0)


def _parse_bbox_to_wsen(text_or_seq) -> Tuple[float, float, float, float]:
    """
    Accepts:
      - 'south,west,north,east' (string)
      - [south, west, north, east] (list/tuple)
    Returns (west, south, east, north).
    """
    if isinstance(text_or_seq, (list, tuple)):
        parts = [float(x) for x in text_or_seq]
    else:
        parts = [float(x.strip()) for x in str(text_or_seq).split(",")]
    if len(parts) != 4:
        raise ValueError("user_inputs.bbox must be 4 values: 'south,west,north,east'")
    s, w, n, e = parts
    return (w, s, e, n)

def clip_layers_to_bbox(layers: Dict[str, Any], bbox_wsen):
    w, s, e, n = bbox_wsen
    bbox_poly = box(w, s, e, n)
    out = {}
    for name, gdf in layers.items():
        if gdf is None or getattr(gdf, "empty", True):
            out[name] = gdf
            continue
        try:
            # ensure geographic CRS; clip expects same CRS
            if getattr(gdf, "crs", None) is not None and gdf.crs.to_string().upper() not in {"EPSG:4326", "WGS84"}:
                gdf = gdf.to_crs(epsg=4326)
            out[name] = gpd.clip(gdf, bbox_poly)
        except Exception:
            out[name] = gdf  # fail-safe: keep original
    return out


# ----------------------------
# User inputs
# ----------------------------
class UserInputs(BaseModel):
    """
    Exactly one of {place | center+radius | bbox} must be provided.

    - place: free-text geocoder string (e.g. "Maastricht, NL")
    - center: "lat,lon" (string) or [lat, lon] / (lat, lon)
    - radius: meters (required if center is set)
    - bbox: "south,west,north,east" or [s, w, n, e] / (s, w, n, e)
    """
    place: Optional[str] = Field(default=None, description="Free-text place name for geocoding.")
    center: Optional[Union[str, Sequence[float], Tuple[float, float]]] = Field(
        default=None,
        description='Center as "lat,lon" (string) or [lat, lon] / (lat, lon).'
    )
    radius: Optional[int] = Field(default=None, description="Radius in meters (required with center).")
    bbox: Optional[Union[str, Sequence[float], Tuple[float, float, float, float]]] = Field(
        default=None,
        description='BBox as "south,west,north,east" or [s,w,n,e] / (s,w,n,e).'
    )

    # Private state (NOT serialized, NOT part of validation schema)
    _bbox_wsen: Optional[Tuple[float, float, float, float]] = PrivateAttr(default=None)
    _center_latlon: Optional[Tuple[float, float]] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_exclusive(self) -> "UserInputs":
        has_place = bool(self.place)
        has_center = self.center is not None
        has_bbox = self.bbox is not None

        if sum((has_place, has_center, has_bbox)) != 1:
            raise ValueError("Specify exactly one of user_inputs.place | user_inputs.center | user_inputs.bbox")

        if has_center and not self.radius:
            raise ValueError("user_inputs.center requires user_inputs.radius (meters)")

        if has_bbox:
            self._bbox_wsen = _parse_bbox_to_wsen(self.bbox)

        if has_center:
            # Normalize center to (lat, lon)
            if isinstance(self.center, (list, tuple)):
                lat, lon = float(self.center[0]), float(self.center[1])
            else:
                lat, lon = [float(x.strip()) for x in str(self.center).split(",")]
            self._center_latlon = (lat, lon)

        return self

    def resolve_bbox_and_center(
        self,
        geocode_func,  # callable: place -> (bbox_wsen, center_lat, center_lon)
        fig_w: float, 
        fig_h: float,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        if self.place:
            bbox_wsen, lat, lon = geocode_func(self.place)
            return bbox_wsen, (lat, lon)

        if self._center_latlon:
            lat, lon = self._center_latlon
            bbox = meters_to_bbox(lat, lon, int(self.radius), fig_w, fig_h)
            return bbox, (lat, lon)

        # else: bbox path
        bbox = self._bbox_wsen  # already normalized
        return bbox, _center_from_bbox(bbox)


# ----------------------------
# Map styling / settings
# ----------------------------
class DrawFlags(BaseModel):
    buildings: bool = Field(default=True, description="Draw buildings layer.")
    landuse_green: bool = Field(default=False, description="Draw parks/green landuse.")
    water: bool = Field(default=True, description="Draw water polygons/lines.")
    rail: bool = Field(default=True, description="Draw railways.")
    road_edges: bool = Field(default=True, description="Draw road edges from drive network.")


class ColorScheme(BaseModel):
    background: str = "#FFFFFF"
    land: str = "#FFFFFF"
    buildings_fill: str = "#FFFFFF"
    buildings_outline: str = "#BEBEBE"
    green: str = "#FFFFFF"
    water: str = "#D9D9D9"
    rail: str = "#BEBEBE"
    motorway: str = "#2B2B2B"
    trunk: str = "#2B2B2B"
    primary: str = "#2B2B2B"
    secondary: str = "#3A3A3A"
    tertiary: str = "#4A4A4A"
    residential: str = "#5A5A5A"
    service: str = "#6A6A6A"
    footway: str = "#7A7A7A"
    other: str = "#808080"


class Widths(BaseModel):
    motorway: float = 2.2
    trunk: float = 2.0
    primary: float = 1.9
    secondary: float = 1.6
    tertiary: float = 1.3
    residential: float = 1.0
    service: float = 0.8
    footway: float = 0.6
    buildings_outline: float = 0.3
    other: float = 0.9


class FigureOpts(BaseModel):
    width_in: float = Field(default=24.0, description="Figure width in inches.")
    height_in: float = Field(default=36.0, description="Figure height in inches.")
    dpi: int = Field(default=300, description="Figure DPI.")
    margin_frac: float = Field(default=0.02, description="Subplot margins fraction (0-1).")


CoordsAuto = Literal["none", "dd", "dms"]


class OutputOpts(BaseModel):
    out: str = Field(default="poster.png", description="Output PNG path.")
    svg: bool = Field(default=False, description="Additionally write an SVG next to PNG.")
    show: bool = Field(default=False, description="Show Matplotlib figure window.")
    no_axes: bool = Field(default=True, description="Hide axes for clean poster look.")
    title: Optional[str] = Field(default=None, description="Optional big title in footer.")
    coords: Optional[str] = Field(default=None, description="Explicit coordinates footer text.")
    coords_auto: CoordsAuto = Field(default="none", description='Auto coord format: "none"|"dd"|"dms".')
    font: Optional[str] = Field(default=None, description="Matplotlib font family.")
    font_size_title: Optional[int] = Field(default=112, description="Font size of the title")
    font_size_subtitle: Optional[int] = Field(default=42, description="Font size of the subtitle")
    icon: Optional[str] = Field(default=None, description="Path to icon positioned in center")
    seed: int = Field(default=42, description="Seed for any randomized operations/caches.")


class MapSettings(BaseModel):
    draw: DrawFlags = Field(default_factory=DrawFlags)
    colors: ColorScheme = Field(default_factory=ColorScheme)
    widths: Widths = Field(default_factory=Widths)
    figure: FigureOpts = Field(default_factory=FigureOpts)
    output: OutputOpts = Field(default_factory=OutputOpts)

    # ---------- instance helpers ----------
    def as_runtime_dict(self) -> Dict[str, Any]:
        """Return the flattened dict your renderer expects (drop-in for get_map_settings)."""
        fig = self.figure
        out = self.output
        return {
            "draw": self.draw.model_dump(),
            "colors": self.colors.model_dump(),
            "widths": self.widths.model_dump(),
            "fig_w": fig.width_in,
            "fig_h": fig.height_in,
            "dpi": fig.dpi,
            "margin": fig.margin_frac,
            "out_path": out.out,
            "save_svg": out.svg,
            "show_fig": out.show,
            "hide_axes": out.no_axes,
            "title": out.title,
            "coords": out.coords,
            "coords_auto": out.coords_auto,
            "font": out.font,
            "font_size_title": out.font_size_title,
            "font_size_subtitle": out.font_size_subtitle,
            "icon": out.icon,
            "seed": out.seed,
        }


# ----------------------------
# Top-level config model
# ----------------------------
class MapPosterConfig(BaseModel):
    """
    Top-level config matching your YAML:
      user_inputs: {place|center+radius|bbox}
      map_settings: {draw, colors, widths, figure, output}
    """
    user_inputs: UserInputs
    map_settings: MapSettings = Field(default_factory=MapSettings)

    # ---------- class helpers ----------
    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "MapPosterConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    # ---------- instance helpers ----------
    def resolve_bbox_and_center(self, geocode_func, fig_w: float, fig_h: float):
        """Proxy to the user_inputs method, using a provided geocoder adapter."""
        return self.user_inputs.resolve_bbox_and_center(geocode_func, fig_w, fig_h)

    def map_settings_runtime(self) -> Dict[str, Any]:
        """Flattened settings for the plotting routine (replaces get_map_settings)."""
        return self.map_settings.as_runtime_dict()
