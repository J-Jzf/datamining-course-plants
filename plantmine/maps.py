"""Small dependency-free administrative maps using bundled Natural Earth polygons."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

CLUSTER_COLORS = ["#2f6b57", "#527a94", "#c28e39", "#926d91", "#84a596", "#c27462", "#7e7e56", "#7998b9", "#bcab78", "#495c75"]
ASSET = Path(__file__).resolve().parents[1] / "assets/geodata/ne_50m_admin_1_us_canada.geojson"


@lru_cache(maxsize=1)
def load_features():
    if not ASSET.exists():
        raise FileNotFoundError(f"Bundled map asset missing: {ASSET}")
    return json.loads(ASSET.read_text(encoding="utf-8"))["features"]


def feature_code(feature):
    props = feature["properties"]
    code = str(props.get("iso_3166_2", "")).split("-")[-1]
    return code.upper() if len(code) == 2 else str(props.get("postal", "")).upper()


def _rings(feature):
    geometry = feature["geometry"]
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    for polygon in polygons:
        # Admin boundaries only: interior water holes are omitted consistently.
        ring = np.asarray(polygon[0], dtype=float)
        # Keep western Alaska islands in the Alaska inset instead of connecting across the dateline.
        ring[:, 0] = np.where(ring[:, 0] > 170, ring[:, 0] - 360, ring[:, 0])
        yield ring


def draw_admin_map(ax, values=None, cmap=None, norm=None, categorical=False, labels=False, region_info=None,
                   extent=(-142, -51, 23, 83), codes=None):
    """Draw lon/lat polygons (display aspect adjusted at 52 degrees N)."""
    values = values or {}
    codes = set(codes) if codes is not None else None
    for feature in load_features():
        code = feature_code(feature)
        if codes is not None and code not in codes:
            continue
        value = values.get(code)
        face = "#e8eeea"
        if value is not None:
            face = CLUSTER_COLORS[int(value) % len(CLUSTER_COLORS)] if categorical else cmap(norm(value))
        for ring in _rings(feature):
            ax.add_patch(Polygon(ring, facecolor=face, edgecolor="white", linewidth=.5, zorder=2))
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(52)))
    ax.set_facecolor("#f2f6f7")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if region_info is not None:
        indexed = region_info.set_index("code")
        for code, row in indexed.iterrows():
            if codes is not None and code not in codes:
                continue
            x, y = row.longitude, row.latitude
            if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
                if code == "DC":
                    ax.scatter([x], [y], s=13, c=[CLUSTER_COLORS[int(values.get(code, 0)) % len(CLUSTER_COLORS)]
                               if categorical else cmap(norm(values.get(code, 0)))], edgecolors="white", lw=.5, zorder=5)
                if labels:
                    # Dense northeastern states are shown in the separate detail inset.
                    if code in {"DC", "DE", "MD", "NJ", "CT", "RI", "MA", "NH", "VT"} and extent[0] < -100:
                        continue
                    callouts={"DC":(-77.05,38.15),"MD":(-77.0,40.0),"DE":(-74.25,38.9),"RI":(-70.3,41.5),"MA":(-70.3,43.05)}
                    if code in callouts and extent[0] > -100:
                        tx,ty=callouts[code]
                        ax.plot([x,tx],[y,ty],color="#6d8074",lw=.45,zorder=5)
                        ax.text(tx,ty,code,fontsize=6.2,ha="center",va="center",color="#24352f",zorder=6,
                                bbox={"facecolor":"white","alpha":.7,"edgecolor":"none","pad":.3})
                    else:
                        ax.text(x, y, code, fontsize=6.2, ha="center", va="center", color="#24352f", zorder=6)
    return ax


def map_with_insets(fig, values, region_info, cmap=None, norm=None, categorical=False, labels=True):
    """Main US/Canada plus Alaska, Hawaii and Northeast detail; all 64 units are represented."""
    ax = fig.add_axes([.025, .18, .94, .79])
    draw_admin_map(ax, values, cmap, norm, categorical, labels, region_info)
    for rect, extent, codes, label in [
        ([.055, .22, .19, .19], (-180, -130, 49, 72), ["AK"], "AK"),
        ([.26, .23, .10, .115], (-161, -154, 18, 23), ["HI"], "HI"),
        ([.79, .25, .17, .22], (-77.7, -69, 37.8, 45.8), ["DC", "DE", "MD", "NJ", "CT", "RI", "MA", "NH", "VT", "NY", "PA", "ME"], "Northeast detail"),
    ]:
        sub = fig.add_axes(rect)
        draw_admin_map(sub, values, cmap, norm, categorical, labels, region_info, extent, codes)
        sub.text(.01, .99, label, transform=sub.transAxes, va="top", fontsize=7, color="#50655c",
                 bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85, "pad":1})
    return ax
