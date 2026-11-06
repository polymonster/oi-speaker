"""Reading cut drawings back as shapes: what a CAM program sees in a DXF or DWG."""
from pathlib import Path

import ezdxf
from ezdxf.addons import odafc
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def read(path):
    path = Path(path)
    return odafc.readfile(str(path)) if path.suffix == ".dwg" else ezdxf.readfile(path)


def depth(layer):
    """POCKET_A_4.5mm -> 4.5"""
    return float(layer.split("_")[-1][:-2])


def shapes(msp, layer):
    """Closed shapes on a layer as polygons, and a count of anything open."""
    polys, open_ = [], 0
    for e in msp.query(f'*[layer=="{layer}"]'):
        if e.dxftype() == "LWPOLYLINE":
            if not e.closed:
                open_ += 1
            pts = [p[:2] for p in e.get_points()]
            polys.append(Polygon(pts).buffer(0))
        elif e.dxftype() == "CIRCLE":
            polys.append(Point(e.dxf.center.x, e.dxf.center.y).buffer(e.dxf.radius, 256))
        else:
            open_ += 1  # lines, arcs, splines: a CAM program can't be sure they join up
    return polys, open_


def parts_on(msp):
    """Group a drawing's cut lines into parts: each outermost outline, less the outlines inside it."""
    contours, open_ = shapes(msp, "CUT_THROUGH")
    contours.sort(key=lambda p: -p.area)
    outers = []
    for c in contours:
        if not any(o.contains(c.representative_point()) and o.area > c.area for o in outers):
            outers.append(c)
    parts = []
    for o in outers:
        holes = [c for c in contours if c is not o and o.contains(c.representative_point())]
        parts.append(o.difference(unary_union(holes)) if holes else o)
    return parts, outers, open_


def pockets_in(msp, region):
    """{layer: shape} of the pockets inside region."""
    out = {}
    for layer in {e.dxf.layer for e in msp if e.dxf.layer.startswith("POCKET_")}:
        polys, _ = shapes(msp, layer)
        mine = [p for p in polys if region.contains(p.representative_point())]
        if mine:
            out[layer] = unary_union(mine)
    return out


def crossings(pockets, thickness):
    """Where a face A pocket and a face B pocket overlap: [(layer A, layer B, area mm², wood left mm)]."""
    found = []
    for ka, pa in pockets.items():
        for kb, pb in pockets.items():
            if ka.startswith("POCKET_A") and kb.startswith("POCKET_B"):
                area = pa.intersection(pb).area
                if area > 0.01:
                    found.append((ka, kb, area, thickness - depth(ka) - depth(kb)))
    return found
