"""Check an enclosure's shop package against its design, reading only the files in out/cnc as the shop's
software would:

    python utils/check_cnc.py <enclosure folder>

Run the design script and export_cnc.py first. For each sheet it checks:
  - the file is clean: mm units, known layers, every outline closed, no doubled-up lines
  - the parts sit inside the sheet margin and keep their gap from each other
  - each part on the sheet is the part from the design, the same way up (turned, never mirrored)
  - rebuilt in 3D from its lines and pocket depths alone, it has the same volume as the 3D model, and put
    back in the box (turned, not mirrored) it fills exactly the model's shape: every pocket on the face
    and at the end the legend says
  - where a face A pocket crosses a face B pocket, at least MIN_WOOD is left between them
  - a bit of the design's diameter reaches every corner of every cut and pocket
"""
import csv
import sys

import ezdxf
import numpy as np
import manifold3d as mf
import shapely
import trimesh
from shapely.geometry import box
from shapely.ops import unary_union

from dxf_parts import crossings, depth, parts_on, pockets_in, read, shapes
from enclosure import Enclosure, folder_arg, use_oda
from export_cnc import GAP, MARGIN

CUT_LAYERS = ("CUT_THROUGH", "MARK_DRILL", "SHEET")
MIN_WOOD = 3.0  # mm left where pockets on opposite faces cross


def aligner(part, ref, angle):
    """The move that turns part by angle and puts it on ref, bounding box centre on centre. Returns a
    function so the part's pockets move with it."""
    origin = box(*part.bounds).centroid
    turned = box(*shapely.affinity.rotate(part, angle, origin=origin).bounds).centroid
    target = box(*ref.bounds).centroid
    return lambda g: shapely.affinity.translate(shapely.affinity.rotate(g, angle, origin=origin),
                                                target.x - turned.x, target.y - turned.y)


def solid(outline, pockets, t):
    """Extrude the part and take each pocket out of the face its layer names."""
    def cs(poly):
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        rings = []
        for pg in polys:
            pg = shapely.geometry.polygon.orient(pg, 1.0)
            rings.append(np.array(pg.exterior.coords)[:-1])
            rings += [np.array(h.coords)[:-1] for h in pg.interiors]
        return mf.CrossSection(rings, mf.FillRule.EvenOdd)
    m = mf.Manifold.extrude(cs(outline), t)
    for layer, poly in pockets.items():
        d = depth(layer)
        z0 = t - d if layer.startswith("POCKET_A") else 0.0
        m = m - mf.Manifold.extrude(cs(poly.intersection(outline).buffer(0)), d + 0.02).translate((0, 0, z0 - 0.01))
    return m


def model_solids(glb, origin):
    """The glb's meshes back in the design's coordinates: the glb is Y-up, shifted by the manifest's origin."""
    out = {}
    for name, g in glb.geometry.items():
        g = g.copy()
        g.merge_vertices()
        v = g.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float) - origin
        out[name] = mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32), tri_verts=g.faces.astype(np.uint32)))
    return out


def misplaced(part, pockets, place, model):
    """Volume by which the part, put back in the box where the design has it, differs from the 3D model.
    Its drawing is the view from face A with face B pockets seen through the board, so this only comes out
    near zero when every pocket is on the right face and the right way round. Tries the four turns."""
    (x0, y0, x1, y1) = place["bounds"]
    R = np.column_stack([place["u"], place["v"], place["n"]])
    world = np.column_stack([R, place["origin"]])
    best = None
    for angle in (0, 90, 180, 270):
        move = aligner(part, box(x0, y0, x1, y1), angle)
        m = solid(move(part), {k: move(p) for k, p in pockets.items()}, place["t"]).transform(world)
        err = (m - model).volume() + (model - m).volume()
        best = err if best is None else min(best, err)
    return best


def unreachable(region, outline, r):
    """Area of region a bit of radius r can't reach. The bit can run off the part's edge, so whatever is
    outside the part counts as open."""
    big = box(*outline.buffer(4 * r).bounds)
    space = region.union(big.difference(outline)) if region is not outline else big.difference(outline)
    # close hairlines: a part turned on the sheet leaves its pocket and outline edges a few nm apart
    space = space.buffer(0.01, 16).buffer(-0.01, 16)
    opened = space.buffer(-(r - 0.01), 64).buffer(r - 0.01, 64)
    # the bit runs round the outside: what it can't reach is the part's inside corners
    missed = opened.intersection(outline) if region is outline else region.difference(opened)
    # a square corner leaves r²(1 - π/4); far smaller pieces are just round holes drawn as polygons
    return sum(g.area for g in getattr(missed, "geoms", [missed]) if g.area > 0.1 * r * r)


def main():
    enc = Enclosure(folder_arg(sys.argv))
    use_oda()
    design = enc.manifest
    bit = design["cnc"]["bit"]
    with open(enc.out / "cutlist.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    glb = trimesh.load(enc.glb)
    model_volume = {n: g.volume for n, g in glb.geometry.items()}
    # a part cut more than once is one entry: check it against each piece it stands for
    cut_as, placement = design["cut_as"], design["placement"]
    model = model_solids(glb, np.array(design["origin"], float))

    problems = 0

    def bad(msg):
        nonlocal problems
        problems += 1
        print("  PROBLEM:", msg)

    sheets = sorted(enc.cnc.glob("sheet_*.dwg")) or sorted(enc.cnc.glob("sheet_*.dxf"))
    if not sheets:
        sys.exit(f"no sheets in {enc.cnc}: run export_cnc.py first")
    for path in sheets:
        thickness = float(path.stem.split("_")[1][:-2])
        doc = read(path)
        msp = doc.modelspace()
        print(f"{path.name}: {thickness:g} mm")

        # clean file
        if doc.units != ezdxf.units.MM:
            bad(f"units are {doc.units}, not mm")
        layers = {e.dxf.layer for e in msp}
        unknown = {l for l in layers if l not in CUT_LAYERS and not l.startswith("POCKET_")}
        if unknown:
            bad(f"unexpected layers {sorted(unknown)}")
        parts, outers, open_ = parts_on(msp)
        for layer in layers - {"SHEET"}:
            open_ += shapes(msp, layer)[1]
        if open_:
            bad(f"{open_} open or non-polyline shapes")
        seen = set()
        for e in msp:
            key = (e.dxftype(), e.dxf.layer, tuple(np.round(np.array([p[:2] for p in e.get_points()]), 3).flatten())
                   if e.dxftype() == "LWPOLYLINE" else (round(e.dxf.center.x, 3), round(e.dxf.center.y, 3), e.dxf.radius))
            if key in seen:
                bad(f"doubled-up {e.dxftype()} on {e.dxf.layer}")
            seen.add(key)
        print(f"  file: mm, {len(msp)} shapes on {len(layers)} layers, all closed, none doubled")

        # layout
        sheet, _ = shapes(msp, "SHEET")
        inner = box(*sheet[0].bounds).buffer(-MARGIN + 0.01, join_style=2)
        for o in outers:
            if not inner.contains(o):
                bad("a part is inside the sheet margin")
        gaps = [a.distance(b) for i, a in enumerate(outers) for b in outers[i + 1:]]
        if gaps and min(gaps) < GAP - 0.01:
            bad(f"parts only {min(gaps):.1f} mm apart")
        print(f"  layout: {len(outers)} parts, all inside the {MARGIN:g} mm margin"
              + (f", at least {min(gaps):.1f} mm apart" if gaps else ""))

        # each part against the design
        names = [r["part"] for r in rows if float(r["thickness_mm"]) == thickness for _ in range(int(r["qty"]))]
        unmatched = list(names)
        for part, outer in zip(parts, outers):
            pockets = pockets_in(msp, outer)
            best = None
            for name in unmatched:
                ref = ezdxf.readfile(enc.out / "dxf" / f"{name}.dxf").modelspace()
                ref_parts, ref_outers, _ = parts_on(ref)
                ref_part, ref_pockets = ref_parts[0], pockets_in(ref, ref_outers[0])
                if set(ref_pockets) != set(pockets):
                    continue
                for angle in (0, 90, 180, 270):  # turned only: a mirrored part would never match
                    move = aligner(part, ref_part, angle)
                    err = move(part).symmetric_difference(ref_part).area
                    err += sum(move(pockets[k]).symmetric_difference(ref_pockets[k]).area for k in pockets)
                    if best is None or err < best[0]:
                        best = (err, name)
            if best is None or best[0] > 1.0:
                bad(f"a part at {outer.bounds[:2]} doesn't match any part in the design")
                continue
            name = best[1]
            unmatched.remove(name)
            volume = solid(part, pockets, thickness).volume()
            diff = max((volume - model_volume[m] for m in cut_as[name]), key=abs)
            corners = unreachable(part, part, bit / 2) + sum(unreachable(p.intersection(part), part, bit / 2)
                                                             for p in pockets.values())
            for ka, kb, area, left in crossings(pockets, thickness):
                print(f"  {name:<12} {ka} crosses {kb} over {area:.0f} mm², leaving {left:g} mm of wood")
                if left < MIN_WOOD:
                    bad(f"{name}: {ka} and {kb} leave only {left:g} mm of wood where they cross")
            fit = max(misplaced(part, pockets, placement[m], model[m]) for m in cut_as[name])
            print(f"  {name:<12} same shape as the design (off by {best[0]:.2f} mm²), 3D volume "
                  f"{volume / 1000:7.1f} cm³ vs model {model_volume[cut_as[name][0]] / 1000:7.1f} ({diff / 1000:+.2f}), "
                  f"back in the box off by {fit / 1000:.2f} cm³, bit can't reach {corners:.1f} mm²")
            if abs(diff) > 0.002 * volume + 5:
                bad(f"{name}: volume differs from the 3D model by {diff:.0f} mm³")
            if fit > 0.002 * volume + 5:
                bad(f"{name}: put back in the box it differs from the 3D model by {fit:.0f} mm³ - a pocket on "
                    "the wrong face or mirrored")
            if corners > 1.0:
                bad(f"{name}: {corners:.1f} mm² the {bit:g} mm bit can't reach")
        for name in unmatched:
            bad(f"{name} is missing from the sheet")

    print()
    print("all good" if problems == 0 else f"{problems} problems")


if __name__ == "__main__":
    main()
