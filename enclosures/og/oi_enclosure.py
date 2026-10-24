#!/usr/bin/env python3
"""
Oi Speaker enclosure — parametric CNC model (v1)

One source of truth for every panel. Running this script regenerates:
  out/dxf/<part>.dxf        one file per part, layered by cut type
  out/dxf/all_parts.dxf     every part laid out (unnested) for the shop
  out/cutlist.csv           sizes, thickness, quantities
  out/oi_speaker.glb        3D assembly (open in Blender / any viewer)
  out/report.txt            volumes, heights, clash check

Axes (world, mm): X = left->right (from the front), Y = front->back, Z = up.

Joinery: every joint is a dado or rabbet DADO mm deep (18mm stock) or T9_RAB
deep (9mm stock). Pockets are computed automatically as the overlap between a
"guest" panel's full size and its "host" panel, so the 3D model and the DXFs
can't drift apart.

DXF layers
  CUT_THROUGH          outline + through holes (cut all the way)
  POCKET_A_<d>mm       pocket <d> deep on face A (the face the drawing shows)
  POCKET_B_<d>mm       pocket <d> deep on face B (flip the part)
  MARK_DRILL           hand-drill / spot marks (too small for the router)
  INFO                 labels only — do not cut
"""
import csv, json, math, os, sys
from dataclasses import dataclass, field
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box as sbox
from shapely.ops import unary_union
import manifold3d as mf
import ezdxf
import trimesh

# ----------------------------------------------------------------------------
# PARAMETERS — change these and re-run
# ----------------------------------------------------------------------------
P = dict(
    # stock
    T=18.0,            # main ply, MEASURE your sheet (often 17.5-18.2)
    T9=9.0,            # cap / Pi tray ply
    DADO=6.0,          # dado/rabbet depth in 18mm stock
    RAB9=4.5,          # rabbet depth in 9mm stock
    BIT=6.0,           # router bit diameter the shop uses (dogbones)
    # outer box
    W=150.0,           # external width
    # acoustic chamber — set the volume from WinISD and the internal height you want; the box
    # depth is derived from the two. 120 high x 114 wide keeps the chamber near square from
    # the front, with room around the driver and PR flanges
    CHAMBER_NET_L=2.2, # net internal volume, litres (before stuffing)
    CHAMBER_H=120.0,   # internal chamber height
    # Pi bay
    BAY_H=68.0,        # internal height of the Pi bay
    FIN_D=40.0,        # depth of the front cross fins
    # driver: Dayton RS100-8
    DRV_OD=98.0, DRV_CUT=77.2, DRV_FLANGE=5.0, DRV_Z_FRAC=0.5,
    # passive radiator: Dayton ND90-PR (4.07" overall, 3.35" cutout per Dayton, 2.39" deep).
    # the flange is not round: its recess is a PR_OD circle cut down to PR_FLATS across, square on
    # to the panel. None = plain round recess at PR_OD, which fits whatever the shape, with gaps.
    # MEASURE: PR_OD across the corners, PR_FLATS across the flat sides, PR_FLANGE its thickness
    PR_OD=103.4, PR_FLATS=None, PR_CUT=85.1, PR_FLANGE=5.0, PR_Z_FRAC=0.5, PR_BASKET=85.1,
    PR_DEPTH=60.7,
    CUT_CLEAR=0.5,     # added to driver/PR cutouts
    # mic cap (9mm ply)
    CAP_GAP=0.0,       # gap under the cap walls. 0 = legs stand on the body, the arches are the only gaps
    CAP_AIR=24.0,      # body top -> underside of grille (arches run up to CAP_BAR below it)
    MIC_STANDOFF=12.0, # XVF3800 board on legs, this far above the body top
    # cap walls are bridges on corner legs, open underneath so the XVF3800's LED light spills out
    CAP_LEG_W=30.0,    # leg width, from the outside corner (hides the corner pad behind it)
    CAP_BAR=8.0,       # wall left standing below the grille, above the opening
    CAP_WINDOW_R=6.0,  # radius on the opening's top corners
    CAP_PAD=20.0,      # the cap's feet, glued into each corner, this square
    SLOT_W=6.5, SLOT_PITCH=11.5, SLOT_LEN=104.0,
    # the cap is located by a hardwood dowel glued into each corner pad, dropping loose into a hole
    # in the body top — held by its own weight, lifts straight off to get at the board
    DOWEL_D=6.0, DOWEL_IN_PAD=8.0, DOWEL_IN_TOP=6.0, DOWEL_CLEAR=0.3,
    # cable conduit through the chamber (16mm PVC electrical conduit, ~13mm ID)
    CONDUIT_OD=16.0,
    SPK_WIRE_HOLE=8.0,
    # Pi tray connectors
    BARREL_HOLE=25.0,        # panel-mount barrel: 23mm thread, 30mm flange, panels <=16mm
    USB_CUT=(15.0, 8.0),     # USB-A panel stub body — CHECK with calipers
    USB_SCREW_PITCH=29.0,    # M3 ears ~29mm apart
)

for k, v in list(P.items()):
    globals()[k] = v

# displacement estimates inside the chamber (litres)
DRV_DISP_L = 0.12   # magnet + basket
PR_DISP_L = 0.04
CH_W = W - 2 * T          # 114
CH_H = CHAMBER_H


def chamber_depth():
    # depth follows from the volume, once the driver, PR and conduit have taken their share
    disp = DRV_DISP_L + PR_DISP_L + math.pi * (CONDUIT_OD / 2) ** 2 * CH_H * 1e-6
    d = (CHAMBER_NET_L + disp) / (CH_W * CH_H * 1e-6)
    return math.ceil(d * 2) / 2


CH_D = chamber_depth()
DEPTH = CH_D + 2 * T
CONDUIT_X = T + CONDUIT_OD / 2   # tight in the rear-left corner
CONDUIT_Y = DEPTH - T - CONDUIT_OD / 2
# the PR's widest extent across the panel — the flats if it has them, else its diameter
PR_HALF = (PR_FLATS if PR_FLATS else PR_OD) / 2
Z_BOT1 = T
Z_SH0 = T + BAY_H
Z_SH1 = Z_SH0 + T
Z_CH0 = Z_SH1
Z_CH1 = Z_CH0 + CH_H
H = Z_CH1 + T                         # body height
ZF = T + BAY_H / 2                    # cross fin centre height
XI0, XI1 = T - DADO, W - T + DADO     # extents of panels housed in the sides

# cap stack — the cap's legs stand on the body top (CAP_GAP above it, if set), pads inside the corners
Z_CAPW0 = H + CAP_GAP                  # cap wall bottom
Z_GR0 = H + CAP_AIR                    # grille underside
Z_CAPW1 = Z_GR0 + RAB9                 # walls go up into grille rabbet
Z_GR1 = Z_GR0 + T9
TOTAL_H = Z_GR1

DRV_Z = Z_CH0 + CH_H * DRV_Z_FRAC
PR_X = max(W / 2, CONDUIT_X + CONDUIT_OD / 2 + PR_BASKET / 2 + 0.5)
PR_Z = Z_CH0 + CH_H * PR_Z_FRAC

# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
AX = {'+X': np.array([1, 0, 0]), '-X': np.array([-1, 0, 0]),
      '+Y': np.array([0, 1, 0]), '-Y': np.array([0, -1, 0]),
      '+Z': np.array([0, 0, 1]), '-Z': np.array([0, 0, -1])}


@dataclass
class Part:
    name: str
    t: float
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    outline: Polygon                    # local uv, before through-cuts
    group: str = 'body'
    qty: int = 1
    note: str = ''
    world_box: tuple = None             # nominal AABB (for hosting)
    through: list = field(default_factory=list)   # shapely polys (local)
    pockets: list = field(default_factory=list)   # (poly, z0, z1)
    marks: list = field(default_factory=list)     # (x, y, d, label)
    circles: list = field(default_factory=list)   # (x, y, r) through circles kept as true arcs
    hosts_for: list = field(default_factory=list)

    def M(self):
        R = np.column_stack([self.u, self.v, self.n]).astype(float)
        return R, self.origin.astype(float)

    def to_local(self, p):
        R, o = self.M()
        return R.T @ (np.asarray(p, float) - o)

    def to_world(self, p):
        R, o = self.M()
        return R @ np.asarray(p, float) + o

    def local_box(self, wb):
        (x0, y0, z0), (x1, y1, z1) = wb
        cs = [self.to_local((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        cs = np.array(cs)
        return cs.min(0), cs.max(0)


def panel(name, wb, n, u, **kw):
    """Make a panel from a world AABB, choosing thickness axis n and width axis u."""
    n = AX[n]; u = AX[u]; v = np.cross(n, u)
    (lo, hi) = np.array(wb[0], float), np.array(wb[1], float)
    origin = np.zeros(3)
    for i in range(3):
        for a in (u, v, n):
            if a[i] != 0:
                origin[i] = lo[i] if a[i] > 0 else hi[i]
    size = hi - lo
    lu = abs(u @ size); lv = abs(v @ size); t = abs(n @ size)
    p = Part(name, t, origin, u, v, n, sbox(0, 0, lu, lv), world_box=(tuple(lo), tuple(hi)), **kw)
    return p


def aabb_inter(a, b):
    lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1])
    if np.any(hi - lo <= 1e-6):
        return None
    return lo, hi


def host(hostp, guestp):
    """Cut a pocket in hostp where guestp's nominal box overlaps it."""
    ib = aabb_inter(np.array(hostp.world_box), np.array(guestp.world_box))
    if ib is None:
        raise ValueError(f'{guestp.name} does not touch {hostp.name}')
    lo, hi = hostp.local_box(ib)
    z0, z1 = max(lo[2], 0), min(hi[2], hostp.t)
    poly = sbox(lo[0], lo[1], hi[0], hi[1])
    if z0 <= 1e-6 and z1 >= hostp.t - 1e-6:
        hostp.through.append(poly)
    else:
        assert z0 <= 1e-6 or z1 >= hostp.t - 1e-6, f'blind pocket mid-panel {hostp.name}'
        hostp.pockets.append((poly, z0, z1))
    hostp.hosts_for.append(guestp.name)


def dogbone_notch(rect, corners, r):
    """Union dogbone circles at given inside-corner points of a notch rect."""
    cx, cy = rect.centroid.x, rect.centroid.y
    shapes = [rect]
    for (x, y) in corners:
        d = np.array([x - cx, y - cy]); d = d / np.linalg.norm(d)
        # centre placed so the circle just covers the corner
        c = np.array([x, y]) - d * r * 0.95
        shapes.append(Point(*c).buffer(r, 64))
    return unary_union(shapes)


def stadium(cx, cy, length, width, horizontal=True):
    r = width / 2
    if horizontal:
        a = Point(cx - length / 2 + r, cy); b = Point(cx + length / 2 - r, cy)
    else:
        a = Point(cx, cy - length / 2 + r); b = Point(cx, cy + length / 2 - r)
    return shapely.geometry.LineString([a, b]).buffer(r, 64)


def wcirc(part, wx, wy_or_z, r, plane):
    """Local (u,v) of a world point on the part's face; plane tells which coords given."""
    if plane == 'XZ':
        p = part.to_local((wx, part.origin[1], wy_or_z))
    elif plane == 'XY':
        p = part.to_local((wx, wy_or_z, part.origin[2]))
    elif plane == 'YZ':
        p = part.to_local((part.origin[0], wx, wy_or_z))
    return p[0], p[1]


# ----------------------------------------------------------------------------
# PARTS
# ----------------------------------------------------------------------------
parts = {}
def add(p): parts[p.name] = p; return p

# --- body, 18mm ------------------------------------------------------------
side_L = add(panel('side_L', ((0, 0, 0), (T, DEPTH, H)), '+X', '+Y', note='face A = inside'))
side_R = add(panel('side_R', ((W - T, 0, 0), (W, DEPTH, H)), '-X', '-Y', note='face A = inside'))
bottom = add(panel('bottom', ((XI0, 0, 0), (XI1, DEPTH, T)), '+Z', '+X'))
top = add(panel('top', ((XI0, 0, H - T), (XI1, DEPTH, H)), '-Z', '+X', note='face A = underside'))
shelf = add(panel('shelf', ((XI0, 0, Z_SH0), (XI1, DEPTH, Z_SH1)), '+Z', '+X',
                  note='seals the chamber; pockets on both faces'))
baffle = add(panel('baffle', ((XI0, 0, Z_SH1 - DADO), (XI1, T, Z_CH1 + DADO)), '-Y', '+X',
                   note='face A = front (driver rebate)'))
back = add(panel('back', ((XI0, DEPTH - T, Z_SH1 - DADO), (XI1, DEPTH, Z_CH1 + DADO)), '+Y', '-X',
                 note='face A = outside (PR rebate)'))
hfin = add(panel('fin_h', ((XI0, 0, ZF - T / 2), (XI1, FIN_D, ZF + T / 2)), '+Z', '+X',
                 note='cross fin, horizontal'))
vfin = add(panel('fin_v', ((W / 2 - T / 2, 0, T - DADO), (W / 2 + T / 2, FIN_D, Z_SH0 + DADO)), '+X', '+Y',
                 note='cross fin, vertical'))

# joinery (host <- guest)
for g in (bottom, top, shelf, baffle, back, hfin):
    host(side_L, g); host(side_R, g)
for g in (baffle, back):
    host(top, g); host(shelf, g)
host(shelf, vfin); host(bottom, vfin)

# stopped fin dados: extend 3mm past the fin so the bit's round end is hidden
def extend_stopped(p, guest_name, extra=BIT / 2):
    pass  # handled below by lengthening pocket polys that end at FIN_D
for p in (side_L, side_R, shelf, bottom):
    new = []
    for (poly, z0, z1) in p.pockets:
        # find pockets that correspond to fins: those whose world Y-extent ends at FIN_D
        b = poly.bounds
        wlo = p.to_world((b[0], b[1], z0)); whi = p.to_world((b[2], b[3], z1))
        ymax = max(wlo[1], whi[1]); ymin = min(wlo[1], whi[1])
        if abs(ymax - FIN_D) < 1e-6 and ymin < 1e-6:
            # extend along whichever local axis maps to Y
            ax = 0 if abs(p.u[1]) else 1
            sign = p.u[1] if ax == 0 else p.v[1]
            x0, y0, x1, y1 = b
            if ax == 0:
                if sign > 0: x1 += BIT / 2
                else: x0 -= BIT / 2
            else:
                if sign > 0: y1 += BIT / 2
                else: y0 -= BIT / 2
            poly = sbox(x0, y0, x1, y1)
        new.append((poly, z0, z1))
    p.pockets = new

# cross-lap halving joint between the fins (through notches with dogbones)
r_bit = BIT / 2
# hfin local: u=X, v=Y. Notch open at the FRONT (y=0) half depth, at x = W/2 +- T/2
hx0 = W / 2 - T / 2 - XI0; hx1 = W / 2 + T / 2 - XI0
n_h = sbox(hx0, -1, hx1, FIN_D / 2)
hfin.through.append(dogbone_notch(n_h, [(hx0, FIN_D / 2), (hx1, FIN_D / 2)], r_bit))
# vfin local: u=Y, v=Z(from T-DADO). Notch open at the BACK (y=FIN_D)
vz0 = ZF - T / 2 - (T - DADO); vz1 = ZF + T / 2 - (T - DADO)
n_v = sbox(FIN_D / 2, vz0, FIN_D + 1, vz1)
vfin.through.append(dogbone_notch(n_v, [(FIN_D / 2, vz0), (FIN_D / 2, vz1)], r_bit))

# driver: through cut + flange rebate on front
bu, bv = wcirc(baffle, W / 2, DRV_Z, 0, 'XZ')
baffle.circles.append((bu, bv, (DRV_CUT + CUT_CLEAR) / 2))
baffle.pockets.append((Point(bu, bv).buffer((DRV_OD + 1) / 2, 128), T - DRV_FLANGE, T))
baffle.marks.append((bu, bv, 0, 'RS100-8: drill 6 pilot holes on assembly'))

# passive radiator on back
pu, pv = wcirc(back, PR_X, PR_Z, 0, 'XZ')
back.circles.append((pu, pv, (PR_CUT + CUT_CLEAR) / 2))
pr_recess = Point(pu, pv).buffer((PR_OD + 1) / 2, 128)
if PR_FLATS:
    f = (PR_FLATS + 1) / 2
    pr_recess = pr_recess.intersection(sbox(pu - f, pv - f, pu + f, pv + f))
back.pockets.append((pr_recess, T - PR_FLANGE, T))
back.marks.append((pu, pv, 0, 'ND90-PR: drill 4 pilots on assembly'))

# conduit + speaker wire holes
for p in (top, shelf):
    cu, cv = wcirc(p, CONDUIT_X, CONDUIT_Y, 0, 'XY')
    p.circles.append((cu, cv, CONDUIT_OD / 2 + 0.25))
su, sv = wcirc(shelf, W - T - 14, DEPTH - T - 14, 0, 'XY')
shelf.circles.append((su, sv, SPK_WIRE_HOLE / 2))

# --- cap, 9mm walls + grille on 18mm corner pads ------------------------------
mu, mv = top.to_local((W / 2, DEPTH / 2, H))[:2]
top.marks.append((mu, mv, 99, 'XVF3800 outline (99mm) — board on legs, rubber grommets, drill on assembly'))

capw_f = add(panel('cap_front', ((0, 0, Z_CAPW0), (W, T9, Z_CAPW1)), '-Y', '+X', group='cap', note='9mm, face A = inside'))
capw_b = add(panel('cap_back', ((0, DEPTH - T9, Z_CAPW0), (W, DEPTH, Z_CAPW1)), '+Y', '-X', group='cap', note='9mm, face A = inside'))
capw_l = add(panel('cap_left', ((0, RAB9, Z_CAPW0), (T9, DEPTH - RAB9, Z_CAPW1)), '+X', '+Y', group='cap', note='9mm'))
capw_r = add(panel('cap_right', ((W - T9, RAB9, Z_CAPW0), (W, DEPTH - RAB9, Z_CAPW1)), '-X', '-Y', group='cap', note='9mm'))
grille = add(panel('cap_grille', ((0, 0, Z_GR0), (W, DEPTH, Z_GR1)), '-Z', '+X', group='cap',
                   note='9mm, face A = underside (wall rabbet)'))
# faces A are the inner faces: flip front/back so n points inward
capw_f = parts['cap_front'] = panel('cap_front', ((0, 0, Z_CAPW0), (W, T9, Z_CAPW1)), '+Y', '+X', group='cap', note='9mm, face A = inside')
capw_b = parts['cap_back'] = panel('cap_back', ((0, DEPTH - T9, Z_CAPW0), (W, DEPTH, Z_CAPW1)), '-Y', '-X', group='cap', note='9mm, face A = inside')
for g in (capw_l, capw_r):
    host(capw_f, g); host(capw_b, g)
for g in (capw_f, capw_b, capw_l, capw_r):
    host(grille, g)
# grille slots
n_slots = int((DEPTH - 2 * T9 - 2 * 8) // SLOT_PITCH)
y0s = DEPTH / 2 - (n_slots - 1) * SLOT_PITCH / 2
for i in range(n_slots):
    gu, gv = wcirc(grille, W / 2, y0s + i * SLOT_PITCH, 0, 'XY')
    grille.through.append(stadium(gu, gv, SLOT_LEN, SLOT_W))

# openings: each wall is cut away from its bottom edge up to CAP_BAR below the grille, leaving a
# leg at each corner. the openings look straight through to the board, so the light its LEDs throw on
# the body top shows all the way round
WIN_TOP = Z_GR0 - CAP_BAR
assert WIN_TOP > Z_CAPW0 + 4, 'CAP_BAR leaves no opening — raise CAP_AIR or lower CAP_BAR'


def cap_opening(p, run_axis, a0, a1):
    """Cut the opening in wall p, between a0 and a1 along world axis run_axis (0 = X, 1 = Y)."""
    r = CAP_WINDOW_R
    pts = []
    for a in (a0, a1):
        for z in (Z_CAPW0 - r - 2, WIN_TOP):  # starts below the wall, so only the top corners round
            w = np.array(p.origin, float)
            w[run_axis] = a
            w[2] = z
            pts.append(p.to_local(w)[:2])
    pts = np.array(pts)
    rect = sbox(*pts.min(0), *pts.max(0))
    p.through.append(rect.buffer(-r, 64).buffer(r, 64))


for wall in (capw_f, capw_b):
    cap_opening(wall, 0, CAP_LEG_W, W - CAP_LEG_W)
for wall in (capw_l, capw_r):
    cap_opening(wall, 1, CAP_LEG_W, DEPTH - CAP_LEG_W)

# corner pads: the cap's feet. 18mm, glued into each corner behind the leg, standing on the body top.
# their height less CAP_GAP is their glue face inside the walls. each carries a dowel that drops into
# the top. the pads straddle the joint between side and top, so the dowel sits on the top's side of
# it — a hole in the side's end grain could not be cut before glue-up
assert T9 + CAP_PAD <= CAP_LEG_W, 'corner pads would show through the openings — widen CAP_LEG_W'
pad_corners = [('fl', T9, T9), ('fr', W - T9 - CAP_PAD, T9),
               ('rl', T9, DEPTH - T9 - CAP_PAD), ('rr', W - T9 - CAP_PAD, DEPTH - T9 - CAP_PAD)]
dowel_x = T + DOWEL_D / 2 + 1.5        # just clear of the side panel
dowel_y = T9 + DOWEL_D / 2 + 2.0       # in from the pad's outer edge
assert dowel_x + DOWEL_D / 2 < T9 + CAP_PAD - 2, 'dowel runs off the pad — widen CAP_PAD'
for tag, x0, y0 in pad_corners:
    pad = add(panel(f'cap_pad_{tag}', ((x0, y0, H), (x0 + CAP_PAD, y0 + CAP_PAD, H + T)), '-Z', '+X',
                    group='cap', note='18mm, face A = top; dowel hole underneath. cut from offcut'))
    dx = dowel_x if tag[1] == 'l' else W - dowel_x
    dy = dowel_y if tag[0] == 'f' else DEPTH - dowel_y
    lu, lv = pad.to_local((dx, dy, H))[:2]
    pad.pockets.append((Point(lu, lv).buffer(DOWEL_D / 2, 64), T - DOWEL_IN_PAD, T))  # glued, snug
    tu, tv = top.to_local((dx, dy, H))[:2]
    # top's face A is its underside, so this lands on face B, the outside. loose, so the cap lifts off
    top.pockets.append((Point(tu, tv).buffer(DOWEL_D / 2 + DOWEL_CLEAR, 64), 0.0, DOWEL_IN_TOP))
    # the cable comes up through the top in the rear-left corner — notch that pad round it
    cu, cv = pad.to_local((CONDUIT_X, CONDUIT_Y, H))[:2]
    hole = Point(cu, cv).buffer(CONDUIT_OD / 2 + 0.25, 64)
    if hole.intersects(pad.outline):
        assert not hole.intersects(Point(lu, lv).buffer(DOWEL_D / 2 + 1)), 'dowel runs into the cable notch'
        pad.through.append(hole)

# --- Pi tray, 9mm ------------------------------------------------------------
TR_X0, TR_X1 = T + 0.5, W - T - 0.5
plate = add(panel('tray_plate', ((TR_X0, DEPTH - T9, T), (TR_X1, DEPTH, Z_SH0 - 1)), '-Y', '+X', group='tray',
                  note='9mm, face A = inside. Connectors mount here'))
floor_ = add(panel('tray_floor', ((TR_X0, FIN_D + 1, T), (TR_X1, DEPTH - RAB9, T + T9)), '+Z', '+X', group='tray',
                   note='9mm; slides on bottom, stops on fins'))
host(plate, floor_)
# plate local u = X - TR_X0? panel(): u=+X origin at x=TR_X0 ; v = n x u = (-Y)x(X) = +Z, origin z=T
def plate_uv(x, z): return plate.to_local((x, DEPTH, z))[:2]
bx, bz = 40.0, 50.0
u_, v_ = plate_uv(bx, bz); plate.circles.append((u_, v_, BARREL_HOLE / 2))
plate.marks.append((u_, v_, 30, 'DC barrel (30mm flange)'))
uw, uh = USB_CUT
for zc in (42.0, 64.0):
    u_, v_ = plate_uv(85.0, zc)
    # plain slot with the bit's own corner radius — the receptacle shell is far smaller than the cut,
    # so it doesn't need square corners and dogbones would just look odd from outside
    rect = sbox(u_ - uw / 2, v_ - uh / 2, u_ + uw / 2, v_ + uh / 2)
    plate.through.append(rect.buffer(-r_bit, 64).buffer(r_bit, 64))
    for s in (-1, 1):
        plate.marks.append((u_ + s * USB_SCREW_PITCH / 2, v_, 3.2, 'M3'))
for xs in (112.5, 122.5):
    u_, v_ = plate_uv(xs, 55.0)
    plate.through.append(stadium(u_, v_, 40, 6.5, horizontal=False))
# finger notch at the top edge
u_, v_ = plate_uv(40.0, Z_SH0 - 1)
plate.through.append(Point(u_, v_).buffer(12, 64))
# Pi 5 standoffs on the floor (85x56 board, 58x49 hole pattern)
PI_X0, PI_Y0 = W / 2 - 42.5, 65.0
for (hx, hy) in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):
    fu, fv = floor_.to_local((PI_X0 + hx, PI_Y0 + hy, T + T9))[:2]
    floor_.marks.append((fu, fv, 2.7, 'M2.5'))
fu, fv = floor_.to_local((PI_X0, PI_Y0, T + T9))[:2]
floor_.marks.append((fu + 42.5, fv + 28, 0, 'Pi 5 (USB/Eth end -> right)'))

# ----------------------------------------------------------------------------
# build solids
# ----------------------------------------------------------------------------
def poly_to_cs(poly):
    polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    contours = []
    for pg in polys:
        ext = np.array(shapely.geometry.polygon.orient(pg, 1.0).exterior.coords)[:-1]
        contours.append(ext)
        for hole in shapely.geometry.polygon.orient(pg, 1.0).interiors:
            contours.append(np.array(hole.coords)[:-1])
    return mf.CrossSection(contours, mf.FillRule.EvenOdd)


def final_outline(p):
    shape = p.outline
    cut = list(p.through) + [Point(x, y).buffer(r, 128) for (x, y, r) in p.circles]
    if cut:
        shape = shape.difference(unary_union(cut))
    return shape


def solid(p):
    m = mf.Manifold.extrude(poly_to_cs(final_outline(p)), p.t)
    for (poly, z0, z1) in p.pockets:
        pm = mf.Manifold.extrude(poly_to_cs(poly.buffer(0)), z1 - z0 + 0.02).translate((0, 0, z0 - 0.01 if z0 > 0 else -0.01))
        m = m - pm
    R, o = p.M()
    mat = np.column_stack([R, o])
    return m.transform(mat)


solids = {k: solid(p) for k, p in parts.items()}

# hardware proxies (for clash check + viewer)
def cyl_y(x, z, r, y0, y1):
    return mf.Manifold.cylinder(y1 - y0, r, r, 96).rotate((-90, 0, 0)).translate((x, y0, z))
hw = {}
hw['hw_driver'] = (cyl_y(W / 2, DRV_Z, DRV_OD / 2, 0, DRV_FLANGE) + cyl_y(W / 2, DRV_Z, 76.5 / 2, DRV_FLANGE, 30)
                   + cyl_y(W / 2, DRV_Z, 35, 30, 51 + 0.0))
pr_front = DEPTH - PR_FLANGE - (PR_DEPTH - PR_FLANGE)
hw['hw_pr'] = (cyl_y(PR_X, PR_Z, PR_HALF, DEPTH - PR_FLANGE, DEPTH) + cyl_y(PR_X, PR_Z, PR_BASKET / 2 - 0.5, DEPTH - 40, DEPTH - PR_FLANGE)
               + cyl_y(PR_X, PR_Z, 30, pr_front, DEPTH - 40))
hw['hw_conduit'] = mf.Manifold.cylinder(H - T - Z_SH0 + 0, CONDUIT_OD / 2, CONDUIT_OD / 2, 48).translate((CONDUIT_X, CONDUIT_Y, Z_SH1)) \
    - mf.Manifold.cylinder(H, 6.5, 6.5, 48).translate((CONDUIT_X, CONDUIT_Y, 0))
hw['hw_conduit'] = mf.Manifold.cylinder(Z_CH1 - Z_CH0, CONDUIT_OD / 2, CONDUIT_OD / 2, 48).translate((CONDUIT_X, CONDUIT_Y, Z_CH0)) \
    - mf.Manifold.cylinder(H * 2, 6.5, 6.5, 48).translate((CONDUIT_X, CONDUIT_Y, 0))
hw['hw_pi'] = mf.Manifold.cube((85, 56, 34)).translate((PI_X0, PI_Y0, T + T9 + 6))
hw['hw_mic'] = mf.Manifold.cylinder(4, 99 / 2, 99 / 2, 96).translate((W / 2, DEPTH / 2, H + MIC_STANDOFF))

# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
report = []
def R(s=''): report.append(s); print(s)

R('OI SPEAKER ENCLOSURE v1 — generated report')
R(f'body W x D x H        : {W:.0f} x {DEPTH:.0f} x {H:.1f} mm')
R(f'with mic cap          : {TOTAL_H:.1f} mm tall')
R(f'chamber internal      : {CH_W:.0f} x {CH_D:.0f} x {CH_H:.1f} mm')
gross = CH_W * CH_D * CH_H * 1e-6
cond = math.pi * (CONDUIT_OD / 2) ** 2 * CH_H * 1e-6
R(f'chamber gross / net   : {gross:.3f} L / {gross - DRV_DISP_L - PR_DISP_L - cond:.3f} L '
  f'(driver {DRV_DISP_L}, PR {PR_DISP_L}, conduit {cond:.3f} L)')
R(f'Pi bay internal       : {CH_W:.0f} x {DEPTH - FIN_D:.0f} (behind fins) x {BAY_H:.0f} mm')
R(f'driver centre         : x={W/2:.1f} z={DRV_Z:.1f} (from floor)')
R(f'PR centre             : x={PR_X:.1f} z={PR_Z:.1f}  (offset {PR_X - W/2:+.1f} to clear conduit)')
R(f'conduit               : {CONDUIT_OD:.0f}mm OD at rear-left corner x={CONDUIT_X:.0f} y={CONDUIT_Y:.0f}')
R(f'cap openings          : {W - 2 * CAP_LEG_W:.0f} (front/back) and {DEPTH - 2 * CAP_LEG_W:.0f} (sides) wide, '
  f'{WIN_TOP - Z_CAPW0:.1f} tall, from {CAP_GAP:.0f} to {WIN_TOP - H:.0f} above the body top; board at {MIC_STANDOFF:.0f}')
R()
R('clash check (all parts + hardware, pairwise):')
names = list(solids) + list(hw)
allm = {**solids, **hw}
clashes = 0
for i, a in enumerate(names):
    for b in names[i + 1:]:
        if a.startswith('hw_conduit') and b in ('top', 'shelf') or b.startswith('hw_conduit') and a in ('top', 'shelf'):
            pass
        v = (allm[a] ^ allm[b]).volume()
        if v > 0.5:
            clashes += 1
            R(f'  CLASH {a} x {b}: {v:.1f} mm^3')
R(f'  {clashes} clashes')
# rim checks
R()
R('material left around cutouts:')
R(f'  baffle: driver rebate edge -> panel edge {min(W/2 - (DRV_OD+1)/2 - XI0, 0+999):.1f} mm (sides), '
  f'top {Z_CH1 + DADO - (DRV_Z + (DRV_OD+1)/2):.1f} mm')
R(f'  back  : PR rebate edge -> right edge {XI1 - (PR_X + PR_HALF + 0.5):.1f} mm, left {PR_X - PR_HALF - 0.5 - XI0:.1f} mm, '
  f'bottom {PR_Z - PR_HALF - 0.5 - (Z_SH1 - DADO):.1f} mm')
R(f'  driver/PR gap inside: {CH_D - (51.0 - DRV_FLANGE) - (PR_DEPTH - PR_FLANGE):.1f} mm front to back')
# glue faces of the chamber: check the chamber is closed — build air volume and test leak
air = mf.Manifold.cube((CH_W, CH_D, CH_H)).translate((T, T, Z_CH0))
body_names = ['side_L', 'side_R', 'top', 'shelf', 'baffle', 'back']
R()

# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(os.path.join(OUT, 'dxf'), exist_ok=True)


def add_poly(msp, poly, layer, dx=0, dy=0):
    polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    for pg in polys:
        for ring in [pg.exterior, *pg.interiors]:
            pts = [(x + dx, y + dy) for x, y in ring.coords]
            msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': layer})


def draw_part(msp, p, dx=0, dy=0, label=True):
    outer = final_outline(p)
    # true circles as CIRCLE entities; everything else as polylines
    base = p.outline.difference(unary_union(p.through)) if p.through else p.outline
    add_poly(msp, base, 'CUT_THROUGH', dx, dy)
    for (x, y, r) in p.circles:
        msp.add_circle((x + dx, y + dy), r, dxfattribs={'layer': 'CUT_THROUGH'})
    for (poly, z0, z1) in p.pockets:
        d = z1 - z0
        face = 'A' if z1 >= p.t - 1e-6 else 'B'
        layer = f'POCKET_{face}_{d:g}mm'
        if poly.geom_type == 'Polygon' and len(poly.exterior.coords) > 60 and abs(poly.area - math.pi * (poly.length / (2 * math.pi)) ** 2) / poly.area < 0.01:
            c = poly.centroid; r = poly.length / (2 * math.pi)
            msp.add_circle((c.x + dx, c.y + dy), r, dxfattribs={'layer': layer})
        else:
            # clip to part outline so pockets never extend past the edge
            add_poly(msp, poly.intersection(p.outline), layer, dx, dy)
    for (x, y, d, lab) in p.marks:
        if d > 0:
            msp.add_circle((x + dx, y + dy), d / 2, dxfattribs={'layer': 'MARK_DRILL'})
        if lab:
            msp.add_text(lab, height=3, dxfattribs={'layer': 'INFO'}).set_placement((x + dx + d / 2 + 1, y + dy + 1))
    if label:
        b = p.outline.bounds
        msp.add_text(f'{p.name}  {p.t:g}mm  x{p.qty}', height=5, dxfattribs={'layer': 'INFO'}).set_placement((b[0] + dx, b[3] + dy + 3))
        if p.note:
            msp.add_text(p.note, height=3, dxfattribs={'layer': 'INFO'}).set_placement((b[0] + dx, b[3] + dy + 10))


def new_doc():
    doc = ezdxf.new('R2010', setup=True)
    doc.units = ezdxf.units.MM
    cols = {'CUT_THROUGH': 7, 'POCKET_A_6mm': 1, 'POCKET_A_5mm': 3, 'POCKET_A_4.5mm': 1, 'POCKET_A_3mm': 6,
            'POCKET_B_6mm': 5, 'POCKET_B_4.5mm': 5, 'POCKET_A_8mm': 6, 'MARK_DRILL': 2, 'INFO': 8}
    for k, c in cols.items():
        doc.layers.add(k, color=c)
    return doc


for p in parts.values():
    doc = new_doc(); draw_part(doc.modelspace(), p); doc.saveas(os.path.join(OUT, 'dxf', f'{p.name}.dxf'))

# combined sheet: rows by thickness
doc = new_doc(); msp = doc.modelspace()
x = 0; y = 0; rowh = 0; sheet_w = 1200
layout = {}
for thick in sorted({p.t for p in parts.values()}, reverse=True):
    for p in parts.values():
        if p.t != thick: continue
        b = p.outline.bounds; w = b[2] - b[0]; h = b[3] - b[1]
        if x + w > sheet_w:
            x = 0; y -= rowh + 40; rowh = 0
        draw_part(msp, p, x - b[0], y - b[1]); layout[p.name] = (x, y, w, h)
        x += w + 40; rowh = max(rowh, h)
    x = 0; y -= rowh + 80; rowh = 0
doc.saveas(os.path.join(OUT, 'dxf', 'all_parts.dxf'))

# cutlist
with open(os.path.join(OUT, 'cutlist.csv'), 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['part', 'qty', 'thickness_mm', 'length_mm', 'width_mm', 'pockets', 'through_cuts', 'note'])
    for p in parts.values():
        b = p.outline.bounds
        wtr.writerow([p.name, p.qty, p.t, round(b[2] - b[0], 1), round(b[3] - b[1], 1), len(p.pockets),
                      len(p.through) + len(p.circles), p.note])

# GLB with per-part meshes + explode metadata
scene = trimesh.Scene()
mats = {'body': [214, 184, 140, 255], 'cap': [196, 160, 112, 255], 'tray': [180, 150, 110, 255],
        'hw': [40, 40, 44, 255]}
centre = np.array([W / 2, DEPTH / 2, H / 2])
meta = {}
for name, m in list(solids.items()) + list(hw.items()):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)
    grp = parts[name].group if name in parts else 'hw'
    tm.visual = trimesh.visual.ColorVisuals(tm, face_colors=mats[grp])
    # glTF is Y-up: map (x, y, z)mm -> (x, z, -y) metres*0.001? keep mm, Y-up
    T_ = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    tm.apply_transform(T_)
    scene.add_geometry(tm, node_name=name, geom_name=name)
    meta[name] = grp
scene.export(os.path.join(OUT, 'oi_speaker.glb'))
with open(os.path.join(OUT, 'parts.json'), 'w') as f:
    json.dump({'groups': meta, 'params': P, 'derived': dict(H=H, CH_H=CH_H, CH_W=CH_W, CH_D=CH_D, DEPTH=DEPTH,
                                                            TOTAL_H=TOTAL_H)}, f, indent=1)

with open(os.path.join(OUT, 'report.txt'), 'w') as f:
    f.write('\n'.join(report) + '\n')
print('wrote', OUT)
