#!/usr/bin/env python3
"""
Oi Speaker enclosure, og design — parametric CNC model (v1)

One source of truth for every panel. Running this script regenerates:
  out/dxf/<part>.dxf        one file per part, layered by cut type
  out/dxf/all_parts.dxf     every part laid out (unnested) for the shop
  out/cutlist.csv           sizes, thickness, quantities
  out/oi_og.glb             3D assembly (open in Blender / any viewer)
  out/parts.json            the manifest the shared tools in ../utils read (see utils/enclosure.py)
  out/report.txt            volumes, heights, clash check

Axes (world, mm): X = left->right (from the front), Y = front->back, Z = up.

Joinery: every joint is a dado or rabbet DADO mm deep (18mm stock) or T9_RAB
deep (9mm stock). Pockets are computed automatically as the overlap between a
"guest" panel's full size and its "host" panel, so the 3D model and the DXFs
can't drift apart.

DXF layers
  CUT_THROUGH          outline + through holes (cut all the way)
  POCKET_A_<d>mm       pocket <d> deep on face A (the face the drawing shows)
  POCKET_B_<d>mm       pocket <d> deep on face B, drawn as seen through the board from face A (not mirrored)
  MARK_DRILL           hand-drill / spot marks (too small for the router)
  INFO                 labels only — do not cut
"""
import csv, json, math, os, sys
from dataclasses import dataclass, field
import numpy as np
import shapely
import shapely.affinity
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
    FIN_D=30.0,        # depth of the front cross fins
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
    CAP_AIR=21.0,      # body top -> underside of grille: the mic's ports top out at 18, so 3mm clear
    MIC_STANDOFF=10.0, # XVF3800 board on M2 x 10mm standoffs, this far above the body top
    MIC_D=100.0,       # ReSpeaker XVF3800 board, from Seeed's 2D mechanical drawing
    # cap walls are bridges on corner legs, open underneath so the XVF3800's LED light spills out
    CAP_LEG_W=30.0,    # leg width, from the outside corner (hides the corner pad behind it)
    CAP_BAR=12.0,      # wall left standing below the grille, above the opening: 9mm tall, under the mic board
    CAP_WINDOW_R=6.0,  # radius on the opening's top corners
    CAP_PAD=20.0,      # the cap's feet, glued into each corner, this square
    SLOT_W=6.5, SLOT_PITCH=11.5, SLOT_LEN=104.0,
    # the cap is located by a hardwood dowel glued into each corner pad, dropping loose into a hole
    # in the body top — held by its own weight, lifts straight off to get at the board
    DOWEL_D=6.0, DOWEL_IN_PAD=8.0, DOWEL_IN_TOP=6.0, DOWEL_CLEAR=0.3,
    # cable conduit through the chamber: 12mm OD tube, ~9mm bore, wide enough for a USB-C plug. any wider
    # and the passive radiator has to move off centre to clear it
    CONDUIT_OD=12.0,
    SPK_WIRE_HOLE=8.0,
    # Pi tray connectors
    # Adafruit 5607 panel-mount barrel lead (drawing C17817): M22 thread 14 long, 28mm flange 2.5 thick,
    # 33.5 overall, 30mm nut. 22mm behind a 9mm panel, plus the cable's strain relief
    BARREL_HOLE=22.5, BARREL_FLANGE=28.0, BARREL_FLANGE_T=2.5, BARREL_LEN=33.5, BARREL_THREAD=14.0,
    BARREL_NUT=30.0, BARREL_NUT_T=4.0,   # nut thickness not on the drawing — CHECK, <=5 fits a 9mm panel
    USB_CUT=(15.0, 8.0),     # USB-A panel stub body — CHECK with calipers
    USB_SCREW_PITCH=29.0,    # M3 ears ~29mm apart
    # screwed from inside: short wood screws through the ears into the plate, nothing shows outside.
    # if the ears are threaded, drill them out to 3.2mm first
    USB_PILOT=2.0,           # No.4 x 6mm pan-head wood screw pilot, ~5mm deep, hand-drilled through the ears
    BUTTON_HOLE=16.2,        # Adafruit 1505 16mm momentary: 15.6mm thread 18.5 long, 18mm bezel, 29.5 overall
    BUTTON_HEAD=18.0,
    BUTTON_CRIMPS=15.0,      # 2.8mm quick-connect crimps on the tags, behind the switch
    # Pi 5 + HiFiBerry Amp4 stack on M2.5 standoffs. PI_ROT turns the board about its own corner:
    #   90 = USB/Ethernet end facing the back plate, amp jack + speaker terminals facing the right wall
    #  180 = USB end facing the left wall, amp edge facing the back
    # PI_AT is where the board's corner (GPIO header end, USB-C edge) lands on the floor
    PI_ROT=90, PI_AT=(78.0, 40.0), PI_STANDOFF=8.0,   # M2.5 x 8mm standoffs
    USB_PLUG=(16.0, 9.0, 38.0),   # chunky USB-A plug overmould + strain relief: wide, tall, long past the port
    DC_PLUG=(10.0, 40.0),         # the barrel lead's far-end plug into the Amp4: diameter, length out of the jack
    BARREL_BOOT=(9.0, 12.0),      # the lead's strain relief behind the panel body: diameter, length
    BARREL_CABLE_D=4.0,
)

NAME = 'oi_og'   # names the outputs: oi_og.glb, oi_og.blend, oi_og_cnc_<REVISION>.zip
# bump whenever files go to the shop changed: it's on every page of drawings.pdf and in the zip's name
REVISION = 'v0'

# for ../utils/export_cnc.py: the brief's own words. {flip} is the parts with face B pockets, {bit} the BIT
CNC_BRIEF = dict(
    title='Oi Speaker enclosure - CNC brief',
    summary='One set of parts for a small speaker enclosure.',
    material=['18 mm and 9 mm birch plywood, BB grade or better (the edges are on show).'],
    questions=[
        "Measured thickness of the 18mm and 9mm sheets you'll cut from - the dados are sized to it, "
        "so the files will be regenerated to match before the final cut.",
        'Bit diameter - the design assumes {bit:g}mm (inside corners have a radius of half that).',
        'Can you machine pockets on both faces (POCKET_B layers, on {flip})? If not, I can cut those by hand.',
        'Will you spot-drill the MARK_DRILL points, or leave them?',
        'Do you stock birch ply (BB grade or better - the edges are on show), or should I supply it?',
    ],
)

# for ../utils/build_blend.py, in mm, design axes (the back is +Y)
BLEND = dict(
    collections={'body': 'Body', 'cap': 'Mic cap', 'tray': 'Pi tray', 'hw': 'Hardware'},
    explode_lift={'cap': 140.0},
    explode_move={'tray': [0.0, 160.0, 0.0]},   # the tray leaves through the back, not outward
    sliders=[
        # far enough that the floor clears the back, taking the hardware fixed to the tray with it
        dict(name='tray', description='0 in, 1 the Pi tray slid out of the back', groups=['tray'],
             meshes=['hw_pi', 'hw_plugs', 'hw_barrel', 'hw_usb_panel', 'hw_button'], move=[0.0, 190.0, 0.0]),
        dict(name='cap', description='0 on, 1 the mic cap lifted off', groups=['cap'], move=[0.0, 0.0, 110.0]),
    ],
    hardware_materials={
        'hw_driver': 'driver', 'hw_pr': 'driver', 'hw_driver_cone': 'cone', 'hw_pr_cone': 'cone',
        'hw_driver_spider': 'spider', 'hw_pr_spider': 'spider', 'hw_driver_terminals': 'brass',
        'hw_pr_stud': 'metal', 'hw_barrel_pin': 'nickel', 'hw_button_tags': 'brass', 'hw_button_crimps': 'crimp',
        'hw_pi': 'pcb', 'hw_mic': 'pcb', 'hw_mic_standoffs': 'brass', 'hw_pi_standoffs': 'brass',
        'hw_conduit': 'conduit', 'hw_barrel': 'driver', 'hw_plugs': 'driver', 'hw_usb_panel': 'driver',
        'hw_button': 'driver'},
    front=[0.0, -1.0, 0.0],
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
CONDUIT_X = T + CONDUIT_OD / 2   # against the left wall
# far enough forward to come up into the cap clear of the rear-left corner pad, so no pad needs a notch
CONDUIT_Y = DEPTH - T9 - CAP_PAD - CONDUIT_OD / 2 - 2.5
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
            # pad: the open end, run past the edge so only the stopped end's corners get rounded
            pad = [0.0, 0.0, 0.0, 0.0]
            if ax == 0:
                if sign > 0: x1 += BIT / 2; pad[0] = -BIT
                else: x0 -= BIT / 2; pad[2] = BIT
            else:
                if sign > 0: y1 += BIT / 2; pad[1] = -BIT
                else: y0 -= BIT / 2; pad[3] = BIT
            # the stopped end drawn round, as the bit cuts it. it's BIT/2 past the fin, so the fin still
            # sits in the full width
            padded = sbox(x0 + pad[0], y0 + pad[1], x1 + pad[2], y1 + pad[3])
            poly = padded.buffer(-BIT / 2, 64).buffer(BIT / 2, 64).intersection(sbox(x0, y0, x1, y1))
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

# passive radiator on back
pu, pv = wcirc(back, PR_X, PR_Z, 0, 'XZ')
back.circles.append((pu, pv, (PR_CUT + CUT_CLEAR) / 2))
pr_recess = Point(pu, pv).buffer((PR_OD + 1) / 2, 128)
if PR_FLATS:
    f = (PR_FLATS + 1) / 2
    pr_recess = pr_recess.intersection(sbox(pu - f, pv - f, pu + f, pv + f))
back.pockets.append((pr_recess, T - PR_FLANGE, T))

# conduit + speaker wire holes
for p in (top, shelf):
    cu, cv = wcirc(p, CONDUIT_X, CONDUIT_Y, 0, 'XY')
    p.circles.append((cu, cv, CONDUIT_OD / 2 + 0.25))
su, sv = wcirc(shelf, W - T - 14, DEPTH - T - 14, 0, 'XY')
shelf.circles.append((su, sv, SPK_WIRE_HOLE / 2))

# --- cap, 9mm walls + grille on 18mm corner pads ------------------------------
# ReSpeaker XVF3800 mounting holes, from Seeed's 2D mechanical drawing: board centre to hole, with the
# board's USB-C socket towards -y. The four rings on a 66mm square are the mics, not holes
MIC_HOLES = ((-21.355, 24.5), (21.355, 24.5), (0.0, -34.0))
MIC_PILOT = 1.6   # for M2 screws into the ply, or an M2 standoff's screw


def mic_world(a, b):
    # turned half round, so the USB-C socket faces the back, towards the cable coming up the conduit
    return W / 2 - a, DEPTH / 2 - b


for a, b in MIC_HOLES:
    hu, hv = top.to_local((*mic_world(a, b), H))[:2]
    top.marks.append((hu, hv, MIC_PILOT, 'M2 pilot, from the top (outside) face'))

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
dowel_y = dowel_x                      # on the pad's diagonal, so the pads are all one part, any way round
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
# layout as seen from behind (so world +X is on the left):
#   ---------------- [notch]
#   | | | grille | | |
#   [jack] [usb] [button]
# the jack is on the Amp4's side, so its lead runs straight down that wall; the USB and button sit
# behind the Pi's own plugs, which they clear
# the bottom row sits as low as the barrel flange allows over the floor rabbet
ROW_Z = T + T9 + 16.0                  # 43: barrel flange clears the floor by 1mm
USB_X, BUTTON_X, BARREL_X = 74.0, 42.0, 112.0
u_, v_ = plate_uv(BARREL_X, ROW_Z); plate.circles.append((u_, v_, BARREL_HOLE / 2))
u_, v_ = plate_uv(BUTTON_X, ROW_Z); plate.circles.append((u_, v_, BUTTON_HOLE / 2))
uw, uh = USB_CUT
u_, v_ = plate_uv(USB_X, ROW_Z)
# plain slot with the bit's own corner radius — the receptacle shell is far smaller than the cut,
# so it doesn't need square corners and dogbones would just look odd from outside
rect = sbox(u_ - uw / 2, v_ - uh / 2, u_ + uw / 2, v_ + uh / 2)
plate.through.append(rect.buffer(-r_bit, 64).buffer(r_bit, 64))
for s in (-1, 1):
    plate.marks.append((u_ + s * USB_SCREW_PITCH / 2, v_, USB_PILOT, 'pilot, inside, not through'))
# finger notch, top right from behind
NOTCH_X, NOTCH_R = TR_X0 + 18.0, 12.0
u_, v_ = plate_uv(NOTCH_X, Z_SH0 - 1)
notch = Point(u_, v_).buffer(NOTCH_R, 64)
plate.through.append(notch)
# vent grille across the top: upright slots on the cap grille's pitch, from above the barrel flange
# (the tallest thing in the row) to the top edge, skipping any that would run into the notch
VENT_Z0, VENT_Z1 = ROW_Z + 15.0 + 4.0, Z_SH0 - 1 - 4.0
for i in range(20):
    xs = TR_X1 - 7.5 - i * SLOT_PITCH
    u_, v_ = plate_uv(xs, (VENT_Z0 + VENT_Z1) / 2)
    slot = stadium(u_, v_, VENT_Z1 - VENT_Z0, SLOT_W, horizontal=False)
    if slot.distance(notch) < 3.0 or xs - SLOT_W / 2 < TR_X0 + 3.0:
        break
    plate.through.append(slot)
# Pi 5 standoffs on the floor (85x56 board, 58x49 hole pattern). board coords: a along the board,
# USB/Ethernet end at a=85; b across, b=0 the USB-C edge (where the Amp4's jack and terminals are)
def pi_world(a, b):
    r = math.radians(PI_ROT)
    return (PI_AT[0] + a * math.cos(r) - b * math.sin(r), PI_AT[1] + a * math.sin(r) + b * math.cos(r))
PI_HOLES = ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5))
for (ha, hb) in PI_HOLES:
    fu, fv = floor_.to_local((*pi_world(ha, hb), T + T9))[:2]
    floor_.marks.append((fu, fv, 2.7, 'M2.5 pilot'))

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
# driver and PR, detailed for looks. Built about local Z = depth into the box from the panel's outside
# face (negative = proud of it), then mounted on the baffle (sign +1, depth runs +Y) or the back (-1).
def revolve(profile):
    """Solid of revolution about local Z from an (r, depth) outline."""
    return mf.Manifold.revolve(mf.CrossSection([np.array(profile, float)], mf.FillRule.EvenOdd), 96)


def mount(m, x, z, y0, sign):
    return m.rotate((-90 * sign, 0, 0)).translate((x, y0, z))


def ring_of(m, r, n, start=90.0):
    out = None
    for k in range(n):
        a = math.radians(start + 360 / n * k)
        piece = m.translate((r * math.cos(a), r * math.sin(a), 0))
        out = piece if out is None else out + piece
    return out


def shell(r0, d0, r1, d1, t):
    """Conical wall t thick from (r0, d0) to (r1, d1), measured inward."""
    return revolve([(r0, d0), (r1, d1), (r1 - t, d1), (r0 - t, d0)])


def roll(r_in, r_out, d_base, height, t=1.0, n=24):
    """Half-roll surround: a half ellipse t thick standing `height` proud of d_base."""
    c, a = (r_in + r_out) / 2, (r_out - r_in) / 2
    outer = [(c - a * math.cos(math.pi * i / n), d_base - height * math.sin(math.pi * i / n)) for i in range(n + 1)]
    inner = [(c - (a - t) * math.cos(math.pi * i / n), d_base - (height - t) * math.sin(math.pi * i / n)) for i in range(n, -1, -1)]
    return revolve(outer + inner)


def windows(r, d0, d1, n, width):
    """n slots through a basket wall between depths d0 and d1, leaving n legs."""
    box = mf.Manifold.cube((r + 5, width, d1 - d0)).translate((0, -width / 2, d0))
    out = None
    for k in range(n):
        piece = box.rotate((0, 0, 360 / n * k + 180 / n))
        out = piece if out is None else out + piece
    return out


def button_head(d=7.6, h=2.2):
    # ISO 7380 M4 button head, dome facing out of the panel (-depth)
    return revolve([(0, 0), (d / 2, 0), (d / 2 - 0.6, -h * 0.6), (d / 4, -h), (0, -h)])


# Dayton RS100-8: round cast frame, 6 screws, black alu cone with a phase plug, yellow spider behind
drv_flange = revolve([(41.0, 0.4), (47.6, 0.0), (DRV_OD / 2, 1.0), (DRV_OD / 2, DRV_FLANGE), (41.0, DRV_FLANGE)])
drv_basket = shell(38.2, 4.0, 25.0, 30.0, 2.5) - windows(40, 9.0, 27.0, 4, 26.0)
drv_motor = (revolve([(0, 30.0), (35.0, 30.0), (35.0, 33.5), (33.0, 34.5), (33.0, 45.0), (35.0, 45.5), (35.0, 48.5),
                      (12.0, 48.5), (12.0, 51.0), (0, 51.0)])
             + revolve([(0, 13.5), (0, 3.0), (3.0, 3.4), (6.5, 6.5), (8.6, 10.5), (9.0, 13.5)]))   # phase plug
drv_screws = ring_of(button_head(), 44.7, 6)
hw['hw_driver'] = mount(drv_flange + drv_basket + drv_motor + drv_screws, W / 2, DRV_Z, 0.0, 1)
hw['hw_driver_cone'] = mount(roll(32.2, 40.9, 1.6, 3.2) + shell(33.0, 1.6, 13.5, 13.0, 1.0), W / 2, DRV_Z, 0.0, 1)
hw['hw_driver_spider'] = mount(revolve([(13.5, 21.0), (25.0, 21.0), (25.0, 22.0), (13.5, 22.0)]), W / 2, DRV_Z, 0.0, 1)
drv_tag = mf.Manifold.cube((3.0, 7.0, 0.8)).translate((-1.5, 27.5, 26.5))   # solder tags off the basket
hw['hw_driver_terminals'] = mount(drv_tag.translate((-5.5, 0, 0)) + drv_tag.translate((5.5, 0, 0)), W / 2, DRV_Z, 0.0, 1)

# Dayton ND90-PR: square flange with round corners, 4 screws, flat diaphragm on a deep roll, open frame
# behind with the spider and the stud the mass discs go on
pr_flats = PR_FLATS or 92.0   # visual only: the recess stays a PR_OD circle while PR_FLATS is None
pr_flange = ((mf.Manifold.cylinder(PR_FLANGE - 1.0, PR_OD / 2, PR_OD / 2, 96).translate((0, 0, 1.0))
              ^ mf.Manifold.cube((pr_flats, pr_flats, PR_FLANGE), center=True).translate((0, 0, PR_FLANGE / 2)))
             + revolve([(41.5, 0.3), (48.0, 0.0), (PR_OD / 2 - 1.0, 1.0), (41.5, 1.0)])
             - mf.Manifold.cylinder(PR_FLANGE + 2, 41.5, 41.5, 96).translate((0, 0, -1)))
pr_basket = shell(PR_BASKET / 2 - 0.5, PR_FLANGE, 30.0, 32.0, 2.0) - windows(45, 9.0, 29.0, 4, 30.0)
pr_hub = revolve([(26.0, 30.5), (30.0, 30.5), (30.0, 34.0), (3.5, 34.0), (3.5, 33.0), (26.0, 33.0)])   # spider ring
pr_screws = ring_of(button_head(), (PR_OD / 2 - 4.5), 4, 45.0)
hw['hw_pr'] = mount(pr_flange + pr_basket + pr_hub + pr_screws, PR_X, PR_Z, DEPTH, -1)
hw['hw_pr_cone'] = mount(roll(31.5, 41.4, 1.6, 4.0) + revolve([(0, 4.2), (32.3, 1.6), (32.3, 2.6), (0, 5.2)]),
                         PR_X, PR_Z, DEPTH, -1)
hw['hw_pr_spider'] = mount(revolve([(6.0, 29.5), (25.5, 29.5), (25.5, 30.5), (6.0, 30.5)]), PR_X, PR_Z, DEPTH, -1)
hw['hw_pr_stud'] = mount(revolve([(0, 5.2), (3.0, 5.2), (3.0, PR_DEPTH), (0, PR_DEPTH)]), PR_X, PR_Z, DEPTH, -1)
hw['hw_conduit'] = mf.Manifold.cylinder(H - T - Z_SH0 + 0, CONDUIT_OD / 2, CONDUIT_OD / 2, 48).translate((CONDUIT_X, CONDUIT_Y, Z_SH1)) \
    - mf.Manifold.cylinder(H, 6.5, 6.5, 48).translate((CONDUIT_X, CONDUIT_Y, 0))
hw['hw_conduit'] = mf.Manifold.cylinder(Z_CH1 - Z_CH0, CONDUIT_OD / 2, CONDUIT_OD / 2, 48).translate((CONDUIT_X, CONDUIT_Y, Z_CH0)) \
    - mf.Manifold.cylinder(H * 2, 6.5, 6.5, 48).translate((CONDUIT_X, CONDUIT_Y, 0))
# barrel lead: flange on the outside, then the thread through the plate, nut and body behind it
PLATE_IN = DEPTH - T9
BARREL_END = DEPTH + BARREL_FLANGE_T - BARREL_LEN   # back of the body, inside the bay


def thread(r_major, r_minor, d0, d1, pitch):
    """Rings of V grooves standing in for a thread, d0..d1 along the axis."""
    n = max(1, int((d1 - d0) / pitch))
    step = (d1 - d0) / n
    pts = [(0, d0), (r_minor, d0)]
    for i in range(n):
        pts += [(r_major, d0 + step * (i + 0.5)), (r_minor, d0 + step * (i + 1))]
    return revolve(pts + [(0, d1)])


def hex_nut(across_corners, d0, d1, chamfer=0.8):
    r = across_corners / 2
    bevel = revolve([(0, d0), (r * 0.9, d0), (r, d0 + chamfer), (r, d1 - chamfer), (r * 0.9, d1), (0, d1)])
    return mf.Manifold.cylinder(d1 - d0, r, r, 6).translate((0, 0, d0)) ^ bevel


# depths below are from the plate's outside face into the bay: outside is negative, the inside face is at T9
BJ_BACK = BARREL_LEN - BARREL_FLANGE_T     # back of the body
BJ_BORE = (3.2, 8.0)                       # socket mouth radius and depth
bj_body = (revolve([(0, -BARREL_FLANGE_T), (BARREL_FLANGE / 2 - 1.2, -BARREL_FLANGE_T),
                    (BARREL_FLANGE / 2, -BARREL_FLANGE_T + 1.2), (BARREL_FLANGE / 2, 0), (0, 0)])
           + revolve([(0, -BARREL_FLANGE_T - 1.0), (5.0, -BARREL_FLANGE_T - 1.0), (5.5, -BARREL_FLANGE_T - 0.5),
                      (5.5, -BARREL_FLANGE_T + 0.1), (0, -BARREL_FLANGE_T + 0.1)])          # collar round the socket
           + thread(11.0, 10.4, 0, BARREL_THREAD, 1.5)
           + hex_nut(BARREL_NUT, T9, T9 + BARREL_NUT_T)
           + revolve([(0, BARREL_THREAD - 0.1), (21.7 / 2, BARREL_THREAD - 0.1), (21.7 / 2, BJ_BACK - 1.0),
                      (21.7 / 2 - 1.0, BJ_BACK), (0, BJ_BACK)])
           - mf.Manifold.cylinder(BJ_BORE[1] + BARREL_FLANGE_T + 5, BJ_BORE[0], BJ_BORE[0], 48).translate((0, 0, -BARREL_FLANGE_T - 5)))
hw['hw_barrel'] = mount(bj_body, BARREL_X, ROW_Z, DEPTH, -1)
# socket contacts: the sleeve lining the mouth and the 2.1mm centre pin
hw['hw_barrel_pin'] = mount(revolve([(2.75, -BARREL_FLANGE_T - 0.6), (BJ_BORE[0] - 0.1, -BARREL_FLANGE_T - 0.6),
                                     (BJ_BORE[0] - 0.1, BJ_BORE[1]), (2.75, BJ_BORE[1])])
                            + revolve([(0, -0.5), (0.8, -0.5), (1.05, 0), (1.05, BJ_BORE[1]), (0, BJ_BORE[1])]),
                            BARREL_X, ROW_Z, DEPTH, -1)

def box(a0, a1, b0, b1, z0, z1):
    return mf.Manifold.cube((a1 - a0, b1 - b0, z1 - z0)).translate((a0, b0, z0))


def on_pi(m):
    """Board coords (z from the underside of the Pi) to world."""
    return m.rotate((0, 0, PI_ROT)).translate((PI_AT[0], PI_AT[1], T + T9 + PI_STANDOFF))


# Pi 5: Ethernet by the USB-C edge, then the two USB stacks; ports stand 2.5 proud of the board
PCB = 1.6
HAT_Z = PCB + 12.0                      # Amp4 underside, on the 2x20 header
USB_B = (29.1, 47.0)                    # USB stack centres across the board
USB_LOW_Z = PCB + 4.5                   # lower port centre
pi_stack = (box(0, 85, 0, 56, 0, PCB)
            + box(66, 87.5, 2.2, 18.2, PCB, PCB + 13.5)
            + sum((box(70, 87.5, b - 7.5, b + 7.5, PCB, PCB + 16) for b in USB_B), mf.Manifold())
            + box(7, 58, 49.5, 54.5, PCB, HAT_Z)
            # Amp4: 65x56.5 board, 6-way speaker/power terminal and DC jack overhanging the USB-C edge
            + box(0, 65, 0, 56.5, HAT_Z, HAT_Z + PCB)
            + box(30.4, 58.8, -3, 8, HAT_Z + PCB, HAT_Z + PCB + 14)
            + box(22.8, 30.8, -5, 9, HAT_Z + PCB, HAT_Z + PCB + 11))
hw['hw_pi'] = on_pi(pi_stack)
# the standoffs it sits on: M2.5, 5mm across the flats
hw['hw_pi_standoffs'] = sum((mf.Manifold.cylinder(PI_STANDOFF, 2.85, 2.85, 6).translate((*pi_world(a, b), T + T9))
                             for a, b in PI_HOLES), mf.Manifold())
# what plugs into it: the mic and the panel USB lead in the two lower USB ports, the barrel lead in the Amp4
uw_, uh_, ul_ = USB_PLUG
dc_d, dc_l = DC_PLUG
DC_Z = HAT_Z + PCB + 6.5
plugs = sum((box(87.5, 87.5 + ul_, b - uw_ / 2, b + uw_ / 2, USB_LOW_Z - uh_ / 2, USB_LOW_Z + uh_ / 2) for b in USB_B),
            mf.Manifold())
plugs += mf.Manifold.cylinder(dc_l, dc_d / 2, dc_d / 2, 48).rotate((90, 0, 0)).translate((26.8, -5, DC_Z))
hw['hw_plugs'] = on_pi(plugs)


def tube(points, d):
    """A cable through the given points: capsules between neighbours, so bends are rounded."""
    balls = [mf.Manifold.sphere(d / 2, 16).translate(tuple(p)) for p in points]
    return sum(((a + b).hull() for a, b in zip(balls, balls[1:])), mf.Manifold())


# barrel lead: boot behind the panel body, then the cable rises over the Pi's plugs, runs to the amp side
# and turns into the back of the DC plug
boot_d, boot_l = BARREL_BOOT
BOOT_END = np.array([BARREL_X, BARREL_END - boot_l, ROW_Z])
hw['hw_barrel'] += mount(thread(boot_d / 2, boot_d / 2 - 0.5, BJ_BACK - 0.1, BJ_BACK + boot_l, 2.0),   # ribbed boot
                         BARREL_X, ROW_Z, DEPTH, -1)
dc_z = T + T9 + PI_STANDOFF + DC_Z
dc_end = np.array([*pi_world(26.8, -5 - dc_l), dc_z])
dc_dir = np.array([*pi_world(26.8, -1), 0]) - np.array([*pi_world(26.8, 0), 0])
TOP_RUN = Z_SH0 - BARREL_CABLE_D / 2 - 6       # just under the shelf
turn = dc_end + dc_dir * 5 + np.array([0, 8, 0])
hw['hw_barrel_cable'] = tube([BOOT_END - [0, BARREL_CABLE_D / 2 + 0.2, 0], BOOT_END - [0, 12, -10], [BARREL_X, BOOT_END[1] - 20, TOP_RUN],
                              [turn[0], BOOT_END[1] - 30, TOP_RUN], turn + [0, 15, 0], turn, dc_end + dc_dir * (BARREL_CABLE_D / 2 + 0.2)],
                             BARREL_CABLE_D)

# panel USB lead (Adafruit 'panel mount USB extension'): ears screwed to the inside face, the receptacle
# shell through the slot, moulding and strain relief behind
hw['hw_usb_panel'] = (mf.Manifold.cube((12.5, T9 - 1, 5.0)).translate((USB_X - 6.25, PLATE_IN, ROW_Z - 2.5))
                      + mf.Manifold.cube((37.0, 3.0, 10.0)).translate((USB_X - 18.5, PLATE_IN - 3, ROW_Z - 5))
                      + mf.Manifold.cube((18.0, 22.0, 13.0)).translate((USB_X - 9, PLATE_IN - 25, ROW_Z - 6.5))
                      + cyl_y(USB_X, ROW_Z, 3.5, PLATE_IN - 33, PLATE_IN - 25))
# button: bezel and cap outside, threaded body through, nut, tags and crimped leads behind
BT_BODY = 18.5   # threaded body behind the bezel
hw['hw_button'] = mount(revolve([(0, -3.0), (6.9, -3.0), (6.9, -2.4), (BUTTON_HEAD / 2 - 0.8, -2.4),   # bezel ring
                                 (BUTTON_HEAD / 2, -1.2), (BUTTON_HEAD / 2, 0), (0, 0)])
                        + revolve([(0, -2.9), (6.5, -2.9), (6.5, -6.6), (5.6, -7.6), (3.5, -8.0), (0, -8.0)])  # domed cap
                        + thread(7.8, 7.3, 0, BT_BODY, 1.0)
                        + hex_nut(21.0, T9, T9 + 3.0)
                        + mf.Manifold.cube((11.0, 6.0, 1.0), center=True).translate((0, 0, BT_BODY + 0.5)),  # tag base
                        BUTTON_X, ROW_Z, DEPTH, -1)
# two 2.8mm quick-connect tags, and the insulated crimps pushed onto them
bt_tag = mf.Manifold.cube((2.8, 0.5, 2.0)).translate((-1.4, -0.25, BT_BODY + 1.0))
hw['hw_button_tags'] = mount(bt_tag.translate((-3.2, 0, 0)) + bt_tag.translate((3.2, 0, 0)), BUTTON_X, ROW_Z, DEPTH, -1)
bt_crimp = revolve([(0, 0), (1.9, 0), (2.4, 2.0), (2.4, BUTTON_CRIMPS - 4), (1.6, BUTTON_CRIMPS), (0, BUTTON_CRIMPS)]) \
    .translate((0, 0, BT_BODY + 3.0))
hw['hw_button_crimps'] = mount(bt_crimp.translate((-3.2, 0, 0)) + bt_crimp.translate((3.2, 0, 0)),
                               BUTTON_X, ROW_Z, DEPTH, -1)
# the board, in its own coords (USB-C edge towards -y) as on Seeed's drawing: 1.2mm PCB, the USB-C and
# 3.5mm jack standing up at that edge, 8mm overall from the board's underside, and the XIAO hanging
# underneath at the opposite edge
MIC_PCB, MIC_PORTS_H, MIC_XIAO_H = 1.2, 8.0, 5.66
mic_board = (mf.Manifold.cylinder(MIC_PCB, MIC_D / 2, MIC_D / 2, 96)
             + box(-14.5, -5.5, -49.0, -41.0, MIC_PCB, MIC_PORTS_H)      # USB-C
             + box(16.0, 28.0, -44.0, -34.0, MIC_PCB, MIC_PORTS_H)       # 3.5mm jack
             + box(-9.0, 9.0, 30.0, 51.0, MIC_PCB - MIC_XIAO_H, 0.0))    # XIAO ESP32S3, underneath
hw['hw_mic'] = mic_board.rotate((0, 0, 180)).translate((W / 2, DEPTH / 2, H + MIC_STANDOFF))   # as mic_world()
# M2 standoffs under its three holes
hw['hw_mic_standoffs'] = sum((mf.Manifold.cylinder(MIC_STANDOFF, 2.0, 2.0, 6).translate((*mic_world(a, b), H))
                              for a, b in MIC_HOLES), mf.Manifold())

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
R(f'barrel jack           : {BARREL_LEN - BARREL_FLANGE_T - T9:.1f} mm behind the plate')
# room left for cables to turn: plugs on the Pi against the back-plate hardware and the walls
R('room for cables to turn — gap from the plugs in the Pi stack to:')
for k in ('hw_barrel', 'hw_button', 'hw_button_crimps', 'hw_usb_panel'):
    R(f'  {k:<19} : {hw["hw_plugs"].min_gap(hw[k], 100):.1f} mm')
for k in ('side_L', 'side_R', 'fin_h', 'fin_v'):
    R(f'  {k:<19} : {hw["hw_plugs"].min_gap(solids[k], 100):.1f} mm')
R(f'conduit               : {CONDUIT_OD:.0f}mm OD at rear-left corner x={CONDUIT_X:.0f} y={CONDUIT_Y:.0f}')
R(f'cap openings          : {W - 2 * CAP_LEG_W:.0f} (front/back) and {DEPTH - 2 * CAP_LEG_W:.0f} (sides) wide, '
  f'{WIN_TOP - Z_CAPW0:.1f} tall, from {CAP_GAP:.0f} to {WIN_TOP - H:.0f} above the body top; board at {MIC_STANDOFF:.0f}')
R(f'mic board             : underside {MIC_STANDOFF - (WIN_TOP - H):.1f} mm above the top of the openings, ports '
  f'{Z_GR0 - (H + MIC_STANDOFF + MIC_PORTS_H):.1f} mm under the grille, XIAO {MIC_STANDOFF - MIC_XIAO_H + MIC_PCB:.1f} mm over the top panel')
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
# the other way round: every pocket that takes a mating part should be filled by it, so joints are tight.
# a pocket nothing fills is a hole or rebate for hardware, or a dowel hole
R()
R('pockets: how much of each is filled by the part or hardware that goes in it')
everything = {**solids, **hw}
for name, p in parts.items():
    others = None
    for k, m in everything.items():
        if k != name:
            others = m if others is None else others + m
    Rm, o = p.M()
    for (poly, z0, z1) in p.pockets:
        region = poly.intersection(final_outline(p))
        if region.is_empty or region.area < 1:
            continue
        pm = mf.Manifold.extrude(poly_to_cs(region.buffer(0)), z1 - z0).translate((0, 0, z0)).transform(np.column_stack([Rm, o]))
        filled = (pm ^ others).volume() / pm.volume()
        b = region.bounds
        R(f'  {name:<12} {z1 - z0:g}mm deep, {b[2] - b[0]:6.1f} x {b[3] - b[1]:6.1f}: {filled * 100:5.1f}% filled')
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
for old in os.listdir(os.path.join(OUT, 'dxf')):   # parts renamed or merged since the last run
    if old.endswith('.dxf'):
        os.remove(os.path.join(OUT, 'dxf', old))


def flat(p):
    """What the router sees of a part: the cut-out shape, and its pockets by face and depth."""
    base = p.outline.difference(unary_union(p.through)) if p.through else p.outline
    for (x, y, r) in p.circles:
        base = base.difference(Point(x, y).buffer(r, 64))
    pockets = {}
    for (poly, z0, z1) in p.pockets:
        k = ('A' if z1 >= p.t - 1e-6 else 'B', round(z1 - z0, 3))
        region = poly.intersection(p.outline)
        pockets[k] = region if k not in pockets else pockets[k].union(region)
    return base, pockets


def same_part(a, b):
    """True when b is a cut the same as a, turned on the sheet but not flipped over."""
    if a.t != b.t:
        return False
    (fa, ka), (fb, kb) = flat(a), flat(b)
    if set(ka) != set(kb) or abs(fa.area - fb.area) > 0.5:
        return False
    for angle in (0, 90, 180, 270):
        origin = sbox(*fa.bounds).centroid
        turned = sbox(*shapely.affinity.rotate(fa, angle, origin=origin).bounds).centroid
        target = sbox(*fb.bounds).centroid
        def move(g):
            return shapely.affinity.translate(shapely.affinity.rotate(g, angle, origin=origin),
                                              target.x - turned.x, target.y - turned.y)
        err = move(fa).symmetric_difference(fb).area + sum(move(ka[k]).symmetric_difference(kb[k]).area for k in ka)
        if err < 0.5:
            return True
    return False


# parts that are cut the same are one entry on the cut list, with a quantity: (name, part, qty, members)
cut_parts = []
for p in parts.values():
    for i, (name, lead, qty, members) in enumerate(cut_parts):
        if same_part(lead, p):
            cut_parts[i] = (name, lead, qty + p.qty, members + [p.name])
            break
    else:
        cut_parts.append((p.name, p, p.qty, [p.name]))
for i, (name, lead, qty, members) in enumerate(cut_parts):
    if len(members) > 1:   # cap_front + cap_back -> cap_front_back
        prefix = os.path.commonprefix(members)
        prefix = prefix[:prefix.rfind('_') + 1]
        cut_parts[i] = (prefix + '_'.join(m[len(prefix):] for m in members), lead, qty, members)


def add_poly(msp, poly, layer, dx=0, dy=0):
    polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    for pg in polys:
        for ring in [pg.exterior, *pg.interiors]:
            pts = [(x + dx, y + dy) for x, y in ring.coords]
            msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': layer})


def draw_part(msp, p, dx=0, dy=0, label=True, name=None, qty=None):
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
    # marks get a number on the drawing and their note in a key under the part, so notes on marks close
    # together don't run into each other. marks sharing a note share a number
    b = p.outline.bounds
    key = []
    for (x, y, d, lab) in p.marks:   # pilot holes, drilled by hand on assembly
        msp.add_circle((x + dx, y + dy), d / 2, dxfattribs={'layer': 'MARK_DRILL'})
        if lab:
            if lab not in key:
                key.append(lab)
            r = max(d / 2, 1.0)
            msp.add_text(str(key.index(lab) + 1), height=3, dxfattribs={'layer': 'INFO'}).set_placement(
                (x + dx + r * 0.7 + 0.5, y + dy + r * 0.7 + 0.5))
    for i, lab in enumerate(key):
        msp.add_text(f'{i + 1}  {lab}', height=3, dxfattribs={'layer': 'INFO'}).set_placement((b[0] + dx, b[1] + dy - 6 - 5 * i))
    if label:
        msp.add_text(f'{name or p.name}  {p.t:g}mm  x{qty or p.qty}', height=5, dxfattribs={'layer': 'INFO'}).set_placement((b[0] + dx, b[3] + dy + 3))
        if p.note:
            msp.add_text(p.note, height=3, dxfattribs={'layer': 'INFO'}).set_placement((b[0] + dx, b[3] + dy + 10))


def new_doc():
    doc = ezdxf.new('R2010', setup=True)
    doc.units = ezdxf.units.MM
    cols = {'CUT_THROUGH': 7, 'POCKET_A_6mm': 1, 'POCKET_A_5mm': 3, 'POCKET_A_4.5mm': 1, 'POCKET_A_3mm': 6,
            'POCKET_B_6mm': 5, 'POCKET_B_3mm': 4, 'POCKET_B_4.5mm': 5, 'POCKET_A_8mm': 6, 'MARK_DRILL': 2, 'INFO': 8}
    for k, c in cols.items():
        doc.layers.add(k, color=c)
    return doc


for name, p, qty, _ in cut_parts:
    doc = new_doc(); draw_part(doc.modelspace(), p, name=name, qty=qty); doc.saveas(os.path.join(OUT, 'dxf', f'{name}.dxf'))

# combined sheet: rows by thickness
doc = new_doc(); msp = doc.modelspace()
x = 0; y = 0; rowh = 0; sheet_w = 1200
layout = {}
for thick in sorted({p.t for p in parts.values()}, reverse=True):
    for name, p, qty, _ in cut_parts:
        if p.t != thick: continue
        b = p.outline.bounds; w = b[2] - b[0]; h = b[3] - b[1]
        if x + w > sheet_w:
            x = 0; y -= rowh + 40; rowh = 0
        draw_part(msp, p, x - b[0], y - b[1], name=name, qty=qty); layout[name] = (x, y, w, h)
        x += w + 40; rowh = max(rowh, h)
    x = 0; y -= rowh + 80; rowh = 0
doc.saveas(os.path.join(OUT, 'dxf', 'all_parts.dxf'))

# cutlist
with open(os.path.join(OUT, 'cutlist.csv'), 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['part', 'qty', 'thickness_mm', 'length_mm', 'width_mm', 'pockets', 'through_cuts', 'note'])
    for name, p, qty, _ in cut_parts:
        b = p.outline.bounds
        wtr.writerow([name, qty, p.t, round(b[2] - b[0], 1), round(b[3] - b[1], 1), len(p.pockets),
                      len(p.through) + len(p.circles), p.note])

# GLB with per-part meshes + explode metadata
scene = trimesh.Scene()
GLB_ORIGIN = (-W / 2, -DEPTH / 2, 0.0)
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
    tm.apply_translation(GLB_ORIGIN)   # scene origin at the middle of the base; DXFs are unaffected
    tm.apply_transform(T_)
    scene.add_geometry(tm, node_name=name, geom_name=name)
    meta[name] = grp
scene.export(os.path.join(OUT, f'{NAME}.glb'))
with open(os.path.join(OUT, 'parts.json'), 'w') as f:
    # where each part sits in the box: local (u, v) is the drawing seen from face A, local z runs from
    # face B (0) to face A (t). check_cnc.py puts the cut parts back with this to prove which way up they are
    placement = {p.name: dict(origin=p.origin.tolist(), u=p.u.tolist(), v=p.v.tolist(), n=p.n.tolist(), t=p.t,
                              bounds=list(p.outline.bounds)) for p in parts.values()}
    json.dump({'name': NAME, 'revision': REVISION, 'groups': meta, 'cut_as': {name: members for name, _, _, members in cut_parts},
               'placement': placement, 'origin': list(GLB_ORIGIN),
               'cnc': dict(bit=BIT, brief=CNC_BRIEF), 'blend': BLEND,
               'params': P, 'derived': dict(H=H, CH_H=CH_H, CH_W=CH_W, CH_D=CH_D, DEPTH=DEPTH,
                                            TOTAL_H=TOTAL_H)}, f, indent=1)

with open(os.path.join(OUT, 'report.txt'), 'w') as f:
    f.write('\n'.join(report) + '\n')
print('wrote', OUT)
