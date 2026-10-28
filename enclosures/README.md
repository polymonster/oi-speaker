# Enclosures

Each enclosure design has its own folder, with a design script that generates everything about it. The tools in
[utils/](utils/) take it from there for any design: a Blender scene to look at it, a package for the CNC shop, and a
check of that package against the 3D model.

| Design | |
|---|---|
| [og/](og/) | the first Oi Speaker: 2.2 L passive-radiator box, Pi bay, lift-off mic cap |

## Building

From this folder, one command builds an enclosure (here `og`) from its design to a checked CNC package:

    uv run --no-project --with-requirements utils/requirements.txt python utils/build.py og

`uv` fetches the packages in [utils/requirements.txt](utils/requirements.txt) the first time. Add `--no-blender` to
skip the Blender step, or `--sheet 2440x1220` for a different sheet size. It stops at the first step that fails,
and ends with the checker's "all good". The package to send is `og/out/oi_og_cnc_v0.zip`, and `og/out/email.txt` has
the text for the email.

[build.py](utils/build.py) runs these, which can also be run one at a time (`python` meaning the `uv run ...`
above):

1. `python <folder>/<name>_enclosure.py`: the design (for og, `og/oi_og_enclosure.py`). Writes `<folder>/out/`.
   `build.py` finds it by that `_enclosure.py` ending.
2. `blender --background --python utils/build_blend.py -- <folder> --render`: `out/<name>.blend` to look at it,
   plus front and back renders for the shop's PDF. Optional, but `drawings.pdf` has no renders without it.
3. `python utils/export_cnc.py <folder>`: the package for the CNC shop. Every run rewrites `out/cnc/` and
   `out/<name>_cnc_<revision>.zip`, and `out/email.txt`.
4. `python utils/check_cnc.py <folder>`: reads the package back as the shop's software would and checks it.
   It must end with "all good".

Re-run all four after any change to the design, including setting the sheet thickness to what the shop measures.

## The tools

- **[build_blend.py](utils/build_blend.py)** imports the glb, sorts parts into collections, gives the wood a plywood
  material (veneers along the thin axis) and hardware its colours. Select the `controls` empty and drag its
  properties (N panel > Item > Properties): `explode`, `cutaway`, and any sliders the design declares (og has
  `tray`, which slides the Pi tray out, and `cap`, which lifts the cap off).
- **[blender_autoreload.py](utils/blender_autoreload.py)** is a Blender add-on: with it on, a `.blend` that
  `build_blend.py` made reloads itself in the open Blender whenever it's rebuilt, keeping your view. Install once:
  `blender --background --python utils/blender_autoreload.py -- --install`, or Edit > Preferences > Add-ons >
  Install from Disk.
- **[export_cnc.py](utils/export_cnc.py)** nests the parts on sheets (default 1220 × 607, a quarter board;
  `--sheet WxH` to change) and writes `out/cnc/`:
  - `sheet_<t>mm_<n>.dwg`: what the router cuts. Needs the free [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter);
    without it, `.dxf`
  - `sheet_<t>mm_<n>.pdf`: the same at 1:1, layers shown by colour
  - `drawings.pdf`: the brief (files, material, layer legend, faces), renders, then each sheet and each part
  - `cutlist.csv`

  `out/email.txt` is the same brief plus the questions for the shop, to paste into an email.
- **[check_cnc.py](utils/check_cnc.py)** checks each sheet: mm units, known layers, closed outlines, nothing doubled,
  parts inside the margin and apart; each part the same shape as the design and not mirrored; rebuilt in 3D from its
  lines and pocket depths and put back in the box, the same as the 3D model; no pockets on opposite faces leaving
  under 3 mm of wood; every inside corner reachable by the bit.
- [enclosure.py](utils/enclosure.py) and [dxf_parts.py](utils/dxf_parts.py) are shared by the others.

## Layers in the cut files

| Layer | |
|---|---|
| `CUT_THROUGH` | outlines and through holes |
| `POCKET_A_<d>mm` | pocket `<d>` mm deep into face A, the face the drawing shows |
| `POCKET_B_<d>mm` | pocket `<d>` mm deep into face B, the underside, drawn as seen through the board from face A (not mirrored) |
| `MARK_DRILL` | pilot holes, drilled by hand or spot-drilled |
| `INFO` | labels, only in the per-part DXFs in `out/dxf`; never on the sheets |

## Making a new design

Copy a design folder and change it, or start fresh. Whatever generates the design has to write to `<folder>/out/`:

- `dxf/<part>.dxf`: one per part on the cut list, on the layers above, in mm
- `cutlist.csv`: `part, qty, thickness_mm, length_mm, width_mm, ...`
- `<name>.glb`: the assembly, Y-up, mm, one mesh per part and piece of hardware
- `parts.json`: the manifest saying how the two relate, plus the design's own CNC brief and Blender settings.
  Its fields are documented in [enclosure.py](utils/enclosure.py), and the Blender ones in
  [build_blend.py](utils/build_blend.py). og's `NAME`, `CNC_BRIEF` and `BLEND` are a worked example.
