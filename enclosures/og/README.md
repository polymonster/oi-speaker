# Oi Speaker enclosure — v1

Edit the `P = dict(...)` block at the top of `oi_enclosure.py`, then:

    pip install ezdxf manifold3d trimesh shapely numpy scipy
    python3 oi_enclosure.py      # regenerates out/

Key knobs: `T` (measured ply thickness), `DADO`, `CHAMBER_NET_L` (from WinISD) and `CHAMBER_H` — box depth is
derived from the two, `BAY_H`, `W`, `PR_OD`/`PR_FLATS` (passive radiator flange), cap gap/heights, connector cutout sizes.

## Outputs
- `out/dxf/*.dxf` one per part; `all_parts.dxf` everything laid out (not nested)
- `out/cutlist.csv`, `out/report.txt` (volumes, clash check)
- `out/oi_speaker.glb` assembly for Blender (Y-up, mm)
- `out/oi_speaker.blend`, made from the glb by `blender --background --python build_blend.py`. Parts are sorted into
  collections with ply/hardware materials. Select the `controls` empty and drag its `explode` and `cutaway`
  properties (N panel > Item > Properties). Add `-- --render` to also write `out/blender_preview.png`.

## DXF layers
CUT_THROUGH · POCKET_A_<d>mm (face shown) · POCKET_B_<d>mm (flip part) · MARK_DRILL (hand drill) · INFO (don't cut)

## Assembly order
1. Glue carcass: sides + bottom + shelf + top + baffle + back (all dado/rabbet). Glue conduit tube between shelf and top, seal both ends.
2. Pre-glue the cross fins (halving joint), slide the cross into the open-front dados.
3. Cap: glue the walls into the grille rabbet, then an 18mm pad into each corner behind the legs (rear-left pad
   is notched for the cable). Glue a 6mm hardwood dowel, cut to 13mm, into the hole under each pad; it drops loose
   into the matching hole in the top panel's outside face (POCKET_B on `top`). The cap is held by its weight and
   lifts off for access.
4. Tray: floor into plate rabbet. Fit barrel jack, 2× USB stubs, Pi on M2.5 standoffs. Slides in from the back, stops on the fins.

## Check before cutting
- USB stub body size (USB_CUT) — measure with calipers.
- Driver/PR flange thickness (DRV_FLANGE/PR_FLANGE) for flush rebates.
- XVF3800 mounting holes aren't in the model — drill on assembly.
