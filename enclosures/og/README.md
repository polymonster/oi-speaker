# Oi Speaker enclosure — og

Parts, hardware and key measurements: [SPEC.md](SPEC.md). Running it, Blender, and the CNC package:
[../README.md](../README.md).

Edit the `P = dict(...)` block at the top of [oi_og_enclosure.py](oi_og_enclosure.py), then from `enclosures/`:

    uv run --no-project --with-requirements utils/requirements.txt python utils/build.py og

That regenerates everything in `og/out/`, ending with "all good" and the package to send, `oi_og_cnc_v0.zip`.
Before sending changed files to the shop, bump `REVISION` (`v0` → `v1`) near the top of the design script: it
names the zip and is stamped on every page of `drawings.pdf`.

Key knobs: `T`/`T9` (measured ply thickness), `DADO`, `BIT`, `CHAMBER_NET_L` (from the box sim) and `CHAMBER_H`:
the box depth is derived from the two. Also `BAY_H`, `W`, `PR_OD`/`PR_FLATS` (passive radiator flange), cap gap
and heights, connector cutout sizes. The CNC brief's wording and questions are `CNC_BRIEF`; Blender's
collections, sliders and hardware colours are `BLEND`.

## Outputs, in `out/`
- `dxf/*.dxf`, one per part; `all_parts.dxf`, everything laid out (not nested)
- `cutlist.csv`, `report.txt` (volumes, clearances, clash check)
- `oi_og.glb`, the assembly (Y-up, mm, origin at the middle of the base), and `parts.json`, its manifest
- from the shared tools: `oi_og.blend`, `preview_front.png`/`preview_back.png`, `cnc/`, `oi_og_cnc_v0.zip`,
  `email.txt`

[og.00sim](og.00sim) is the box simulation (simulator.00aud.io): RS100-8 and ND90-PR in 2.2 L.

## Assembly order
1. Glue carcass: sides + bottom + shelf + top + baffle + back (all dado/rabbet). Glue the conduit tube between
   shelf and top, seal both ends.
2. Pre-glue the cross fins (halving joint), slide the cross into the open-front dados.
3. Cap: glue the walls into the grille rabbet, then an 18mm pad into each corner behind the legs (all four
   the same). Glue a 6mm hardwood dowel, cut to 13mm, into the hole under each pad; it drops loose into the
   matching hole in the top panel's outside face (POCKET_B on `top`). The cap is held by its weight and lifts
   off for access. The XVF3800 goes on the top on M2 × 10mm standoffs, in the three marked pilots.
4. Tray: floor into plate rabbet. Fit barrel jack, USB stub (from inside: two No.4 × 6mm wood screws through its
   ears into 2mm pilots, not through the plate), button, Pi on M2.5 × 8mm standoffs. Slides in from the back,
   stops on the fins.

## Check before cutting
- USB stub body size (`USB_CUT`): measure with calipers.
- Driver/PR flange thickness (`DRV_FLANGE`/`PR_FLANGE`) for flush rebates, and the PR's width across its flats
  (`PR_FLATS`).
- The shop's measured sheet thickness (`T`/`T9`) and bit (`BIT`).
