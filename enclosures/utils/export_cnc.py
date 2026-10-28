"""Package an enclosure's out/dxf for the CNC shop, in out/cnc/ plus out/<name>_cnc.zip (made every run),
with the brief as plain text in out/email.txt to paste into the email, with the questions for the shop added:

    python utils/export_cnc.py <enclosure folder> [--sheet 1220x607]

Run the design script first; the brief's title, material and questions come from its manifest (see
enclosure.py). Parts are nested onto one sheet size per thickness:

    sheet_<t>mm_<n>.dwg       what the router cuts: outlines, pockets and drill marks, no text. Needs the
                              free ODA File Converter; without it the sheet is written as .dxf instead
    sheet_<t>mm_<n>.pdf       the same as a 1:1 vector PDF in mm
    drawings.pdf              cover page with the brief (files, material, layers), renders,
                              then each sheet, then each part with its notes
    cutlist.csv               copied from out/
"""
import argparse
import csv
import math
import shutil
import textwrap
from datetime import date

import ezdxf
from ezdxf import bbox
from ezdxf.math import Matrix44
from ezdxf.addons import odafc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from dxf_parts import crossings, parts_on, pockets_in
from enclosure import Enclosure, use_oda

GAP = 15.0     # between parts: room for a 6mm bit's path on both and a web that holds the sheet together
MARGIN = 20.0  # off the sheet edge, for clamping and the factory edge
PDF_BORDER = 10.0  # paper round the sheet in the 1:1 PDFs

LEGEND = [
    ("CUT_THROUGH", "outlines and through holes - cut all the way through"),
    ("POCKET_A_<d>mm", "pocket <d> mm deep, into face A (the face up, the one you are looking at)"),
    ("POCKET_B_<d>mm", "pocket <d> mm deep, into face B (the underside) - see FACES"),
    ("MARK_DRILL", "pilot holes, {drill} mm, not through: spot-drill them if you can, or leave them for me"),
]
FACES = [
    "Every part is drawn looking down on face A: cut the sheet as drawn, face A up.",
    "POCKET_B shapes are drawn in that same view, where they would be if you could see through the board",
    "from face A. They are NOT mirrored. To cut them, flip the part or sheet over and mirror the POCKET_B",
    "shapes about the same axis you flipped it about.",
    "Parts are only turned on the sheet to nest, never flipped: a part and its mirror image are drawn as",
    "two separate parts. No pocket goes through.",
]
# one colour per layer on the sheets, so the 1:1 PDFs (which have no layers) still tell the operations
# apart. AutoCAD colour index: all readable on white
COLOURS = [(1, "red"), (5, "blue"), (3, "green"), (6, "magenta"), (30, "orange"), (200, "purple"),
           (94, "dark green"), (12, "dark red"), (150, "navy")]
FIXED_COLOURS = {"CUT_THROUGH": (7, "black"), "MARK_DRILL": (8, "grey"),
                 "SHEET": (9, "light grey")}


def read_cutlist(enc):
    with open(enc.out / "cutlist.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_part(enc, name):
    doc = ezdxf.readfile(enc.out / "dxf" / f"{name}.dxf")
    cut = [e for e in doc.modelspace() if e.dxf.layer != "INFO"]
    return doc, cut, bbox.extents(cut)


def crossing_notes(parts, rows):
    """A sentence for each place a face A pocket crosses a face B pocket, so the shop knows it's meant."""
    notes = []
    for r in rows:
        msp = parts[r["part"]][0].modelspace()
        _, outers, _ = parts_on(msp)
        for ka, kb, area, left in crossings(pockets_in(msp, outers[0]), float(r["thickness_mm"])):
            notes.append(f"On {r['part']}, {ka} crosses {kb} over {area:.0f} mm2, leaving {left:g} mm of "
                         "wood between them - that is intended.")
    return notes


def drill_sizes(parts):
    sizes = sorted({round(2 * e.dxf.radius, 1) for _, cut, _ in parts.values() for e in cut
                    if e.dxf.layer == "MARK_DRILL" and e.dxftype() == "CIRCLE"})
    return f"{sizes[0]:g}" if len(sizes) == 1 else f"{sizes[0]:g}-{sizes[-1]:g}" if sizes else "-"


def nest(items, sheet_w, sheet_h):
    """Shelf packing: long side along the sheet, tallest first. items are (name, w, h);
    returns [(sheet, name, x, y, rotated)] with x, y the part's lower-left corner on the sheet."""
    usable_w, usable_h = sheet_w - 2 * MARGIN, sheet_h - 2 * MARGIN
    oriented = []
    for name, w, h in items:
        rotated = h > w
        if rotated:
            w, h = h, w
        assert w <= usable_w and h <= usable_h, f"{name} ({w:.0f} x {h:.0f}) doesn't fit the sheet"
        oriented.append((name, w, h, rotated))
    oriented.sort(key=lambda it: (-it[2], -it[1]))
    placed, sheet, x, y, shelf_h = [], 0, 0.0, 0.0, 0.0
    for name, w, h, rotated in oriented:
        if x + w > usable_w:
            x, y, shelf_h = 0.0, y + shelf_h + GAP, 0.0
        if y + h > usable_h:
            sheet, x, y, shelf_h = sheet + 1, 0.0, 0.0, 0.0
        placed.append((sheet, name, MARGIN + x, MARGIN + y, rotated))
        x += w + GAP
        shelf_h = max(shelf_h, h)
    return placed


def new_sheet_doc(sources, sheet_w, sheet_h):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    for src in sources:
        for layer in src.layers:
            if layer.dxf.name not in doc.layers:
                doc.layers.add(layer.dxf.name, color=layer.color)
    if "SHEET" not in doc.layers:
        doc.layers.add("SHEET")
    # the stock outline, on its own layer so it's obvious it isn't a cut
    doc.modelspace().add_lwpolyline([(0, 0), (sheet_w, 0), (sheet_w, sheet_h), (0, sheet_h)], close=True,
                                    dxfattribs={"layer": "SHEET"})
    return doc


def placement(box, x, y, rotated):
    w, h = box.size.x, box.size.y
    to_centre = Matrix44.translate(-box.center.x, -box.center.y, 0)
    if rotated:
        return Matrix44.chain(to_centre, Matrix44.z_rotate(math.pi / 2), Matrix44.translate(x + h / 2, y + w / 2, 0))
    return Matrix44.chain(to_centre, Matrix44.translate(x + w / 2, y + h / 2, 0))


def save_page(pdf, fig, stamp):
    """Every page carries the revision in its top right corner, so pages from two revisions can't mix."""
    if stamp:
        fig.text(0.97, 0.975, stamp, ha="right", va="top", fontsize=12, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def draw_page(pdf, doc, title, stamp):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes((0.03, 0.03, 0.94, 0.88))
    config = Configuration().with_changes(background_policy=BackgroundPolicy.WHITE)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=config).draw_layout(doc.modelspace(), finalize=True)
    # wrapped short of the stamp: a sheet's title lists every part on it
    fig.suptitle("\n".join(textwrap.wrap(title, 100)), fontsize=13, x=0.03, ha="left")
    save_page(pdf, fig, stamp)


def draw_full_scale(path, doc, sheet_w, sheet_h):
    """The sheet as a vector PDF at 1:1 in mm, with a PDF_BORDER of paper round the stock outline."""
    w, h = sheet_w + 2 * PDF_BORDER, sheet_h + 2 * PDF_BORDER
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    config = Configuration().with_changes(background_policy=BackgroundPolicy.WHITE)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=config).draw_layout(doc.modelspace(), finalize=True)
    ax.set_xlim(-PDF_BORDER, sheet_w + PDF_BORDER)
    ax.set_ylim(-PDF_BORDER, sheet_h + PDF_BORDER)
    fig.set_size_inches(w / 25.4, h / 25.4)  # ezdxf's backend resizes the figure while drawing
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("enclosure", nargs="?", default=".", help="the enclosure's folder (default: here)")
    ap.add_argument("--sheet", default="1220x607", help="sheet size in mm, WxH (default: a quarter of a 2440x1220 board)")
    args = ap.parse_args()
    sheet_w, sheet_h = (float(v) for v in args.sheet.lower().split("x"))
    enc = Enclosure(args.enclosure)
    use_oda()
    CNC = enc.cnc
    brief_in = enc.manifest["cnc"]["brief"]

    # empty the folder rather than delete it: a viewer or Explorer window open on it would block the delete
    CNC.mkdir(parents=True, exist_ok=True)
    for old in CNC.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    rows = read_cutlist(enc)
    parts = {r["part"]: load_part(enc, r["part"]) for r in rows}

    used = sorted({e.dxf.layer for _, cut, _ in parts.values() for e in cut} - set(FIXED_COLOURS),
                  key=lambda n: (n[:8], float(n.split("_")[-1][:-2])))
    assert len(used) <= len(COLOURS), "more pocket layers than colours"
    colours = {**FIXED_COLOURS, **dict(zip(used, COLOURS))}
    for doc, _, _ in parts.values():  # the part pages in drawings.pdf match the sheets
        for layer in doc.layers:
            if layer.dxf.name in colours:
                layer.color = colours[layer.dxf.name][0]

    sheets = []  # (thickness, index, file stem, [part names])
    for thickness in sorted({float(r["thickness_mm"]) for r in rows}, reverse=True):
        names = [r["part"] for r in rows if float(r["thickness_mm"]) == thickness for _ in range(int(r["qty"]))]
        items = [(n, parts[n][2].size.x, parts[n][2].size.y) for n in names]
        placed = nest(items, sheet_w, sheet_h)
        for index in sorted({p[0] for p in placed}):
            stem = f"sheet_{thickness:g}mm_{index + 1}"
            on_sheet = [p for p in placed if p[0] == index]
            sources = [parts[p[1]][0] for p in on_sheet]
            cnc_doc = new_sheet_doc(sources, sheet_w, sheet_h)
            ref_doc = new_sheet_doc(sources, sheet_w, sheet_h)
            if "INFO" not in ref_doc.layers:
                ref_doc.layers.add("INFO", color=8)
            for _, name, x, y, rotated in on_sheet:
                _, cut, box = parts[name]
                m = placement(box, x, y, rotated)
                for doc in (cnc_doc, ref_doc):
                    for e in cut:
                        copy = e.copy()
                        copy.transform(m)
                        doc.modelspace().add_entity(copy)
                w, h = (box.size.y, box.size.x) if rotated else (box.size.x, box.size.y)
                ref_doc.modelspace().add_text(
                    f"{name}  {box.size.x:.1f} x {box.size.y:.1f}", height=min(8.0, h / 5),
                    dxfattribs={"layer": "INFO"}).set_placement((x + 2, y + 2))
            for doc in (cnc_doc, ref_doc):
                for layer in doc.layers:
                    if layer.dxf.name in colours:
                        layer.color = colours[layer.dxf.name][0]
            draw_full_scale(CNC / f"{stem}.pdf", cnc_doc, sheet_w, sheet_h)
            if odafc.is_installed():
                odafc.export_dwg(cnc_doc, str(CNC / f"{stem}.dwg"), version="R2018", replace=True)
            else:
                cnc_doc.saveas(CNC / f"{stem}.dxf")
            sheets.append((thickness, index, stem, [p[1] for p in on_sheet], ref_doc))
            print(f"{stem}: {len(on_sheet)} parts")

    shutil.copy(enc.out / "cutlist.csv", CNC / "cutlist.csv")

    # one plain-text brief: the PDF's cover page, and out/email.txt to paste into the email
    board = " (a quarter of a 2440 x 1220 board)" if (sheet_w, sheet_h) == (1220, 607) else ""
    brief = [brief_in["title"], f"Revision {enc.revision or '-'}, built {date.today().isoformat()}. "
             f"Package: {enc.zip.name}.zip", "",
             f"{brief_in['summary']} All files are in mm, at 1:1.", "",
             "FILES",
             "  Each sheet comes in these formats, all with the same geometry:"]
    brief += ["    .dwg   AutoCAD 2018, each operation on its own layer" if odafc.is_installed() else
              "    .dxf   each operation on its own layer",
              "    .pdf   vector, 1:1 - operations shown by colour (see LAYERS)", "", "MATERIAL"]
    brief += [f"  {line}" for line in brief_in["material"]]
    brief += [f"  Nested on {sheet_w:.0f} x {sheet_h:.0f} mm sheets{board}:"]
    brief += [f"    {stem} - {thickness:g} mm, {len(names)} parts" for thickness, _, stem, names, _ in sheets]
    brief += [f"  Parts are {GAP:.0f} mm apart and {MARGIN:.0f} mm from the sheet edge. drawings.pdf",
              "  shows each sheet with the parts labelled; cutlist.csv lists them with sizes.", "", "LAYERS"]
    drill = drill_sizes(parts)
    brief += [f"  {layer:<16} {meaning.format(drill=drill)}" for layer, meaning in LEGEND]
    brief += [f"  {'SHEET':<16} the stock outline - not a cut", "",
              "  Colour of each layer on these sheets (also how the 1:1 PDFs show them):"]
    brief += [f"    {name:<16} {colours[name][1]}" for name in ["CUT_THROUGH", *used, "MARK_DRILL", "SHEET"]]
    brief += ["", "FACES"] + [f"  {line}" for line in FACES]
    for note in crossing_notes(parts, rows):
        brief += textwrap.wrap(note, 100, initial_indent="  ", subsequent_indent="  ")
    # the questions go in the email only, not on the PDF that travels with the files
    questions = ["", "QUESTIONS BEFORE CUTTING"]
    flip = [n for n, (_, cut, _) in parts.items() if any(e.dxf.layer.startswith("POCKET_B") for e in cut)]
    for i, q in enumerate(brief_in["questions"], 1):
        q = q.format(flip=", ".join(flip), bit=enc.manifest["cnc"]["bit"])
        questions += textwrap.wrap(q, 100, initial_indent=f"  {i}. ", subsequent_indent="     ")
    enc.email.write_text("\n".join(brief + questions) + "\n", encoding="utf-8")

    with PdfPages(CNC / "drawings.pdf") as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.05, 0.94, "\n".join(brief), va="top", family="monospace", fontsize=9.5)
        save_page(pdf, fig, enc.revision)
        renders = [r for r in enc.renders if r.exists()]  # from build_blend.py -- --render
        if renders:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle("The finished enclosure, front and back - for reference only, not to scale", fontsize=13,
                         x=0.03, ha="left")
            for i, path in enumerate(renders):
                ax = fig.add_axes((0.02 + i * 0.49, 0.03, 0.47, 0.88))
                ax.imshow(plt.imread(path))
                ax.axis("off")
            save_page(pdf, fig, enc.revision)
        for thickness, _, stem, names, ref_doc in sheets:
            counted = ", ".join(n if names.count(n) == 1 else f"{n} x{names.count(n)}" for n in dict.fromkeys(names))
            draw_page(pdf, ref_doc, f"{stem} — {thickness:g} mm ply — {counted}", enc.revision)
        for r in rows:
            draw_page(pdf, parts[r["part"]][0],
                      f"{r['part']} — {float(r['thickness_mm']):g} mm, {r['length_mm']} x {r['width_mm']} mm, qty {r['qty']}",
                      enc.revision)

    if not odafc.is_installed():
        print("no ODA File Converter: wrote the sheets as DXF instead of DWG")
    archive = shutil.make_archive(str(enc.zip), "zip", CNC)
    print(f"wrote {CNC} and {archive}")


if __name__ == "__main__":
    main()
