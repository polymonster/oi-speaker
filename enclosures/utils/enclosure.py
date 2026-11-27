"""What the shared tools know about an enclosure, so they work for any design.

Each enclosure has its own folder under enclosures/, with a design script that writes out/:

    out/parts.json      the manifest (below)
    out/<name>.glb      the 3D assembly: Y-up, mm, one mesh per part or piece of hardware
    out/dxf/<part>.dxf  one per part as it is cut, on the layers export_cnc.py documents
    out/cutlist.csv     part, qty, thickness_mm, length_mm, width_mm, ...

The manifest is a JSON object with:

    name        names the outputs: <name>.glb, <name>.blend, <name>_cnc_<revision>.zip
    groups      {mesh: group} for every mesh in the glb. "hw" is hardware; every other group is wood
    cut_as      {part on the cut list: [the meshes it is cut as]}, for parts cut more than once
    placement   {mesh: {origin, u, v, n, t, bounds}}: where each part sits in the design's coordinates.
                Local (u, v) is its drawing seen from face A, local z runs from face B (0) to face A (t)
    origin      [x, y, z] added to the design's coordinates to get the glb's (before Y-up)
    params      the design's parameters, for reference
    cnc         {bit, brief: {title, summary, material: [lines], questions: [lines]}, colours}. Questions may
                use {flip} (the parts with face B pockets) and {bit}. colours is optional: {layer: [AutoCAD
                colour index, name]}, pinned so new layers don't move them; other layers get spare colours
    blend       optional, see build_blend.py
"""
import json
import re
import sys
from pathlib import Path


class Enclosure:
    def __init__(self, folder):
        self.dir = Path(folder).resolve()
        self.out = self.dir / "out"
        path = self.out / "parts.json"
        if not path.exists():
            sys.exit(f"{path} not found: run the enclosure's design script first")
        self.manifest = json.loads(path.read_text())
        self.name = self.manifest["name"]
        self.glb = self.out / f"{self.name}.glb"
        self.blend = self.out / f"{self.name}.blend"
        self.renders = [self.out / "preview_front.png", self.out / "preview_back.png"]
        self.cnc = self.out / "cnc"
        self.email = self.out / "email.txt"

    def new_revision(self):
        """Take the next revision after the last one, so a package is never overwritten and a number the shop
        has seen is never reused. The last is kept in <folder>/cnc_revision.txt, outside out/, so it survives
        out/ being deleted; the zips in out/ count too. It goes on every page of drawings.pdf and in the zip's
        name."""
        pattern = re.compile(rf"{re.escape(self.name)}_cnc_v(\d+)\.zip")
        taken = [int(m[1]) for m in (pattern.fullmatch(p.name) for p in self.out.iterdir()) if m]
        record = self.dir / "cnc_revision.txt"
        if record.exists():
            taken.append(int(record.read_text().strip().lstrip("v")))
        n = max(taken, default=-1) + 1
        record.write_text(f"v{n}\n")
        self.revision = f"v{n}"
        self.zip = self.out / f"{self.name}_cnc_{self.revision}"   # shutil.make_archive adds the .zip


def folder_arg(argv):
    """The enclosure folder from the command line, or the current folder. Under Blender, only what comes
    after "--" is ours."""
    args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    args = [a for a in args if not a.startswith("-")]
    return args[0] if args else "."


def use_oda():
    """Point ezdxf at the ODA File Converter, for reading and writing DWG. ezdxf looks in an unversioned
    folder; the installer puts the version in the name."""
    import ezdxf
    found = sorted(Path("C:/Program Files/ODA").glob("ODAFileConverter*/ODAFileConverter.exe"))
    if found:
        ezdxf.options.set("odafc-addon", "win_exec_path", str(found[-1]))
