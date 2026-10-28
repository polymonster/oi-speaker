"""Build an enclosure end to end, from its design script to a checked CNC package:

    python utils/build.py <enclosure folder> [--no-blender] [--sheet WxH]

Runs, stopping at the first that fails:
  1. the design script, <folder>/*_enclosure.py      -> out/ (dxf, cutlist, glb, parts.json)
  2. build_blend.py in Blender, with renders           -> out/<name>.blend, out/preview_*.png
  3. export_cnc.py                                     -> out/cnc/, out/<name>_cnc.zip, out/email.txt
  4. check_cnc.py, which must end "all good"

Blender is found on the PATH or in Program Files; without it, or with --no-blender, step 2 is skipped and
drawings.pdf keeps no renders.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parent


def find_blender():
    found = shutil.which("blender")
    if found:
        return found
    installs = sorted(Path("C:/Program Files/Blender Foundation").glob("Blender*/blender.exe"),
                      key=lambda p: [int(n) if n.isdigit() else 0 for n in p.parent.name.split()[-1].split(".")])
    return str(installs[-1]) if installs else None


def run(title, cmd):
    print(f"\n== {title}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"\nstopped: {title} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("enclosure", help="the enclosure's folder, e.g. og")
    ap.add_argument("--no-blender", action="store_true", help="skip the Blender scene and renders")
    ap.add_argument("--sheet", help="sheet size in mm, WxH, passed to export_cnc.py")
    args = ap.parse_args()

    folder = Path(args.enclosure).resolve()
    designs = sorted(folder.glob("*_enclosure.py"))
    if len(designs) != 1:
        sys.exit(f"expected one *_enclosure.py in {folder}, found {len(designs)}")
    py = sys.executable   # the same Python and packages this is running under

    run(f"design: {designs[0].name}", [py, str(designs[0])])
    blender = None if args.no_blender else find_blender()
    if blender:
        run("Blender scene and renders", [blender, "--background", "--factory-startup",
                                          "--python", str(UTILS / "build_blend.py"), "--", str(folder), "--render"])
    else:
        print("\n== Blender skipped" + ("" if args.no_blender else ": not found"))
    run("CNC package", [py, str(UTILS / "export_cnc.py"), str(folder)] + (["--sheet", args.sheet] if args.sheet else []))
    run("check", [py, str(UTILS / "check_cnc.py"), str(folder)])


if __name__ == "__main__":
    main()
