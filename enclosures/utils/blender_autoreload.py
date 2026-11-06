"""Blender add-on: reload the open .blend when it changes on disk, keeping the current view.

build_blend.py rewrites an enclosure's out/<name>.blend; with this enabled, the copy already open in
Blender picks up the new one by itself instead of needing a new window. Install it with

    blender --background --python utils/blender_autoreload.py -- --install

or by hand: Edit > Preferences > Add-ons > Install from Disk, then tick it.
Only files build_blend.py made (their scene has an "autoreload" property) are watched, so other work is
never reloaded under you.
"""
import os
import sys

import bpy

bl_info = {
    "name": "Auto-reload generated .blend",
    "description": "Reloads the open file when a script rewrites it on disk, keeping the view",
    "author": "oi-speaker",
    "version": (1, 1),
    "blender": (4, 2, 0),
    "category": "System",
}

POLL_SECONDS = 1.0

_seen = {}  # path -> (mtime, size) when last loaded or checked


def _stat(path):
    try:
        st = os.stat(path)
        return st.st_mtime, st.st_size
    except OSError:
        return None


def _poll():
    path = bpy.data.filepath
    if not path or not any(s.get("autoreload") for s in bpy.data.scenes):
        return POLL_SECONDS
    now = _stat(path)
    if now is None:
        return POLL_SECONDS
    if path not in _seen:
        _seen[path] = now
        return POLL_SECONDS
    if now != _seen[path]:
        # wait one more tick for the writer to finish, then load if it has settled
        _seen[path] = now
        bpy.app.timers.register(lambda: _reload(path, now), first_interval=POLL_SECONDS)
    return POLL_SECONDS


def _reload(path, expected):
    if _stat(path) != expected:
        return None  # still being written; _poll will see the next change
    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window):
        # load_ui=False keeps this window's layout and 3D view rather than the file's saved one
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    print(f"auto-reload: reloaded {path}")
    return None


def register():
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_SECONDS, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)


if __name__ == "__main__" and "--install" in sys.argv:
    # copy into the user's add-ons folder and enable it for every session
    import shutil
    import addon_utils

    target = bpy.utils.user_resource("SCRIPTS", path="addons", create=True)
    shutil.copy(__file__, os.path.join(target, "oi_autoreload.py"))
    addon_utils.enable("oi_autoreload", default_set=True, persistent=True)
    bpy.ops.wm.save_userpref()
    print(f"installed and enabled oi_autoreload in {target}")
