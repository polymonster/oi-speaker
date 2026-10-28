"""Build out/<name>.blend from an enclosure's generated assembly, for looking at the design in Blender.

    blender --background --python utils/build_blend.py -- <enclosure folder> [--render]

Reads out/<name>.glb and out/parts.json (run the design script first), so the .blend always matches the
DXFs. --render also writes out/preview_front.png and out/preview_back.png, which export_cnc.py puts in
drawings.pdf. In the saved file, select the "controls" empty and use its custom properties
(N panel > Item > Properties):

    explode   0 = assembled, 1 = pulled apart
    cutaway   1 = slice the wooden parts down the middle to see inside
    plus one slider per entry in the manifest's blend.sliders

The manifest's optional "blend" section, all in mm in the design's axes (Z up):

    collections         {group: collection name}. Default: the group's name
    explode_spread      how far each part travels, as a multiple of its distance from the body centre
    explode_lift        {group: mm}: extra lift when exploded
    explode_move        {group: [x, y, z]}: the group travels this far instead of outward
    sliders             [{name, description, groups: [...], meshes: [name prefixes], move: [x, y, z]}]:
                        a 0-1 property that moves those groups and meshes by move
    hardware_materials  {mesh: material}, from MATERIALS or the section's own "materials"
    materials           {name: [[r, g, b, a], roughness]}, added to MATERIALS
    front               [x, y, z]: the way the front faces, for the camera. Default [0, -1, 0]
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enclosure import Enclosure, folder_arg  # noqa: E402

EXPLODE_SPREAD = 0.9

# base colours, linear. used for hardware, and as the solid-mode colour of the ply
MATERIALS = {
    "ply_18": ((0.72, 0.52, 0.32, 1.0), 0.6),
    "ply_9": ((0.80, 0.62, 0.42, 1.0), 0.6),
    "driver": ((0.03, 0.03, 0.03, 1.0), 0.4),
    "metal": ((0.35, 0.35, 0.37, 1.0), 0.35),
    "pcb": ((0.02, 0.18, 0.06, 1.0), 0.45),
    "conduit": ((0.55, 0.55, 0.55, 1.0), 0.5),
    "cone": ((0.012, 0.012, 0.013, 1.0), 0.65),
    "spider": ((0.62, 0.42, 0.08, 1.0), 0.75),
    "brass": ((0.75, 0.55, 0.22, 1.0), 0.3),
    "nickel": ((0.6, 0.6, 0.62, 1.0), 0.2),
    "crimp": ((0.45, 0.02, 0.02, 1.0), 0.5),
}

# birch ply: odd number of ~1.4mm veneers, grain alternating, so the edge shows light long-grain plies
# between darker end-grain ones with a thin glue line at each joint
PLY_VENEER = 1.4
PLY_FACE = (0.78, 0.58, 0.36, 1.0)
PLY_CROSS = (0.55, 0.37, 0.21, 1.0)
PLY_GLUE = (0.20, 0.12, 0.07, 1.0)
PLY_GLUE_WIDTH = 0.12  # fraction of a veneer


def make_material(name: str, colour, roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = colour  # solid-mode viewport colour
    return mat


def _socket(sockets, name: str, kind: str = None):
    # Mix nodes carry one A/B pair per data type, so names alone are ambiguous
    return next(s for s in sockets if s.name == name and (kind is None or s.type == kind))


def make_ply_material(thickness: float, axis: int) -> bpy.types.Material:
    """Plywood whose veneers stack along local `axis`. Driven by object coordinates, which run
    -thickness/2..thickness/2 across the sheet because every part's origin is at its bounds centre."""
    plies = max(3, round(thickness / PLY_VENEER) | 1)
    mat = bpy.data.materials.new(f"ply_{thickness:g}mm_{'xyz'[axis]}")
    mat.use_nodes = True
    mat.diffuse_color = MATERIALS["ply_9" if thickness < 12 else "ply_18"][0]
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.65

    def node(kind, **settings):
        n = nodes.new(kind)
        for key, value in settings.items():
            setattr(n, key, value)
        return n

    def feed(socket, value):
        if isinstance(value, bpy.types.NodeSocket):
            links.new(value, socket)
        else:
            socket.default_value = value

    def math(op, a, b=0.0, c=0.0):
        n = node("ShaderNodeMath", operation=op)
        for i, value in enumerate((a, b, c)):
            feed(n.inputs[i], value)
        return n.outputs[0]

    def mix(blend, factor, a, b):
        n = node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        feed(_socket(n.inputs, "Factor", "VALUE"), factor)
        feed(_socket(n.inputs, "A", "RGBA"), a)
        feed(_socket(n.inputs, "B", "RGBA"), b)
        return _socket(n.outputs, "Result", "RGBA")

    coord = node("ShaderNodeTexCoord")
    sep = node("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Object"], sep.inputs[0])
    # u counts veneers from one face. clamped off the faces themselves, so a face reads as its outer
    # veneer rather than landing exactly on a glue line
    u = math("MULTIPLY_ADD", sep.outputs["XYZ"[axis]], plies / thickness, plies / 2)
    u = math("MINIMUM", math("MAXIMUM", u, PLY_GLUE_WIDTH + 0.05), plies - 0.05)
    index = math("FLOOR", u)
    cross = math("FLOORED_MODULO", index, 2.0)
    glue = math("LESS_THAN", math("FRACT", u), PLY_GLUE_WIDTH)

    # each veneer a slightly different tone, like real stock
    noise = node("ShaderNodeTexWhiteNoise", noise_dimensions="1D")
    links.new(index, noise.inputs["W"])
    tone = math("MULTIPLY_ADD", noise.outputs["Value"], 0.25, 0.85)
    # soft grain streaks over everything
    wave = node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X")
    links.new(coord.outputs["Object"], wave.inputs["Vector"])
    wave.inputs["Scale"].default_value = 0.04
    wave.inputs["Distortion"].default_value = 8.0
    wave.inputs["Detail"].default_value = 3.0

    colour = mix("MIX", cross, PLY_FACE, PLY_CROSS)
    colour = mix("MULTIPLY", 1.0, colour, tone)
    colour = mix("MULTIPLY", math("MULTIPLY", wave.outputs["Fac"], 0.3), colour, (0.7, 0.7, 0.7, 1.0))
    colour = mix("MIX", glue, colour, PLY_GLUE)
    links.new(colour, bsdf.inputs["Base Color"])
    return mat


def add_property(obj, name: str, default: float, description: str):
    obj[name] = default
    ui = obj.id_properties_ui(name)
    ui.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, description=description)


def drive(target, path: str, index: int, expression: str, controls, *props: str):
    """Drive target.path[index] from controls' properties, each a variable of the same name in the
    expression. Kept to plain arithmetic so Blender's simple expression evaluator runs it — no
    "auto-run python scripts" prompt when the file opens."""
    fcurve = target.driver_add(path, index) if index >= 0 else target.driver_add(path)
    driver = fcurve.driver
    driver.type = "SCRIPTED"
    for prop in props:
        var = driver.variables.new()
        var.name = prop
        var.targets[0].id = controls
        var.targets[0].data_path = f'["{prop}"]'
    driver.expression = expression


def main():
    enc = Enclosure(folder_arg(sys.argv))
    groups = enc.manifest["groups"]
    cfg = enc.manifest.get("blend", {})
    library = {**MATERIALS, **{k: (tuple(c), r) for k, (c, r) in cfg.get("materials", {}).items()}}
    hardware_materials = cfg.get("hardware_materials", {})
    names = sorted(set(groups.values()), key=lambda g: (g == "hw", g))   # hardware's collection last
    collection_names = {g: cfg.get("collections", {}).get(g, g) for g in names}
    sliders = cfg.get("sliders", [])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # tells blender_autoreload.py this file is generated, so it may reload it when it's rewritten
    scene["autoreload"] = True
    bpy.ops.import_scene.gltf(filepath=str(enc.glb))
    # the glb holds mm as plain units. label them mm — after the import, because the importer scales
    # by the scene's unit scale and would otherwise blow everything up a thousandfold
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    imported = [o for o in scene.objects if o.type == "MESH"]

    # every part comes in with its origin at the world origin. centre each one so its location means
    # something — exploding moves locations
    for obj in imported:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.object.select_all(action="DESELECT")

    materials = {name: make_material(name, colour, rough) for name, (colour, rough) in library.items()}
    ply_materials = {}
    collections = {}
    for group, label in collection_names.items():
        coll = bpy.data.collections.new(label)
        scene.collection.children.link(coll)
        collections[group] = coll

    for obj in imported:
        group = groups.get(obj.name, "hw")
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collections[group].objects.link(obj)
        obj.data.materials.clear()
        if group == "hw":
            obj.data.materials.append(materials[hardware_materials.get(obj.name, "metal")])
        else:
            # the sheet thickness is the part's thinnest local extent
            size = [max(c[i] for c in obj.bound_box) - min(c[i] for c in obj.bound_box) for i in range(3)]
            axis = size.index(min(size))
            key = (round(size[axis], 1), axis)
            if key not in ply_materials:
                ply_materials[key] = make_ply_material(*key)
            obj.data.materials.append(ply_materials[key])
        # the exporter's normals are per-face already; smooth shading would round the ply edges off.
        # hardware is mostly round, so smooth it but keep its hard edges
        for poly in obj.data.polygons:
            poly.use_smooth = group == "hw"
        if group == "hw":
            obj.data.set_sharp_from_angle(angle=math.radians(35))

    body = [o for o in imported if groups.get(o.name) == "body"]
    lo = Vector((min(o.location.x - o.dimensions.x / 2 for o in body),
                 min(o.location.y - o.dimensions.y / 2 for o in body),
                 min(o.location.z - o.dimensions.z / 2 for o in body)))
    hi = Vector((max(o.location.x + o.dimensions.x / 2 for o in body),
                 max(o.location.y + o.dimensions.y / 2 for o in body),
                 max(o.location.z + o.dimensions.z / 2 for o in body)))
    centre = (lo + hi) / 2
    front = Vector(cfg.get("front", (0, -1, 0))).normalized()

    controls = bpy.data.objects.new("controls", None)
    controls.empty_display_type = "ARROWS"
    controls.empty_display_size = 40
    controls.location = (centre.x, centre.y, hi.z + 120)
    scene.collection.objects.link(controls)
    add_property(controls, "explode", 0.0, "0 assembled, 1 pulled apart")
    add_property(controls, "cutaway", 0.0, "1 slices the wooden parts to show inside")
    for s in sliders:
        add_property(controls, s["name"], 0.0, s.get("description", ""))

    spread = cfg.get("explode_spread", EXPLODE_SPREAD)
    for obj in imported:
        group = groups.get(obj.name, "hw")
        if group in cfg.get("explode_move", {}):
            offset = Vector(cfg["explode_move"][group])
        else:
            offset = (obj.location - centre) * spread
            offset.z += cfg.get("explode_lift", {}).get(group, 0.0)
        mine = [s for s in sliders if group in s.get("groups", []) or obj.name.startswith(tuple(s.get("meshes", [])))]
        for axis in range(3):
            expression = f"{obj.location[axis]:.4f} + explode * {offset[axis]:.4f}"
            for s in mine:
                if s["move"][axis]:
                    expression += f" + {s['name']} * {s['move'][axis]:.4f}"
            drive(obj, "location", axis, expression, controls, "explode", *(s["name"] for s in mine))

    # cutaway: one box covering the +x half, subtracted from every wooden part while cutaway is on
    cutter = bpy.data.objects.new("cutaway_box", bpy.data.meshes.new("cutaway_box"))
    scene.collection.objects.link(cutter)
    import bmesh
    mesh = bmesh.new()
    bmesh.ops.create_cube(mesh, size=1.0)
    mesh.to_mesh(cutter.data)
    mesh.free()
    span = (hi - lo) * 3
    cutter.scale = (span.x / 2, span.y, span.z)
    cutter.location = (centre.x + span.x / 4, centre.y, centre.z + 60)
    cutter.display_type = "WIRE"
    cutter.hide_render = True
    cutter.hide_viewport = True
    for obj in imported:
        if groups.get(obj.name, "hw") != "hw":
            mod = obj.modifiers.new("cutaway", "BOOLEAN")
            mod.operation = "DIFFERENCE"
            mod.object = cutter
            mod.solver = "EXACT"
            drive(mod, "show_viewport", -1, "cutaway > 0.5", controls, "cutaway")
            drive(mod, "show_render", -1, "cutaway > 0.5", controls, "cutaway")

    # light and camera for a quick render — three-quarter view from the front left, above
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.9, 0.2, -0.6)
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.5
    scene.world = world

    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    cam.data.lens = 50
    cam.data.clip_end = 20000
    scene.collection.objects.link(cam)
    scene.camera = cam
    # three-quarter view: from in front, off to the left of the front, and above
    left = front.cross(Vector((0, 0, 1)))
    cam.location = centre + front * 720.0 + left * 520.0 + Vector((0, 0, 330.0))
    direction = centre + Vector((0, 0, 30)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # 3d views clip at 1000 units by default, which is 1m in mm units — zooming out cuts the model off.
    # set on every view saved in the file; the views open framed around the assembly too
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start = 1.0
                    space.clip_end = 100000.0
                    space.region_3d.view_location = centre
                    space.region_3d.view_distance = 900.0
                    space.shading.type = "MATERIAL"

    scene.render.resolution_x = 900
    scene.render.resolution_y = 1100
    scene.render.film_transparent = False
    preview_front, preview_back = enc.renders
    scene.render.filepath = str(preview_front)

    bpy.ops.wm.save_as_mainfile(filepath=str(enc.blend))
    print(f"saved {enc.blend}")
    if "--render" in sys.argv:
        bpy.ops.render.render(write_still=True)
        print(f"rendered {preview_front}")
        # and the back, from the other corner with the sun swung round to light it. after the save, so
        # the .blend keeps the front view
        cam.location = centre - front * 720.0 - left * 520.0 + Vector((0, 0, 330.0))
        cam.rotation_euler = (centre + Vector((0, 0, 30)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sun.rotation_euler.z += math.pi
        scene.render.filepath = str(preview_back)
        bpy.ops.render.render(write_still=True)
        print(f"rendered {preview_back}")


main()
