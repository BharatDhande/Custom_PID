import bpy
import json
import os

# ==============================
# CONFIG
# ==============================
JSON_PATH = r"D:\Rohit Python\P&ID Diagram 3D\json_op.json"
EXPORT_PATH = r"D:\Rohit Python\P&ID Diagram 3D\pid_output.glb"

SCALE = 0.01   # pixel → meter
Z_HEIGHT = 0.0

TEXT_SCALE = 0.08   # 🔹 SMALL TEXT (as you asked)

# ==============================
# COMPONENT LIBRARY
# ==============================
COMPONENT_LIBRARY = {
    "Plug Valve": "CYLINDER",
    "Screw Pump": "CUBE"
}

# ==============================
# LOAD JSON
# ==============================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

detections = data["detections"]

# ==============================
# CLEAN SCENE
# ==============================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ==============================
# HELPERS
# ==============================
COLOR_LIBRARY = {
    "Plug Valve": (0.1, 0.3, 0.8, 1.0),   # Blue
    "Screw Pump": (0.8, 0.1, 0.1, 1.0),   # Red
    "PIPE": (0.1, 0.7, 0.2, 1.0),         # Green
    "DEFAULT": (0.7, 0.7, 0.7, 1.0)       # Grey
}

def apply_color(obj, rgba, mat_name):
    mat = bpy.data.materials.get(mat_name)

    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Metallic"].default_value = 0.1

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_component(name, location, comp_type, w, h):
    size_x = max(w * SCALE, 0.15)
    size_y = max(h * SCALE, 0.15)

    if comp_type == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=size_x / 2,
            depth=size_y,
            location=location
        )
    else:
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=location
        )
        bpy.context.object.scale = (size_x, size_x, size_y)

    obj = bpy.context.object
    obj.name = name

    # 🔹 COLOR APPLY
    base_name = name.split("_")[0]
    color = COLOR_LIBRARY.get(base_name, COLOR_LIBRARY["DEFAULT"])
    apply_color(obj, color, f"{base_name}_MAT")

    # 🔹 SMALL LABEL
    bpy.ops.object.text_add(
        location=(location[0], location[1], location[2] + size_y + 0.1)
    )
    text = bpy.context.object
    text.data.body = name
    text.scale = (TEXT_SCALE, TEXT_SCALE, TEXT_SCALE)

# ==============================
# MAIN LOOP
# ==============================
for det in detections:
    bbox = det["bbox"]

    cx = (bbox["x1"] + bbox["x2"]) / 2 * SCALE
    cy = (bbox["y1"] + bbox["y2"]) / 2 * SCALE

    width = bbox.get("width", bbox["x2"] - bbox["x1"])
    height = bbox.get("height", bbox["y2"] - bbox["y1"])

    raw_name = det["class_name"]
    base_name = raw_name.split("_")[0]

    comp_type = COMPONENT_LIBRARY.get(base_name, "CUBE")

    create_component(
        name=raw_name,
        location=(cx, -cy, Z_HEIGHT),
        comp_type=comp_type,
        w=width,
        h=height
    )

# ==============================
# EXPORT GLB
# ==============================
bpy.ops.export_scene.gltf(
    filepath=EXPORT_PATH,
    export_format="GLB",
    export_apply=True,
    export_yup=True
)

print("✅ GLB exported:", EXPORT_PATH)
