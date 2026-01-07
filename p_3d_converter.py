"""
3D Converter for P&ID JSON to GLB
This script converts the JSON output from the pipeline to a 3D GLB file.
"""
import bpy
import json
import os
import mathutils
from mathutils import Vector

# ==============================
# CONFIG
# ==============================
JSON_PATH = r"/workspace/code/pid_graph.json"
EXPORT_PATH = r"/workspace/pid_output.glb"

# Scale factors
PIXEL_TO_METER = 0.01  # Convert pixels to meters
Z_HEIGHT = 0.1  # Height for symbols above the ground plane
PIPE_HEIGHT = 0.05  # Height for pipes
TEXT_HEIGHT_OFFSET = 0.2  # Height offset for text labels

# Text scale
TEXT_SCALE = 0.05  # Smaller text scale

# Component library mapping
COMPONENT_LIBRARY = {
    "Plug Valve": "CYLINDER",
    "Screw Pump": "CUBE",
    "Centrifugal Pump": "CUBE",
    "Ball Valve": "CYLINDER",
    "Gate Valve": "CUBE",
    "Globe Valve": "CYLINDER",
    "Pressure Indicator": "SPHERE",
    "Flow Transmitter": "SPHERE",
    "Temperature Transmitter": "SPHERE",
    "Orifice": "CYLINDER",
    "Angle Valve": "CYLINDER",
    "Pneumatic-Diaphragm Gate Valve": "CUBE",
    "default": "CUBE"
}

# Color library
COLOR_LIBRARY = {
    "Plug Valve": (0.1, 0.3, 0.8, 1.0),   # Blue
    "Screw Pump": (0.8, 0.1, 0.1, 1.0),   # Red
    "Centrifugal Pump": (0.8, 0.1, 0.1, 1.0),  # Red
    "Ball Valve": (0.1, 0.8, 0.3, 1.0),   # Green
    "Gate Valve": (0.8, 0.6, 0.2, 1.0),   # Orange
    "Globe Valve": (0.8, 0.2, 0.8, 1.0),  # Purple
    "Pressure Indicator": (0.9, 0.9, 0.2, 1.0),  # Yellow
    "Flow Transmitter": (0.9, 0.5, 0.2, 1.0),  # Orange-red
    "Temperature Transmitter": (0.2, 0.9, 0.9, 1.0),  # Cyan
    "Orifice": (0.5, 0.5, 0.5, 1.0),      # Gray
    "Angle Valve": (0.1, 0.8, 0.3, 1.0),  # Green
    "Pneumatic-Diaphragm Gate Valve": (0.8, 0.6, 0.2, 1.0),  # Orange
    "PIPE": (0.1, 0.7, 0.2, 1.0),         # Green
    "JUNCTION": (0.8, 0.8, 0.8, 1.0),     # Light Gray
    "DEFAULT": (0.7, 0.7, 0.7, 1.0)       # Gray
}

def apply_color(obj, rgba, mat_name):
    """Apply color material to an object"""
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

def create_symbol(symbol_data):
    """Create a 3D representation of a symbol"""
    bbox = symbol_data["bbox"]
    
    # Calculate center position
    cx = ((bbox["x1"] + bbox["x2"]) / 2) * PIXEL_TO_METER
    cy = ((bbox["y1"] + bbox["y2"]) / 2) * PIXEL_TO_METER
    
    # Calculate dimensions
    width = (bbox["x2"] - bbox["x1"]) * PIXEL_TO_METER
    height = (bbox["y2"] - bbox["y1"]) * PIXEL_TO_METER
    
    # Get base name for classification
    raw_name = symbol_data["class_name"]
    base_name = raw_name.split("_")[0] if "_" in raw_name else raw_name
    
    # Determine component type
    comp_type = COMPONENT_LIBRARY.get(base_name, "default")
    
    # Create the appropriate geometry
    if comp_type == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add(
            radius=max(width, 0.1) / 2,
            depth=max(height, 0.1),
            location=(cx, -cy, Z_HEIGHT)
        )
    elif comp_type == "SPHERE":
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=max(width, 0.1) / 2,
            location=(cx, -cy, Z_HEIGHT)
        )
    else:  # Default to cube
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(cx, -cy, Z_HEIGHT)
        )
        bpy.context.object.scale = (width, width, height)
    
    obj = bpy.context.object
    obj.name = raw_name
    
    # Apply color
    color = COLOR_LIBRARY.get(base_name, COLOR_LIBRARY["DEFAULT"])
    apply_color(obj, color, f"{base_name}_MAT")
    
    # Add label
    bpy.ops.object.text_add(
        location=(cx, -cy, Z_HEIGHT + TEXT_HEIGHT_OFFSET)
    )
    text_obj = bpy.context.object
    text_obj.data.body = raw_name
    text_obj.scale = (TEXT_SCALE, TEXT_SCALE, TEXT_SCALE)
    
    # Apply color to text
    text_color = COLOR_LIBRARY.get("DEFAULT")
    apply_color(text_obj, text_color, "TEXT_MAT")

def create_line(line_data):
    """Create a 3D representation of a line (pipe)"""
    points = line_data["points"]
    
    # Extract start and end points
    start_x, start_y = points[0]
    end_x, end_y = points[1]
    
    # Convert to 3D coordinates
    start_pos = Vector((start_x * PIXEL_TO_METER, -start_y * PIXEL_TO_METER, PIPE_HEIGHT))
    end_pos = Vector((end_x * PIXEL_TO_METER, -end_y * PIXEL_TO_METER, PIPE_HEIGHT))
    
    # Calculate direction and length
    direction = end_pos - start_pos
    length = direction.length
    direction.normalize()
    
    if length > 0:
        # Create cylinder for the line
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,  # Pipe radius
            depth=length,
            location=start_pos.lerp(end_pos, 0.5)  # Midpoint between start and end
        )
        
        obj = bpy.context.object
        obj.name = line_data["id"]
        
        # Rotate to align with direction
        obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
        # Apply color
        color = COLOR_LIBRARY.get("PIPE", COLOR_LIBRARY["DEFAULT"])
        apply_color(obj, color, "PIPE_MAT")

def create_junction(junction_data, idx):
    """Create a 3D representation of a junction"""
    x, y = junction_data
    
    # Create a small sphere for the junction
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=0.03,  # Small radius for junction
        location=(x * PIXEL_TO_METER, -y * PIXEL_TO_METER, PIPE_HEIGHT)
    )
    
    obj = bpy.context.object
    obj.name = f"J{idx}"
    
    # Apply color
    color = COLOR_LIBRARY.get("JUNCTION", COLOR_LIBRARY["DEFAULT"])
    apply_color(obj, color, "JUNCTION_MAT")

def main():
    """Main function to convert JSON to 3D GLB"""
    # Clean the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Load JSON data
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Process symbols
    print(f"Processing {len(data['symbols'])} symbols...")
    for symbol in data['symbols']:
        create_symbol(symbol)
    
    # Process lines
    print(f"Processing {len(data['lines'])} lines...")
    for line in data['lines']:
        create_line(line)
    
    # Process junctions
    print(f"Processing {len(data['junctions'])} junctions...")
    for idx, junction in enumerate(data['junctions']):
        create_junction(junction, idx)
    
    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=EXPORT_PATH,
        export_format="GLB",
        export_apply=True,
        export_yup=True
    )
    
    print(f"✅ GLB exported: {EXPORT_PATH}")

if __name__ == "__main__":
    main()