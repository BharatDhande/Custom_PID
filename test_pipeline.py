"""
Test pipeline to demonstrate the updated change line detection model
with pretrained model and accurate JSON output for GLB conversion.
"""
import json
import numpy as np
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.append('/workspace/code')

# Import the updated modules
from line_detection import detect_lines
from graph_builder import build_connections

# ===== NUMPY SAFE JSON ENCODER =====
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def run_test_pipeline():
    print("🚀 Starting P&ID → JSON pipeline (TEST MODE)")
    
    # Mock symbol data (in a real scenario, this would come from the inference model)
    symbols = [
        {
            "class_id": 558,
            "class_name": "Plug Valve_009",
            "confidence": 0.999,
            "bbox": {
                "x1": 390.0,
                "y1": 90.0,
                "x2": 410.0,
                "y2": 110.0,
                "width": 20.0,
                "height": 20.0
            },
            "embedding_distance": 0.998
        },
        {
            "class_id": 708,
            "class_name": "Screw Pump_001",
            "confidence": 0.998,
            "bbox": {
                "x1": 390.0,
                "y1": 240.0,
                "x2": 410.0,
                "y2": 260.0,
                "width": 20.0,
                "height": 20.0
            },
            "embedding_distance": 1.008
        }
    ]

    # --------------------------------------------------
    # LINE + JUNCTION DETECTION (USING IMPROVED METHOD)
    # --------------------------------------------------
    print("🔹 Detecting pipes and junctions...")
    image_path = '/workspace/test_pid.png'
    lines, junctions = detect_lines(image_path)
    print(f"✅ Detected {len(lines)} lines")
    print(f"✅ Detected {len(junctions)} junctions")

    # --------------------------------------------------
    # BUILD CONNECTION GRAPH
    # --------------------------------------------------
    print("🔹 Building symbol ↔ pipe connections...")
    connections = build_connections(symbols, lines)
    print(f"✅ Created {len(connections)} connections")

    # --------------------------------------------------
    # FINAL JSON (STRUCTURED FOR GLB CONVERSION)
    # --------------------------------------------------
    final_json = {
        "symbols": symbols,
        "lines": lines,
        "junctions": junctions,
        "connections": connections,
        "metadata": {
            "image_path": image_path,
            "processing_timestamp": json.dumps(str(np.datetime64('now')), cls=NumpyEncoder).strip('"'),
            "total_symbols": len(symbols),
            "total_lines": len(lines),
            "total_junctions": len(junctions),
            "total_connections": len(connections)
        }
    }

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------
    output_path = Path(__file__).parent / "test_pid_graph.json"
    print("🔹 Saving JSON output...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, cls=NumpyEncoder)

    print("🎉 PIPELINE COMPLETE")
    print(f"📄 Output saved to: {output_path}")
    
    # Show sample of output
    print("\n📊 Sample of output structure:")
    print(f"  - Symbols: {len(final_json['symbols'])}")
    print(f"  - Lines: {len(final_json['lines'])}")
    print(f"  - Junctions: {len(final_json['junctions'])}")
    print(f"  - Connections: {len(final_json['connections'])}")
    
    if final_json['lines']:
        print(f"  - First line: {final_json['lines'][0]}")
    if final_json['connections']:
        print(f"  - First connection: {final_json['connections'][0]}")


if __name__ == "__main__":
    run_test_pipeline()