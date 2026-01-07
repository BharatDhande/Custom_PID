import json
import base64
import numpy as np
from pathlib import Path

# ===== IMPORT YOUR EXISTING FUNCTIONS =====
from inference import model_fn, input_fn, predict_fn, output_fn
from line_detection import detect_lines
from graph_builder import build_connections


from lcnn import run_lcnn
from line_detection_lcnn import detect_lines_lcnn
from junction import merge_junctions

# ===== PATH CONFIG (SAFE FOR WINDOWS) =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = r"C:\Users\Admin\Downloads\inference\models"
IMAGE_PATH = r"C:\Users\Admin\Downloads\inference\p_id_diagram.png"
OUTPUT_JSON = BASE_DIR / "pid_graph.json"


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


def run_pipeline():
    print("🚀 Starting P&ID → JSON pipeline")

    # --------------------------------------------------
    # 1. LOAD MODELS (RCNN + SIAMESE)
    # --------------------------------------------------
    print("🔹 Loading models...")
    models = model_fn(str(MODEL_DIR))

    # --------------------------------------------------
    # 2. READ IMAGE
    # --------------------------------------------------
    print("🔹 Reading image...")
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "score_threshold": 0.5,
        "n_closest": 3
    }

    # --------------------------------------------------
    # 3. SYMBOL DETECTION (YOUR EXISTING PIPELINE)
    # --------------------------------------------------
    print("🔹 Running symbol detection...")
    input_data = input_fn(json.dumps(payload))
    prediction = predict_fn(input_data, models)
    symbol_output = json.loads(output_fn(prediction))

    symbols = symbol_output.get("detections", [])
    print(f"✅ Detected {len(symbols)} symbols")

    # --------------------------------------------------
    # 4. LINE + JUNCTION DETECTION
    # --------------------------------------------------
    print("🔹 Detecting pipes and junctions...")
    print("🔹 Running LCNN line detection...")
    lcnn_output = run_lcnn(IMAGE_PATH)

    lines, raw_junctions = detect_lines_lcnn(lcnn_output)
    junctions = merge_junctions(raw_junctions)
    print(f"✅ Detected {len(lines)} lines")
    print(f"✅ Detected {len(junctions)} junctions")

    # --------------------------------------------------
    # 5. BUILD CONNECTION GRAPH
    # --------------------------------------------------
    print("🔹 Building symbol ↔ pipe connections...")
    connections = build_connections(symbols, lines)
    print(f"✅ Created {len(connections)} connections")

    # --------------------------------------------------
    # 6. FINAL JSON
    # --------------------------------------------------
    final_json = {
        "symbols": symbols,
        "lines": lines,
        "junctions": junctions,
        "connections": connections
    }

    # --------------------------------------------------
    # 7. SAVE OUTPUT
    # --------------------------------------------------
    print("🔹 Saving JSON output...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, cls=NumpyEncoder)

    print("🎉 PIPELINE COMPLETE")
    print(f"📄 Output saved to: {OUTPUT_JSON}")


# ===== ENTRY POINT =====
if __name__ == "__main__":
    run_pipeline()
