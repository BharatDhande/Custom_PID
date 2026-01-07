# lcnn_runner.py
import subprocess
import json
from pathlib import Path

LCNN_DIR = Path("external/lcnn")
LCNN_CONFIG = LCNN_DIR / "config/lcnn.yaml"
LCNN_MODEL = LCNN_DIR / "checkpoints/lcnn.pth"
OUTPUT_JSON = Path("lcnn_output.json")

def run_lcnn(image_path):
    """
    Runs LCNN inference and saves raw line predictions
    """
    cmd = [
        "python", "demo.py",
        "--config", str(LCNN_CONFIG),
        "--model", str(LCNN_MODEL),
        "--input", str(image_path),
        "--output", str(OUTPUT_JSON)
    ]

    subprocess.run(cmd, cwd=LCNN_DIR, check=True)

    with open(OUTPUT_JSON, "r") as f:
        return json.load(f)
