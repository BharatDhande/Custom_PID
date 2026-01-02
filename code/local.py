import json
import base64
from inference import model_fn, input_fn, predict_fn, output_fn
import json
MODEL_DIR = r"C:\Users\Admin\Downloads\inference\models"
IMAGE_PATH = r"C:\Users\Admin\Downloads\inference\p_id_diagram.png"

def run_local_inference():
    # 1. Load models
    models = model_fn(MODEL_DIR)

    # 2. Read image
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    payload = {
        "image": base64.b64encode(image_bytes).decode(),
        "score_threshold": 0.5,
        "n_closest": 3
    }

    # 3. Run inference pipeline
    input_data = input_fn(json.dumps(payload))
    prediction = predict_fn(input_data, models)
    output = output_fn(prediction)
    output_json = json.loads(output)
    print("=== FINAL JSON OUTPUT ===")
    with open("json_op.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)
    

if __name__ == "__main__":
    run_local_inference()
