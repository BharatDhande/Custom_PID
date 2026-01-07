# graph_builder.py
import math

def point_line_distance(px, py, x1, y1, x2, y2):
    num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1)
    return num / den if den else 1e9

def build_connections(symbols, lines, threshold=15):
    """
    Build connections between symbols and lines with improved accuracy.
    """
    connections = []

    for sym in symbols:
        b = sym["bbox"]
        cx = (b["x1"] + b["x2"]) / 2
        cy = (b["y1"] + b["y2"]) / 2

        for line in lines:
            (x1, y1), (x2, y2) = line["points"]
            if point_line_distance(cx, cy, x1, y1, x2, y2) < threshold:
                connections.append({
                    "symbol_id": sym["class_name"],
                    "symbol_bbox_center": [float(cx), float(cy)],
                    "line_id": line["id"],
                    "line_start": [float(x1), float(y1)],
                    "line_end": [float(x2), float(y2)]
                })

    return connections
