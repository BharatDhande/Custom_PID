# line_detection_lcnn.py
def detect_lines_lcnn(lcnn_lines):
    lines = []
    junctions = set()

    for i, item in enumerate(lcnn_lines):
        p1, p2 = item["line"]

        lines.append({
            "id": f"L{i}",
            "points": [p1, p2]
        })

        junctions.add(tuple(p1))
        junctions.add(tuple(p2))

    return lines, list(junctions)
