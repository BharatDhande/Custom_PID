import cv2
import numpy as np

def detect_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    pipes = []
    junctions = set()

    if lines is not None:
        for i, l in enumerate(lines):
            x1, y1, x2, y2 = l[0]
            pipes.append({
                "id": f"L{i}",
                "points": [(x1, y1), (x2, y2)]
            })
            junctions.add((x1, y1))
            junctions.add((x2, y2))

    return pipes, list(junctions)
