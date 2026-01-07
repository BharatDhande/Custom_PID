import cv2
import numpy as np
from typing import List, Tuple, Dict

def detect_lines(image_path: str) -> Tuple[List[Dict], List[Tuple[float, float]]]:
    """
    Enhanced line detection using improved Hough Transform with better parameters
    and multi-scale detection for more accurate results.
    """
    # Load image in color to preserve more information
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better edge detection
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    # Apply morphological operations to enhance line structures
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Detect lines using HoughLinesP with improved parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,  # Lower threshold for more sensitivity
        minLineLength=30,  # Minimum line length
        maxLineGap=15  # Maximum gap between line segments
    )

    pipes = []
    junctions = set()

    if lines is not None:
        for i, l in enumerate(lines):
            x1, y1, x2, y2 = l[0]
            pipes.append({
                "id": f"L{i}",
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "length": float(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            })
            junctions.add((float(x1), float(y1)))
            junctions.add((float(x2), float(y2)))

    return pipes, list(junctions)
