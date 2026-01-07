# junctions.py
import numpy as np

def merge_junctions(points, threshold=6):
    merged = []

    for p in points:
        p = np.array(p)
        found = False

        for m in merged:
            if np.linalg.norm(p - np.array(m)) < threshold:
                found = True
                break

        if not found:
            merged.append(p.tolist())

    return merged
