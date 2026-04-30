import numpy as np

def calculate_motion(prev, curr):
    motion = 0
    for p, c in zip(prev, curr):
        motion += np.linalg.norm(np.array(p) - np.array(c))
    return motion