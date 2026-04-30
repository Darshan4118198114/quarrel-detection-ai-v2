import numpy as np

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))