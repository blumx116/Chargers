import numpy as np

from env import State

def get_distances(
        query: np.ndarray,
        stations: np.ndarray) -> np.ndarray:
    return np.linalg.norm(stations - query, axis=1)