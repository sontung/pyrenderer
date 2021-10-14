from mathematics.constants import MAX_F
import numpy as np


class Ray:
    def __init__(self, position, direction, depth=0):
        self.position = position
        self.direction = direction
        self.depth = depth
        self.bounds = np.array([0.0, MAX_F])
        self.inv_direction = 1.0/self.direction
        self.position_tile = {}
        self.direction_tile = {}
        self.s_array_cached = {}

    def reset_bounds(self):
        self.bounds = np.array([0.0, MAX_F])

    def __str__(self):
        return f"Ray: pos={self.position} dir={self.direction}"
