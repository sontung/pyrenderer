from mathematics.constants import MAX_F
import numpy as np


class Ray:
    def __init__(self, position, direction, depth=0, bounds=np.array([0.0, MAX_F])):
        self.position = position
        self.direction = direction
        self.dir = direction
        self.depth = depth
        self.bounds = bounds
        self.inv_direction = 1.0/self.direction

