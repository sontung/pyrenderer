from mathematics.vec3 import normalize_vector, norm
from mathematics.quat import euler2quaternion
from numpy import dot, array

class Camera:
    def __init__(self, position, looking_at, up):
        self.position = position
        self.looking_at = looking_at
        self.up = up
        self.radius = None
        self.rot = None

    def front(self):
        return normalize_vector(self.looking_at-self.position)

    def look_at(self, cent, pos):
        self.position = pos
        self.looking_at = cent
        self.radius = norm(pos - cent)
        if dot(self.front(), self.up) == -1.0:
            self.rot = euler2quaternion(array([270.0, 0.0, 0.0]))
        else:
