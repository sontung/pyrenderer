class RayLogger:
    def __init__(self):
        self.points = []
        self.lines = []
        self.colors = []

    def add(self, ray, t=5, color=[1, 0, 0]):
        self.points.extend([ray.position, ray.position + t * ray.direction])
        self.lines.append([len(self.points) - 2, len(self.points) - 1])
        self.colors.append(color)

    def add_line(self, p1, p2, color=[1, 0, 0]):
        self.points.extend([p1, p2])
        self.lines.append([len(self.points) - 2, len(self.points) - 1])
        self.colors.append(color)