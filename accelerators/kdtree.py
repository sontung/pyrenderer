from mathematics.bbox import BBox


class KDtree:
    def __init__(self):
        self.bounds = BBox(None, None)

    def build(self, primitives):
        # init bounds
        self.bounds.copy(primitives[0].bounds)
        for prim in primitives:
            self.bounds.enclose(prim.bounds)

        # split tree