from mathematics.bbox import BBox
import numpy as np


class BVHnode:
    def __init__(self, parent, start, size, left, right):
        self.left = left
        self.right = right
        self.start = start
        self.size = size
        self.parent = parent
        self.prims_idx_vec = []
        self.box = BBox(None, None)


class BVH:
    def __init__(self, max_leaf_size=1):
        self.bounds = BBox(None, None)
        self.max_leaf_size = max_leaf_size
        self.ordered_prims = []
        self.nodes = []
        self.primitives = []
        self.primitive_centroids = []

    def create_tree_node(self):
        a_node = BVHnode(0, 0, -1, 0, 0)
        self.nodes.append(a_node)
        return len(self.nodes) - 1

    def build_helper(self, parent_index):
        start = self.nodes[parent_index].start
        size = self.nodes[parent_index].size

        if size < 1:
            return
        elif size == 1:
            self.ordered_prims.append(self.nodes[parent_index].prims_idx_vec[0])
            return
        else:
            left_child_idx = self.create_tree_node()
            right_child_idx = self.create_tree_node()

            if size >= self.max_leaf_size:
                nb_buckets = size
                if size > 12:
                    nb_buckets = 12

                # init bounds
                global_bounds = BBox()
                centroid_bounds = BBox()
                for i in range(start, start+size):
                    prim_id = self.nodes[parent_index].prims_idx_vec[i]
                    global_bounds.enclose(self.primitives[prim_id].bounds)
                    centroid_bounds.enclose_point(self.primitive_centroids[prim_id])

                # find best split

            return

    def build(self, primitives):
        self.primitives = primitives
        self.primitive_centroids = np.zeros((len(primitives), 3))

        # pre-compute
        for i in range(len(primitives)):
            self.primitive_centroids[i] = primitives[i].bounds.center()

        # split tree
        root = self.create_tree_node()
        self.nodes[root].size = len(primitives)
        self.ordered_prims = list(range(len(primitives)))
        self.build_helper(root)
