from mathematics.bbox import BBox
from mathematics.constants import MAX_F
import numpy as np
import sys


class BVHnode:
    def __init__(self, parent, start, size, left, right):
        self.left = left
        self.right = right
        self.start = start
        self.size = size
        self.parent = parent
        self.prims_idx_vec = []
        self.box = BBox()
        self.split_cost = 0

    def is_leaf(self):
        return self.left == self.right


class Bucket:
    def __init__(self):
        self.bbox = BBox()
        self.bounds_for_centroids = [0.0, 0.0]
        self.prims_idx_vec = []


def compute_partition_cost(buckets, partition_idx, nb_buckets, bound_area):
    sa1 = 0.0
    sa2 = 0.0
    box1 = BBox()
    box2 = BBox()
    for i1 in range(partition_idx):
        box1.enclose(buckets[i1].bbox)
        sa1 += len(buckets[i1].prims_idx_vec)
    for i2 in range(partition_idx, nb_buckets):
        box2.enclose(buckets[i2].bbox)
        sa2 += len(buckets[i2].prims_idx_vec)

    cost1 = box1.surface_area()*sa1
    cost2 = box2.surface_area()*sa2
    total_cost = (cost2+cost1)/bound_area+1.0
    return total_cost


class BVH:
    def __init__(self, max_leaf_size=1):
        self.bounds = BBox(None, None)
        self.max_leaf_size = max_leaf_size
        self.ordered_prims = []
        self.nodes = []
        self.primitives = []
        self.primitive_centroids = []
        self.trace = {"origin": np.array([0.0, 0.0, 0.0]), "hit": False, "t": MAX_F,
                      "position": np.array([0.0, 0.0, 0.0])}

    def create_tree_node(self):
        a_node = BVHnode(0, 0, -1, 0, 0)
        self.nodes.append(a_node)
        return len(self.nodes) - 1

    def compute_bucket_index(self, buckets, prim_idx, dim):
        for i in range(len(buckets)):
            if self.primitive_centroids[prim_idx][dim] >= buckets[i].bounds_for_centroids[0]:
                if self.primitive_centroids[prim_idx][dim] <= buckets[i].bounds_for_centroids[1]:
                    return i
        return len(buckets)

    def sah_heuristic(self, parent_index, dim, nb_buckets, size,
                      global_bounds, centroid_bounds):
        buckets = [Bucket() for _ in range(nb_buckets)]
        max_centroid = centroid_bounds.max_coord[dim]
        min_centroid = centroid_bounds.min_coord[dim]
        max_centroid += 0.000001
        min_centroid -= 0.000001
        step = (max_centroid - min_centroid) / float(nb_buckets)

        for i in range(nb_buckets):
            buckets[i].bounds_for_centroids[0] = min_centroid+step*i
            buckets[i].bounds_for_centroids[1] = min_centroid+step*(i+1)
            assert buckets[i].bounds_for_centroids[0] < buckets[i].bounds_for_centroids[1]

        # compute bucket for each primitive
        for i in range(size):
            prim_id = self.nodes[parent_index].prims_idx_vec[i]
            bucket_id = self.compute_bucket_index(buckets, prim_id, dim)
            assert bucket_id != nb_buckets
            buckets[bucket_id].bbox.enclose(self.primitives[prim_id].bounds)
            buckets[bucket_id].prims_idx_vec.append(prim_id)

        # compute the SAH cost
        global_bound_area = global_bounds.surface_area()
        min_cost = compute_partition_cost(buckets, 1, nb_buckets, global_bound_area)
        best_pid = 1
        for pid in range(2, nb_buckets):
            cost = compute_partition_cost(buckets, pid, nb_buckets, global_bound_area)
            if cost < min_cost:
                min_cost = cost
                best_pid = pid
        res = {
            "best_split": best_pid,
            "best_cost": min_cost,
            "buckets": buckets
        }
        return res

    def create_nodes_for_split(self, left_child_idx, right_child_idx, best_bucket_split, start, nb_buckets):
        self.nodes[left_child_idx].split_cost = best_bucket_split["best_cost"]
        self.nodes[right_child_idx].split_cost = best_bucket_split["best_cost"]
        bucket_size1 = 0
        bucket_size2 = 0
        best_pid = best_bucket_split["best_split"]
        for i1 in range(best_pid):
            for u in range(len(best_bucket_split["buckets"][i1].prims_idx_vec)):
                prim_id = best_bucket_split["buckets"][i1].prims_idx_vec[u]
                self.nodes[left_child_idx].prims_idx_vec.append(prim_id)
            bucket_size1 += len(best_bucket_split["buckets"][i1].prims_idx_vec)
        for i2 in range(best_pid, nb_buckets):
            for u in range(len(best_bucket_split["buckets"][i2].prims_idx_vec)):
                prim_id = best_bucket_split["buckets"][i2].prims_idx_vec[u]
                self.nodes[right_child_idx].prims_idx_vec.append(prim_id)
            bucket_size2 += len(best_bucket_split["buckets"][i2].prims_idx_vec)

        self.nodes[left_child_idx].start = start
        self.nodes[left_child_idx].size = bucket_size1
        self.nodes[right_child_idx].start = start + bucket_size1
        self.nodes[right_child_idx].size = bucket_size2

    def build_helper(self, parent_index):
        start = self.nodes[parent_index].start
        size = self.nodes[parent_index].size

        if size < 1:
            return
        elif size == 1:
            self.ordered_prims.append(self.nodes[parent_index].prims_idx_vec[0])
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
                for i in range(size):
                    prim_id = self.nodes[parent_index].prims_idx_vec[i]
                    global_bounds.enclose(self.primitives[prim_id].bounds)
                    centroid_bounds.enclose_point(self.primitive_centroids[prim_id])

                # find best split
                all_res = []
                if centroid_bounds.min_coord[0] != centroid_bounds.max_coord[0]:
                    res1 = self.sah_heuristic(parent_index, 0, nb_buckets, size, global_bounds, centroid_bounds)
                    all_res.append(res1)
                if centroid_bounds.min_coord[1] != centroid_bounds.max_coord[1]:
                    res2 = self.sah_heuristic(parent_index, 1, nb_buckets, size, global_bounds, centroid_bounds)
                    all_res.append(res2)

                if centroid_bounds.min_coord[2] != centroid_bounds.max_coord[2]:
                    res3 = self.sah_heuristic(parent_index, 2, nb_buckets, size, global_bounds, centroid_bounds)
                    all_res.append(res3)

                if len(all_res) == 0:
                    self.ordered_prims.extend(self.nodes[parent_index].prims_idx_vec)
                best_bucket_split = min(all_res, key=lambda du1: du1["best_cost"])

                # dont split if the cost is higher than earlier
                if best_bucket_split["best_cost"] > self.nodes[parent_index].split_cost:
                    self.ordered_prims.extend(self.nodes[parent_index].prims_idx_vec)

                # create nodes for the best split
                self.create_nodes_for_split(left_child_idx, right_child_idx, best_bucket_split, start, nb_buckets)

            else:
                self.ordered_prims.extend(self.nodes[parent_index].prims_idx_vec)

            self.nodes[parent_index].left = left_child_idx
            self.nodes[parent_index].right = right_child_idx
            self.nodes[left_child_idx].left = left_child_idx
            self.nodes[left_child_idx].right = left_child_idx
            self.nodes[right_child_idx].left = right_child_idx
            self.nodes[right_child_idx].right = right_child_idx

            self.build_helper(left_child_idx)
            self.build_helper(right_child_idx)

    def build(self, primitives):
        self.primitives = primitives
        self.primitive_centroids = np.zeros((len(primitives), 3))

        # pre-compute
        for i in range(len(primitives)):
            self.primitive_centroids[i] = primitives[i].bounds.center()

        # split tree
        root = self.create_tree_node()
        for i in range(len(primitives)):
            self.nodes[root].prims_idx_vec.append(i)
        self.nodes[root].size = len(primitives)

        self.ordered_prims = list(range(len(primitives)))
        self.build_helper(root)
        new_primitives = [self.primitives[i] for i in self.ordered_prims]
        self.primitives = new_primitives
        for i in range(len(self.nodes)):
            start = self.nodes[i].start
            size = self.nodes[i].size
            for j in range(start, start+size):
                self.nodes[i].box.enclose(self.primitives[j].bounds)

    def hit_helper(self, ray, node_id):
        if self.nodes[node_id].is_leaf() or self.nodes[node_id].box.is_empty():
            start = self.nodes[node_id].start
            size = self.nodes[node_id].size
            for i in range(start, start+size):
                ret2 = self.primitives[i].hit(ray)
                if ret2["hit"] and ret2["t"] < self.trace["t"]:
                    self.trace = ret2
        else:
            box_hit = self.nodes[node_id].box.hit(ray)
            if not box_hit["hit"]:
                return
            self.hit_helper(ray, self.nodes[node_id].left)
            self.hit_helper(ray, self.nodes[node_id].right)

    def hit(self, ray):
        self.trace = {"origin": ray.position, "hit": False, "t": MAX_F,
                      "position": np.array([0.0, 0.0, 0.0])}
        self.hit_helper(ray, 0)
        return self.trace
