import taichi as ti
import copy
from ..mathematics.constants import GAMMA2_3


def surrounding_box(box1, box2):
    ''' Calculates the surround bbox of two bboxes '''
    box1_min, box1_max = box1
    box2_min, box2_max = box2

    small = [
        min(box1_min[0], box2_min[0]),
        min(box1_min[1], box2_min[1]),
        min(box1_min[2], box2_min[2])
    ]
    big = [
        max(box1_max[0], box2_max[0]),
        max(box1_max[1], box2_max[1]),
        max(box1_max[2], box2_max[2])
    ]
    return small, big


def sort_obj_list(obj_list):
    ''' Sort the list of objects along the longest directional span '''
    def get_x(e):
        return e.center[0]

    def get_y(e):
        return e.center[1]

    def get_z(e):
        return e.center[2]

    centers = [obj.center for obj in obj_list]
    min_center = [
        min([center[0] for center in centers]),
        min([center[1] for center in centers]),
        min([center[2] for center in centers])
    ]
    max_center = [
        max([center[0] for center in centers]),
        max([center[1] for center in centers]),
        max([center[2] for center in centers])
    ]
    span_x, span_y, span_z = (max_center[0] - min_center[0],
                              max_center[1] - min_center[1],
                              max_center[2] - min_center[2])
    if span_x >= span_y and span_x >= span_z:
        obj_list.sort(key=get_x)
    elif span_y >= span_z:
        obj_list.sort(key=get_y)
    else:
        obj_list.sort(key=get_z)
    return obj_list


class BVHNode:
    """ A bvh node for constructing the bvh tree.  Note this is done on CPU """

    left = None
    right = None
    obj = None
    box_min = box_max = []
    id = 0
    parent = None
    total = 0

    def __init__(self, object_list, parent):
        self.parent = parent
        obj_list = copy.copy(object_list)

        span = len(object_list)
        if span == 1:
            # one obj, set to sphere bbox
            self.obj = obj_list[0]
            self.box_min, self.box_max = obj_list[0].bounding_box
            self.total = 1
        else:
            # set left and right child and this bbox is the sum of two
            mid = int(span / 2)
            self.left = BVHNode(obj_list[:mid], self)
            self.right = BVHNode(obj_list[mid:], self)
            self.box_min, self.box_max = surrounding_box(
                self.left.bounding_box, self.right.bounding_box)
            self.total = self.left.total + self.right.total + 1

    @property
    def bounding_box(self):
        return self.box_min, self.box_max

    @property
    def next(self):
        ''' Returns the next node to walk '''
        node = self

        while True:
            if node.parent is not None and node.parent.right is not node:
                return node.parent.right
            elif node.parent is None:
                return None
            else:
                node = node.parent
        return None


@ti.data_oriented
class BVH:
    """ The BVH class takes a list of objects and creates a bvh from them.
        The bvh structure contains a "next" pointer for walking the tree. """
    def __init__(self, object_list):
        self.root = BVHNode(object_list, None)

        total = self.root.total

        self.bvh_obj_id = ti.field(ti.i32)
        self.bvh_left_id = ti.field(ti.i32)
        self.bvh_right_id = ti.field(ti.i32)
        self.bvh_next_id = ti.field(ti.i32)
        self.bvh_min = ti.Vector.field(3, dtype=ti.f32)
        self.bvh_max = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, total).place(self.bvh_obj_id, self.bvh_left_id,
                                         self.bvh_right_id, self.bvh_next_id,
                                         self.bvh_min, self.bvh_max)

    def build(self):
        """ building function. Compress the object list to structure"""
        i = 0

        # first walk tree and give ids
        def walk_bvh(node):
            nonlocal i
            node.id = i
            i += 1
            if node.left:
                walk_bvh(node.left)
            if node.right:
                walk_bvh(node.right)

        walk_bvh(self.root)

        def save_bvh(node):
            id = node.id

            self.bvh_obj_id[id] = node.obj.id if node.obj is not None else -1
            self.bvh_left_id[
                id] = node.left.id if node.left is not None else -1
            self.bvh_right_id[
                id] = node.right.id if node.right is not None else -1
            self.bvh_next_id[
                id] = node.next.id if node.next is not None else -1
            self.bvh_min[id] = node.box_min
            self.bvh_max[id] = node.box_max

            if node.left is not None:
                save_bvh(node.left)
            if node.right is not None:
                save_bvh(node.right)

        save_bvh(self.root)
        self.bvh_root = 0

    @ti.func
    def get_id(self, bvh_id):
        """ Get the obj id for a bvh node """
        return self.bvh_obj_id[bvh_id]

    @ti.func
    def hit_aabb(self, bvh_id, ray_origin, ray_direction, t0, t1):
        """ Use the slab method to do aabb test"""
        intersect = 1
        min_aabb = self.bvh_min[bvh_id]
        max_aabb = self.bvh_max[bvh_id]

        for i in ti.static(range(3)):
            t_near = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
            t_far = (max_aabb[i] - ray_origin[i]) / ray_direction[i]

            if t_near > t_far:
                t_near, t_far = t_far, t_near

            t_far *= 1 + 2 * GAMMA2_3

            if t_near > t0:
                t0 = t_near
            if t_far < t1:
                t1 = t_far
            if t0 > t1:
                intersect = 0
        return intersect

    @ti.func
    def get_full_id(self, i):
        """ Gets the obj id, left_id, right_id, next_id for a bvh node """
        return self.bvh_obj_id[i], self.bvh_left_id[i], self.bvh_right_id[i], self.bvh_next_id[i]
