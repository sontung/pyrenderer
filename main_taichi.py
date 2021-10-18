import sys

import taichi as ti
from mathematics.vec3_taichi import *
import core.ray_taichi as ray
from time import time
from mathematics.intersection_taichi import World, Sphere
from core.camera_taichi import Camera
from core.bsdf_taichi import *
from io_utils.read_tungsten import read_file

import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu)


@ti.func
def get_background(dir):
    ''' Returns the background color for a given direction vector '''
    unit_direction = dir.normalized()
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * WHITE + t * BLUE


if __name__ == '__main__':
    a_scene, a_camera = read_file("media/cornell-box/scene.json")

    # image data
    image_width, image_height = a_camera.resolution
    rays = ray.Rays(image_width, image_height)
    pixels = ti.Vector.field(3, dtype=float)
    sample_count = ti.field(dtype=ti.i32)
    needs_sample = ti.field(dtype=ti.i32)
    ti.root.dense(ti.ij,
                  (image_width, image_height)).place(pixels, sample_count,
                                                     needs_sample)

    samples_per_pixel = 512
    max_depth = 16

    # materials
    mat_ground = Lambert([0.5, 0.5, 0.5])
    mat2 = Lambert([0.4, 0.2, 0.2])
    mat1 = Dielectric(1.5)
    mat3 = Metal([0.7, 0.6, 0.5], 0.0)

    # world
    R = math.cos(math.pi / 4.0)
    world = World()
    for p in a_scene.primitives:
        world.add(p)
    world.commit()

    # camera
    vfrom = Point(13.0, 2.0, 3.0)
    at = Point(0.0, 0.0, 0.0)
    up = Vector(0.0, 1.0, 0.0)
    focus_dist = 10.0
    aperture = 0.1
    cam = a_camera.convert_to_taichi_camera()
    start_attenuation = Vector(1.0, 1.0, 1.0)
    initial = True

    @ti.kernel
    def finish():
        for x, y in pixels:
            pixels[x, y] = ti.sqrt(pixels[x, y] / samples_per_pixel)

    @ti.kernel
    def wavefront_initial():
        for x, y in pixels:
            sample_count[x, y] = 0
            needs_sample[x, y] = 1

    @ti.kernel
    def wavefront_big() -> ti.i32:
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample backgound
            return pixels that hit max samples
        '''
        num_completed = 0
        for x, y in pixels:
            if sample_count[x, y] == samples_per_pixel:
                continue

            # gen sample
            ray_org = Point(0.0, 0.0, 0.0)
            ray_dir = Vector(0.0, 0.0, 0.0)
            depth = max_depth
            pdf = start_attenuation

            if needs_sample[x, y] == 1:
                needs_sample[x, y] = 0
                u = (x + ti.random()) / (image_width - 1)
                v = (y + ti.random()) / (image_height - 1)
                ray_org, ray_dir = cam.get_ray(u, v)
                rays.set(x, y, ray_org, ray_dir, depth, pdf)
            else:
                ray_org, ray_dir, depth, pdf = rays.get(x, y)

            # intersect
            hit, p, n, front_facing, index = world.hit_all(ray_org, ray_dir)
            depth -= 1
            rays.depth[x, y] = depth
            if hit:
                reflected, out_origin, out_direction, attenuation = world.materials.scatter(
                    index, ray_dir, p, n, front_facing)
                rays.set(x, y, out_origin, out_direction, depth,
                         pdf * attenuation)
                ray_dir = out_direction

            if not hit or depth == 0:
                pixels[x, y] += pdf * get_background(ray_dir)
                sample_count[x, y] += 1
                needs_sample[x, y] = 1

                if sample_count[x, y] == samples_per_pixel:
                    num_completed += 1

        return num_completed

    num_pixels = image_width * image_height

    t = time()
    print('starting big wavefront')
    wavefront_initial()
    num_completed = 0
    while num_completed < num_pixels:
        num_completed += wavefront_big()
        # print('completed', num_completed)

    finish()
    print(time() - t)
    ti.imwrite(pixels.to_numpy(), 'out.png')
