import sys
from taichi_glsl.vector import normalize, dot
from core.tracing import PathTracer
import taichi as ti
from mathematics.vec3_taichi import *
import core.ray_taichi as ray
from time import time
from mathematics.intersection_taichi import World, Sphere
from core.camera_taichi import Camera
from core.bsdf_taichi import *
from io_utils.read_tungsten import read_file
import numpy as np
import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu, debug=True)


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
    ray_o_stored = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
    normal_stored = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))

    ray_d_stored = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))
    uv_stored = ti.Vector.field(n=2, dtype=ti.f32, shape=(image_width, image_height))
    t_stored = ti.field(dtype=ti.f32, shape=(image_width, image_height))
    hit_stored = ti.field(dtype=ti.i8, shape=(image_width, image_height))

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
    cam = a_camera.convert_to_taichi_camera()
    start_attenuation = Vector(1.0, 1.0, 1.0)
    initial = True
    path_tracer = PathTracer(world)

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
            depth = max_depth
            pdf = start_attenuation

            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.gen_ray(u, v)
            rays.set(x, y, ray_org, ray_dir, depth, pdf)

            e, r = path_tracer.trace(ray_org, ray_dir, depth)
            pixels[x, y] += e+r
            sample_count[x, y] += 1
            needs_sample[x, y] = 1

            if sample_count[x, y] == samples_per_pixel:
                num_completed += 1

        return num_completed

    @ti.kernel
    def debug():
        for x, y in pixels:
            if x != 100 or y != 100:
                continue

            needs_sample[x, y] = 0
            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.gen_ray(u, v)
            e, r = path_tracer.trace(ray_org, ray_dir, max_depth)


    num_pixels = image_width * image_height
    debugging = False
    if not debugging:
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
    else:
        debug()
        for x in range(0, image_width, 100):
            for y in range(0, image_height, 100):
                # if x != 50 or y != 700:
                #     continue
                u, v = uv_stored[x, y]
                ray_org = ray_o_stored[x, y]
                ray_dir = ray_d_stored[x, y]
                n = normal_stored[x, y]
                ray = a_camera.generate_ray(np.array([u, v]))
                trace = a_scene.hit(ray)

                normal_ref = trace["normal"]
                diff = np.sum(np.abs(n.to_numpy()-normal_ref))
                if diff > 0.01:
                    print(n, normal_ref)
                    print(x, y, diff)
                    sys.exit()

                try:
                    assert trace["hit"]-hit_stored[x, y] == 0
                except AssertionError:
                    print(u, v)
                    sys.exit()
                if trace["hit"]:
                    diff = trace["t"]-t_stored[x, y]
                    if abs(diff) > 0.01:
                        print(x, y, trace["t"], t_stored[x, y])
                        sys.exit()
