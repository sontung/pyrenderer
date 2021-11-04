import cv2
from core.tracing import PathTracer
import core.ray_taichi as ray
from time import time
from mathematics.intersection_taichi import World, Sphere
from core.bsdf_taichi import *
from io_utils.read_tungsten import read_file
import math
import random


# switch to cpu if needed
ti.init(arch=ti.gpu, debug=True)
# ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)


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
    pixels = ti.Vector.field(3, dtype=float)
    buffer = ti.Vector.field(3, dtype=float)
    samples = ti.field(dtype=ti.f32)

    ti.root.dense(ti.ij,
                  (image_width, image_height)).place(pixels, buffer, samples)

    debugging = False
    samples_per_pixel = 16
    if debugging:
        samples_per_pixel = 1
        import random
        xc = random.randint(0, image_width)
        yc = random.randint(0, image_height)
        print("debugging with", xc, yc)

    max_depth = 32

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
    path_tracer = PathTracer(world, max_depth, image_width, image_height)

    @ti.kernel
    def finish():
        for x, y in pixels:
            buffer[x, y] = ti.sqrt(pixels[x, y] / samples[x, y])

    @ti.kernel
    def render():
        ''' Loops over pixels
            for each pixel:
                generate ray if needed
                intersect scene with ray
                if miss or last bounce sample background
            return pixels that hit max samples
        '''
        for x, y in pixels:

            # gen sample
            depth = max_depth
            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.gen_ray(u, v)

            color = path_tracer.trace2(ray_org, ray_dir, depth, x, y)
            pixels[x, y] += color
            samples[x, y] += 1.0


    @ti.kernel
    def debug():
        for x, y in pixels:
            if x != xc or y != yc:
                continue
            u = (x + ti.random()) / (image_width - 1)
            v = (y + ti.random()) / (image_height - 1)
            ray_org, ray_dir = cam.gen_ray(u, v)
            color = path_tracer.trace2(ray_org, ray_dir, max_depth, x, y)
            if color[0]+color[1]+color[2] < 0.01:
                print("black color", x, y)


    num_pixels = image_width * image_height

    gui = ti.GUI('Cornell Box', (image_width, image_height), fast_gui=True)
    gui.fps_limit = 300
    last_t = time()
    iteration = 0
    while gui.running:
        render()
        interval = 10
        if iteration % interval == 0:
            finish()
            print("{:.2f} samples/s ({} iters)".format(
                interval / (time() - last_t), iteration))
            last_t = time()
            gui.set_image(buffer)
            gui.show()
        iteration += 1

    # if not debugging:
    #     t = time()
    #     print('starting big wavefront')
    #     wavefront_initial()
    #     num_completed = 0
    #     while num_completed < num_pixels:
    #         num_completed += wavefront_big()
    #     finish2()
    #     print("completed in", time() - t)
    #     ti.imwrite(pixels.to_numpy(), 'out.png')
    #     cv2.imshow("t", pixels.to_numpy())
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    # else:
    #     debug()
