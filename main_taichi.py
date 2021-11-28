from core.tracing import PathTracer
from time import time
from mathematics.intersection_taichi import World, Sphere
from core.bsdf_taichi import *
from io_utils.read_tungsten import read_file
from taichi_glsl.scalar import clamp, isnan
import math
import numpy as np


# switch to cpu if needed
ti.init(arch=ti.gpu)
# ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)

if __name__ == '__main__':
    a_scene, a_camera = read_file("media/cornell-box/scene.json")

    # image data
    image_width, image_height = a_camera.resolution
    pixels = ti.Vector.field(3, dtype=ti.f32)
    buffer = ti.Vector.field(3, dtype=ti.f32)
    luminances = ti.field(dtype=ti.f32)
    samples = ti.field(dtype=ti.f32)

    ti.root.dense(ti.ij, (image_width, image_height)).place(pixels, buffer,
                                                            samples, luminances)

    debugging = False
    samples_per_pixel = 64
    if debugging:
        samples_per_pixel = 1
        import random
        xc = random.randint(0, image_width)
        yc = random.randint(0, image_height)
        print("debugging with", xc, yc)

    max_depth = 16

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
    num_pixels = float(image_width * image_height)

    @ti.kernel
    def tonemap():
        for i, j in pixels:
            radiance = pixels[i, j] / samples[i, j]
            if isnan(radiance[0]) or isnan(radiance[1]) or isnan(radiance[2]):
                continue
            luminances[i, j] = radiance[0] * 0.2126 + radiance[1] * 0.7152 + radiance[2] * 0.0722

    @ti.kernel
    def finish():
        for x, y in pixels:
            buffer[x, y] = ti.sqrt(pixels[x, y] / samples[x, y])


    def finishing_tonemap():
        rgb_mat = pixels.to_numpy()
        lumi_mat = luminances.to_numpy()
        max_white_l = np.max(lumi_mat)
        numerator = lumi_mat * (1.0 + (lumi_mat / (max_white_l * max_white_l)))
        l_new = numerator / (1.0 + lumi_mat)
        l_scale = l_new/lumi_mat
        for i in range(3):
            rgb_mat[:, :, i] = rgb_mat[:, :, i] * l_scale / samples[0, 0]

        print(f"max, min: {np.max(rgb_mat), np.min(rgb_mat), max_white_l}")
        buffer.from_numpy(rgb_mat)

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

            color = path_tracer.trace(ray_org, ray_dir, depth, x, y)
            pixels[x, y] += color
            samples[x, y] += 1.0


    gui = ti.GUI('Cornell Box', (image_width, image_height), fast_gui=True)
    gui.fps_limit = 300
    last_t = time()
    iteration = 0
    interval = 10

    while gui.running:
        render()
        if iteration % interval == 0:
            # tonemap()
            # finishing_tonemap()
            finish()
            print("{:.2f} samples/s ({} iterations)".format(interval / (time() - last_t), iteration))
            last_t = time()
            gui.set_image(buffer)
            gui.show()
        iteration += 1
        if iteration % 100 == 0:
            # np_mat = pixels.to_numpy()
            # np.save("hdr", np_mat)
            # np_mat = samples.to_numpy()
            # np.save("spp", np_mat)

            ti.imwrite(buffer, 'out.png')
        if iteration > 1000:
            break
