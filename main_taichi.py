import taichi as ti
from io_utils.read_tungsten import read_file

ti.init(arch=ti.gpu)


@ti.kernel
def render(buf: ti.template()):
    for u, v in buf:
        buf[u, v] = ti.Vector([0.678, 0.063, v / 800])


a_scene, a_camera = read_file("media/cornell-box/scene.json")
x_dim, y_dim = a_camera.get_resolution()
color_buffer = ti.Vector.field(3, dtype=ti.f64, shape=(x_dim, y_dim))

gui = ti.GUI('Cornell Box', (x_dim, y_dim))
for i in range(10):
    render(color_buffer)
    img = color_buffer.to_numpy()
    gui.set_image(img)
    gui.show()

input("Press any key to quit")
