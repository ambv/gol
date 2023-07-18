import math
import time

from pyodide.ffi import create_proxy
from js import window

from libthree import get_scene, get_camera, get_renderer
from libthree import get_controls, get_lights
from libthree import generate_cubes, get_stats

scene = get_scene()
camera = get_camera()
renderer = get_renderer()
controls = get_controls(camera, renderer)
lights = get_lights()
stats = get_stats()
cubes = []


def init():
    for cube in generate_cubes(scale=1000):
        scene.add(cube)
        cubes.append(cube)
        if len(cubes) == 50000:
            break

    for light in lights:
        scene.add(light)


def animate(now):
    stats.begin()
    controls.update()
    camera.position.x += math.sin(now * 0.0001)
    camera.position.z += math.cos(now * 0.0001)
    light_back_green, light_back_white = lights
    light_back_green.position.x = camera.position.x
    light_back_green.position.y = camera.position.y
    light_back_green.position.z = camera.position.z
    light_back_white.position.x = camera.position.x
    light_back_white.position.y = camera.position.y
    light_back_white.position.z = camera.position.z
    camera.lookAt(scene.position)
    renderer.render(scene, camera)
    stats.end()
    window.requestAnimationFrame(animate_js)


animate_js = create_proxy(animate)

if __name__ == "__main__":
    init()
    animate(time.time())
