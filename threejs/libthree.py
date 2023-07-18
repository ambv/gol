import random

from js import document
from js import window
from js import THREE
from js import TrackballControls
from js import Stats


def get_scene():
    return THREE.Scene.new()


def get_camera():
    camera = THREE.PerspectiveCamera.new(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000,
    )
    camera.position.z = 5
    return camera


def get_renderer():
    renderer = THREE.WebGLRenderer.new()
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.shadowMap.enabled = False
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.shadowMap.needsUpdate = True
    document.body.appendChild(renderer.domElement)
    return renderer


def get_lights():
    light_back_green = THREE.PointLight.new(0x00FF00, 1, 1000)
    light_back_green.decay = 3.0
    light_back_green.position.set(5, 0, 2)

    light_back_white = THREE.PointLight.new(0xFFFFFF, 5, 1000)
    light_back_white.decay = 20.0
    light_back_white.position.set(5, 0, 2)

    return light_back_green, light_back_white


def get_controls(camera, renderer):
    controls = TrackballControls.new(camera, renderer.domElement)
    controls.rotateSpeed = 1.0
    controls.zoomSpeed = 1.2
    controls.panSpeed = 0.8
    controls.noZoom = False
    controls.noPan = False
    controls.staticMoving = True
    controls.dynamicDampingFactor = 0.3
    return controls


def get_stats():
    stats = Stats.new()
    stats.showPanel(0)
    document.getElementById("stats").appendChild(stats.dom)
    return stats


def generate_cubes(scale):
    geometry = THREE.BoxGeometry.new(1, 1)
    while True:
        x = random.random() * scale - (scale / 2)
        y = random.random() * scale - (scale / 2)
        z = random.random() * scale - (scale / 2)
        material = THREE.MeshPhongMaterial.new(color=0xFFFFFF)
        cube = THREE.Mesh.new(geometry, material)
        cube.position.set(x, y, z)
        yield cube

