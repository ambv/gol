import numpy as np
from PIL import Image

from reduce_climb import get_average_color


def test_average_color():
    img = Image.open(".local/input.jpg")
    arr = np.array(img)
    triangle = ((100, 200), (300, 400), (500, 600))
    assert (133, 135, 126) == get_average_color(arr, triangle)
    triangle = ((111, 222), (333, 444), (555, 666))
    assert (116, 117, 108) == get_average_color(arr, triangle)
    half1 = ((0, 0), (1599, 0), (0, 899))
    half2 = ((1599, 0), (0, 899), (1599, 899))
    assert (137, 152, 163) == get_average_color(arr, half1)
    assert (108, 111, 111) == get_average_color(arr, half2)
