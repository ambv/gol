from typing import cast

import imageio.v3 as iio

from reduce_climb import get_average_color, Pixels


def test_average_color():
    arr = cast(Pixels, iio.imread(".local/input.jpg"))
    triangle = ((100, 200), (300, 500), (500, 900))
    assert (137, 127, 92) == get_average_color(arr, triangle)
    triangle = ((111, 222), (333, 666), (555, 444))
    assert (144, 136, 107) == get_average_color(arr, triangle)
    half1 = ((0, 0), (1599, 0), (0, 899))
    half2 = ((1599, 0), (0, 899), (1599, 899))
    assert (137, 152, 163) == get_average_color(arr, half1)
    assert (108, 111, 111) == get_average_color(arr, half2)
