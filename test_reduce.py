from typing import cast

import imageio.v3 as iio

from reduce_climb import get_average_color, Pixels, ColorTriangle
from reduce_climb import calculate_full_difference, calculate_triangle_difference
from reduce_climb import apply_triangle

import skimage.util


def test_average_color():
    arr = cast(Pixels, iio.imread(".local/input1.jpg"))
    triangle = ColorTriangle([100, 300, 500], [200, 500, 900])
    assert (137, 127, 92) == get_average_color(arr, triangle)
    triangle = ColorTriangle([111, 333, 555], [222, 666, 444])
    assert (144, 136, 107) == get_average_color(arr, triangle)
    half1 = ColorTriangle([0, 1599, 0], [0, 0, 899])
    half2 = ColorTriangle([1599, 0, 1599], [0, 899, 899])
    assert (137, 152, 163) == get_average_color(arr, half1)
    assert (108, 111, 111) == get_average_color(arr, half2)


def test_difference():
    tr_half1 = ColorTriangle([0, 1599, 0], [0, 0, 899])
    tr_half2 = ColorTriangle([1599, 0, 1599], [0, 899, 899])
    input = cast(Pixels, iio.imread(".local/input1.jpg"))
    inverted = skimage.util.invert(input)
    total = calculate_full_difference(input, inverted)
    assert total == 538896328

    with_th1 = apply_triangle(inverted, tr_half1)
    total_th1 = calculate_full_difference(input, with_th1)
    assert total_th1 == 499533150
    diff_th1 = calculate_triangle_difference(total, input, inverted, with_th1, tr_half1)
    assert total_th1 == diff_th1

    with_th2 = apply_triangle(inverted, tr_half2)
    total_th2 = calculate_full_difference(input, with_th2)
    assert total_th2 == 497742548
    diff_th2 = calculate_triangle_difference(total, input, inverted, with_th2, tr_half2)
    assert total_th2 == diff_th2

    with_both = apply_triangle(with_th1, tr_half2)
    total_both = calculate_full_difference(input, with_both)
    assert total_both == 458379384
    diff_both = calculate_triangle_difference(
        total_th1, input, with_th1, with_both, tr_half2
    )
    assert total_both == diff_both
