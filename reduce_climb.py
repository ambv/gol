import functools
import random
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
from rich.progress import Progress
from rich.console import Console


Point = tuple[int, int]
Triangle = tuple[Point, Point, Point]
RGB = tuple[int, int, int]


console = Console()
print = console.print


def calculate_difference(image1: Image.Image, image2: Image.Image):
    """Calculate the difference between two images."""
    diff = np.array(image1) - np.array(image2)
    squared_diff = np.square(diff)
    return np.sum(squared_diff)


def generate_candidate_triangle(
    max_x: int, max_y: int, mutate: Triangle | None = None
) -> Triangle:
    if mutate is not None:
        x = [p[0] for p in mutate]
        y = [p[1] for p in mutate]
        which_point = random.randint(0, 2)
        if random.random() < 0.5:
            x[which_point] = random.randint(0, max_x)
        else:
            y[which_point] = random.randint(0, max_y)
    else:
        x = [random.randint(0, max_x) for _ in range(3)]
        y = [random.randint(0, max_y) for _ in range(3)]
    return ((x[0], y[0]), (x[1], y[1]), (x[2], y[2]))


def get_average_color(image: Image.Image, triangle_coords: Triangle) -> RGB:
    width, height = image.size
    pixels = np.array(image)
    mask = np.zeros((height, width), dtype=bool)
    mask_img = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(triangle_coords, outline=1, fill=1)
    mask_pixels = np.array(mask_img, dtype=bool)
    mask[:, :] = mask_pixels

    num_pixels = np.sum(mask)
    if num_pixels == 0:
        raise LookupError("No pixels under the triangle")

    masked_pixels = pixels * mask[..., np.newaxis]
    average_color = np.sum(masked_pixels, axis=(0, 1)) // num_pixels
    return tuple(average_color.tolist())  # type: ignore


def calculate_triangle_area(triangle: Triangle) -> float:
    """Calculate the area of a triangle."""
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)


def apply_triangle(image: Image.Image, triangle: Triangle, color: RGB) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.polygon(triangle, fill=color)
    return output


def choose_next_triangle(
    input_image: Image.Image,
    output_image: Image.Image,
    attempts: int = 1000,
    callback: Callable[[], None] = lambda: None,
) -> Image.Image:
    width, height = input_image.size

    # Calculate the difference between the input and the output image
    smallest_difference = calculate_difference(input_image, output_image)
    best_triangle = None
    best_image = None

    iterations = 0
    while smallest_difference > 0 and iterations < attempts:
        iterations += 1
        callback()

        candidate_triangle = generate_candidate_triangle(
            width, height, mutate=best_triangle
        )
        candidate_color = get_average_color(input_image, candidate_triangle)
        new_output_image = apply_triangle(
            output_image, candidate_triangle, candidate_color
        )

        new_difference = calculate_difference(input_image, new_output_image)

        if new_difference < smallest_difference:
            smallest_difference = new_difference
            best_triangle = candidate_triangle
            best_image = new_output_image
            print(
                f"  Difference improved to {smallest_difference}"
                f" at iteration {iterations}"
            )

    if best_image is not None:
        return best_image

    raise LookupError(f"Couldn't improve the score in {attempts} iterations")


def reduce_image_to_output(
    input_path: str,
    output_path: str,
    num_triangles: int = 500,
    attempts_per_triangle: int = 1000,
) -> None:
    # Load the input image
    input_image = Image.open(input_path)
    width, height = input_image.size

    # Create the output image with the average color of the input as the background
    average_color = input_image.resize((1, 1), resample=Image.LANCZOS).getpixel((0, 0))
    output_image = Image.new("RGB", (width, height), average_color)

    candidate_images = [output_image]
    # Hill-climbing algorithm to improve the output image
    with Progress(console=console) as progress:
        task_id = progress.add_task(
            "Generating...", total=num_triangles * attempts_per_triangle
        )
        update = functools.partial(progress.update, task_id, advance=1)
        while len(candidate_images) < num_triangles:
            print(f"{len(candidate_images)} triangles so far...")
            try:
                candidate_images.append(
                    choose_next_triangle(
                        input_image,
                        candidate_images[-1],
                        attempts=attempts_per_triangle,
                        callback=update,
                    )
                )
            except LookupError:
                break  # we can't improve anymore

    # Save the final output image
    output_image = candidate_images[-1]
    output_image.save(output_path)
    print(f"Output image saved to {output_path}")


if __name__ == "__main__":
    reduce_image_to_output("input.jpg", "output_image8.jpg")
