import asyncio
import concurrent.futures
from contextvars import ContextVar
from dataclasses import dataclass
import os
import random

import click
import numpy as np
from PIL import Image, ImageDraw
from rich.progress import Progress
from rich.console import Console


Point = tuple[int, int]
Triangle = tuple[Point, Point, Point]
RGB = tuple[int, int, int]


@dataclass
class ColorTriangle:
    color: RGB
    triangle: Triangle
    difference: int


console = Console()
print = console.print
CPU_COUNT: int = os.cpu_count() or 2
INPUT_IMG: ContextVar[Image.Image] = ContextVar("input_image")
OUTPUT_IMG: ContextVar[Image.Image] = ContextVar("output_image")


def calculate_difference(image1: Image.Image, image2: Image.Image) -> int:
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
    current_difference: int,
    last_triangle: ColorTriangle | None,
    attempts: int,
) -> ColorTriangle:
    input_image = INPUT_IMG.get()
    output_image = OUTPUT_IMG.get()
    width, height = input_image.size

    # Calculate the difference between the input and the output image
    if last_triangle is not None:
        output_image = apply_triangle(
            output_image, last_triangle.triangle, last_triangle.color
        )
        OUTPUT_IMG.set(output_image)

    smallest_difference = calculate_difference(input_image, output_image)
    assert smallest_difference == current_difference, "Workers out of sync with manager"

    best_triangle = None
    best_color = None

    iterations = 0
    while smallest_difference > 0 and iterations < attempts:
        iterations += 1
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
            best_color = candidate_color

    if best_triangle is not None and best_color is not None:
        return ColorTriangle(
            color=best_color, triangle=best_triangle, difference=smallest_difference
        )

    raise LookupError(f"Couldn't improve the score in {attempts} iterations")


def init_images(input_path: str) -> None:
    input_image = Image.open(input_path)
    width, height = input_image.size
    average_color: RGB = input_image.resize((1, 1), resample=Image.LANCZOS).getpixel((0, 0))  # type: ignore
    output_image = Image.new("RGB", (width, height), average_color)
    INPUT_IMG.set(input_image)
    OUTPUT_IMG.set(output_image)


async def reduce_image_to_output(
    input_path: str,
    output_path: str,
    num_triangles: int = 500,
    attempts_per_triangle: int = 1000,
) -> None:
    init_images(input_path)
    triangles: list[ColorTriangle] = []
    loop = asyncio.get_running_loop()

    input_image = INPUT_IMG.get()
    output_image = OUTPUT_IMG.get()
    current_difference = calculate_difference(input_image, output_image)

    with (
        concurrent.futures.ProcessPoolExecutor(
            initializer=init_images,
            initargs=(input_path,),
            max_workers=CPU_COUNT,
        ) as pool,
        Progress(console=console, auto_refresh=False) as progress,
    ):
        task_id = progress.add_task("Generating...", total=num_triangles)
        while len(triangles) < num_triangles:
            print(f"{len(triangles)} triangles so far...")
            last_triangle = triangles[-1] if triangles else None
            attempts = [
                loop.run_in_executor(
                    pool,
                    choose_next_triangle,
                    current_difference,
                    last_triangle,
                    attempts_per_triangle,
                )
                for _ in range(CPU_COUNT)
            ]

            triangle: ColorTriangle | None = None
            for attempt in asyncio.as_completed(attempts):
                try:
                    candidate = await attempt
                    if triangle is None or triangle.difference > candidate.difference:
                        triangle = candidate
                except LookupError:
                    print("  One worker didn't find anything")
                    continue
            if triangle is None:
                print("  No candidates in this round")
                break

            triangles.append(triangle)
            current_difference = triangle.difference
            print(f"  Difference improved to {current_difference}")
            progress.update(task_id, advance=1)

    # Save the final output image
    for t in triangles:
        apply_triangle(output_image, t.triangle, t.color)
    output_image.save(output_path)
    print(f"Output image saved to {output_path}")


@click.command()
@click.argument("input_path")
@click.argument("output_path")
def diff_images(input_path: str, output_path: str) -> None:
    input_image = Image.open(input_path)
    output_image = Image.open(output_path)
    print(calculate_difference(input_image, output_image))


@click.command()
@click.argument("input_path")
@click.argument("output_path")
def reduce_main(input_path: str, output_path: str) -> None:
    asyncio.run(reduce_image_to_output(input_path, output_path))


if __name__ == "__main__":
    reduce_main()
    # diff_images()
