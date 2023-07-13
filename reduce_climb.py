from __future__ import annotations

import asyncio
import concurrent.futures
from contextvars import ContextVar
from dataclasses import dataclass, field
import multiprocessing
from pathlib import Path
import os
import random
from typing import cast, TypeVar

import click
import numpy as np
import numpy.typing as npt
import imageio.v3 as iio
from PIL import Image
from rich.progress import Progress, TaskID
from rich.console import Console
from skimage.draw import polygon, polygon_perimeter  # type: ignore

import profiling


Point = tuple[int, int]
RGB = tuple[int, int, int]
Num = TypeVar("Num", int, float)
Pixels = npt.NDArray[np.uint8]


BLACK: RGB = (0, 0, 0)


@dataclass
class ColorTriangle:
    xs: list[int]
    ys: list[int]
    color: RGB = field(default=BLACK)
    difference: int = field(default=0)

    def copy(
        self,
        *,
        xs: list[int] | None = None,
        ys: list[int] | None = None,
        color: RGB | None = None,
        difference: int | None = None,
    ) -> ColorTriangle:
        return ColorTriangle(
            xs=list(self.xs) if xs is None else xs,
            ys=list(self.ys) if ys is None else ys,
            color=self.color if color is None else color,
            difference=self.difference if difference is None else difference,
        )


@dataclass
class ProgressUpdate:
    completed: int
    total: int
    profile_this: bool = field(default=False)

    def done(self):
        return self.completed == self.total


ProgressDict = dict[TaskID, ProgressUpdate]


console = Console()
print = console.print
CPU_COUNT: int = os.cpu_count() or 2
INPUT_PIXELS: ContextVar[Pixels] = ContextVar("input_pixels")
OUTPUT_PIXELS: ContextVar[Pixels] = ContextVar("output_pixels")


def clamp(num: Num, minimum: Num, maximum: Num) -> Num:
    return min(maximum, max(minimum, num))


def calculate_difference(image1: Pixels, image2: Pixels) -> int:
    """Calculate the difference between two images."""
    diff = image1 - image2
    squared_diff = np.square(diff)
    return np.sum(squared_diff)


def generate_candidate_triangle(
    pixels: Pixels,
    delta: int | None = None,
    mutate: ColorTriangle | None = None,
) -> ColorTriangle:
    max_y, max_x = pixels.shape[:2]
    delta = clamp(max_x // 100, 16, 128)

    if mutate is not None:
        result = mutate.copy(color=BLACK, difference=0)
        idx = random.randint(0, 2)
        result.xs[idx] = clamp(
            result.xs[idx] + random.randint(-delta, delta), -delta, max_x + delta
        )
        result.ys[idx] = clamp(
            result.ys[idx] + random.randint(-delta, delta), -delta, max_y + delta
        )
    else:
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        x2 = x1 + random.randint(-delta, delta)
        y2 = y1 + random.randint(-delta, delta)
        x3 = x1 + random.randint(-delta, delta)
        y3 = y1 + random.randint(-delta, delta)
        result = ColorTriangle(
            xs=[x1, x2, x3],
            ys=[y1, y2, y3],
        )
    result.color = get_average_color(pixels, result)
    return result


def get_average_color(pixels: Pixels, triangle: ColorTriangle) -> RGB:
    rr: npt.NDArray[np.int64]
    cc: npt.NDArray[np.int64]

    rr, cc = polygon(triangle.ys, triangle.xs, shape=pixels.shape[:2])
    num_pixels = len(rr)
    if num_pixels == 0:
        raise LookupError("No pixels under the triangle")

    average_color = np.sum(pixels[rr, cc], axis=0) // num_pixels
    return tuple(average_color.tolist())  # type: ignore


def apply_triangle(pixels: Pixels, triangle: ColorTriangle) -> Pixels:
    output = np.copy(pixels)  # FIXME: avoid this with an Undo
    rr, cc = polygon(triangle.ys, triangle.xs, shape=pixels.shape[:2])
    output[rr, cc] = triangle.color
    rr, cc = polygon_perimeter(triangle.ys, triangle.xs, shape=pixels.shape[:2])
    output[rr, cc] = triangle.color
    return output


def choose_next_triangle(
    current_difference: int,
    last_triangle: ColorTriangle | None,
    attempts: int,
    progress_dict: ProgressDict,
    task_id: TaskID,
) -> ColorTriangle:
    with profiling.maybe(progress_dict[task_id].profile_this):
        return _choose_next_triangle(
            current_difference, last_triangle, attempts, progress_dict, task_id
        )


def _choose_next_triangle(
    current_difference: int,
    last_triangle: ColorTriangle | None,
    attempts: int,
    progress_dict: ProgressDict,
    task_id: TaskID,
) -> ColorTriangle:
    input_pixels = INPUT_PIXELS.get()
    output_pixels = OUTPUT_PIXELS.get()

    # Calculate the difference between the input and the output image
    if last_triangle is not None:
        output_pixels = apply_triangle(output_pixels, last_triangle)
        OUTPUT_PIXELS.set(output_pixels)

    smallest_difference = calculate_difference(input_pixels, output_pixels)
    assert smallest_difference == current_difference, "Workers out of sync with manager"

    best_triangle = None

    update = progress_dict[task_id]
    iterations = 0
    while iterations < attempts:
        iterations += 1
        try:
            candidate_triangle = generate_candidate_triangle(
                input_pixels, mutate=best_triangle
            )
        except LookupError:
            continue  # triangle entirely out of bounds
        new_output_pixels = apply_triangle(output_pixels, candidate_triangle)
        new_difference = calculate_difference(input_pixels, new_output_pixels)
        if new_difference < smallest_difference:
            candidate_triangle.difference = new_difference
            smallest_difference = new_difference
            best_triangle = candidate_triangle
        update.completed = iterations
        progress_dict[task_id] = update  # shared dict needs explicit assignment

    if best_triangle is None:
        raise LookupError(f"Couldn't improve the score in {attempts} iterations")

    return best_triangle


def save_image_with_triangles(
    pixels: Pixels,
    path: Path | str,
    *,
    diff_with: Pixels,
    triangles: list[ColorTriangle],
) -> None:
    for i, t in enumerate(triangles):
        pixels = apply_triangle(pixels, t)
        actual = calculate_difference(diff_with, pixels)
        if actual != t.difference:
            print(
                f"warning: invalid difference at triangle {i}."
                f" Expected: {t.difference}, actual: {actual}"
            )
    iio.imwrite(path, pixels)


def path_with_index(path: str | Path, index: int) -> Path:
    p = Path(path)
    s = p.suffix
    n = p.with_suffix("").name
    return p.with_name(n + f"_{index:03}").with_suffix(s)


def init_images(input_path: str) -> None:
    input_image = Image.open(input_path)
    width, height = input_image.size
    average_color: RGB = input_image.resize((1, 1), resample=Image.LANCZOS).getpixel((0, 0))  # type: ignore
    output_image = Image.new("RGB", (width, height), average_color)  # type: ignore
    INPUT_PIXELS.set(np.array(input_image))
    OUTPUT_PIXELS.set(np.array(output_image))


async def reduce_image_to_output(
    input_path: str,
    output_path: str,
    num_triangles: int = 500,
    attempts_per_triangle: int = 1000,
    max_retries: int = 3,
) -> None:
    init_images(input_path)
    triangles: list[ColorTriangle] = []
    loop = asyncio.get_running_loop()

    input_pixels = INPUT_PIXELS.get()
    output_pixels = OUTPUT_PIXELS.get()
    current_difference = calculate_difference(input_pixels, output_pixels)

    with (
        multiprocessing.Manager() as manager,
        concurrent.futures.ProcessPoolExecutor(
            initializer=init_images,
            initargs=(input_path,),
            max_workers=CPU_COUNT,
        ) as pool,
        Progress(console=console) as progress,
    ):
        _progress = cast(ProgressDict, manager.dict())
        _tasks: list[TaskID] = []
        for i in range(CPU_COUNT):
            _tasks.append(progress.add_task(f"task {i+1}", visible=False))
            _progress[_tasks[i]] = ProgressUpdate(
                completed=0, total=attempts_per_triangle, profile_this=i == 0
            )

        retries = 0
        overall = progress.add_task(
            "Generating...", total=num_triangles * CPU_COUNT * attempts_per_triangle
        )
        while len(triangles) < num_triangles and retries < max_retries:
            last_triangle = triangles[-1] if triangles and retries == 0 else None
            attempts = [
                loop.run_in_executor(
                    pool,
                    choose_next_triangle,
                    current_difference,
                    last_triangle,
                    attempts_per_triangle,
                    _progress,
                    _tasks[i],
                )
                for i in range(CPU_COUNT)
            ]

            async def monitor() -> None:
                base = len(triangles) * CPU_COUNT * attempts_per_triangle
                for task_id in _progress:
                    progress.reset(task_id)
                while True:
                    await asyncio.sleep(0.1)
                    current = 0
                    for task_id, update_data in _progress.items():
                        # update the progress bar for this task:
                        progress.update(
                            task_id,
                            completed=update_data.completed,
                            total=update_data.total,
                            visible=not update_data.done(),
                        )
                        current += update_data.completed
                    progress.update(
                        overall,
                        description=f"{len(triangles)} triangles so far...",
                        completed=base + current,
                    )

            empty = 0
            triangle: ColorTriangle | None = None
            monitor_task = asyncio.create_task(monitor(), name="monitor")
            try:
                for attempt in asyncio.as_completed(attempts):
                    try:
                        candidate = await attempt
                        if (
                            triangle is None
                            or triangle.difference > candidate.difference
                        ):
                            triangle = candidate
                    except LookupError:
                        empty += 1
                        continue
            finally:
                monitor_task.cancel()
            if triangle is None:
                print("  No candidates in this round")
                retries += 1
                continue

            retries = 0
            triangles.append(triangle)
            delta_difference = current_difference - triangle.difference
            current_difference = triangle.difference

            maybe_empty = ""
            if empty:
                maybe_empty = f" with {empty} empty results"
            print(
                f"{len(triangles):03}: Difference improved to {current_difference}"
                f" (delta {delta_difference}) {maybe_empty}"
            )
            p = path_with_index(output_path, len(triangles))
            save_image_with_triangles(
                output_pixels, p, diff_with=input_pixels, triangles=triangles
            )

    # Save the final output image
    print(f"Output image saved to {output_path}")
    save_image_with_triangles(
        output_pixels, output_path, diff_with=input_pixels, triangles=triangles
    )


@click.command()
@click.argument("input_path")
@click.argument("output_path")
def diff_images(input_path: str, output_path: str) -> None:
    input_pixels = iio.imread(input_path)
    output_pixels = iio.imread(output_path)
    print(calculate_difference(input_pixels, output_pixels))


@click.command()
@click.argument("input_path")
@click.argument("output_path")
def reduce_main(input_path: str, output_path: str) -> None:
    asyncio.run(reduce_image_to_output(input_path, output_path))


if __name__ == "__main__":
    reduce_main()
    # diff_images()
